use std::cell::RefCell;
use std::num::NonZero;

use pgrx::debug1;

use crate::access_method::build::maintenance_work_mem_bytes;
use crate::util::lru::LruCacheWithStats;
use crate::util::{IndexPointer, ItemPointer};

use crate::access_method::graph::neighbor_with_distance::*;
use crate::access_method::labels::LabelSet;
use crate::access_method::meta_page::MetaPage;
use crate::access_method::stats::PruneNeighborStats;
use crate::access_method::storage::Storage;

use super::Graph;

#[cfg(any(test, feature = "pg_test"))]
#[path = "neighbor_store_tests.rs"]
mod tests;

/// Preserve candidate ordering and pruning policy at each writeback site.
#[derive(Clone, Copy)]
pub(crate) enum NeighborMergeMode {
    Disk,
    Builder { always_prune: bool },
}

/// A builderGraph is a graph that keep the neighbors in-memory in the neighbor_map below
/// The idea is that during the index build, you don't want to update the actual Postgres
/// pages every time you change the neighbors. Instead you change the neighbors in memory
/// until the build is done. Afterwards, calling the `write` method, will write out all
/// the neighbors to the right pages.
///
pub struct NeighborCacheEntry {
    pub labels: Option<LabelSet>,
    pub neighbors: Vec<NeighborWithDistance>,
}

impl NeighborCacheEntry {
    pub fn new(labels: Option<LabelSet>, neighbors: Vec<NeighborWithDistance>) -> Self {
        Self { labels, neighbors }
    }

    /// Estimate of the size of an entry in the cache in bytes.
    pub fn size(num_neighbors: usize, has_labels: bool) -> usize {
        std::mem::size_of::<Self>()
            + num_neighbors * std::mem::size_of::<NeighborWithDistance>()
            + if has_labels {
                // Heuristic: assume around 4 labels per vector, or 8 bytes of payload
                8
            } else {
                0
            }
    }
}

pub struct BuilderNeighborCache {
    /// Map of node pointer to neighbor cache entry
    neighbor_map: RefCell<LruCacheWithStats<ItemPointer, NeighborCacheEntry>>,
    num_neighbors: usize,
    max_alpha: f64,
}

impl BuilderNeighborCache {
    pub fn new(memory_budget: f64, meta_page: &MetaPage, worker_count: usize) -> Self {
        let total_memory = maintenance_work_mem_bytes() as f64;
        let memory_budget = (total_memory * memory_budget).ceil() as usize;
        let capacity = memory_budget
            / NeighborCacheEntry::size(meta_page.get_num_neighbors() as _, meta_page.has_labels());
        let capacity = if worker_count > 0 {
            capacity / worker_count
        } else {
            capacity
        };

        Self {
            neighbor_map: RefCell::new(LruCacheWithStats::new(
                NonZero::new(capacity.max(1)).unwrap(),
                "Builder neighbor",
            )),
            num_neighbors: meta_page.get_num_neighbors() as _,
            max_alpha: meta_page.get_max_alpha(),
        }
    }

    /// Convert cache to a sorted vector of neighbors
    fn into_sorted(self) -> Vec<(ItemPointer, NeighborCacheEntry)> {
        let (neighbor_map, stats) = self.neighbor_map.into_inner().into_parts();
        debug1!(
            "Builder neighbor cache teardown: capacity {}, stats: {:?}",
            neighbor_map.cap(),
            stats
        );
        let mut vec = neighbor_map.into_iter().collect::<Vec<_>>();
        vec.sort_by_key(|(key, _)| *key);
        vec
    }

    pub fn get_neighbors<S: Storage>(
        &self,
        neighbors_of: ItemPointer,
        storage: &S,
        stats: &mut PruneNeighborStats,
    ) -> Vec<IndexPointer> {
        let neighbors = self.get_neighbors_with_full_vector_distances(neighbors_of, storage, stats);
        neighbors
            .iter()
            .map(|n| n.get_index_pointer_to_neighbor())
            .collect()
    }

    pub fn get_neighbors_with_full_vector_distances<S: Storage>(
        &self,
        neighbors_of: ItemPointer,
        storage: &S,
        stats: &mut PruneNeighborStats,
    ) -> Vec<NeighborWithDistance> {
        let mut neighbor_map = self.neighbor_map.borrow_mut();
        let neighbors = neighbor_map.get(&neighbors_of);
        if let Some(entry) = neighbors {
            return entry.neighbors.clone();
        }
        drop(neighbor_map);
        let neighbors = storage.get_neighbors_with_distances_from_disk(neighbors_of, stats);

        self.set_neighbors(neighbors_of, None, neighbors.clone(), storage, stats);
        neighbors
    }

    pub fn set_neighbors<S: Storage>(
        &self,
        neighbors_of: ItemPointer,
        labels: Option<LabelSet>,
        mut new_neighbors: Vec<NeighborWithDistance>,
        storage: &S,
        stats: &mut PruneNeighborStats,
    ) {
        // Cache fills must retain the source labels used by label-aware pruning.
        let labels = labels.or_else(|| storage.get_labels(neighbors_of, stats));
        new_neighbors.shrink_to_fit();
        let evictee = {
            let mut neighbor_map = self.neighbor_map.borrow_mut();
            neighbor_map.push(neighbors_of, NeighborCacheEntry::new(labels, new_neighbors))
        };
        if let Some((key, value)) = evictee {
            Graph::merge_neighbors_on_disk(
                storage,
                key,
                value.labels.as_ref(),
                &value.neighbors,
                self.max_alpha,
                self.num_neighbors,
                NeighborMergeMode::Builder { always_prune: true },
                stats,
            );
        }
    }

    /// Reconcile and flush all cached entries, publishing changes to other workers.
    pub fn flush_neighbor_cache<S: Storage>(&self, storage: &S, stats: &mut PruneNeighborStats) {
        loop {
            let entry = self.neighbor_map.borrow_mut().pop_lru();
            let Some((neighbors_of, entry)) = entry else {
                break;
            };
            Graph::merge_neighbors_on_disk(
                storage,
                neighbors_of,
                entry.labels.as_ref(),
                &entry.neighbors,
                self.max_alpha,
                self.num_neighbors,
                NeighborMergeMode::Builder {
                    always_prune: false,
                },
                stats,
            );
        }
    }
}

pub enum GraphNeighborStore {
    Builder(BuilderNeighborCache),
    Disk,
}

impl GraphNeighborStore {
    pub fn get_neighbors_with_full_vector_distances<S: Storage>(
        &self,
        neighbors_of: ItemPointer,
        storage: &S,
        stats: &mut PruneNeighborStats,
    ) -> Vec<NeighborWithDistance> {
        match self {
            GraphNeighborStore::Builder(b) => {
                b.get_neighbors_with_full_vector_distances(neighbors_of, storage, stats)
            }
            GraphNeighborStore::Disk => {
                storage.get_neighbors_with_distances_from_disk(neighbors_of, stats)
            }
        }
    }

    /// Newly created disk nodes already have empty adjacency. Only the builder
    /// needs a cache entry before the node is published as a start node.
    pub fn initialize_neighbors<S: Storage>(
        &self,
        storage: &S,
        neighbors_of: ItemPointer,
        labels: Option<LabelSet>,
        stats: &mut PruneNeighborStats,
    ) {
        match self {
            GraphNeighborStore::Builder(b) => {
                b.set_neighbors(neighbors_of, labels, Vec::new(), storage, stats)
            }
            GraphNeighborStore::Disk => {}
        }
    }

    pub fn max_neighbors(&self, meta_page: &MetaPage) -> usize {
        match self {
            GraphNeighborStore::Builder(_) => meta_page.get_max_neighbors_during_build(),
            GraphNeighborStore::Disk => meta_page.get_num_neighbors() as _,
        }
    }

    pub fn into_sorted(self) -> Vec<(ItemPointer, NeighborCacheEntry)> {
        match self {
            GraphNeighborStore::Builder(b) => b.into_sorted(),
            GraphNeighborStore::Disk => {
                panic!("Should not be using the disk neighbor store during build")
            }
        }
    }
}
