use std::num::NonZero;

use pgrx::debug1;

use crate::access_method::storage::Storage;
use crate::util::lru::{CacheStats, LruCacheWithStats};

use crate::{
    access_method::{build::maintenance_work_mem_bytes, stats::StatsNodeRead},
    util::{IndexPointer, ItemPointer},
};

use super::{node::SbqNode, SbqSpeedupStorage, SbqVectorElement};

pub struct QuantizedVectorCache {
    cache: LruCacheWithStats<ItemPointer, Vec<SbqVectorElement>>,
}

impl QuantizedVectorCache {
    pub fn new(memory_budget: f64, sbq_vec_len: usize, min_capacity: usize) -> Self {
        let total_memory = maintenance_work_mem_bytes() as f64;
        let memory_budget = (total_memory * memory_budget).ceil() as usize;
        let capacity = std::cmp::max(memory_budget / Self::entry_size(sbq_vec_len), min_capacity);

        Self {
            cache: LruCacheWithStats::new(NonZero::new(capacity).unwrap(), "Quantized vector"),
        }
    }

    pub fn new_for_insert(memory_budget: f64, sbq_vec_len: usize, min_capacity: usize) -> Self {
        let total_memory = maintenance_work_mem_bytes() as f64;
        let memory_budget = (total_memory * memory_budget).ceil() as usize;
        let capacity = std::cmp::max(memory_budget / Self::entry_size(sbq_vec_len), min_capacity);

        Self {
            cache: LruCacheWithStats::new_lazy(NonZero::new(capacity).unwrap(), "Quantized vector"),
        }
    }

    pub fn remember(&mut self, index_pointer: IndexPointer, vector: Vec<SbqVectorElement>) {
        self.cache.push(index_pointer, vector);
    }

    pub fn stats(&self) -> &CacheStats {
        self.cache.stats()
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn cap(&self) -> NonZero<usize> {
        self.cache.cap()
    }

    /// Estimate of the size of an entry in the cache in bytes.
    pub fn entry_size(sbq_vec_len: usize) -> usize {
        std::mem::size_of::<ItemPointer>()
            + std::mem::size_of::<Vec<SbqVectorElement>>()
            + (std::mem::size_of::<SbqVectorElement>() * sbq_vec_len)
    }

    pub fn get<S: StatsNodeRead>(
        &mut self,
        index_pointer: IndexPointer,
        storage: &SbqSpeedupStorage,
        stats: &mut S,
    ) -> &[SbqVectorElement] {
        self.cache.get_or_insert(index_pointer, || {
            let node = unsafe {
                SbqNode::read(
                    storage.index,
                    index_pointer,
                    storage.get_has_labels(),
                    stats,
                )
            };
            node.get_archived_node().get_bq_vector().to_vec()
        })
    }
}

impl Drop for QuantizedVectorCache {
    fn drop(&mut self) {
        debug1!(
            "Quantized vector cache teardown: capacity {}, stats: {:?}",
            self.cache.cap(),
            self.cache.stats()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_size_accounts_for_quantized_elements() {
        let overhead =
            std::mem::size_of::<ItemPointer>() + std::mem::size_of::<Vec<SbqVectorElement>>();
        assert_eq!(QuantizedVectorCache::entry_size(0), overhead);
        assert_eq!(
            QuantizedVectorCache::entry_size(3),
            overhead + 3 * std::mem::size_of::<SbqVectorElement>()
        );
    }
}
