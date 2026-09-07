use pgrx::{pg_sys::AttrNumber, PgBox, PgRelation};

use crate::{
    access_method::{
        distance::DistanceFn,
        graph::{
            neighbor_store::GraphNeighborStore,
            neighbor_with_distance::{DistanceWithTieBreak, NeighborWithDistance},
            ListSearchNeighbor, ListSearchResult,
        },
        labels::{LabelSet, LabeledVector},
        meta_page::MetaPage,
        node::{ReadableNode, WriteableNode},
        pg_vector::PgVector,
        stats::{
            GreedySearchStats, StatsDistanceComparison, StatsHeapNodeRead, StatsNodeModify,
            StatsNodeRead, StatsNodeWrite,
        },
        storage::{ArchivedData, NodeDistanceMeasure, Storage},
        storage_common::get_index_vector_attribute,
    },
    util::{
        page::PageType, table_slot::TableSlot, tape::Tape, HeapPointer, IndexPointer, ItemPointer,
    },
};

use super::{
    node::{ArchivedPlainNode, PlainNode},
    IndexFullDistanceMeasure, PlainDistanceMeasure,
};

pub struct PlainStorage<'a> {
    pub index: &'a PgRelation,
    pub distance_fn: DistanceFn,
    heap_rel: &'a PgRelation,
    heap_attr: AttrNumber,
    num_neighbors: u32,
}

impl<'a> PlainStorage<'a> {
    pub fn new_for_build(
        index: &'a PgRelation,
        heap_rel: &'a PgRelation,
        meta_page: &MetaPage,
    ) -> PlainStorage<'a> {
        Self {
            index,
            distance_fn: meta_page.get_distance_function(),
            heap_rel,
            heap_attr: get_index_vector_attribute(index),
            num_neighbors: meta_page.get_num_neighbors(),
        }
    }

    pub fn load_for_insert(
        index_relation: &'a PgRelation,
        heap_rel: &'a PgRelation,
        meta_page: &MetaPage,
    ) -> PlainStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: meta_page.get_distance_function(),
            heap_rel,
            heap_attr: get_index_vector_attribute(index_relation),
            num_neighbors: meta_page.get_num_neighbors(),
        }
    }

    pub fn load_for_search(
        index_relation: &'a PgRelation,
        heap_rel: &'a PgRelation,
        meta_page: &MetaPage,
    ) -> PlainStorage<'a> {
        Self {
            index: index_relation,
            distance_fn: meta_page.get_distance_function(),
            heap_rel,
            heap_attr: get_index_vector_attribute(index_relation),
            num_neighbors: meta_page.get_num_neighbors(),
        }
    }
}

//todo move to storage_common
pub struct PlainStorageLsnPrivateData {
    pub heap_pointer: HeapPointer,
    pub neighbors: Vec<ItemPointer>,
}

impl PlainStorageLsnPrivateData {
    pub fn new(node: &ArchivedPlainNode) -> Self {
        let heap_pointer = node.heap_item_pointer.deserialize_item_pointer();
        let neighbors = node.get_index_pointer_to_neighbors();
        Self {
            heap_pointer,
            neighbors,
        }
    }
}

impl Storage for PlainStorage<'_> {
    type QueryDistanceMeasure = PlainDistanceMeasure;
    type NodeDistanceMeasure<'b>
        = IndexFullDistanceMeasure<'b>
    where
        Self: 'b;
    type ArchivedType<'b>
        = ArchivedPlainNode
    where
        Self: 'b;
    type LSNPrivateData = PlainStorageLsnPrivateData;

    fn page_type() -> PageType {
        PageType::Node
    }

    fn create_node<S: StatsNodeWrite>(
        &self,
        full_vector: &[f32],
        _labels: Option<LabelSet>,
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        tape: &mut Tape,
        stats: &mut S,
    ) -> ItemPointer {
        //OPT: avoid the clone?
        let node = PlainNode::new_for_full_vector(full_vector.to_vec(), heap_pointer, meta_page);
        let index_pointer: IndexPointer = node.write(tape, stats);
        index_pointer
    }

    unsafe fn get_node_distance_measure<'b, S: StatsNodeRead + StatsNodeWrite>(
        &'b self,
        index_pointer: IndexPointer,
        stats: &mut S,
    ) -> Self::NodeDistanceMeasure<'b> {
        IndexFullDistanceMeasure::with_index_pointer(self, index_pointer, stats)
    }

    fn get_query_distance_measure(&self, query: LabeledVector) -> PlainDistanceMeasure {
        PlainDistanceMeasure::Full(query)
    }

    fn get_full_distance_for_resort<S: StatsHeapNodeRead + StatsDistanceComparison>(
        &self,
        scan: &PgBox<pgrx::pg_sys::IndexScanDescData>,
        qdm: &Self::QueryDistanceMeasure,
        _index_pointer: IndexPointer,
        heap_pointer: HeapPointer,
        meta_page: &MetaPage,
        stats: &mut S,
    ) -> Option<f32> {
        /* Plain storage only needs to resort when the index is using less dimensions than the underlying data. */
        assert!(meta_page.get_num_dimensions() > meta_page.get_num_dimensions_to_index());

        let slot_opt = unsafe {
            TableSlot::from_index_heap_pointer(self.heap_rel, heap_pointer, scan.xs_snapshot, stats)
        };
        let slot = slot_opt?;
        match qdm {
            PlainDistanceMeasure::Full(query) => {
                let datum = unsafe {
                    slot.get_attribute(self.heap_attr)
                        .expect("vector attribute should exist in the heap")
                };
                let vec = unsafe { PgVector::from_datum(datum, meta_page, false, true) };
                Some(self.get_distance_function()(
                    vec.to_full_slice(),
                    query.vec().to_full_slice(),
                ))
            }
        }
    }

    fn get_neighbors_with_distances_from_disk<
        S: StatsNodeRead + StatsDistanceComparison + StatsNodeWrite + StatsNodeModify,
    >(
        &self,
        neighbors_of: ItemPointer,
        stats: &mut S,
    ) -> Vec<NeighborWithDistance> {
        let rn = unsafe { PlainNode::read(self.index, neighbors_of, stats) };
        // Capture adjacency before the distance state copies the vector and releases rn.
        let neighbors: Vec<_> = rn.get_archived_node().iter_neighbors().collect();
        let dist_state = unsafe { IndexFullDistanceMeasure::with_readable_node(self, rn) };
        neighbors
            .into_iter()
            .map(|n| {
                // TODO: we are reading node twice
                let dist = unsafe { dist_state.get_distance(n, stats) };
                NeighborWithDistance::new(n, DistanceWithTieBreak::new(dist, neighbors_of, n), None)
            })
            .collect()
    }

    /* get_lsn and visit_lsn are different because the distance
    comparisons for SBQ get the vector from different places */
    fn create_lsn_for_start_node(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        index_pointer: ItemPointer,
        gns: &mut GraphNeighborStore,
    ) -> Option<ListSearchNeighbor<Self::LSNPrivateData>> {
        if !lsr.prepare_insert(index_pointer) {
            // Node already processed, skip it
            return None;
        }

        let rn = unsafe { PlainNode::read(self.index, index_pointer, &mut lsr.stats) };
        let node = rn.get_archived_node();

        let distance = match lsr.sdm.as_ref().unwrap() {
            PlainDistanceMeasure::Full(query) => PlainDistanceMeasure::calculate_distance(
                self.distance_fn,
                query.vec().to_index_slice(),
                node.vector.as_slice(),
                &mut lsr.stats,
            ),
        };

        let mut private_data = PlainStorageLsnPrivateData::new(node);
        drop(rn);
        // A cache miss may evict and write a node on the page we just read.
        if let GraphNeighborStore::Builder(b) = gns {
            private_data.neighbors = b.get_neighbors(index_pointer, self, &mut lsr.prune_stats);
        }

        Some(ListSearchNeighbor::new(
            index_pointer,
            lsr.create_distance_with_tie_break(distance, index_pointer),
            private_data,
            None,
        ))
    }

    fn visit_lsn(
        &self,
        lsr: &mut ListSearchResult<Self::QueryDistanceMeasure, Self::LSNPrivateData>,
        lsn_idx: usize,
        gns: &mut GraphNeighborStore,
        no_filter: bool,
    ) {
        assert!(no_filter, "Plain storage does not support label filters");

        let lsn = lsr.get_lsn_by_idx(lsn_idx);
        //clone needed so we don't continue to borrow lsr
        let neighbors = lsn.get_private_data().neighbors.clone();

        for &neighbor_index_pointer in neighbors.iter() {
            if !lsr.prepare_insert(neighbor_index_pointer) {
                continue;
            }

            let rn_neighbor =
                unsafe { PlainNode::read(self.index, neighbor_index_pointer, &mut lsr.stats) };
            let node_neighbor = rn_neighbor.get_archived_node();

            let distance = match lsr.sdm.as_ref().unwrap() {
                PlainDistanceMeasure::Full(query) => PlainDistanceMeasure::calculate_distance(
                    self.distance_fn,
                    query.vec().to_index_slice(),
                    node_neighbor.vector.as_slice(),
                    &mut lsr.stats,
                ),
            };
            let mut private_data = PlainStorageLsnPrivateData::new(node_neighbor);
            drop(rn_neighbor);
            if let GraphNeighborStore::Builder(b) = gns {
                private_data.neighbors =
                    b.get_neighbors(neighbor_index_pointer, self, &mut lsr.prune_stats);
            }
            let lsn = ListSearchNeighbor::new(
                neighbor_index_pointer,
                lsr.create_distance_with_tie_break(distance, neighbor_index_pointer),
                private_data,
                None,
            );

            lsr.insert_neighbor(lsn);
        }
    }

    fn return_lsn(
        &self,
        lsn: &ListSearchNeighbor<Self::LSNPrivateData>,
        _stats: &mut GreedySearchStats,
    ) -> HeapPointer {
        lsn.get_private_data().heap_pointer
    }

    fn try_set_neighbors_on_disk<S: StatsNodeModify + StatsNodeRead>(
        &self,
        index_pointer: IndexPointer,
        expected: &[IndexPointer],
        neighbors: &[NeighborWithDistance],
        stats: &mut S,
    ) -> bool {
        assert!(neighbors.len() <= self.num_neighbors as usize);
        let mut node = unsafe { PlainNode::modify(self.index, index_pointer, stats) };
        let mut archived = node.get_archived_node();
        if !archived.iter_neighbors().eq(expected.iter().copied()) {
            return false;
        }
        if archived.iter_neighbors().eq(neighbors
            .iter()
            .map(NeighborWithDistance::get_index_pointer_to_neighbor))
        {
            stats.record_unchanged();
            return true;
        }
        archived
            .as_mut()
            .set_neighbors(neighbors, self.num_neighbors);
        node.commit();
        true
    }

    fn get_distance_function(&self) -> DistanceFn {
        self.distance_fn
    }

    fn get_labels<S: StatsNodeRead>(
        &self,
        _index_pointer: IndexPointer,
        _stats: &mut S,
    ) -> Option<LabelSet> {
        None
    }

    fn get_has_labels(&self) -> bool {
        false
    }
}
