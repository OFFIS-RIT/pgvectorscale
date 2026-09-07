use std::time::Instant;

pub trait StatsNodeRead {
    fn record_read(&mut self);
}

pub trait StatsHeapNodeRead {
    fn record_heap_read(&mut self);
}

pub trait StatsNodeModify {
    fn record_modify(&mut self);
    fn record_unchanged(&mut self) {}
}

pub trait StatsNodeWrite {
    fn record_write(&mut self);
}

pub trait StatsDistanceComparison {
    fn record_full_distance_comparison(&mut self);
    fn record_quantized_distance_comparison(&mut self);
}

pub trait StatsNodeVisit {
    fn record_visit(&mut self);
    fn record_candidate(&mut self);
}

#[derive(Debug, Default)]
pub struct PruneNeighborStats {
    pub adjacency_attempts: usize,
    pub adjacency_retries: usize,
    pub unchanged_writes: usize,
    pub calls: usize,
    pub distance_comparisons: usize,
    pub node_reads: usize,
    pub node_modify: usize,
    pub node_writes: usize,
    pub num_neighbors_before_prune: usize,
    pub num_neighbors_after_prune: usize,
}

impl StatsDistanceComparison for PruneNeighborStats {
    fn record_full_distance_comparison(&mut self) {
        self.distance_comparisons += 1;
    }

    fn record_quantized_distance_comparison(&mut self) {
        self.distance_comparisons += 1;
    }
}

impl StatsNodeRead for PruneNeighborStats {
    fn record_read(&mut self) {
        self.node_reads += 1;
    }
}

impl StatsNodeModify for PruneNeighborStats {
    fn record_modify(&mut self) {
        self.node_modify += 1;
    }

    fn record_unchanged(&mut self) {
        self.unchanged_writes += 1;
    }
}

impl StatsNodeWrite for PruneNeighborStats {
    fn record_write(&mut self) {
        self.node_writes += 1;
    }
}

#[derive(Debug, Default)]
pub struct GreedySearchStats {
    calls: usize,
    full_distance_comparisons: usize,
    node_reads: usize,
    node_writes: usize,
    node_modify: usize,
    node_heap_reads: usize,
    quantized_distance_comparisons: usize,
    visited_nodes: usize,
    candidate_nodes: usize,
}

impl GreedySearchStats {
    pub fn combine(&mut self, other: &Self) {
        self.calls += other.calls;
        self.full_distance_comparisons += other.full_distance_comparisons;
        self.node_reads += other.node_reads;
        self.node_heap_reads += other.node_heap_reads;
        self.quantized_distance_comparisons += other.quantized_distance_comparisons;
    }

    pub fn get_calls(&self) -> usize {
        self.calls
    }

    pub fn get_node_reads(&self) -> usize {
        self.node_reads
    }

    pub fn get_node_heap_reads(&self) -> usize {
        self.node_heap_reads
    }

    pub fn get_total_distance_comparisons(&self) -> usize {
        self.full_distance_comparisons + self.quantized_distance_comparisons
    }

    pub fn get_quantized_distance_comparisons(&self) -> usize {
        self.quantized_distance_comparisons
    }

    pub fn get_visited_nodes(&self) -> usize {
        self.visited_nodes
    }

    pub fn get_candidate_nodes(&self) -> usize {
        self.candidate_nodes
    }

    pub fn get_full_distance_comparisons(&self) -> usize {
        self.full_distance_comparisons
    }

    pub fn record_call(&mut self) {
        self.calls += 1;
    }
}

impl StatsNodeRead for GreedySearchStats {
    fn record_read(&mut self) {
        self.node_reads += 1;
    }
}

impl StatsHeapNodeRead for GreedySearchStats {
    fn record_heap_read(&mut self) {
        self.node_heap_reads += 1;
    }
}

impl StatsNodeWrite for GreedySearchStats {
    fn record_write(&mut self) {
        self.node_writes += 1;
    }
}

impl StatsNodeModify for GreedySearchStats {
    fn record_modify(&mut self) {
        self.node_modify += 1;
    }
}

impl StatsDistanceComparison for GreedySearchStats {
    fn record_full_distance_comparison(&mut self) {
        self.full_distance_comparisons += 1;
    }

    fn record_quantized_distance_comparison(&mut self) {
        self.quantized_distance_comparisons += 1;
    }
}

impl StatsNodeVisit for GreedySearchStats {
    fn record_visit(&mut self) {
        self.visited_nodes += 1;
    }

    fn record_candidate(&mut self) {
        self.candidate_nodes += 1;
    }
}

#[derive(Debug, Default)]
pub struct QuantizerStats {
    pub node_reads: usize,
    pub node_writes: usize,
}

impl StatsNodeRead for QuantizerStats {
    fn record_read(&mut self) {
        self.node_reads += 1;
    }
}

impl StatsNodeWrite for QuantizerStats {
    fn record_write(&mut self) {
        self.node_writes += 1;
    }
}

#[derive(Debug, Default)]
pub struct InsertStats {
    pub total_us: u128,
    pub metadata_us: u128,
    pub storage_us: u128,
    pub node_write_us: u128,
    pub graph_us: u128,
    pub prune_neighbor_stats: PruneNeighborStats,
    pub greedy_search_stats: GreedySearchStats,
    pub quantizer_stats: QuantizerStats,
    pub node_reads: usize,
    pub node_modify: usize,
    pub node_writes: usize,
}

impl StatsNodeRead for InsertStats {
    fn record_read(&mut self) {
        self.node_reads += 1;
    }
}

impl StatsNodeModify for InsertStats {
    fn record_modify(&mut self) {
        self.node_modify += 1;
    }
}

impl StatsNodeWrite for InsertStats {
    fn record_write(&mut self) {
        self.node_writes += 1;
    }
}

impl InsertStats {
    pub fn log(
        &self,
        index: pgrx::pg_sys::Oid,
        cache: Option<(usize, usize, crate::util::lru::CacheStats)>,
    ) {
        use pgrx::pg_sys;
        let message = std::ffi::CString::new(format!(
            "diskann insert stats index={}: {:?}, cache_entries_capacity_stats={:?}",
            index, self, cache
        ))
        .unwrap();
        // pgrx 0.16 does not expose statement/context suppression on ErrorReport.
        // Keep this LOG-only FFI sequence behind its error boundary, with all
        // Rust-owned allocations outside the trivially-deallocated closure.
        unsafe extern "C-unwind" {
            fn errstart(level: std::ffi::c_int, domain: *const std::ffi::c_char) -> bool;
            fn errhidestmt(hide: bool) -> std::ffi::c_int;
            fn errhidecontext(hide: bool) -> std::ffi::c_int;
            fn errmsg_internal(format: *const std::ffi::c_char, ...) -> std::ffi::c_int;
            fn errfinish(
                file: *const std::ffi::c_char,
                line: std::ffi::c_int,
                function: *const std::ffi::c_char,
            );
        }
        let message_ptr = message.as_ptr();
        unsafe {
            pg_sys::ffi::pg_guard_ffi_boundary(|| {
                if errstart(pg_sys::LOG as _, std::ptr::null()) {
                    errhidestmt(true);
                    errhidecontext(true);
                    errmsg_internal(c"%s".as_ptr(), message_ptr);
                    errfinish(
                        c"vectorscale".as_ptr(),
                        line!() as _,
                        c"insert_stats".as_ptr(),
                    );
                }
            });
        }
    }

    pub fn merge(&mut self, other: &InsertStats) {
        self.total_us += other.total_us;
        self.metadata_us += other.metadata_us;
        self.storage_us += other.storage_us;
        self.node_write_us += other.node_write_us;
        self.graph_us += other.graph_us;
        self.prune_neighbor_stats.adjacency_attempts +=
            other.prune_neighbor_stats.adjacency_attempts;
        self.prune_neighbor_stats.adjacency_retries += other.prune_neighbor_stats.adjacency_retries;
        self.prune_neighbor_stats.unchanged_writes += other.prune_neighbor_stats.unchanged_writes;
        // Merge individual stats
        self.prune_neighbor_stats.calls += other.prune_neighbor_stats.calls;
        self.prune_neighbor_stats.distance_comparisons +=
            other.prune_neighbor_stats.distance_comparisons;
        self.prune_neighbor_stats.node_reads += other.prune_neighbor_stats.node_reads;
        self.prune_neighbor_stats.node_modify += other.prune_neighbor_stats.node_modify;
        self.prune_neighbor_stats.node_writes += other.prune_neighbor_stats.node_writes;
        self.prune_neighbor_stats.num_neighbors_before_prune +=
            other.prune_neighbor_stats.num_neighbors_before_prune;
        self.prune_neighbor_stats.num_neighbors_after_prune +=
            other.prune_neighbor_stats.num_neighbors_after_prune;

        self.greedy_search_stats.combine(&other.greedy_search_stats);

        self.quantizer_stats.node_reads += other.quantizer_stats.node_reads;
        self.quantizer_stats.node_writes += other.quantizer_stats.node_writes;

        self.node_reads += other.node_reads;
        self.node_modify += other.node_modify;
        self.node_writes += other.node_writes;
    }
}

#[derive(Debug)]
pub struct WriteStats {
    pub started: Instant,
    pub num_nodes: usize,
    pub nodes_read: usize,
    pub nodes_modified: usize,
    pub nodes_written: usize,
    pub prune_stats: PruneNeighborStats,
    pub num_neighbors: usize,
}

impl Default for WriteStats {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            num_nodes: 0,
            nodes_read: 0,
            nodes_modified: 0,
            nodes_written: 0,
            prune_stats: PruneNeighborStats::default(),
            num_neighbors: 0,
        }
    }
}

impl StatsNodeRead for WriteStats {
    fn record_read(&mut self) {
        self.nodes_read += 1;
    }
}

impl StatsNodeModify for WriteStats {
    fn record_modify(&mut self) {
        self.nodes_modified += 1;
    }
}

impl StatsNodeWrite for WriteStats {
    fn record_write(&mut self) {
        self.nodes_written += 1;
    }
}
