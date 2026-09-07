# Concurrent DiskANN Writes

DiskANN insertions no longer take a per-index transaction-level advisory lock.
Graph search and distance calculations can run concurrently across PostgreSQL
backends. This does not eliminate contention on hot graph nodes, physical pages,
relation extension, CPU, storage, or WAL.

## Synchronization

- Each disk adjacency update reads an owned neighbor snapshot, merges candidates,
  and prunes outside the exclusive buffer lock. Under the exclusive lock it
  compares the complete ordered adjacency with the snapshot. A mismatch retries
  the entire operation with the original additions and the latest disk state.
- Only neighbor fields are replaced. Vectors, labels, and VACUUM's heap-TID
  tombstones are preserved. Graph nodes are not physically recycled; changing
  that invariant would require revisiting snapshot validation and reclamation.
- Default and label entrypoints only grow. An operation-scoped PostgreSQL page
  lock on block zero coordinates metadata read/merge/publication. Existing roots
  take the fast path without acquiring this heavyweight lock. Error cleanup is
  handled by PostgreSQL; successful publication explicitly releases the lock.
- A metadata replacement changes the root in one Generic WAL operation. Any
  overflow suffix is completed before publishing its pointer. Readers retain a
  shared root buffer lock while reading the chain. Old and abandoned suffixes
  are not reclaimed, as with the previous storage scheme.
- Buffer locks and relation-extension locks remain in place. Physical scans
  skip uninitialized pages left by interrupted allocation, but still validate
  initialized pages normally.
- Builder eviction, flush, and final draining use the same validated adjacency
  merge. Parallel workers do not overwrite metadata with their local snapshots.
  Initialization publishes seed edges before waking the other workers. Parallel
  label builds remain disabled.

Adjacency validation prevents accidental stale replacement, not intentional
edge removal by pruning. ANN recall and graph connectivity still require
workload-specific validation. Retries check PostgreSQL interrupts; there is no
guarantee of starvation-free progress under unlimited contention.

## Deployment

The current version-3 page layout is unchanged; valid existing indexes do not
need rebuilding solely for this change. Legacy metadata reads remain read-only,
and explicit updates use the current metadata representation.

**Do not run old and new writer binaries against the same index concurrently.**
Old writers do not participate in adjacency validation or the new metadata
protocol. Drain writers and index builds, deploy the binary, and restart
PostgreSQL (or otherwise replace every backend that could retain the old
library) before admitting writes. Validate the deployment on a disposable copy
first. This change does not repair indexes already damaged by older races.

## Verification

Rust tests cover adjacency comparison for Plain, SBQ, and labeled SBQ, stale
cache writeback, same-page lock scopes, metadata page boundaries, legacy reads,
aborting prepared root publication, and abandoned page allocation.

The Python regressions in `tests/test_write_concurrency.py` cover overlapping
transactions including rollback, opposite index order, concurrent startup and
label growth, multi-page metadata, concurrent readers and VACUUM, and parallel
build finalization. Recall checks compare against exact scans and verify the
executed DiskANN query plan. Run against a disposable database only:

```bash
DB_NAME=vectorscale_test DB_USER=postgres \
  python -m pytest tests/test_write_concurrency.py -v
```

The optional benchmark uses deterministic data, fixed total work, independent
backends, committed transactions, and synchronous commit. It reports throughput,
transaction latency, cluster-wide WAL volume, and recall without imposing a
performance threshold:

```bash
DB_NAME=vectorscale_bench DB_USER=postgres \
  python scripts/benchmark-concurrent-inserts.py \
  --dimensions 128 --initial-rows 1000 --rows 2000 \
  --writers 1 2 4 8 --batch-size 10 --repeats 3
```

Use the same release build, resources, data size, transaction size, and search
settings for comparisons. A small resident index is not representative of a
terabyte-scale workload. Production qualification should include sustained
hotspot-heavy ingestion, cold-cache/high-dimensional data, crash recovery,
physical replication, cancellation, and recall at the required query latency.

## Local Results

Validation used PostgreSQL 18.6, pgvector 0.8.1, pgrx 0.16.1, Rust 1.91.1, and
an isolated container limited to eight CPUs and 16 GiB RAM:

- Rust release tests: 98 passed, 11 ignored, with `build_parallel` both enabled
  and disabled. Historical upgrade tests remain ignored.
- Python tests: 42 passed. The 31 new concurrency cases also passed five
  consecutive runs.
- An immediate PostgreSQL shutdown/restart smoke test recovered 512 committed
  labeled vectors and retained traversal through 16 aborted nodes without
  returning their heap rows. Multi-page metadata and subsequent VACUUM worked.
  This is not exhaustive crash injection at every write boundary or replication
  qualification.

The benchmark command above, with two repetitions, produced these mean insert
rates. The serialized control used the new adjacency implementation with the
transaction advisory lock still enabled; it was not an untouched upstream build.

| Writers | Serialized Control (vectors/s) | Concurrent (vectors/s) |
| --- | ---: | ---: |
| 1 | 640 | 673 |
| 2 | 746 | 1,142 |
| 4 | 745 | 1,760 |
| 8 | 731 | 3,194 |

These were small resident indexes, 128-dimensional SBQ vectors, 1,000 initial
rows plus 2,000 inserts, and ten rows per committed transaction. Recall@10 was
0.71-0.73 in both groups at the fixed query settings; these settings are not a
high-recall production recommendation. Eight-writer throughput was about 4.4x
the serialized control, not a general scaling guarantee.

A shared-label fixture at degree 30 occasionally returned 15 of 16 matching
rows even with writers explicitly serialized. A large query search budget
cannot recover nodes unreachable in the filtered graph. Exact-cardinality
publication tests therefore use a no-pruning construction budget; other tests
retain bounded-degree pruning and recall checks. This existing algorithmic
limitation needs separate evaluation for workloads with overlapping labels.

## Insert-Path Optimization

A subsequent performance pass retains the synchronization and durability rules
above, with the following changes:

- A validated, unchanged adjacency returns without committing a page update.
  Snapshot validation still occurs under the exclusive lock. Tests verify both
  page contents/LSN and backend-local WAL position stay unchanged.
- SBQ traversal releases its source page after copying neighbor pointers.
  VACUUM delays occur outside buffer locks; cleanup locking and revalidation
  remain in place for node modifications.
- Online SBQ vector caches allocate lazily up to their existing entry limit,
  rather than reserving a maintenance-sized hash table for every row. Build
  caches retain eager allocation. This is an estimated memory budget, not a
  strict RSS/byte limit; hash-table and allocator overhead still apply.
- Cache lookups use one probe and count one hit/miss. Node creation and graph
  traversal populate immutable vector payloads for reuse during pruning. Classic
  SBQ adjacency-distance calculation also reuses these payloads. Labeled nodes
  continue reading the labels required for pruning.
- Cache sizing uses indexed dimensions and rounds up to complete quantized
  words. No vector arithmetic, pruning rules, persisted layout, heap visibility,
  or WAL durability settings are changed.

All online caches remain local to one insertion. No borrowed relation, mutable
adjacency, or heap visibility state is retained across calls. Cross-row quantizer
caching was not added: sampled metadata/storage setup was small relative to graph
work, so its additional invalidation/cleanup machinery was not justified.

### Diagnostics

`diskann.log_insert_stats` is a superuser-settable, default-off diagnostic GUC.
When enabled, it emits one record per non-skipped inserted vector containing
timings, adjacency attempts/retries, unchanged writes, distance/read counters,
and SBQ cache occupancy/counters. It does not aggregate across backends.

`total_us` includes local cache/metadata teardown but excludes the diagnostic
record itself and the SQL transaction commit. `storage_us` includes datum
preparation. Stage times include waits and are not a CPU profiler. Timers and
record formatting are disabled when the setting is off.

The records explicitly suppress attached SQL statements and function context.
Independent PostgreSQL statement/audit logging is not disabled or overridden.
Enable diagnostics only for bounded investigations; logging affects timings and
can generate significant output.

### Further Validation

PostgreSQL 18 release tests passed with parallel builds both on and off:
105 passed, 11 ignored in each configuration. All 69 Python tests passed,
including actual server-log privacy assertions. The 40 concurrency/diagnostic
cases passed three additional consecutive runs. Cache tests include bounded
eviction, initialization-panic recovery, and exactly-once owned-value cleanup.
The immediate-shutdown recovery smoke test above also passed again.
Python CI now supplies the fixture's actual `DB_*` settings and the server-log
path, and includes a PostgreSQL 18 parallel-build job. For an installation built
without `build_parallel`, set `VECTORSCALE_TEST_PARALLEL_BUILD=0` to skip only the
two forced parallel-build cases; the other write-concurrency tests still run.

Screening runs used the same eight-CPU container and synchronous commit, with
three repetitions per configuration. The pre-optimization baseline already had
concurrent writes; these are not comparisons against advisory serialization:

| Dimensions | Initial / Inserted Rows | Writers | Before (vectors/s) | After (vectors/s) |
| --- | --- | ---: | ---: | ---: |
| 128 | 2,000 / 4,000 | 1 | 226 | 273 |
| 128 | 2,000 / 4,000 | 8 | 533 | 1,985 |
| 1,536 | 1,000 / 2,000 | 1 | 162 | 430 |
| 1,536 | 1,000 / 2,000 | 8 | 659 | 2,253 |

These are medians of short resident-index trials. Shared-host timing varied
materially between runs; do not interpret the ratios as isolated causal effects
or production guarantees. WAL volume in the 128-dimensional screening workload
fell from about 19.0 MB to 16.7 MB per 4,000 inserts. Fixed low query budgets still
had low recall, so separate checks increased the query budget rather than
presenting those low-recall settings as production recommendations.

With query search/rescore both 1,000, 30 held-out queries, and a 0.95 mean-recall
gate, both repetitions of each configuration passed. Eight-writer observations:

| Dimensions | Insert Throughput | Mean Recall@10 | ANN Query p95 |
| --- | --- | --- | --- |
| 128 | 3,237-3,346 vectors/s | 97.3% | 11.8-11.9 ms |
| 1,536 | 2,675-2,738 vectors/s | 96.7% | 23.2-24.2 ms |

Query latency was measured after ingestion, not during concurrent writes. The
worst individual query recall was 90% and 80%, respectively: a mean-recall gate
is not a per-query guarantee. The different row counts also prevent treating
this table as a direct dimensionality comparison.

To reproduce the 1,536-dimensional high-recall check:

```bash
DB_NAME=vectorscale_bench DB_USER=postgres \
  python scripts/benchmark-concurrent-inserts.py \
  --dimensions 1536 --initial-rows 1000 --rows 2000 \
  --writers 1 8 --batch-size 10 --repeats 2 \
  --query-search-list-size 1000 --query-rescore 1000 \
  --recall-queries 30 --min-recall 0.95
```

The benchmark reports every trial and exits nonzero if any trial misses the
requested mean-recall gate. `--log-insert-stats` enables diagnostics for writers;
without it, inherited profiling must be off or successfully disabled before
timing starts. `--maintenance-work-mem-mb` controls the setup and writer setting
for cache-budget investigations. Neither flag should be mixed into an otherwise
unprofiled comparison without recording the difference.
