#!/usr/bin/env python3
"""Benchmark DiskANN inserts in an explicitly chosen, dedicated test database.

From the repository root inside the container (extensions must already exist):
    DB_NAME=vectorscale_bench DB_USER=postgres python3 scripts/benchmark-concurrent-inserts.py

Uses DB_HOST/DB_PORT/DB_USER/DB_PASSWORD like tests/conftest.py, but requires
DB_NAME and rejects maintenance/template databases. Never point at production.
Each trial creates and finally drops a uniquely named logged table in public;
SQL TEMP tables cannot be shared by the writer backends. Requires only the
existing tests/requirements.txt dependencies. WAL deltas are cluster-wide, so
other database activity can inflate them. --min-recall controls acceptance of
mean recall, not automatic tuning; failed trials still emit measurements and
clean up, and the run exits nonzero after all trials. Insert-stat profiling
(--log-insert-stats, requires superuser) changes timings. Optional
--maintenance-work-mem-mb applies to setup and each writer's insertion cache;
omitting it leaves PostgreSQL's configured default unchanged.
Query latency measures the actual ANN SELECT and fetch, excluding EXPLAIN and
the exact baseline; queries run after the baseline and are not cold-cache tests.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, closing
import json
import os
import signal
from threading import Barrier, Event
from time import perf_counter
from uuid import uuid4

import numpy as np
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def bounded_int(lower, upper):
    def parse(value):
        number = int(value)
        if not lower <= number <= upper:
            raise argparse.ArgumentTypeError(f"must be between {lower} and {upper}")
        return number
    return parse


def recall_threshold(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def recall(cur, table, index_name, queries):
    scores = []
    latency_ms = []
    for vector in queries:
        cur.execute("SET enable_seqscan = on; SET enable_indexscan = off; "
                    "SET enable_indexonlyscan = off; SET enable_bitmapscan = off")
        cur.execute(sql.SQL(
            "SELECT id FROM {} ORDER BY (embedding <=> %s::vector) + 0 LIMIT 10"
        ).format(table), (vector,))
        exact = {row[0] for row in cur.fetchall()}
        cur.execute("SET enable_seqscan = off; SET enable_indexscan = on")
        query = sql.SQL(
            "SELECT id FROM {} ORDER BY embedding <=> %s::vector LIMIT 10"
        ).format(table)
        cur.execute(sql.SQL("EXPLAIN (FORMAT JSON) ") + query, (vector,))
        pending = [cur.fetchone()[0][0]["Plan"]]
        found = False
        while pending:
            node = pending.pop()
            found |= (node.get("Index Name") == index_name and "Order By" in node)
            pending.extend(node.get("Plans", []))
        if not found:
            raise RuntimeError("ANN query did not use the benchmark DiskANN index")
        begin = perf_counter()
        cur.execute(query, (vector,))
        rows = cur.fetchall()
        latency_ms.append((perf_counter() - begin) * 1000)
        ann = {row[0] for row in rows}
        scores.append(len(exact & ann) / len(exact))
    return {
        "recall_at_10": float(np.mean(scores)),
        "recall_at_10_min": float(np.min(scores)),
        "query_latency_ms_p50": float(np.percentile(latency_ms, 50)),
        "query_latency_ms_p95": float(np.percentile(latency_ms, 95)),
        "query_latency_ms_p99": float(np.percentile(latency_ms, 99)),
    }


def configure_writer(cur, args):
    if args.maintenance_work_mem_mb is not None:
        cur.execute("SET maintenance_work_mem = %s", (f"{args.maintenance_work_mem_mb}MB",))
    # A fresh backend may not have loaded vectorscale yet; inherited custom
    # settings still exist as placeholders and PostgreSQL parses their booleans.
    cur.execute(
        "SELECT COALESCE(current_setting('diskann.log_insert_stats', true)::boolean, false)"
    )
    if cur.fetchone()[0] != args.log_insert_stats:
        cur.execute("SET diskann.log_insert_stats = %s",
                    ("on" if args.log_insert_stats else "off",))


def trial(params, args, writers, rows, queries):
    name = "bench_inserts_" + uuid4().hex
    table = sql.Identifier("public", name)
    index_name = name + "_idx"
    insert = sql.SQL("INSERT INTO {} (id, embedding) VALUES %s").format(table)
    with closing(psycopg2.connect(**params)) as setup:
        setup.autocommit = True
        with setup.cursor() as cur:
            try:
                if args.maintenance_work_mem_mb is not None:
                    cur.execute("SET maintenance_work_mem = %s",
                                (f"{args.maintenance_work_mem_mb}MB",))
                cur.execute(sql.SQL(
                    "CREATE TABLE {} (id bigint PRIMARY KEY, embedding vector({}) NOT NULL) "
                    "WITH (autovacuum_enabled = false)"
                ).format(table, sql.Literal(args.dimensions)))
                execute_values(cur, insert, rows[:args.initial_rows], page_size=1000)
                cur.execute("SET max_parallel_maintenance_workers = 0; "
                            "SET diskann.force_parallel_workers = 0")
                cur.execute(sql.SQL(
                    "CREATE INDEX {} ON {} USING diskann (embedding vector_cosine_ops) "
                    "WITH (storage_layout = {}, num_neighbors = {}, search_list_size = {})"
                ).format(sql.Identifier(index_name), table, sql.Literal(
                    "memory_optimized" if args.layout == "sbq" else "plain"
                ), sql.Literal(args.num_neighbors), sql.Literal(args.search_list_size)))
                cur.execute(sql.SQL("ANALYZE {}").format(table))

                started = []
                barrier = Barrier(writers + 1, timeout=60,
                                  action=lambda: started.append(perf_counter()))
                stop = Event()

                def writer(conn, work):
                    latencies = []
                    try:
                        with conn.cursor() as writer_cur:
                            configure_writer(writer_cur, args)
                            conn.commit()
                            barrier.wait()
                            for offset in range(0, len(work), args.batch_size):
                                if stop.is_set():
                                    raise RuntimeError("benchmark cancelled")
                                batch = work[offset:offset + args.batch_size]
                                begin = perf_counter()
                                execute_values(writer_cur, insert, batch, page_size=len(batch))
                                conn.commit()
                                finished = perf_counter()
                                latencies.append(finished - begin)
                        return latencies, finished
                    except BaseException:
                        stop.set()
                        barrier.abort()
                        raise

                with ExitStack() as stack:
                    connections = [stack.enter_context(closing(psycopg2.connect(**params)))
                                   for _ in range(writers)]
                    backend_pids = [conn.get_backend_pid() for conn in connections]
                    if len(set(backend_pids)) != writers:
                        raise RuntimeError("writers must use distinct PostgreSQL backends")
                    with ThreadPoolExecutor(max_workers=writers) as pool:
                        futures = []
                        try:
                            for worker, conn in enumerate(connections):
                                first = args.initial_rows + args.rows * worker // writers
                                last = args.initial_rows + args.rows * (worker + 1) // writers
                                futures.append(pool.submit(writer, conn, rows[first:last]))
                            cur.execute("SELECT pg_current_wal_insert_lsn()")
                            wal_start = cur.fetchone()[0]
                            barrier.wait()
                            completed = [future.result() for future in as_completed(futures)]
                        except BaseException:
                            stop.set()
                            barrier.abort()
                            for conn in connections:
                                try:
                                    conn.cancel()
                                except psycopg2.Error:
                                    pass  # Still close every connection after workers exit.
                            raise
                    cur.execute(
                        "SELECT pg_wal_lsn_diff(pg_current_wal_insert_lsn(), %s::pg_lsn)",
                        (wal_start,),
                    )
                    wal_bytes = int(cur.fetchone()[0])

                elapsed = max(end for _, end in completed) - started[0]
                latency_ms = np.array([t for times, _ in completed for t in times]) * 1000
                cur.execute(sql.SQL("SELECT count(*) FROM {}").format(table))
                row_count = cur.fetchone()[0]
                if row_count != args.initial_rows + args.rows:
                    raise RuntimeError(f"unexpected final row count: {row_count}")
                cur.execute(sql.SQL("ANALYZE {}").format(table))
                cur.execute("SET diskann.query_search_list_size = %s; "
                            "SET diskann.query_rescore = %s",
                            (args.query_search_list_size, args.query_rescore))
                return {
                    "writers": writers,
                    "backend_pids": backend_pids,
                    "rows_committed": args.rows,
                    "transactions": len(latency_ms),
                    "elapsed_seconds": elapsed,
                    "rows_per_second": args.rows / elapsed,
                    "transaction_latency_ms_p95": float(np.percentile(latency_ms, 95)),
                    "transaction_latency_ms_p99": float(np.percentile(latency_ms, 99)),
                    "wal_bytes_clusterwide": wal_bytes,
                    **recall(cur, table, index_name, queries),
                }
            finally:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dimensions", type=positive_int, default=128)
    parser.add_argument("--initial-rows", type=positive_int, default=1000)
    parser.add_argument("--rows", type=positive_int, default=2000,
                        help="total inserted rows per trial, not per writer (default: 2000)")
    parser.add_argument("--writers", type=positive_int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--batch-size", type=positive_int, default=1,
                        help="rows per committed transaction (default: 1)")
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--layout", choices=["sbq", "plain"], default="sbq")
    parser.add_argument("--num-neighbors", type=bounded_int(10, 1000), default=30,
                        help="build graph degree, 10..1000 (default: 30)")
    parser.add_argument("--search-list-size", type=bounded_int(10, 1000), default=100,
                        help="build search size, 10..1000 (default: 100)")
    parser.add_argument("--query-search-list-size", type=bounded_int(1, 10000), default=100,
                        help="query search size, 1..10000 (default: 100)")
    parser.add_argument("--query-rescore", type=bounded_int(0, 1000), default=100,
                        help="query rescore count, 0..1000; 0 disables rescoring (default: 100)")
    parser.add_argument("--recall-queries", type=positive_int, default=10,
                        help="held-out queries per trial (default: 10)")
    parser.add_argument("--min-recall", type=recall_threshold,
                        help="minimum mean recall@10, 0..1; acceptance only, no tuning (default: unset)")
    parser.add_argument("--maintenance-work-mem-mb", type=bounded_int(1, 1024),
                        help="setup and per-writer maintenance_work_mem, 1..1024 MB (default: unchanged)")
    parser.add_argument("--log-insert-stats", action="store_true",
                        help="enable diskann.log_insert_stats on writers only; requires superuser, "
                             "changes timings (default: off)")
    args = parser.parse_args()
    database = os.environ.get("DB_NAME", "").strip()
    if not database or database in {"postgres", "template0", "template1"}:
        parser.error("set DB_NAME explicitly to a dedicated test database (not postgres/template*)")
    if max(args.writers) > args.rows:
        parser.error("--writers cannot exceed --rows")
    params = {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": int(os.environ.get("DB_PORT", 5432)),
        "user": os.environ.get("DB_USER", os.environ.get("USER", "postgres")),
        "database": database,
        "password": os.environ.get("DB_PASSWORD", ""),
        "connect_timeout": 10,
        "options": "-c statement_timeout=60s -c synchronous_commit=on -c jit=off",
    }

    def terminate(signum, frame):
        raise KeyboardInterrupt("termination requested")

    signal.signal(signal.SIGTERM, terminate)
    rng = np.random.default_rng(42)
    vectors = ["[" + ",".join(map(str, vector)) + "]" for vector in
               rng.normal(size=(args.initial_rows + args.rows, args.dimensions)).astype(np.float32)]
    queries = ["[" + ",".join(map(str, vector)) + "]" for vector in
               np.random.default_rng(43).normal(
                   size=(args.recall_queries, args.dimensions)).astype(np.float32)]
    rows = list(enumerate(vectors))
    print(json.dumps({
        "type": "config", **vars(args),
        "database": database, "host": params["host"], "port": params["port"],
        "seed": 42, "query_seed": 43, "recall_k": 10,
        "build_parallel_workers": 0,
        "statement_timeout_seconds": 60, "synchronous_commit": "on",
        "wal_scope": "cluster-wide; includes unrelated concurrent activity",
    }), flush=True)
    failed = False
    for writers in args.writers:
        for repeat in range(1, args.repeats + 1):
            result = trial(params, args, writers, rows, queries)
            result["recall_passed"] = (None if args.min_recall is None else
                                       result["recall_at_10"] >= args.min_recall)
            failed |= result["recall_passed"] is False
            print(json.dumps({"type": "result", "repeat": repeat, **result}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
