"""Bounded, multi-backend regressions for DiskANN writes and root publication."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier
import os

import numpy as np
import psycopg2
import pytest
from psycopg2.extensions import TRANSACTION_STATUS_INTRANS
from psycopg2.extras import execute_values


pytestmark = [pytest.mark.concurrency, pytest.mark.integration]

STORAGE = [
    pytest.param("plain", False, id="plain"),
    pytest.param("memory_optimized", False, id="sbq"),
    pytest.param("memory_optimized", True, id="sbq-labels"),
]


@pytest.fixture(scope="session")
def db_connection_params(db_connection_params):
    # Also bound the existing db_setup and clean_db fixture connections.
    return {
        **db_connection_params,
        "connect_timeout": 10,
        "options": (
            db_connection_params.get("options", "")
            + " -c statement_timeout=10s -c jit=off"
        ),
    }


@contextmanager
def _connection(params, autocommit=True):
    conn = psycopg2.connect(**params)
    conn.autocommit = autocommit
    try:
        yield conn
    finally:
        # Unlike psycopg2's transaction context, this closes on every exit path.
        conn.close()


def _vectors(count, seed):
    return [
        "[" + ",".join(map(str, vector)) + "]"
        for vector in np.random.default_rng(seed).normal(size=(count, 16))
    ]


def _create_table(cur):
    cur.execute(
        """
        CREATE TABLE test_concurrent (
            id integer PRIMARY KEY,
            embedding vector(16) NOT NULL,
            labels smallint[] NOT NULL,
            bucket integer NOT NULL DEFAULT 0
        ) WITH (autovacuum_enabled = false)
        """
    )


def _create_index(
    cur, layout, labeled, bucket=None, concurrent=False,
    num_neighbors=30, search_list_size=100,
):
    suffix = "" if bucket is None else f"_{bucket}"
    predicate = "" if bucket is None else f" WHERE bucket = {bucket}"
    columns = "embedding vector_cosine_ops" + (", labels" if labeled else "")
    cur.execute(
        f"CREATE INDEX {'CONCURRENTLY ' if concurrent else ''}"
        f"test_concurrent_idx{suffix} ON test_concurrent "
        f"USING diskann ({columns}) "
        f"WITH (storage_layout = {layout}, num_neighbors = {num_neighbors}, "
        f"search_list_size = {search_list_size}){predicate}"
    )


def _insert(cur, rows):
    execute_values(
        cur,
        "INSERT INTO test_concurrent (id, embedding, labels, bucket) VALUES %s",
        rows,
        page_size=16,
    )


def _scan_ids(cur, vector, labels=None, bucket=None, limit=10, exact=False):
    predicates, params = [], []
    if labels is not None:
        predicates.append("labels && %s::smallint[]")
        params.append(labels)
    if bucket is not None:
        predicates.append("bucket = %s")
        params.append(bucket)
    predicate = " AND ".join(predicates) or "true"
    # The extra arithmetic prevents the exact query from using a KNN path.
    distance = "(embedding <=> %s::vector)" + (" + 0" if exact else "")
    query = (
        f"SELECT id FROM test_concurrent WHERE {predicate} "
        f"ORDER BY {distance} LIMIT %s"
    )
    params.extend([vector, limit])
    cur.execute(
        f"SET enable_seqscan = {'on' if exact else 'off'}; "
        f"SET enable_indexscan = {'off' if exact else 'on'}; "
        "SET enable_indexonlyscan = off; SET enable_bitmapscan = off; "
        "SET diskann.query_search_list_size = 1000; "
        "SET diskann.query_rescore = 1000"
    )
    cur.execute("EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) " + query, params)
    plan = cur.fetchone()[0][0]["Plan"]
    pending, scans = [plan], []
    while pending:
        node = pending.pop()
        if node["Node Type"] == ("Seq Scan" if exact else "Index Scan"):
            scans.append(node)
        pending.extend(node.get("Plans", []))
    assert len(scans) == 1, plan
    assert scans[0]["Actual Loops"] > 0, plan
    if not exact:
        suffix = "" if bucket is None else f"_{bucket}"
        assert scans[0]["Index Name"] == f"test_concurrent_idx{suffix}", plan
        assert "Order By" in scans[0], plan
        if labels is not None:
            assert "labels" in scans[0].get("Index Cond", ""), plan
    cur.execute(query, params)
    ids = [row[0] for row in cur.fetchall()]
    assert len(ids) == len(set(ids)), ids
    return set(ids)


def _assert_recall(cur, vector, **kwargs):
    exact = _scan_ids(cur, vector, exact=True, **kwargs)
    indexed = _scan_ids(cur, vector, **kwargs)
    assert len(indexed) == len(exact), (indexed, exact, kwargs)
    if exact:
        recall = len(indexed & exact) / len(exact)
        assert recall >= 0.9, (recall, indexed, exact, kwargs)
    return indexed


@pytest.mark.parametrize("layout,labeled", STORAGE)
@pytest.mark.parametrize("empty", [True, False], ids=["empty", "built"])
@pytest.mark.parametrize("outcome", ["commit", "rollback"])
def test_insert_finishes_before_other_transaction_ends(
    db_setup, clean_db, layout, labeled, empty, outcome
):
    vectors = _vectors(66, seed=110)
    with _connection(db_setup) as setup, setup.cursor() as cur:
        _create_table(cur)
        if not empty:
            _insert(cur, [(i, vectors[i], [0], 0) for i in range(64)])
        _create_index(cur, layout, labeled)

    with (
        _connection(db_setup, autocommit=False) as first,
        _connection(db_setup) as second,
    ):
        assert first.get_backend_pid() != second.get_backend_pid()
        with first.cursor() as cur:
            _insert(cur, [(1001, vectors[64], [0, 101], 0)])
        with second.cursor() as cur:
            # A transaction-scoped writer lock times out here. Do not release
            # first until second has both committed and searched its own row.
            _insert(cur, [(1002, vectors[65], [0, 102], 0)])
            assert first.get_transaction_status() == TRANSACTION_STATUS_INTRANS
            expected = set() if empty else set(range(64))
            expected.add(1002)
            assert _assert_recall(cur, vectors[65], limit=128) == expected
            if labeled:
                assert _assert_recall(cur, vectors[65], labels=[102]) == {1002}
                assert _assert_recall(cur, vectors[64], labels=[101]) == set()
            assert first.get_transaction_status() == TRANSACTION_STATUS_INTRANS
            if outcome == "commit":
                first.commit()
                expected.add(1001)
            else:
                first.rollback()
            assert _assert_recall(cur, vectors[64], limit=128) == expected
            if labeled:
                visible = {1001} if outcome == "commit" else set()
                assert _assert_recall(cur, vectors[64], labels=[101]) == visible
            cur.execute("VACUUM test_concurrent")
            assert _assert_recall(cur, vectors[65], limit=128) == expected


@pytest.mark.parametrize("layout,labeled", STORAGE)
def test_opposite_index_order_in_open_transactions(db_setup, clean_db, layout, labeled):
    vectors = _vectors(4, seed=120)
    with _connection(db_setup) as setup, setup.cursor() as cur:
        _create_table(cur)
        for bucket in (0, 1):
            _create_index(cur, layout, labeled, bucket=bucket)

    with (
        _connection(db_setup, autocommit=False) as first,
        _connection(db_setup, autocommit=False) as second,
    ):
        with first.cursor() as a, second.cursor() as b:
            # Partial indexes let two backends visit A->B and B->A without
            # introducing extra tables outside clean_db's cleanup list.
            _insert(a, [(1, vectors[0], [0], 0)])
            _insert(b, [(2, vectors[1], [0], 1)])
            _insert(a, [(3, vectors[2], [0], 1)])
            _insert(b, [(4, vectors[3], [0], 0)])
            assert first.get_transaction_status() == TRANSACTION_STATUS_INTRANS
            assert second.get_transaction_status() == TRANSACTION_STATUS_INTRANS
        first.commit()
        second.commit()

    with _connection(db_setup) as conn, conn.cursor() as cur:
        for bucket, expected in [(0, {1, 4}), (1, {2, 3})]:
            assert _assert_recall(cur, vectors[bucket], bucket=bucket) == expected
            if labeled:
                assert (
                    _assert_recall(cur, vectors[bucket], bucket=bucket, labels=[0])
                    == expected
                )


@pytest.mark.parametrize("layout,labeled", STORAGE)
@pytest.mark.parametrize("empty", [True, False], ids=["empty", "built"])
def test_concurrent_startup_and_new_labels(db_setup, clean_db, layout, labeled, empty):
    vectors = _vectors(256, seed=130)
    with _connection(db_setup) as setup, setup.cursor() as cur:
        _create_table(cur)
        if not empty:
            _insert(cur, [(i, vectors[i], [0], 0) for i in range(64)])
        # Isolate update/publication correctness from ANN pruning: this test
        # requires every row, so all 256 nodes must fit the construction budget.
        # Shared-label filtered reachability can fall short even with serialized
        # writers at degree 30. Other tests retain that bounded-degree workload.
        _create_index(cur, layout, labeled, num_neighbors=256, search_list_size=1000)

    barrier = Barrier(4, timeout=15)

    def writer(worker):
        with _connection(db_setup) as conn, conn.cursor() as cur:
            for batch in range(3):
                start = 64 + worker * 48 + batch * 16
                label = 1 + worker * 3 + batch
                barrier.wait()
                _insert(
                    cur,
                    [(i, vectors[i], [0, label], 0) for i in range(start, start + 16)],
                )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(writer, range(4)))

    with _connection(db_setup) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM test_concurrent")
        assert cur.fetchone()[0] == (192 if empty else 256)
        for vector in _vectors(4, seed=131):
            _assert_recall(cur, vector)
        if labeled:
            for label in range(1, 13):
                start = 64 + (label - 1) * 16
                assert (
                    _assert_recall(cur, vectors[start], labels=[label], limit=16)
                    == set(range(start, start + 16))
                )


@pytest.mark.parametrize(
    "build_first", [False, True], ids=["online-growth", "built-metadata"]
)
def test_multi_page_label_metadata(db_setup, clean_db, build_first):
    vectors = _vectors(182, seed=140)
    # 1500 distinct smallint label -> ItemPointer entries exceed an 8KB page.
    # Ten labels per vector exercise the metadata chain with only 150 nodes.
    rows = [
        (i, vectors[i], [0] + list(range(1 + i * 10, 11 + i * 10)), 0)
        for i in range(182)
    ]
    with _connection(db_setup) as setup, setup.cursor() as cur:
        _create_table(cur)
        if build_first:
            _insert(cur, rows[:150])
        _create_index(cur, "memory_optimized", True)

    barrier = Barrier(4, timeout=15)

    def writer(worker):
        with _connection(db_setup) as conn, conn.cursor() as cur:
            if not build_first:
                barrier.wait()
                _insert(cur, rows[worker:150:4])
            # In both modes, modify an already multi-page metadata chain.
            barrier.wait()
            _insert(cur, rows[150 + worker::4])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(writer, range(4)))

    with _connection(db_setup) as conn, conn.cursor() as cur:
        for i, vector, labels, _ in rows:
            for label in (labels[1], labels[-1]):
                assert _assert_recall(cur, vector, labels=[label]) == {i}
        for vector in _vectors(4, seed=141):
            _assert_recall(cur, vector)


@pytest.mark.parametrize("layout,labeled", STORAGE)
def test_readers_with_label_writers_and_vacuum(db_setup, clean_db, layout, labeled):
    vectors = _vectors(256, seed=150)
    queries = _vectors(4, seed=151)
    with _connection(db_setup) as setup, setup.cursor() as cur:
        _create_table(cur)
        _insert(cur, [(i, vectors[i], [0], 0) for i in range(128)])
        _create_index(cur, layout, labeled)

    barrier = Barrier(5, timeout=15)

    def worker(role):
        with _connection(db_setup, autocommit=role not in (2, 3)) as conn:
            if role in (2, 3):
                conn.set_session(isolation_level="REPEATABLE READ", readonly=True)
            for batch in range(4):
                barrier.wait()
                if role < 2:
                    start = 128 + role * 64 + batch * 16
                    label = 1 + role * 4 + batch
                    with conn.cursor() as cur:
                        _insert(
                            cur,
                            [(i, vectors[i], [0, label], 0) for i in range(start, start + 16)],
                        )
                elif role < 4:
                    # Exact and ANN queries must share a snapshot while other
                    # sessions commit inserts/deletes between these statements.
                    with conn, conn.cursor() as cur:
                        _assert_recall(cur, queries[batch])
                        if labeled and batch:
                            for label in (batch, batch + 4):
                                assert (
                                    len(_assert_recall(cur, queries[batch], labels=[label]))
                                    == 10
                                )
                else:
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM test_concurrent WHERE id >= %s AND id < %s",
                            (batch * 8, (batch + 1) * 8),
                        )
                        cur.execute("VACUUM test_concurrent")

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(worker, range(5)))

    with _connection(db_setup) as conn, conn.cursor() as cur:
        cur.execute("VACUUM test_concurrent")
        cur.execute("SELECT count(*) FROM test_concurrent")
        assert cur.fetchone()[0] == 224
        for vector in queries:
            _assert_recall(cur, vector)
        if labeled:
            for label in range(1, 9):
                assert len(_assert_recall(cur, queries[0], labels=[label])) == 10


@pytest.mark.parametrize(
    "layout,labeled,workers,concurrent",
    [
        pytest.param("plain", False, 0, False, id="plain-serial"),
        pytest.param("memory_optimized", False, 0, False, id="sbq-serial"),
        pytest.param("memory_optimized", True, 0, False, id="sbq-labels-serial"),
        pytest.param("memory_optimized", False, 2, False, id="sbq-parallel"),
        pytest.param("memory_optimized", False, 2, True, id="sbq-parallel-concurrent"),
    ],
)
def test_final_build_cache_reconciliation(
    db_setup, clean_db, layout, labeled, workers, concurrent
):
    if workers and os.environ.get("VECTORSCALE_TEST_PARALLEL_BUILD", "1") == "0":
        pytest.skip("parallel build feature disabled for this test installation")
    vectors = _vectors(512, seed=160)
    with _connection(db_setup) as conn, conn.cursor() as cur:
        _create_table(cur)
        _insert(cur, [(i, vector, [0, 1 + i % 8], 0) for i, vector in enumerate(vectors)])
        cur.execute("ANALYZE test_concurrent")
        cur.execute(f"SET diskann.force_parallel_workers = {workers}")
        cur.execute("SET diskann.parallel_initial_start_nodes_count = 16")
        # Retain worker-local updates until the final reconciliation.
        cur.execute("SET diskann.parallel_flush_interval = 1.0")
        conn.notices.clear()
        _create_index(cur, layout, labeled, concurrent=concurrent)
        if workers:
            assert any("Parallel build with 2 workers" in notice for notice in conn.notices)
            assert not any("No workers launched" in notice for notice in conn.notices)

    # A new backend cannot accidentally use any builder-local cache.
    with _connection(db_setup) as conn, conn.cursor() as cur:
        for vector in _vectors(8, seed=161):
            _assert_recall(cur, vector)
            if labeled:
                for label in range(1, 9):
                    _assert_recall(cur, vector, labels=[label])
