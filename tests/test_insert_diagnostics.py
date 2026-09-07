"""Opt-in insert diagnostics and bounded online-insert cache regressions."""

import os
import time
from uuid import uuid4

import pytest

from tests.test_write_concurrency import (
    STORAGE,
    _assert_recall,
    _connection,
    _create_index,
    _create_table,
    _insert,
    _vectors,
    db_connection_params,  # Reuse the timeout for setup and cleanup connections too.
)


pytestmark = pytest.mark.integration


@pytest.mark.parametrize("layout,labeled", STORAGE)
def test_insert_diagnostics_are_opt_in_and_preserve_results(
    db_setup, clean_db, layout, labeled
):
    vectors = _vectors(24, seed=170)
    rows = [(i, vector, [0, 1 + i % 2], 0) for i, vector in enumerate(vectors)]
    with _connection(db_setup) as conn, conn.cursor() as cur:
        _create_table(cur)
        _create_index(cur, layout, labeled)
        cur.execute(
            "SELECT boot_val, context FROM pg_settings "
            "WHERE name = 'diskann.log_insert_stats'"
        )
        assert cur.fetchone() == ("off", "superuser")
        cur.execute("RESET diskann.log_insert_stats")
        cur.execute("SHOW diskann.log_insert_stats")
        assert cur.fetchone()[0] == "off"
        cur.execute("SET client_min_messages = LOG")
        baseline = None
        for enabled in (False, True, False):
            cur.execute("TRUNCATE test_concurrent")
            cur.execute(f"SET diskann.log_insert_stats = {'on' if enabled else 'off'}")
            conn.notices.clear()
            _insert(cur, rows)
            diagnostics = [
                notice for notice in conn.notices if "diskann insert stats index=" in notice
            ]
            assert len(diagnostics) == (len(rows) if enabled else 0)
            for diagnostic in diagnostics:
                assert "metadata_us:" in diagnostic
                assert "graph_us:" in diagnostic
                assert "cache_entries_capacity_stats=" in diagnostic
            cur.execute("SELECT count(*) FROM test_concurrent")
            assert cur.fetchone()[0] == len(rows)
            assert _assert_recall(cur, vectors[0], limit=32) == set(range(24))
            results = [_assert_recall(cur, vector) for vector in vectors[:3]]
            if labeled:
                assert _assert_recall(cur, vectors[0], labels=[1], limit=32) == set(
                    range(0, 24, 2)
                )
                results.append(_assert_recall(cur, vectors[0], labels=[1]))
            if baseline is None:
                baseline = results
            else:
                assert results == baseline


@pytest.mark.parametrize("layout,labeled", STORAGE)
def test_insert_diagnostics_server_log_privacy(db_setup, clean_db, layout, labeled):
    # Never guess a server path or read historical log contents. Without an
    # explicit path, only exercise client delivery, not server-log privacy.
    log_path = os.environ.get("VECTORSCALE_TEST_LOG_FILE")
    sentinel = f"private_insert_{uuid4().hex}"
    fence = f"insert_diagnostics_end_{uuid4().hex}"
    vector = _vectors(1, seed=171)[0]
    with _connection(db_setup) as conn, conn.cursor() as cur:
        cur.execute("SET log_statement = 'none'")
        cur.execute("SET log_min_duration_statement = -1")
        cur.execute("SET log_min_error_statement = debug5")
        cur.execute("SET log_min_messages = LOG")
        cur.execute("SET client_min_messages = LOG")
        _create_table(cur)
        _create_index(cur, layout, labeled)
        cur.execute("SELECT 'test_concurrent_idx'::regclass::oid")
        diagnostic_prefix = f"diskann insert stats index={cur.fetchone()[0]}:"
        cur.execute("SET diskann.log_insert_stats = on")
        if log_path:
            cur.execute("SELECT size FROM pg_stat_file(%s)", (log_path,))
            offset = cur.fetchone()[0]
        conn.notices.clear()
        cur.execute(
            f"""DO $$ BEGIN
                /* {sentinel} */
                INSERT INTO test_concurrent (id, embedding, labels, bucket)
                VALUES (1, %s::vector, ARRAY[0, 171]::smallint[], 0);
            END $$""",
            (vector,),
        )
        assert sum(diagnostic_prefix in notice for notice in conn.notices) == 1
        if log_path:
            # A subsequent LOG record fences off the complete diagnostic,
            # including any leaked continuation lines, despite collector lag.
            cur.execute("DO $$ BEGIN RAISE LOG %s; END $$", (fence,))
            deadline = time.monotonic() + 5
            while True:
                cur.execute("SELECT size FROM pg_stat_file(%s)", (log_path,))
                size = cur.fetchone()[0]
                assert size >= offset, "Server log was truncated during the test"
                cur.execute(
                    "SELECT pg_read_file(%s, %s, %s)",
                    (log_path, offset, size - offset),
                )
                captured = cur.fetchone()[0]
                if fence in captured:
                    captured = captured.split(fence, 1)[0]
                    break
                assert time.monotonic() < deadline, "Diagnostic log fence did not arrive"
                time.sleep(0.05)
            assert captured.count(diagnostic_prefix) == 1
            assert sentinel not in captured
            assert vector not in captured
            assert "INSERT INTO test_concurrent" not in captured
            assert "CONTEXT:" not in captured
            assert "STATEMENT:" not in captured
            assert "PL/pgSQL function inline_code_block" not in captured
        assert _assert_recall(cur, vector) == {1}
        if labeled:
            assert _assert_recall(cur, vector, labels=[171]) == {1}


@pytest.mark.parametrize("layout,labeled", STORAGE)
def test_online_insert_with_large_maintenance_work_mem(
    db_setup, clean_db, layout, labeled
):
    vectors = _vectors(96, seed=172)
    with _connection(db_setup) as conn, conn.cursor() as cur:
        _create_table(cur)
        # Keep build-time allocation out of this online lazy-cache regression.
        _create_index(cur, layout, labeled, num_neighbors=96, search_list_size=200)
        cur.execute("SET maintenance_work_mem = '1GB'")
        _insert(
            cur,
            [(i, vector, [0, 1 + i % 4], 0) for i, vector in enumerate(vectors)],
        )
        cur.execute("SELECT count(*) FROM test_concurrent")
        assert cur.fetchone()[0] == len(vectors)
        assert _assert_recall(cur, vectors[0], limit=128) == set(range(96))
        for vector in vectors[:3]:
            _assert_recall(cur, vector)
        if labeled:
            for label in range(1, 5):
                assert (
                    _assert_recall(cur, vectors[label - 1], labels=[label], limit=32)
                    == set(range(label - 1, 96, 4))
                )
