"""CLI-only benchmark tests: no PostgreSQL connections or extension required."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark-concurrent-inserts.py"
    spec = importlib.util.spec_from_file_location("benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("DB_NAME", "vectorscale_bench")
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)

    def no_database(*args, **kwargs):
        pytest.fail("CLI tests must not connect to PostgreSQL")

    monkeypatch.setattr(module.psycopg2, "connect", no_database)
    return module


def test_help(benchmark, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--help"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 0
    assert "--maintenance-work-mem-mb" in capsys.readouterr().out


@pytest.mark.parametrize("inherited,requested", [(False, False), (False, True), (True, False), (True, True)])
def test_writer_logging_matches_reported_setting(benchmark, inherited, requested):
    class Cursor:
        statements = []

        def execute(self, query, args=None):
            self.statements.append((query, args))

        def fetchone(self):
            return (inherited,)

    cur = Cursor()
    benchmark.configure_writer(cur, SimpleNamespace(
        maintenance_work_mem_mb=None, log_insert_stats=requested,
    ))
    changes = [(query, args) for query, args in cur.statements if query.startswith("SET")]
    expected = [] if inherited == requested else [
        ("SET diskann.log_insert_stats = %s", ("on" if requested else "off",))
    ]
    assert changes == expected


@pytest.mark.parametrize("flag,values", [
    ("--num-neighbors", ["9", "1001"]),
    ("--search-list-size", ["9", "1001"]),
    ("--query-search-list-size", ["0", "10001"]),
    ("--query-rescore", ["-1", "1001"]),
    ("--recall-queries", ["0", "-1"]),
    ("--min-recall", ["-0.1", "1.1", "nan", "inf"]),
    ("--maintenance-work-mem-mb", ["0", "1025", "1.5"]),
])
def test_invalid_options(benchmark, monkeypatch, flag, values):
    for value in values:
        monkeypatch.setattr(sys, "argv", ["benchmark", flag, value])
        with pytest.raises(SystemExit) as exc:
            benchmark.main()
        assert exc.value.code == 2


@pytest.mark.parametrize("threshold,exit_code,passed", [
    (None, None, None), ("0", None, True), ("0.75", None, True), ("1", 1, False),
])
def test_defaults_and_acceptance(benchmark, monkeypatch, capsys, threshold, exit_code, passed):
    argv = ["benchmark", "--writers", "1", "--repeats", "2"]
    if threshold is not None:
        argv += ["--min-recall", threshold]
    monkeypatch.setattr(sys, "argv", argv)

    def trial(params, args, writers, rows, queries):
        assert len(queries) == 10
        assert not set(queries) & {vector for _, vector in rows}
        return {"recall_at_10": 0.75, "rows_per_second": 123}

    monkeypatch.setattr(benchmark, "trial", trial)
    if exit_code is None:
        benchmark.main()
    else:
        with pytest.raises(SystemExit) as exc:
            benchmark.main()
        assert exc.value.code == exit_code
    config, *results = map(json.loads, capsys.readouterr().out.splitlines())
    for key, value in {
        "num_neighbors": 30, "search_list_size": 100, "query_search_list_size": 100,
        "query_rescore": 100, "recall_queries": 10, "maintenance_work_mem_mb": None,
        "log_insert_stats": False,
    }.items():
        assert config[key] == value
    assert len(results) == 2
    assert all(result["recall_passed"] is passed for result in results)
    assert all(result["rows_per_second"] == 123 for result in results)


@pytest.mark.parametrize("degree,search,query_search,rescore,memory", [
    (10, 10, 1, 0, 1), (1000, 1000, 10000, 1000, 1024),
])
def test_explicit_options(benchmark, monkeypatch, capsys, degree, search, query_search,
                          rescore, memory):
    options = {
        "num_neighbors": degree, "search_list_size": search,
        "query_search_list_size": query_search, "query_rescore": rescore,
        "maintenance_work_mem_mb": memory, "recall_queries": 3,
    }
    argv = ["benchmark", "--writers", "1", "--repeats", "1", "--log-insert-stats"]
    for key, value in options.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    monkeypatch.setattr(sys, "argv", argv)

    def trial(params, args, writers, rows, queries):
        assert len(queries) == 3
        assert args.log_insert_stats
        assert all(getattr(args, key) == value for key, value in options.items())
        return {"recall_at_10": 1.0}

    monkeypatch.setattr(benchmark, "trial", trial)
    benchmark.main()
    config = json.loads(capsys.readouterr().out.splitlines()[0])
    assert config["log_insert_stats"] is True
    assert all(config[key] == value for key, value in options.items())
