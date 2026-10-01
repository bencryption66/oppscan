import json
from datetime import datetime, timedelta

from oppscan import db

T = datetime(2026, 10, 1, 9, 30, 0)


def test_connect_creates_tables(con):
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert {"runs", "raw_api", "search_hits", "seed_counts", "listing_snapshots", "shop_snapshots",
            "reviews", "llm_cache", "niches", "seed_niche", "listing_niche", "complaint_runs",
            "complaints", "niche_scores", "briefs", "run_summary"} <= tables


def test_start_and_get_run(con):
    run_id = db.start_run(con, "abc", T)
    assert run_id == "20261001T093000"
    run = db.get_run(con, run_id)
    assert run["status"] == "running"
    assert run["status_reasons"] == []
    assert run["api_calls"] == 0
    assert run["suspect"] is False
    assert run["started_at"] == T


def test_reasons_status_and_finish(con):
    run_id = db.start_run(con, "abc", T)
    db.add_reason(con, run_id, "one")
    db.add_reason(con, run_id, "two")
    db.set_suspect(con, run_id)
    db.finish_run(con, run_id, "partial", T + timedelta(minutes=5))
    run = db.get_run(con, run_id)
    assert run["status_reasons"] == ["one", "two"]
    assert run["suspect"] is True
    assert run["status"] == "partial"
    assert run["finished_at"] == T + timedelta(minutes=5)


def test_clear_reasons(con):
    run_id = db.start_run(con, "h", T)
    db.add_reason(con, run_id, "one")
    db.clear_reasons(con, run_id)
    assert db.get_run(con, run_id)["status_reasons"] == []
    db.add_reason(con, run_id, "two")
    assert db.get_run(con, run_id)["status_reasons"] == ["two"]


def test_previous_run_skips_unfinished_and_failed(con):
    first = db.start_run(con, "h", T)
    db.finish_run(con, first, "complete", T)
    failed = db.start_run(con, "h", T + timedelta(days=1))
    db.finish_run(con, failed, "failed", T)
    current = db.start_run(con, "h", T + timedelta(days=2))
    assert db.previous_run(con, current) == first
    assert db.previous_run(con, first) is None


def test_cluster_version(con):
    assert db.cluster_version(con, "r1") is None
    con.execute("INSERT INTO listing_niche VALUES ('r1', 1, 'a', 'v9')")
    assert db.cluster_version(con, "r1") == "v9"


def test_prune_raw_archives_old_runs(con, tmp_path):
    for day in range(10):
        run_id = db.start_run(con, "h", T + timedelta(days=day))
        con.execute("INSERT INTO raw_api VALUES (?, '/x', '{}', ?, ?)", [run_id, T, json.dumps({"d": day})])
    archived = db.prune_raw(con, tmp_path / "archive", keep=8)
    assert archived == ["20261001T093000", "20261002T093000"]
    assert (tmp_path / "archive" / "raw_api_20261001T093000.parquet").exists()
    assert con.execute("SELECT count(DISTINCT run_id) FROM raw_api").fetchone()[0] == 8
