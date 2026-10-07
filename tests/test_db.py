import json
from datetime import datetime, timedelta

import duckdb

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


def test_reopen_run_resets_failed_run_state(con):
    run_id = db.start_run(con, "h", T)
    db.set_suspect(con, run_id)
    db.add_reason(con, run_id, "failed: boom")
    db.finish_run(con, run_id, "failed", T + timedelta(minutes=5))
    db.reopen_run(con, run_id)
    run = db.get_run(con, run_id)
    assert (run["status"], run["status_reasons"], run["suspect"], run["finished_at"]) == ("running", [], False, None)


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


OLD_SCHEMA = """
CREATE TABLE listing_snapshots (
    run_id VARCHAR, listing_id BIGINT, title VARCHAR, tags VARCHAR[], price_usd DOUBLE,
    num_favorers INTEGER, views INTEGER, created_at TIMESTAMP, shop_id BIGINT,
    listing_type VARCHAR, url VARCHAR,
    PRIMARY KEY (run_id, listing_id)
);
CREATE TABLE niche_scores (
    run_id VARCHAR, niche_id VARCHAR,
    reviews_90d INTEGER, active_listings INTEGER, fav_delta INTEGER, entry_share DOUBLE,
    gap_per_100 DOUBLE, gap_missing BOOLEAN, median_price DOUBLE, listing_count INTEGER,
    top3_share DOUBLE,
    pct_demand DOUBLE, pct_entry DOUBLE, pct_gap DOUBLE, pct_price DOUBLE, pct_crowding DOUBLE,
    score DOUBLE, confidence VARCHAR, dropped BOOLEAN, dropped_reason VARCHAR, rank INTEGER,
    PRIMARY KEY (run_id, niche_id)
);
"""


def columns(con, table):
    return [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]


def test_connect_migrates_old_schema_to_new_column_order(tmp_path):
    old_path = tmp_path / "old.duckdb"
    old = duckdb.connect(str(old_path))
    old.execute(OLD_SCHEMA)
    old.execute("INSERT INTO listing_snapshots VALUES ('r1', 1, 't', ['a'], 9.5, 3, 0, NULL, 7, 'download', 'u')")
    old.close()

    migrated = db.connect(old_path)
    fresh = db.connect(":memory:")
    for table, added in [("listing_snapshots", ["language"]), ("niche_scores", ["fav_rate", "engaged_listings"])]:
        assert columns(migrated, table)[-len(added):] == added
        assert columns(migrated, table) == columns(fresh, table)
    assert migrated.execute("SELECT listing_id, language FROM listing_snapshots").fetchall() == [(1, None)]
    migrated.close()
    db.connect(old_path).close()  # migrating twice is a no-op
