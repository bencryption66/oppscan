"""DuckDB schema, run bookkeeping and raw-data retention."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    seeds_hash VARCHAR NOT NULL,
    api_calls INTEGER NOT NULL DEFAULT 0,
    status VARCHAR NOT NULL,
    status_reasons VARCHAR[] NOT NULL,
    suspect BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS raw_api (
    run_id VARCHAR NOT NULL,
    endpoint VARCHAR NOT NULL,
    request_key VARCHAR NOT NULL,
    fetched_at TIMESTAMP NOT NULL,
    payload JSON NOT NULL,
    PRIMARY KEY (run_id, endpoint, request_key)
);
CREATE TABLE IF NOT EXISTS search_hits (
    run_id VARCHAR, seed VARCHAR, listing_id BIGINT, search_rank INTEGER,
    PRIMARY KEY (run_id, seed, listing_id)
);
CREATE TABLE IF NOT EXISTS seed_counts (
    run_id VARCHAR, seed VARCHAR, total_count INTEGER,
    PRIMARY KEY (run_id, seed)
);
CREATE TABLE IF NOT EXISTS listing_snapshots (
    run_id VARCHAR, listing_id BIGINT, title VARCHAR, tags VARCHAR[], price_usd DOUBLE,
    num_favorers INTEGER, views INTEGER, created_at TIMESTAMP, shop_id BIGINT,
    listing_type VARCHAR, url VARCHAR, language VARCHAR,
    PRIMARY KEY (run_id, listing_id)
);
CREATE TABLE IF NOT EXISTS shop_snapshots (
    run_id VARCHAR, shop_id BIGINT, shop_name VARCHAR, transaction_sold_count INTEGER,
    review_count INTEGER, review_average DOUBLE, created_at TIMESTAMP,
    PRIMARY KEY (run_id, shop_id)
);
CREATE TABLE IF NOT EXISTS reviews (
    review_hash VARCHAR PRIMARY KEY, listing_id BIGINT, rating INTEGER, text VARCHAR,
    created_at TIMESTAMP, first_seen_run VARCHAR
);
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key VARCHAR PRIMARY KEY, task VARCHAR, response JSON, created_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS niches (
    niche_id VARCHAR, cluster_version VARCHAR, name VARCHAR, description VARCHAR,
    PRIMARY KEY (niche_id, cluster_version)
);
CREATE TABLE IF NOT EXISTS seed_niche (
    cluster_version VARCHAR, seed VARCHAR, niche_id VARCHAR,
    PRIMARY KEY (cluster_version, seed)
);
CREATE TABLE IF NOT EXISTS listing_niche (
    run_id VARCHAR, listing_id BIGINT, niche_id VARCHAR, cluster_version VARCHAR,
    PRIMARY KEY (run_id, listing_id)
);
CREATE TABLE IF NOT EXISTS complaint_runs (
    run_id VARCHAR, niche_id VARCHAR, status VARCHAR,
    PRIMARY KEY (run_id, niche_id)
);
CREATE TABLE IF NOT EXISTS complaints (
    run_id VARCHAR, niche_id VARCHAR, theme VARCHAR, fixable BOOLEAN, mentions INTEGER,
    quotes VARCHAR[], review_hashes VARCHAR[]
);
CREATE TABLE IF NOT EXISTS niche_scores (
    run_id VARCHAR, niche_id VARCHAR,
    reviews_90d INTEGER, active_listings INTEGER, fav_delta INTEGER, entry_share DOUBLE,
    gap_per_100 DOUBLE, gap_missing BOOLEAN, median_price DOUBLE, listing_count INTEGER,
    top3_share DOUBLE,
    pct_demand DOUBLE, pct_entry DOUBLE, pct_gap DOUBLE, pct_price DOUBLE, pct_crowding DOUBLE,
    score DOUBLE, confidence VARCHAR, dropped BOOLEAN, dropped_reason VARCHAR, rank INTEGER,
    fav_rate DOUBLE, engaged_listings INTEGER,
    PRIMARY KEY (run_id, niche_id)
);
CREATE TABLE IF NOT EXISTS briefs (
    run_id VARCHAR, niche_id VARCHAR, brief JSON,
    PRIMARY KEY (run_id, niche_id)
);
CREATE TABLE IF NOT EXISTS run_summary (
    run_id VARCHAR PRIMARY KEY, summary VARCHAR
);
"""

# Columns added after the first release. Each is also at the end of its CREATE TABLE above,
# so new and migrated databases end up with the same column order.
MIGRATIONS = """
ALTER TABLE listing_snapshots ADD COLUMN IF NOT EXISTS language VARCHAR;
ALTER TABLE niche_scores ADD COLUMN IF NOT EXISTS fav_rate DOUBLE;
ALTER TABLE niche_scores ADD COLUMN IF NOT EXISTS engaged_listings INTEGER;
"""

RUN_COLUMNS = ["run_id", "started_at", "finished_at", "seeds_hash", "api_calls",
               "status", "status_reasons", "suspect"]


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def connect(path: Path | str) -> duckdb.DuckDBPyConnection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    con.execute(MIGRATIONS)
    return con


def start_run(con, seeds_hash: str, now: datetime) -> str:
    run_id = now.strftime("%Y%m%dT%H%M%S")
    con.execute(
        "INSERT INTO runs (run_id, started_at, seeds_hash, status, status_reasons) "
        "VALUES (?, ?, ?, 'running', CAST([] AS VARCHAR[]))",
        [run_id, now, seeds_hash],
    )
    return run_id


def get_run(con, run_id: str) -> dict | None:
    row = con.execute(f"SELECT {', '.join(RUN_COLUMNS)} FROM runs WHERE run_id = ?", [run_id]).fetchone()
    return dict(zip(RUN_COLUMNS, row)) if row else None


def add_reason(con, run_id: str, reason: str) -> None:
    con.execute("UPDATE runs SET status_reasons = list_append(status_reasons, ?) WHERE run_id = ?",
                [reason, run_id])


def reopen_run(con, run_id: str) -> None:
    """Reset a run's outcome fields before resuming it."""
    con.execute("UPDATE runs SET status = 'running', status_reasons = []::VARCHAR[], suspect = FALSE, "
                "finished_at = NULL WHERE run_id = ?", [run_id])


def set_suspect(con, run_id: str) -> None:
    con.execute("UPDATE runs SET suspect = TRUE WHERE run_id = ?", [run_id])


def finish_run(con, run_id: str, status: str, now: datetime) -> None:
    con.execute("UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?", [status, now, run_id])


def previous_run(con, run_id: str) -> str | None:
    row = con.execute(
        "SELECT run_id FROM runs WHERE run_id < ? AND status IN ('complete', 'partial') "
        "ORDER BY run_id DESC LIMIT 1",
        [run_id],
    ).fetchone()
    return row[0] if row else None


def cluster_version(con, run_id: str) -> str | None:
    row = con.execute("SELECT any_value(cluster_version) FROM listing_niche WHERE run_id = ?",
                      [run_id]).fetchone()
    return row[0] if row else None


def prune_raw(con, archive_dir: Path, keep: int = 8) -> list[str]:
    """Move raw_api rows for all but the newest `keep` runs into zstd Parquet files."""
    old = [r[0] for r in con.execute(
        "SELECT DISTINCT run_id FROM raw_api WHERE run_id NOT IN "
        "(SELECT run_id FROM runs ORDER BY run_id DESC LIMIT ?) ORDER BY run_id",
        [keep],
    ).fetchall()]
    if not old:
        return []
    Path(archive_dir).mkdir(parents=True, exist_ok=True)
    for run_id in old:
        target = str(Path(archive_dir) / f"raw_api_{run_id}.parquet").replace("'", "''")
        safe_run = run_id.replace("'", "''")
        con.execute(f"COPY (SELECT * FROM raw_api WHERE run_id = '{safe_run}') "
                    f"TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.execute("DELETE FROM raw_api WHERE run_id = ?", [run_id])
    return old
