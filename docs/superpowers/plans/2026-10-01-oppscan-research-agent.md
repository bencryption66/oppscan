# oppscan Research Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `oppscan`, a CLI that scans Etsy for digital spreadsheet/Notion template niches and writes a ranked opportunity report with draft product briefs.

**Architecture:** A deterministic Python pipeline (collect → stage → cluster → complaints → metrics → score → briefs → report) over one DuckDB file. Claude is called only for clustering, complaint mining, briefs and the summary, through a cached, schema-validated `LLMClient`. Fakes for Etsy and the LLM let the whole pipeline run with no network (`--fixtures`).

**Tech Stack:** Python 3.12, uv, httpx, DuckDB, anthropic SDK, jsonschema, Jinja2, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-oppscan-research-agent-design.md`

## Global Constraints

- Etsy Open API v3 only, base URL `https://openapi.etsy.com/v3/application`. No scraping of any kind.
- Keep only listings with `listing_type == "download"` and `price.currency_code == "USD"`. Drop everything else.
- Models: `claude-sonnet-5-5` for clustering and complaint mining; `claude-opus-5-5` for briefs and the summary.
- Every LLM response is validated against its JSON schema. Invalid output is retried once, then raises `LLMOutputError`.
- Scoring weights: demand 0.35, entry 0.25, gap 0.20, price 0.10, crowding −0.10. Each is a percentile across niches.
- Demand floor: drop niches below the 25th percentile of Demand. High confidence: ≥ 30 reviews in the last 90 days across ≥ 8 listings.
- All components are computed over the top 20 search results per niche. Review window: 90 days. A "new listing" is under 365 days old.
- Run status is one of `running`, `paused`, `complete`, `partial` or `failed`. A partial run still produces a report.
- Raw API JSON is kept for the last 8 runs. Older runs are archived to zstd Parquet.
- Tests never touch the network. Live checks carry `@pytest.mark.live` and are excluded by default.
- All timestamps are naive UTC (`db.utcnow()`).

## Refinements to the spec (decided while planning)

1. **Clustering works on seeds, not individual listings.** Claude gets each seed with its top 10 titles and groups the *seeds*. Each listing then inherits the niche of the seed where it ranks best. This keeps the prompt small (about 150 × 10 titles) and gives stable niche IDs across runs.
2. **The cluster cache is keyed by the seeds-file hash.** Niches only change when `seeds.yaml` changes, which keeps rank changes between runs meaningful.
3. **Shops are fetched only for the top-20 listings per seed.** Shop data isn't used in scoring, so fetching more would only use up API quota.
4. **A new `paused` status** covers a run stopped by the API quota. It resumes with `oppscan run --resume <run_id>`.
5. **A new `complaint_runs` table** records per-niche mining status (`ok` / `no_reviews` / `failed`), so a missing Gap score can be told apart from a zero one.
6. **Clustering failure fails the whole run.** Nothing after it can work without niches. All other LLM failures mark the run `partial`.

## File Structure

```
oppscan/
  pyproject.toml
  .gitignore
  README.md
  config/
    seeds.yaml          # seed search terms
    scoring.yaml        # weights, thresholds, known-big seeds
    etsy.yaml           # qps, quota, depth
  src/oppscan/
    __init__.py
    config.py           # load the three YAML files into dataclasses
    db.py               # schema, run bookkeeping, utcnow, raw pruning
    etsy.py             # EtsyClient: pacing, retries, quota, raw_api cache
    fake_etsy.py        # deterministic httpx.MockTransport for tests / --fixtures
    collector.py        # seeds -> API calls (search, shops, reviews)
    staging.py          # raw_api JSON -> staged tables
    prompts.py          # JSON schemas + prompt text per LLM task
    llm.py              # LLMClient protocol, AnthropicLLM, validate()
    fake_llm.py         # deterministic FakeLLM
    clustering.py       # seeds -> niches, listings -> niches
    metrics.py          # top_listings(), compute_metrics()
    complaints.py       # complaint mining per niche
    scoring.py          # percentiles, score, confidence, floor, ranks, sanity
    briefs.py           # competitors(), brief + summary generation
    report.py           # report context + rendering
    pipeline.py         # run_pipeline(): orchestration + status
    cli.py              # `oppscan run`
    templates/
      report.html.j2
      report.md.j2
  tests/
    conftest.py         # `con` fixture
    helpers.py          # row builders + make_cfg()
    test_config.py  test_db.py  test_etsy.py  test_collector.py  test_staging.py
    test_llm.py  test_clustering.py  test_metrics.py  test_complaints.py
    test_scoring.py  test_report.py  test_pipeline.py  test_live.py
```

All commands run from the repo root: `~/Projects/AI & MCP/oppscan`.

---

### Task 1: Project scaffold and config loading

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `config/seeds.yaml`, `config/scoring.yaml`, `config/etsy.yaml`, `src/oppscan/__init__.py`, `src/oppscan/config.py`, `tests/conftest.py`, `tests/helpers.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Seeds(terms: tuple[str, ...], file_hash: str)`, `Weights(demand, entry, gap, price, crowding)`, `ScoringConfig(weights, demand_floor_pct, high_conf_min_reviews, high_conf_min_listings, top_n_per_niche, review_window_days, new_listing_days, report_top_k, known_big_seeds)`, `EtsySettings(api_key, qps, daily_quota, search_depth, reviews_for_top, max_review_pages)`; `load_seeds(path) -> Seeds`, `load_scoring(path) -> ScoringConfig`, `load_etsy(path, api_key: str | None = None) -> EtsySettings`. `tests/helpers.make_cfg(**over) -> ScoringConfig`.

- [ ] **Step 1: Create the project files**

`pyproject.toml`:
```toml
[project]
name = "oppscan"
version = "0.1.0"
description = "Etsy digital-template opportunity scanner"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.70",
    "duckdb>=1.1",
    "httpx>=0.27",
    "jinja2>=3.1",
    "jsonschema>=4.23",
    "pyyaml>=6.0",
]

[project.scripts]
oppscan = "oppscan.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/oppscan"]

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
markers = ["live: calls real external APIs"]
addopts = "-m 'not live'"
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
data/
reports/
tests/fixtures/etsy_live_sample.json
```

`config/etsy.yaml`:
```yaml
# Confirm qps and daily_quota against the limits shown for your approved Etsy app.
qps: 5
daily_quota: 5000
search_depth: 300       # listings fetched per seed (3 pages of 100)
reviews_for_top: 20     # listings per seed whose reviews and shop are fetched
max_review_pages: 3     # pages of 100 reviews per listing
```

`config/scoring.yaml`:
```yaml
weights:
  demand: 0.35
  entry: 0.25
  gap: 0.20
  price: 0.10
  crowding: 0.10        # subtracted
demand_floor_pct: 0.25
high_confidence:
  min_reviews_90d: 30
  min_listings: 8
top_n_per_niche: 20
review_window_days: 90
new_listing_days: 365
report_top_k: 10
known_big_seeds:
  - monthly budget spreadsheet
```

`config/seeds.yaml`:
```yaml
seeds:
  - monthly budget spreadsheet
  - budget spreadsheet google sheets
  - budget planner excel
  - paycheck budget spreadsheet
  - zero based budget template
  - debt snowball spreadsheet
  - savings tracker spreadsheet
  - sinking funds tracker
  - net worth tracker spreadsheet
  - wedding budget spreadsheet
  - wedding planner spreadsheet
  - small business bookkeeping spreadsheet
  - small business expense tracker
  - self employed tax spreadsheet
  - invoice template excel
  - inventory tracker spreadsheet
  - etsy seller spreadsheet
  - social media content calendar
  - rental property spreadsheet
  - airbnb host spreadsheet
  - meal planner spreadsheet
  - grocery list template
  - habit tracker spreadsheet
  - workout tracker spreadsheet
  - weight loss tracker spreadsheet
  - reading tracker spreadsheet
  - job application tracker
  - student planner spreadsheet
  - teacher gradebook spreadsheet
  - homeschool planner
  - cleaning schedule template
  - baby tracker spreadsheet
  - travel planner spreadsheet
  - project management spreadsheet
  - gantt chart excel template
  - crm spreadsheet template
  - notion template
  - notion life planner
  - notion adhd planner
  - notion student planner
  - notion finance tracker
  - notion budget template
  - notion business dashboard
  - notion content planner
  - notion habit tracker
  - notion reading list
  - notion wedding planner
  - notion fitness tracker
  - notion meal planner
  - notion second brain
  - notion small business planner
  - notion freelancer dashboard
  - notion job search tracker
  - notion travel planner
  - notion goal planner
  - google sheets dashboard template
  - excel dashboard template
  - kpi dashboard excel
  - stock portfolio tracker spreadsheet
  - dividend tracker spreadsheet
```

`src/oppscan/__init__.py`:
```python
"""oppscan: Etsy digital-template opportunity scanner."""
```

- [ ] **Step 2: Install dependencies**

Run: `uv sync`
Expected: `.venv` is created and the dependencies install without errors.

- [ ] **Step 3: Write the failing tests**

`tests/test_config.py`:
```python
from pathlib import Path

import pytest

from oppscan.config import load_etsy, load_scoring, load_seeds

REPO = Path(__file__).resolve().parents[1]


def test_load_seeds_normalises_and_dedupes(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("seeds:\n  - Budget  Spreadsheet\n  - budget spreadsheet\n  - '  '\n  - notion planner\n")
    seeds = load_seeds(path)
    assert seeds.terms == ("budget spreadsheet", "notion planner")
    assert len(seeds.file_hash) == 16


def test_load_seeds_hash_changes_with_content(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("seeds: [a]\n")
    first = load_seeds(path).file_hash
    path.write_text("seeds: [a, b]\n")
    assert load_seeds(path).file_hash != first


def test_load_seeds_rejects_empty(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("seeds: []\n")
    with pytest.raises(ValueError):
        load_seeds(path)


def test_repo_scoring_config_loads():
    cfg = load_scoring(REPO / "config" / "scoring.yaml")
    assert cfg.weights.demand == 0.35
    assert cfg.weights.crowding == 0.10
    assert cfg.demand_floor_pct == 0.25
    assert cfg.high_conf_min_reviews == 30
    assert cfg.high_conf_min_listings == 8
    assert cfg.top_n_per_niche == 20
    assert cfg.known_big_seeds == ("monthly budget spreadsheet",)


def test_repo_seeds_include_known_big_seeds():
    seeds = load_seeds(REPO / "config" / "seeds.yaml")
    cfg = load_scoring(REPO / "config" / "scoring.yaml")
    assert 50 <= len(seeds.terms) <= 150
    assert set(cfg.known_big_seeds) <= set(seeds.terms)


def test_load_etsy_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ETSY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ETSY_API_KEY"):
        load_etsy(REPO / "config" / "etsy.yaml")


def test_load_etsy_reads_env_key(monkeypatch):
    monkeypatch.setenv("ETSY_API_KEY", "abc")
    settings = load_etsy(REPO / "config" / "etsy.yaml")
    assert settings.api_key == "abc"
    assert settings.search_depth == 300
    assert settings.reviews_for_top == 20


def test_load_etsy_explicit_key_wins(monkeypatch):
    monkeypatch.setenv("ETSY_API_KEY", "env")
    assert load_etsy(REPO / "config" / "etsy.yaml", api_key="explicit").api_key == "explicit"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oppscan.config'`

- [ ] **Step 5: Implement `src/oppscan/config.py`**

```python
"""Load oppscan's YAML configuration files."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Seeds:
    terms: tuple[str, ...]
    file_hash: str


@dataclass(frozen=True)
class Weights:
    demand: float
    entry: float
    gap: float
    price: float
    crowding: float


@dataclass(frozen=True)
class ScoringConfig:
    weights: Weights
    demand_floor_pct: float
    high_conf_min_reviews: int
    high_conf_min_listings: int
    top_n_per_niche: int
    review_window_days: int
    new_listing_days: int
    report_top_k: int
    known_big_seeds: tuple[str, ...]


@dataclass(frozen=True)
class EtsySettings:
    api_key: str
    qps: float
    daily_quota: int
    search_depth: int
    reviews_for_top: int
    max_review_pages: int


def normalise(term: str) -> str:
    return " ".join(str(term).lower().split())


def load_seeds(path: Path) -> Seeds:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw) or {}
    terms = tuple(dict.fromkeys(t for t in (normalise(s) for s in data.get("seeds", [])) if t))
    if not terms:
        raise ValueError(f"{path}: no seeds defined")
    return Seeds(terms=terms, file_hash=hashlib.sha256(raw).hexdigest()[:16])


def load_scoring(path: Path) -> ScoringConfig:
    d = yaml.safe_load(Path(path).read_text())
    hc = d["high_confidence"]
    return ScoringConfig(
        weights=Weights(**{k: float(v) for k, v in d["weights"].items()}),
        demand_floor_pct=float(d["demand_floor_pct"]),
        high_conf_min_reviews=int(hc["min_reviews_90d"]),
        high_conf_min_listings=int(hc["min_listings"]),
        top_n_per_niche=int(d["top_n_per_niche"]),
        review_window_days=int(d["review_window_days"]),
        new_listing_days=int(d["new_listing_days"]),
        report_top_k=int(d["report_top_k"]),
        known_big_seeds=tuple(normalise(s) for s in d.get("known_big_seeds", [])),
    )


def load_etsy(path: Path, api_key: str | None = None) -> EtsySettings:
    d = yaml.safe_load(Path(path).read_text())
    key = api_key if api_key is not None else os.environ.get("ETSY_API_KEY", "")
    if not key:
        raise ValueError("ETSY_API_KEY is not set")
    return EtsySettings(
        api_key=key,
        qps=float(d["qps"]),
        daily_quota=int(d["daily_quota"]),
        search_depth=int(d["search_depth"]),
        reviews_for_top=int(d["reviews_for_top"]),
        max_review_pages=int(d["max_review_pages"]),
    )
```

- [ ] **Step 6: Create the shared test helpers (used from Task 7 onwards)**

`tests/helpers.py`:
```python
"""Row builders shared by tests."""
from __future__ import annotations

from datetime import datetime

from oppscan.config import ScoringConfig, Weights


def make_cfg(**over) -> ScoringConfig:
    base = dict(
        weights=Weights(demand=0.35, entry=0.25, gap=0.20, price=0.10, crowding=0.10),
        demand_floor_pct=0.25,
        high_conf_min_reviews=30,
        high_conf_min_listings=8,
        top_n_per_niche=20,
        review_window_days=90,
        new_listing_days=365,
        report_top_k=10,
        known_big_seeds=(),
    )
    base.update(over)
    return ScoringConfig(**base)


def add_run(con, run_id: str, started_at: datetime, status: str = "running") -> None:
    con.execute(
        "INSERT INTO runs (run_id, started_at, seeds_hash, status, status_reasons) "
        "VALUES (?, ?, 'h', ?, CAST([] AS VARCHAR[]))",
        [run_id, started_at, status],
    )


def add_listing(con, run_id, listing_id, *, seed, rank, price=10.0, favs=0,
                created=datetime(2024, 1, 1), shop_id=1, title=None) -> None:
    con.execute(
        "INSERT OR IGNORE INTO listing_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [run_id, listing_id, title or f"Listing {listing_id}", ["tag"], price, favs, 0,
         created, shop_id, "download", f"https://www.etsy.com/listing/{listing_id}"],
    )
    con.execute("INSERT INTO search_hits VALUES (?, ?, ?, ?)", [run_id, seed, listing_id, rank])


def add_review(con, listing_id, created: datetime, rating=5, text="Great", run_id="r1") -> str:
    review_hash = f"{listing_id}-{created.isoformat()}-{text}"[:64]
    con.execute("INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?)",
                [review_hash, listing_id, rating, text, created, run_id])
    return review_hash


def add_niche(con, niche_id, seeds, version="v1", name=None, description="desc") -> None:
    con.execute("INSERT INTO niches VALUES (?, ?, ?, ?)",
                [niche_id, version, name or niche_id.title(), description])
    for seed in seeds:
        con.execute("INSERT INTO seed_niche VALUES (?, ?, ?)", [version, seed, niche_id])


def assign(con, run_id, listing_id, niche_id, version="v1") -> None:
    con.execute("INSERT INTO listing_niche VALUES (?, ?, ?, ?)", [run_id, listing_id, niche_id, version])
```

- [ ] **Step 7: Run the config tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: 8 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .gitignore config src tests
git commit -m "feat: project scaffold and config loading"
```

---

### Task 2: DuckDB schema and run bookkeeping

**Files:**
- Create: `src/oppscan/db.py`, `tests/conftest.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces: `utcnow() -> datetime`; `connect(path) -> DuckDBPyConnection` (creates all tables); `start_run(con, seeds_hash, now) -> str` (run_id `YYYYmmddTHHMMSS`); `get_run(con, run_id) -> dict | None`; `add_reason(con, run_id, reason)`; `set_status(con, run_id, status)`; `set_suspect(con, run_id)`; `finish_run(con, run_id, status, now)`; `previous_run(con, run_id) -> str | None`; `cluster_version(con, run_id) -> str | None`; `prune_raw(con, archive_dir, keep=8) -> list[str]`.
- Table columns (exact order, used by positional INSERTs in later tasks):
  - `raw_api(run_id, endpoint, request_key, fetched_at, payload)`
  - `search_hits(run_id, seed, listing_id, search_rank)`
  - `seed_counts(run_id, seed, total_count)`
  - `listing_snapshots(run_id, listing_id, title, tags, price_usd, num_favorers, views, created_at, shop_id, listing_type, url)`
  - `shop_snapshots(run_id, shop_id, shop_name, transaction_sold_count, review_count, review_average, created_at)`
  - `reviews(review_hash, listing_id, rating, text, created_at, first_seen_run)`
  - `llm_cache(cache_key, task, response, created_at)`
  - `niches(niche_id, cluster_version, name, description)`
  - `seed_niche(cluster_version, seed, niche_id)`
  - `listing_niche(run_id, listing_id, niche_id, cluster_version)`
  - `complaint_runs(run_id, niche_id, status)`
  - `complaints(run_id, niche_id, theme, fixable, mentions, quotes, review_hashes)`
  - `niche_scores(run_id, niche_id, reviews_90d, active_listings, fav_delta, entry_share, gap_per_100, gap_missing, median_price, listing_count, top3_share, pct_demand, pct_entry, pct_gap, pct_price, pct_crowding, score, confidence, dropped, dropped_reason, rank)`
  - `briefs(run_id, niche_id, brief)`
  - `run_summary(run_id, summary)`

- [ ] **Step 1: Write the shared fixture and the failing tests**

`tests/conftest.py` (gives every later test an in-memory database with the schema):
```python
import pytest

from oppscan import db


@pytest.fixture
def con():
    connection = db.connect(":memory:")
    yield connection
    connection.close()
```

`tests/test_db.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py -v`
Expected: ERROR while loading `conftest.py`: `ImportError: cannot import name 'db' from 'oppscan'`

- [ ] **Step 3: Implement `src/oppscan/db.py`**

```python
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
    listing_type VARCHAR, url VARCHAR,
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

RUN_COLUMNS = ["run_id", "started_at", "finished_at", "seeds_hash", "api_calls",
               "status", "status_reasons", "suspect"]


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def connect(path: Path | str) -> duckdb.DuckDBPyConnection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
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


def set_status(con, run_id: str, status: str) -> None:
    con.execute("UPDATE runs SET status = ? WHERE run_id = ?", [status, run_id])


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py tests/test_config.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/oppscan/db.py tests/test_db.py tests/conftest.py
git commit -m "feat: duckdb schema and run bookkeeping"
```

---

### Task 3: Etsy API client

**Files:**
- Create: `src/oppscan/etsy.py`
- Test: `tests/test_etsy.py`

**Interfaces:**
- Consumes: `db.utcnow`, the `raw_api` and `runs` tables.
- Produces: `EtsyClient(con, run_id, *, api_key, qps, daily_quota, transport=None, sleep=time.sleep, clock=time.monotonic, max_retries=5)` with `.get(endpoint: str, params: dict | None = None) -> dict`, `.calls_last_24h() -> int` and `.close()`. Raises `QuotaExhausted` or `EtsyError(endpoint, status, body)` (`.status: int`). `endpoint` is relative to the base URL, e.g. `"/listings/active"`, `"/shops/123"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_etsy.py`:
```python
from datetime import datetime

import httpx
import pytest

from oppscan import db
from oppscan.etsy import EtsyClient, EtsyError, QuotaExhausted

T = datetime(2026, 10, 1)


def make_client(con, handler, **kw):
    run_id = db.start_run(con, "h", T)
    sleeps: list[float] = []
    client = EtsyClient(
        con, run_id, api_key="key123",
        qps=kw.pop("qps", 1000.0), daily_quota=kw.pop("daily_quota", 100),
        transport=httpx.MockTransport(handler), sleep=sleeps.append, **kw,
    )
    return client, run_id, sleeps


def test_get_stores_and_reuses_cached_response(con):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"count": 1, "results": [{"listing_id": 1}]})

    client, run_id, _ = make_client(con, handler)
    first = client.get("/listings/active", {"keywords": "budget", "offset": 0})
    second = client.get("/listings/active", {"offset": 0, "keywords": "budget"})
    assert first == second == {"count": 1, "results": [{"listing_id": 1}]}
    assert len(calls) == 1
    assert calls[0].url.path == "/v3/application/listings/active"
    assert calls[0].headers["x-api-key"] == "key123"
    assert db.get_run(con, run_id)["api_calls"] == 1


def test_retries_on_429_then_succeeds(con):
    responses = iter([httpx.Response(429), httpx.Response(200, json={"ok": True})])
    client, _, sleeps = make_client(con, lambda r: next(responses))
    assert client.get("/shops/1") == {"ok": True}
    assert 1.0 in sleeps


def test_uses_retry_after_header(con):
    responses = iter([httpx.Response(503, headers={"retry-after": "7"}), httpx.Response(200, json={})])
    client, _, sleeps = make_client(con, lambda r: next(responses))
    client.get("/shops/1")
    assert 7.0 in sleeps


def test_gives_up_after_max_retries(con):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="down")

    client, _, _ = make_client(con, handler, max_retries=2)
    with pytest.raises(EtsyError) as err:
        client.get("/shops/1")
    assert err.value.status == 503
    assert len(calls) == 3


def test_non_retryable_status_raises_immediately(con):
    client, _, _ = make_client(con, lambda r: httpx.Response(404, text="nope"))
    with pytest.raises(EtsyError) as err:
        client.get("/shops/1")
    assert err.value.status == 404


def test_quota_exhausted(con):
    client, _, _ = make_client(con, lambda r: httpx.Response(200, json={}), daily_quota=1)
    client.get("/shops/1")
    with pytest.raises(QuotaExhausted):
        client.get("/shops/2")


def test_cached_reads_do_not_count_against_quota(con):
    client, _, _ = make_client(con, lambda r: httpx.Response(200, json={}), daily_quota=1)
    client.get("/shops/1")
    client.get("/shops/1")  # cached, no QuotaExhausted


def test_paces_calls(con):
    client, _, sleeps = make_client(con, lambda r: httpx.Response(200, json={}), qps=4.0,
                                    clock=lambda: 0.0)
    client.get("/shops/1")
    client.get("/shops/2")
    assert sleeps == [0.25]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_etsy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oppscan.etsy'`

- [ ] **Step 3: Implement `src/oppscan/etsy.py`**

```python
"""Etsy Open API v3 client: pacing, retries, a rolling daily quota and a per-run response cache."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import timedelta

import httpx

from oppscan.db import utcnow

BASE_URL = "https://openapi.etsy.com/v3/application"
RETRYABLE = {429, 500, 502, 503, 504}


class QuotaExhausted(Exception):
    """The rolling 24-hour call budget is used up."""


class EtsyError(Exception):
    """A non-retryable API failure, or retries exhausted."""

    def __init__(self, endpoint: str, status: int, body: str):
        super().__init__(f"{endpoint} -> HTTP {status}: {body[:200]}")
        self.endpoint = endpoint
        self.status = status


class EtsyClient:
    def __init__(self, con, run_id: str, *, api_key: str, qps: float, daily_quota: int,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 max_retries: int = 5):
        self._con = con
        self._run_id = run_id
        self._daily_quota = daily_quota
        self._min_interval = 1.0 / qps
        self._sleep = sleep
        self._clock = clock
        self._max_retries = max_retries
        self._last_call: float | None = None
        self._http = httpx.Client(base_url=BASE_URL, headers={"x-api-key": api_key},
                                  transport=transport, timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        params = params or {}
        request_key = json.dumps(params, sort_keys=True)
        row = self._con.execute(
            "SELECT payload FROM raw_api WHERE run_id = ? AND endpoint = ? AND request_key = ?",
            [self._run_id, endpoint, request_key],
        ).fetchone()
        if row is not None:
            return json.loads(row[0])
        if self.calls_last_24h() >= self._daily_quota:
            raise QuotaExhausted(f"Etsy daily quota of {self._daily_quota} calls reached")
        payload = self._fetch(endpoint, params)
        self._con.execute("INSERT INTO raw_api VALUES (?, ?, ?, ?, ?)",
                          [self._run_id, endpoint, request_key, utcnow(), json.dumps(payload)])
        self._con.execute("UPDATE runs SET api_calls = api_calls + 1 WHERE run_id = ?", [self._run_id])
        return payload

    def calls_last_24h(self) -> int:
        return self._con.execute("SELECT count(*) FROM raw_api WHERE fetched_at >= ?",
                                 [utcnow() - timedelta(hours=24)]).fetchone()[0]

    def _pace(self) -> None:
        if self._last_call is not None:
            wait = self._min_interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    def _fetch(self, endpoint: str, params: dict) -> dict:
        for attempt in range(self._max_retries + 1):
            self._pace()
            response = self._http.get(endpoint, params=params)
            if response.status_code == 200:
                return response.json()
            if response.status_code in RETRYABLE and attempt < self._max_retries:
                self._sleep(self._backoff(response, attempt))
                continue
            raise EtsyError(endpoint, response.status_code, response.text)
        raise AssertionError("unreachable")

    @staticmethod
    def _backoff(response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after", "")
        if retry_after.isdigit():
            return float(retry_after)
        return min(60.0, 2.0 ** attempt)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_etsy.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/oppscan/etsy.py tests/test_etsy.py
git commit -m "feat: etsy client with pacing, retries, quota and cache"
```

---

### Task 4: Fake Etsy transport and collector

**Files:**
- Create: `src/oppscan/fake_etsy.py`, `src/oppscan/collector.py`
- Test: `tests/test_collector.py`

**Interfaces:**
- Consumes: `EtsyClient`, `EtsyError`, `Seeds`, `EtsySettings`.
- Produces: `fake_transport(now: datetime) -> httpx.MockTransport`. It serves `GET /v3/application/listings/active`, `/shops/{id}` and `/listings/{id}/reviews` with deterministic data, and favourites and reviews grow as `now` moves forward. Also `CollectStats(seeds: int, listings_seen: int, errors: list[str])` and `collect(client, seeds, settings) -> CollectStats`. `collect` lets `QuotaExhausted` propagate.

- [ ] **Step 1: Write the failing tests**

`tests/test_collector.py`:
```python
import json
from datetime import datetime

import httpx
import pytest

from oppscan import db
from oppscan.collector import collect
from oppscan.config import EtsySettings, Seeds
from oppscan.etsy import EtsyClient, QuotaExhausted
from oppscan.fake_etsy import fake_transport

T = datetime(2026, 10, 1)
SETTINGS = EtsySettings(api_key="k", qps=1e6, daily_quota=10_000, search_depth=200,
                        reviews_for_top=3, max_review_pages=2)
SEEDS = Seeds(terms=("budget spreadsheet", "notion planner"), file_hash="h")


def client_for(con, transport, quota=10_000):
    run_id = db.start_run(con, "h", T)
    return EtsyClient(con, run_id, api_key="k", qps=1e6, daily_quota=quota,
                      transport=transport, sleep=lambda s: None)


def endpoints(con):
    return [r[0] for r in con.execute("SELECT endpoint FROM raw_api").fetchall()]


def test_collect_fetches_search_pages_shops_and_reviews(con):
    stats = collect(client_for(con, fake_transport(T)), SEEDS, SETTINGS)
    assert stats.seeds == 2
    assert stats.errors == []
    eps = endpoints(con)
    assert eps.count("/listings/active") == 4  # 2 pages x 2 seeds
    review_eps = {e for e in eps if e.endswith("/reviews")}
    assert len(review_eps) == 6  # 3 listings x 2 seeds, no overlap between families
    assert 1 <= sum(e.startswith("/shops/") for e in eps) <= 6


def test_collect_only_fetches_reviews_for_download_listings(con):
    collect(client_for(con, fake_transport(T)), SEEDS, SETTINGS)
    types = {}
    for (payload,) in con.execute("SELECT payload FROM raw_api WHERE endpoint = '/listings/active'").fetchall():
        for listing in json.loads(payload)["results"]:
            types[listing["listing_id"]] = listing["listing_type"]
    reviewed = {int(e.split("/")[2]) for e in endpoints(con) if e.endswith("/reviews")}
    assert reviewed and all(types[i] == "download" for i in reviewed)


def test_collect_records_errors_and_continues(con):
    inner = fake_transport(T)

    def handler(request):
        if request.url.path.startswith("/v3/application/shops/"):
            return httpx.Response(404, text="gone")
        return inner.handle_request(request)

    stats = collect(client_for(con, httpx.MockTransport(handler)), SEEDS, SETTINGS)
    assert stats.seeds == 2
    assert stats.errors and stats.errors[0].startswith("shop ")


def test_collect_propagates_quota_exhausted(con):
    with pytest.raises(QuotaExhausted):
        collect(client_for(con, fake_transport(T), quota=3), SEEDS, SETTINGS)


def test_fake_transport_is_deterministic_and_grows_with_time():
    def first_listing(now):
        with httpx.Client(transport=fake_transport(now), base_url="https://x/v3/application") as c:
            return c.get("/listings/active", params={"keywords": "budget spreadsheet", "limit": 5, "offset": 0}).json()

    a, b = first_listing(T), first_listing(T)
    assert a == b
    later = first_listing(datetime(2026, 11, 1))
    assert later["results"][0]["listing_id"] == a["results"][0]["listing_id"]
    assert later["results"][0]["num_favorers"] >= a["results"][0]["num_favorers"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_collector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oppscan.collector'`

- [ ] **Step 3: Implement `src/oppscan/fake_etsy.py`**

```python
"""Deterministic fake of the Etsy endpoints oppscan uses (tests and `oppscan run --fixtures`).

Listings belong to a "family" (the first word of the search keywords), so related seeds
share listings. Listings and reviews are fixed relative to ANCHOR. Requests only see
reviews dated on or before `now`, so later runs see more reviews and favourites.
"""
from __future__ import annotations

import hashlib
import random
import re
from datetime import UTC, datetime, timedelta

import httpx

ANCHOR = datetime(2026, 9, 1)
FUTURE = ANCHOR + timedelta(days=120)
POOL_SIZE = 500
ADJECTIVES = ["Simple", "Ultimate", "Aesthetic", "Minimal", "Editable", "Printable", "Digital"]
NOUNS = ["Spreadsheet", "Planner", "Tracker", "Dashboard", "Template", "Workbook"]
POSITIVE = ["Love it, easy to use", "Exactly what I needed", "Beautiful and practical",
            "Wish it had a dark mode, otherwise great"]
NEGATIVE = ["Instructions were confusing", "Does not work in Google Sheets on my phone",
            "Seller never replied to my message", "Formulas broke when I added rows"]


def _rng(*parts) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _unix(dt: datetime) -> int:
    return int(dt.replace(tzinfo=UTC).timestamp())


def _family(keywords: str) -> str:
    words = keywords.split()
    return words[0] if words else "misc"


def _family_base(family: str) -> int:
    return 1_000_000 + int(hashlib.sha256(family.encode()).hexdigest()[:6], 16) * 1000


def _listing_attrs(listing_id: int) -> dict:
    rng = _rng("listing", listing_id)
    return {
        "listing_type": rng.choices(["download", "physical", "both"], [85, 10, 5])[0],
        "currency": rng.choices(["USD", "EUR"], [90, 10])[0],
        "amount": rng.randint(299, 2999),
        "created": ANCHOR - timedelta(days=rng.randint(1, 1200)),
        "popularity": rng.paretovariate(1.5),
        "shop_id": 50_000 + rng.randint(0, 120),
        "adjective": rng.choice(ADJECTIVES),
        "noun": rng.choice(NOUNS),
    }


def _listing(listing_id: int, family: str, now: datetime) -> dict:
    a = _listing_attrs(listing_id)
    age_days = max(0, (now - a["created"]).days)
    return {
        "listing_id": listing_id,
        "title": f"{a['adjective']} {family.title()} {a['noun']}",
        "tags": [family, a["noun"].lower(), "digital download"],
        "price": {"amount": a["amount"], "divisor": 100, "currency_code": a["currency"]},
        "num_favorers": int(a["popularity"] * 50 + age_days * a["popularity"] * 0.5),
        "views": int(a["popularity"] * 500 + age_days * a["popularity"] * 5),
        "original_creation_timestamp": _unix(a["created"]),
        "created_timestamp": _unix(a["created"]),
        "shop_id": a["shop_id"],
        "listing_type": a["listing_type"],
        "url": f"https://www.etsy.com/listing/{listing_id}",
    }


def _reviews(listing_id: int, now: datetime) -> list[dict]:
    a = _listing_attrs(listing_id)
    rng = _rng("reviews", listing_id)
    span = max(1, (FUTURE - a["created"]).days)
    out = []
    for _ in range(min(400, int(a["popularity"] * 15))):
        created = a["created"] + timedelta(days=rng.randint(0, span), seconds=rng.randint(0, 86_399))
        rating = rng.choices([5, 4, 3, 2, 1], [75, 12, 7, 3, 3])[0]
        text = rng.choice(POSITIVE if rating == 5 else NEGATIVE)
        if created <= now:
            out.append({"listing_id": listing_id, "rating": rating, "review": text,
                        "create_timestamp": _unix(created), "created_timestamp": _unix(created)})
    out.sort(key=lambda r: r["create_timestamp"], reverse=True)
    return out


def fake_transport(now: datetime) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v3/application")
        params = request.url.params
        limit = int(params.get("limit", 25))
        offset = int(params.get("offset", 0))

        if path == "/listings/active":
            keywords = params.get("keywords", "")
            family = _family(keywords)
            base = _family_base(family)
            order = list(range(POOL_SIZE))
            _rng("order", keywords).shuffle(order)
            count = 150 + _rng("count", keywords).randint(0, 2000)
            available = order[: min(count, POOL_SIZE)]
            page = available[offset: offset + limit]
            return httpx.Response(200, json={
                "count": count,
                "results": [_listing(base + i, family, now) for i in page],
            })

        if m := re.fullmatch(r"/shops/(\d+)", path):
            shop_id = int(m.group(1))
            rng = _rng("shop", shop_id)
            return httpx.Response(200, json={
                "shop_id": shop_id,
                "shop_name": f"Shop{shop_id}",
                "transaction_sold_count": rng.randint(10, 50_000),
                "review_count": rng.randint(1, 5_000),
                "review_average": round(rng.uniform(4.0, 5.0), 2),
                "create_date": _unix(ANCHOR - timedelta(days=rng.randint(100, 3000))),
            })

        if m := re.fullmatch(r"/listings/(\d+)/reviews", path):
            reviews = _reviews(int(m.group(1)), now)
            return httpx.Response(200, json={"count": len(reviews),
                                             "results": reviews[offset: offset + limit]})

        return httpx.Response(404, json={"error": f"no fake for {path}"})

    return httpx.MockTransport(handler)
```

- [ ] **Step 4: Implement `src/oppscan/collector.py`**

```python
"""Fetch search results, shops and reviews for every seed. Responses land in raw_api via EtsyClient."""
from __future__ import annotations

from dataclasses import dataclass, field

from oppscan.config import EtsySettings, Seeds
from oppscan.etsy import EtsyClient, EtsyError

PAGE = 100


@dataclass
class CollectStats:
    seeds: int = 0
    listings_seen: int = 0
    errors: list[str] = field(default_factory=list)


def collect(client: EtsyClient, seeds: Seeds, settings: EtsySettings) -> CollectStats:
    stats = CollectStats()
    for seed in seeds.terms:
        try:
            listings = _search(client, seed, settings.search_depth)
        except EtsyError as e:
            stats.errors.append(f"search '{seed}': {e}")
            continue
        stats.seeds += 1
        downloads = [l for l in listings if l.get("listing_type") == "download"]
        stats.listings_seen += len(downloads)
        top = downloads[: settings.reviews_for_top]
        for shop_id in dict.fromkeys(l["shop_id"] for l in top):
            try:
                client.get(f"/shops/{shop_id}")
            except EtsyError as e:
                stats.errors.append(f"shop {shop_id}: {e}")
        for listing in top:
            try:
                _reviews(client, listing["listing_id"], settings.max_review_pages)
            except EtsyError as e:
                stats.errors.append(f"reviews {listing['listing_id']}: {e}")
    return stats


def _search(client: EtsyClient, seed: str, depth: int) -> list[dict]:
    out: list[dict] = []
    for offset in range(0, depth, PAGE):
        page = client.get("/listings/active",
                          {"keywords": seed, "limit": PAGE, "offset": offset, "sort_on": "score"})
        results = page.get("results", [])
        out.extend(results)
        if len(results) < PAGE or offset + PAGE >= page.get("count", 0):
            break
    return out


def _reviews(client: EtsyClient, listing_id: int, max_pages: int) -> None:
    for page_no in range(max_pages):
        page = client.get(f"/listings/{listing_id}/reviews", {"limit": PAGE, "offset": page_no * PAGE})
        if len(page.get("results", [])) < PAGE:
            break
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_collector.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/oppscan/fake_etsy.py src/oppscan/collector.py tests/test_collector.py
git commit -m "feat: collector and deterministic fake etsy transport"
```

---

### Task 5: Staging raw JSON into tables

**Files:**
- Create: `src/oppscan/staging.py`
- Test: `tests/test_staging.py`

**Interfaces:**
- Consumes: `raw_api` rows written by `EtsyClient`.
- Produces: `stage_run(con, run_id) -> None` (idempotent; fills `search_hits`, `seed_counts`, `listing_snapshots`, `shop_snapshots`, `reviews`), `review_hash(listing_id, created_ts, text) -> str`, `price_usd(price: dict | None) -> float | None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_staging.py`:
```python
import json
from datetime import datetime

from oppscan.staging import price_usd, review_hash, stage_run


def raw(con, run_id, endpoint, params, payload):
    con.execute("INSERT INTO raw_api VALUES (?, ?, ?, ?, ?)",
                [run_id, endpoint, json.dumps(params, sort_keys=True), datetime(2026, 10, 1), json.dumps(payload)])


def listing(lid, **over):
    base = {"listing_id": lid, "title": f"T{lid}", "tags": ["a"],
            "price": {"amount": 1299, "divisor": 100, "currency_code": "USD"},
            "num_favorers": 5, "views": 50, "original_creation_timestamp": 1700000000,
            "shop_id": 9, "listing_type": "download", "url": f"u{lid}"}
    base.update(over)
    return base


def search(con, run_id, seed, offset, results, count=999):
    raw(con, run_id, "/listings/active",
        {"keywords": seed, "limit": 100, "offset": offset, "sort_on": "score"},
        {"count": count, "results": results})


def test_price_usd():
    assert price_usd({"amount": 1299, "divisor": 100, "currency_code": "USD"}) == 12.99
    assert price_usd({"amount": 1299, "divisor": 100, "currency_code": "EUR"}) is None
    assert price_usd(None) is None


def test_stages_search_results_filtering_type_and_currency(con):
    search(con, "r1", "budget", 0, [
        listing(1),
        listing(2, listing_type="physical"),
        listing(3, price={"amount": 500, "divisor": 100, "currency_code": "EUR"}),
    ])
    search(con, "r1", "budget", 100, [listing(4)])
    stage_run(con, "r1")
    hits = con.execute("SELECT seed, listing_id, search_rank FROM search_hits ORDER BY listing_id").fetchall()
    assert hits == [("budget", 1, 1), ("budget", 4, 101)]
    assert con.execute("SELECT total_count FROM seed_counts").fetchall() == [(999,)]
    row = con.execute("SELECT price_usd, created_at, shop_id FROM listing_snapshots WHERE listing_id = 1").fetchone()
    assert row == (12.99, datetime(2023, 11, 14, 22, 13, 20), 9)


def test_same_listing_in_two_seeds(con):
    search(con, "r1", "a", 0, [listing(1)])
    search(con, "r1", "b", 0, [listing(9), listing(1)])
    stage_run(con, "r1")
    assert con.execute("SELECT seed, search_rank FROM search_hits WHERE listing_id = 1 ORDER BY seed").fetchall() == [("a", 1), ("b", 2)]
    assert con.execute("SELECT count(*) FROM listing_snapshots").fetchone()[0] == 2


def test_stages_shop(con):
    raw(con, "r1", "/shops/9", {}, {"shop_id": 9, "shop_name": "S", "transaction_sold_count": 100,
                                     "review_count": 10, "review_average": 4.8, "create_date": 1600000000})
    stage_run(con, "r1")
    assert con.execute("SELECT shop_name, transaction_sold_count, review_average FROM shop_snapshots").fetchone() == ("S", 100, 4.8)


def test_reviews_dedupe_across_runs(con):
    review = {"rating": 2, "review": "Confusing", "create_timestamp": 1750000000}
    raw(con, "r1", "/listings/1/reviews", {"limit": 100, "offset": 0}, {"count": 1, "results": [review]})
    raw(con, "r2", "/listings/1/reviews", {"limit": 100, "offset": 0}, {"count": 1, "results": [review]})
    stage_run(con, "r1")
    stage_run(con, "r2")
    rows = con.execute("SELECT review_hash, listing_id, rating, text, first_seen_run FROM reviews").fetchall()
    assert rows == [(review_hash(1, 1750000000, "Confusing"), 1, 2, "Confusing", "r1")]


def test_restaging_is_idempotent(con):
    search(con, "r1", "budget", 0, [listing(1)])
    stage_run(con, "r1")
    stage_run(con, "r1")
    assert con.execute("SELECT count(*) FROM search_hits").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM listing_snapshots").fetchone()[0] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_staging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oppscan.staging'`

- [ ] **Step 3: Implement `src/oppscan/staging.py`**

```python
"""Parse a run's raw_api JSON into the staged tables."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime

SHOP_RE = re.compile(r"^/shops/(\d+)$")
REVIEWS_RE = re.compile(r"^/listings/(\d+)/reviews$")
STAGED_TABLES = ("search_hits", "seed_counts", "listing_snapshots", "shop_snapshots")


def _ts(value) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value), UTC).replace(tzinfo=None)


def review_hash(listing_id: int, created_ts: int, text: str) -> str:
    return hashlib.sha256(f"{listing_id}|{created_ts}|{text}".encode()).hexdigest()[:16]


def price_usd(price: dict | None) -> float | None:
    if not price or price.get("currency_code") != "USD":
        return None
    return price["amount"] / price["divisor"]


def stage_run(con, run_id: str) -> None:
    for table in STAGED_TABLES:
        con.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_id])

    hits: dict[tuple[str, int], int] = {}
    counts: dict[str, int] = {}
    listings: dict[int, tuple] = {}
    shops: dict[int, tuple] = {}
    reviews: dict[str, tuple] = {}

    rows = con.execute("SELECT endpoint, request_key, payload FROM raw_api WHERE run_id = ?", [run_id]).fetchall()
    for endpoint, request_key, payload_json in rows:
        payload = json.loads(payload_json)
        params = json.loads(request_key)
        if endpoint == "/listings/active":
            seed, offset = params["keywords"], int(params["offset"])
            if offset == 0:
                counts[seed] = int(payload.get("count", 0))
            for i, item in enumerate(payload.get("results", [])):
                usd = price_usd(item.get("price"))
                if item.get("listing_type") != "download" or usd is None:
                    continue
                lid = int(item["listing_id"])
                rank = offset + i + 1
                hits[(seed, lid)] = min(rank, hits.get((seed, lid), rank))
                created = (item.get("original_creation_timestamp") or item.get("created_timestamp")
                           or item.get("creation_timestamp"))
                listings[lid] = (run_id, lid, item.get("title", ""), item.get("tags") or [], usd,
                                 int(item.get("num_favorers") or 0), int(item.get("views") or 0),
                                 _ts(created), int(item["shop_id"]), item["listing_type"], item.get("url"))
        elif SHOP_RE.match(endpoint):
            shop_id = int(payload["shop_id"])
            shops[shop_id] = (run_id, shop_id, payload.get("shop_name"),
                              int(payload.get("transaction_sold_count") or 0),
                              int(payload.get("review_count") or 0),
                              payload.get("review_average"), _ts(payload.get("create_date")))
        elif m := REVIEWS_RE.match(endpoint):
            lid = int(m.group(1))
            for r in payload.get("results", []):
                created = r.get("create_timestamp") or r.get("created_timestamp")
                text = r.get("review") or ""
                h = review_hash(lid, created, text)
                reviews[h] = (h, lid, int(r.get("rating") or 0), text, _ts(created), run_id)

    if hits:
        con.executemany("INSERT INTO search_hits VALUES (?, ?, ?, ?)",
                        [(run_id, seed, lid, rank) for (seed, lid), rank in hits.items()])
    if counts:
        con.executemany("INSERT INTO seed_counts VALUES (?, ?, ?)",
                        [(run_id, seed, n) for seed, n in counts.items()])
    if listings:
        con.executemany("INSERT INTO listing_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        list(listings.values()))
    if shops:
        con.executemany("INSERT INTO shop_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)", list(shops.values()))
    if reviews:
        con.executemany("INSERT OR IGNORE INTO reviews VALUES (?, ?, ?, ?, ?, ?)", list(reviews.values()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_staging.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/oppscan/staging.py tests/test_staging.py
git commit -m "feat: stage raw etsy json into tables"
```

---

### Task 6: LLM layer (schemas, prompts, Anthropic client, fake)

**Files:**
- Create: `src/oppscan/prompts.py`, `src/oppscan/llm.py`, `src/oppscan/fake_llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `prompts.SCHEMAS: dict[str, dict]` for tasks `"cluster"`, `"complaints"`, `"brief"`, `"summary"`; `prompts.render(task, payload) -> tuple[str, str]` (system, user).
  - `llm.LLMOutputError`, `llm.LLMClient` protocol with `call(task: str, payload: dict) -> dict`, `llm.validate(task, data) -> dict`, `llm.AnthropicLLM(con, client=None)`, `llm.TASKS: dict[str, tuple[str, str]]` (model, effort).
  - `fake_llm.FakeLLM()` with `.calls: list[tuple[str, dict]]`.
- Payload shapes (callers in Tasks 7, 9 and 11 must build exactly these):
  - cluster: `{"seeds": [{"seed": str, "titles": [str]}]}` → `{"niches": [{"name", "description", "seeds": [str]}]}`
  - complaints: `{"niche": str, "reviews": [{"id": str, "rating": int, "text": str}]}` → `{"themes": [{"theme", "fixable_by_product": bool, "review_ids": [str], "quotes": [str]}]}`
  - brief: `{"niche": {"name", "description"}, "metrics": {"reviews_90d", "entry_share", "median_price", "price_min", "price_max", "listing_count", "top3_share", "confidence"}, "complaints": [{"theme", "mentions", "quotes"}], "competitors": [{"title", "price_usd", "url", "reviews_90d"}]}` → `{"target_buyer", "core_features": [str], "differentiators": [str], "suggested_price_usd": number, "title_ideas": [str], "tags": [str]}`
  - summary: `{"top": [{"rank", "niche", "score", "confidence", "reviews_90d", "median_price"}], "dropped": int, "suspect": bool, "status_reasons": [str]}` → `{"summary": str}`

- [ ] **Step 1: Write the failing tests**

`tests/test_llm.py`:
```python
import json
from types import SimpleNamespace

import pytest

from oppscan.fake_llm import FakeLLM
from oppscan.llm import AnthropicLLM, LLMOutputError, validate
from oppscan.prompts import SCHEMAS, render

GOOD_CLUSTER = {"niches": [{"name": "Budget", "description": "d", "seeds": ["budget spreadsheet"]}]}
PAYLOAD = {"seeds": [{"seed": "budget spreadsheet", "titles": ["Budget Spreadsheet"]}]}


def message(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


class FakeAnthropic:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def test_schemas_are_strict():
    def check(schema):
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
            for sub in schema["properties"].values():
                check(sub)
        if schema.get("type") == "array":
            check(schema["items"])

    for schema in SCHEMAS.values():
        check(schema)


def test_render_embeds_payload():
    system, user = render("cluster", PAYLOAD)
    assert "Etsy" in system
    assert '"budget spreadsheet"' in user


def test_validate_rejects_bad_shape():
    with pytest.raises(LLMOutputError):
        validate("cluster", {"niches": [{"name": "x"}]})


def test_anthropic_llm_returns_and_caches(con):
    client = FakeAnthropic(message(json.dumps(GOOD_CLUSTER)))
    llm = AnthropicLLM(con, client=client)
    assert llm.call("cluster", PAYLOAD) == GOOD_CLUSTER
    assert llm.call("cluster", PAYLOAD) == GOOD_CLUSTER
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["model"] == "claude-sonnet-5-5"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["format"]["schema"] == SCHEMAS["cluster"]
    assert request["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in request["betas"]


def test_anthropic_llm_retries_once_on_invalid_output(con):
    client = FakeAnthropic(message("not json"), message(json.dumps(GOOD_CLUSTER)))
    assert AnthropicLLM(con, client=client).call("cluster", PAYLOAD) == GOOD_CLUSTER
    assert len(client.requests) == 2


def test_anthropic_llm_raises_after_two_failures(con):
    client = FakeAnthropic(message('{"niches": [{}]}'), message("still bad"))
    with pytest.raises(LLMOutputError):
        AnthropicLLM(con, client=client).call("cluster", PAYLOAD)


def test_anthropic_llm_treats_refusal_as_failure(con):
    client = FakeAnthropic(message("", "refusal"), message("", "refusal"))
    with pytest.raises(LLMOutputError, match="refusal"):
        AnthropicLLM(con, client=client).call("cluster", PAYLOAD)


def test_fake_llm_outputs_validate():
    fake = FakeLLM()
    clusters = fake.call("cluster", {"seeds": [{"seed": "budget a", "titles": []},
                                               {"seed": "budget b", "titles": []},
                                               {"seed": "notion c", "titles": []}]})
    assert [n["seeds"] for n in clusters["niches"]] == [["budget a", "budget b"], ["notion c"]]
    themes = fake.call("complaints", {"niche": "x", "reviews": [{"id": "a", "rating": 2, "text": "bad"},
                                                                {"id": "b", "rating": 5, "text": "wish"}]})
    assert themes["themes"][0]["review_ids"] == ["a"]
    brief = fake.call("brief", {"niche": {"name": "Budget", "description": "d"},
                                "metrics": {"median_price": 12.5}, "complaints": [], "competitors": []})
    assert brief["suggested_price_usd"] == 12.5
    assert fake.call("summary", {"top": [], "dropped": 0, "suspect": False, "status_reasons": []})["summary"]
    assert [t for t, _ in fake.calls] == ["cluster", "complaints", "brief", "summary"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oppscan.fake_llm'`

- [ ] **Step 3: Implement `src/oppscan/prompts.py`**

```python
"""JSON schemas and prompt text for each LLM task."""
from __future__ import annotations

import json


def _obj(**properties) -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def _arr(items: dict) -> dict:
    return {"type": "array", "items": items}


STR = {"type": "string"}

SCHEMAS: dict[str, dict] = {
    "cluster": _obj(niches=_arr(_obj(name=STR, description=STR, seeds=_arr(STR)))),
    "complaints": _obj(themes=_arr(_obj(theme=STR, fixable_by_product={"type": "boolean"},
                                        review_ids=_arr(STR), quotes=_arr(STR)))),
    "brief": _obj(target_buyer=STR, core_features=_arr(STR), differentiators=_arr(STR),
                  suggested_price_usd={"type": "number"}, title_ideas=_arr(STR), tags=_arr(STR)),
    "summary": _obj(summary=STR),
}

SYSTEM = ("You analyse Etsy marketplace data for a seller of digital spreadsheet and Notion "
          "templates. Base every statement on the data provided and be specific.")

INSTRUCTIONS: dict[str, str] = {
    "cluster": (
        "Group these search seeds into product niches. Each seed comes with the titles of its "
        "top-ranked listings. A niche is a specific product a buyer searches for, for example "
        "'ADHD-friendly Notion life planner' rather than 'planner'. Put every seed in exactly one "
        "niche and copy each seed string exactly as given. Aim for roughly one niche per two or "
        "three seeds; a seed may be a niche on its own."
    ),
    "complaints": (
        "These are reviews of the top listings in one niche. Identify recurring complaint or wish "
        "themes. For each theme give the ids of the reviews that mention it, up to three short "
        "verbatim quotes (under 20 words each), and set fixable_by_product to true only if a "
        "better-designed template would fix it (for example confusing instructions, a missing "
        "feature, breaking on mobile). Seller-service issues (slow replies, delivery, refunds) are "
        "false. Ignore pure praise. Return an empty list if there are no complaints."
    ),
    "brief": (
        "Write a product brief for a new digital template in this niche that would beat the "
        "current top listings. Use the complaints to decide what to do better. Keep the price "
        "within the observed price band unless the data justifies otherwise. title_ideas: three "
        "Etsy titles under 140 characters. tags: up to 13 Etsy tags, each under 20 characters."
    ),
    "summary": (
        "Write a three to five sentence plain summary of this research run for the seller: which "
        "niches look most promising and why, and any caveats about data quality. Factual tone."
    ),
}


def render(task: str, payload: dict) -> tuple[str, str]:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return SYSTEM, f"{INSTRUCTIONS[task]}\n\n<data>\n{data}\n</data>"
```

- [ ] **Step 4: Implement `src/oppscan/llm.py`**

```python
"""LLM access: one JSON-returning call per task, validated against its schema and cached."""
from __future__ import annotations

import hashlib
import json
from typing import Protocol

import jsonschema

from oppscan import prompts
from oppscan.db import utcnow

# task -> (model, effort)
TASKS: dict[str, tuple[str, str]] = {
    "cluster": ("claude-sonnet-5-5", "medium"),
    "complaints": ("claude-sonnet-5-5", "low"),
    "brief": ("claude-opus-5-5", "high"),
    "summary": ("claude-opus-5-5", "medium"),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMOutputError(Exception):
    """The model's output could not be used (refusal, truncation, invalid JSON or schema)."""


class LLMClient(Protocol):
    def call(self, task: str, payload: dict) -> dict: ...


def validate(task: str, data: dict) -> dict:
    try:
        jsonschema.validate(data, prompts.SCHEMAS[task])
    except jsonschema.ValidationError as e:
        raise LLMOutputError(f"{task}: {e.message}") from e
    return data


class AnthropicLLM:
    def __init__(self, con, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self._con = con
        self._client = client

    def call(self, task: str, payload: dict) -> dict:
        model, effort = TASKS[task]
        system, user = prompts.render(task, payload)
        key = hashlib.sha256(json.dumps([model, effort, system, user]).encode()).hexdigest()
        row = self._con.execute("SELECT response FROM llm_cache WHERE cache_key = ?", [key]).fetchone()
        if row is not None:
            return json.loads(row[0])
        last_error: LLMOutputError | None = None
        for _ in range(2):
            try:
                data = validate(task, self._request(model, effort, system, user, prompts.SCHEMAS[task]))
            except LLMOutputError as e:
                last_error = e
                continue
            self._con.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?)",
                              [key, task, json.dumps(data), utcnow()])
            return data
        raise last_error

    def _request(self, model: str, effort: str, system: str, user: str, schema: dict) -> dict:
        response = self._client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMOutputError(f"stop_reason={response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMOutputError("response had no text block")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMOutputError(f"invalid JSON: {e}") from e
```

- [ ] **Step 5: Implement `src/oppscan/fake_llm.py`**

```python
"""Deterministic stand-in for AnthropicLLM, used by tests and `oppscan run --fixtures`."""
from __future__ import annotations

from oppscan.llm import validate


class FakeLLM:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def call(self, task: str, payload: dict) -> dict:
        self.calls.append((task, payload))
        return validate(task, getattr(self, f"_{task}")(payload))

    def _cluster(self, payload: dict) -> dict:
        groups: dict[str, list[str]] = {}
        for item in payload["seeds"]:
            groups.setdefault(item["seed"].split()[0], []).append(item["seed"])
        return {"niches": [{"name": f"{word.title()} templates",
                            "description": f"Templates found by '{word}' searches",
                            "seeds": seeds} for word, seeds in sorted(groups.items())]}

    def _complaints(self, payload: dict) -> dict:
        negative = [r for r in payload["reviews"] if r["rating"] <= 4]
        if not negative:
            return {"themes": []}
        return {"themes": [{"theme": "Confusing setup instructions", "fixable_by_product": True,
                            "review_ids": [r["id"] for r in negative],
                            "quotes": [negative[0]["text"][:80]]}]}

    def _brief(self, payload: dict) -> dict:
        name = payload["niche"]["name"]
        fixes = [f"Fix: {c['theme']}" for c in payload["complaints"]][:3]
        return {"target_buyer": f"Buyers searching for {name.lower()}",
                "core_features": ["Clear setup guide", "Works in Excel and Google Sheets"],
                "differentiators": fixes or ["Cleaner design than current listings"],
                "suggested_price_usd": round(payload["metrics"].get("median_price") or 9.99, 2),
                "title_ideas": [f"{name} | Editable Digital Template"],
                "tags": ["template", "spreadsheet"]}

    def _summary(self, payload: dict) -> dict:
        return {"summary": f"Fixture run with {len(payload['top'])} ranked niches."}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v`
Expected: 8 passed

- [ ] **Step 7: Commit**

```bash
git add src/oppscan/prompts.py src/oppscan/llm.py src/oppscan/fake_llm.py tests/test_llm.py
git commit -m "feat: schema-validated, cached LLM layer with fake"
```

---

### Task 7: Niche clustering

**Files:**
- Create: `src/oppscan/clustering.py`
- Test: `tests/test_clustering.py`

**Interfaces:**
- Consumes: `LLMClient.call("cluster", ...)`, `Seeds`, `normalise` from config, and the `search_hits` and `listing_snapshots` tables.
- Produces: `cluster(con, run_id, seeds, llm) -> str` (returns the cluster_version, which equals `seeds.file_hash`; fills `niches`, `seed_niche`, `listing_niche`); `slugify(name) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_clustering.py`:
```python
from helpers import add_listing

from oppscan.clustering import cluster, slugify
from oppscan.config import Seeds
from oppscan.fake_llm import FakeLLM

SEEDS = Seeds(terms=("budget spreadsheet", "budget planner", "notion planner"), file_hash="v1")


class StubLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def call(self, task, payload):
        self.calls.append((task, payload))
        return self.response


def seed_data(con):
    add_listing(con, "r1", 1, seed="budget spreadsheet", rank=1, title="Budget Sheet")
    add_listing(con, "r1", 2, seed="budget spreadsheet", rank=2)
    add_listing(con, "r1", 2, seed="budget planner", rank=1)
    add_listing(con, "r1", 4, seed="notion planner", rank=1)
    add_listing(con, "r1", 5, seed="budget planner", rank=7)
    add_listing(con, "r1", 5, seed="notion planner", rank=2)


def niche_of(con, listing_id):
    return con.execute("SELECT niche_id FROM listing_niche WHERE run_id = 'r1' AND listing_id = ?",
                       [listing_id]).fetchone()[0]


def test_slugify():
    assert slugify("ADHD-friendly Notion Planner!") == "adhd-friendly-notion-planner"
    assert slugify("!!!") == "niche"


def test_cluster_groups_seeds_and_assigns_listings(con):
    seed_data(con)
    llm = FakeLLM()
    assert cluster(con, "r1", SEEDS, llm) == "v1"
    seed_map = dict(con.execute("SELECT seed, niche_id FROM seed_niche WHERE cluster_version = 'v1'").fetchall())
    assert seed_map == {"budget spreadsheet": "budget-templates", "budget planner": "budget-templates",
                        "notion planner": "notion-templates"}
    assert niche_of(con, 1) == "budget-templates"
    assert niche_of(con, 4) == "notion-templates"
    assert niche_of(con, 5) == "notion-templates"  # best rank (2) is under a notion seed
    payload = llm.calls[0][1]
    assert payload["seeds"][0] == {"seed": "budget spreadsheet", "titles": ["Budget Sheet", "Listing 2"]}


def test_cluster_is_cached_per_seed_file(con):
    seed_data(con)
    llm = FakeLLM()
    cluster(con, "r1", SEEDS, llm)
    cluster(con, "r1", SEEDS, llm)
    assert len(llm.calls) == 1


def test_unassigned_seeds_get_their_own_niche_and_unknown_seeds_are_ignored(con):
    seed_data(con)
    llm = StubLLM({"niches": [{"name": "Budget", "description": "d",
                               "seeds": ["Budget  Spreadsheet", "not a seed"]}]})
    cluster(con, "r1", SEEDS, llm)
    seed_map = dict(con.execute("SELECT seed, niche_id FROM seed_niche").fetchall())
    assert seed_map == {"budget spreadsheet": "budget", "budget planner": "budget-planner",
                        "notion planner": "notion-planner"}


def test_duplicate_niche_names_get_unique_ids(con):
    seed_data(con)
    llm = StubLLM({"niches": [
        {"name": "Budget", "description": "a", "seeds": ["budget spreadsheet"]},
        {"name": "Budget", "description": "b", "seeds": ["budget planner"]},
        {"name": "Notion", "description": "c", "seeds": ["notion planner"]},
    ]})
    cluster(con, "r1", SEEDS, llm)
    ids = sorted(r[0] for r in con.execute("SELECT niche_id FROM niches").fetchall())
    assert ids == ["budget", "budget-2", "notion"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_clustering.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oppscan.clustering'`

- [ ] **Step 3: Implement `src/oppscan/clustering.py`**

```python
"""Group seeds into niches (one LLM call, cached per seed file) and assign listings to niches."""
from __future__ import annotations

import re

from oppscan.config import Seeds, normalise
from oppscan.llm import LLMClient

TITLES_PER_SEED = 10


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:48] or "niche"


def cluster(con, run_id: str, seeds: Seeds, llm: LLMClient) -> str:
    version = seeds.file_hash
    exists = con.execute("SELECT count(*) FROM seed_niche WHERE cluster_version = ?", [version]).fetchone()[0]
    if not exists:
        _cluster_seeds(con, run_id, seeds, llm, version)
    _assign_listings(con, run_id, version)
    return version


def _cluster_seeds(con, run_id: str, seeds: Seeds, llm: LLMClient, version: str) -> None:
    payload = {"seeds": [{"seed": s, "titles": _top_titles(con, run_id, s)} for s in seeds.terms]}
    result = llm.call("cluster", payload)
    known = set(seeds.terms)
    assigned: dict[str, str] = {}
    niches: dict[str, tuple[str, str]] = {}
    for niche in result["niches"]:
        members = [s for s in (normalise(x) for x in niche["seeds"]) if s in known and s not in assigned]
        if not members:
            continue
        niche_id = _unique(slugify(niche["name"]), niches)
        niches[niche_id] = (niche["name"], niche["description"])
        for seed in members:
            assigned[seed] = niche_id
    for seed in seeds.terms:
        if seed not in assigned:
            niche_id = _unique(slugify(seed), niches)
            niches[niche_id] = (seed, f"Unclustered seed '{seed}'")
            assigned[seed] = niche_id
    con.executemany("INSERT INTO niches VALUES (?, ?, ?, ?)",
                    [(nid, version, name, desc) for nid, (name, desc) in niches.items()])
    con.executemany("INSERT INTO seed_niche VALUES (?, ?, ?)",
                    [(version, seed, nid) for seed, nid in assigned.items()])


def _unique(base: str, taken: dict) -> str:
    niche_id, i = base, 2
    while niche_id in taken:
        niche_id = f"{base}-{i}"
        i += 1
    return niche_id


def _top_titles(con, run_id: str, seed: str) -> list[str]:
    rows = con.execute(
        "SELECT l.title FROM search_hits h JOIN listing_snapshots l USING (run_id, listing_id) "
        "WHERE h.run_id = ? AND h.seed = ? ORDER BY h.search_rank LIMIT ?",
        [run_id, seed, TITLES_PER_SEED],
    ).fetchall()
    return [r[0] for r in rows]


def _assign_listings(con, run_id: str, version: str) -> None:
    con.execute("DELETE FROM listing_niche WHERE run_id = ?", [run_id])
    con.execute(
        """
        INSERT INTO listing_niche
        SELECT run_id, listing_id, niche_id, ? FROM (
            SELECT h.run_id, h.listing_id, sn.niche_id,
                   ROW_NUMBER() OVER (PARTITION BY h.listing_id ORDER BY h.search_rank, h.seed) AS rn
            FROM search_hits h
            JOIN seed_niche sn ON sn.seed = h.seed AND sn.cluster_version = ?
            WHERE h.run_id = ?
        ) WHERE rn = 1
        """,
        [version, version, run_id],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_clustering.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/oppscan/clustering.py tests/test_clustering.py
git commit -m "feat: seed clustering into niches and listing assignment"
```

---

### Task 8: Per-niche metrics

**Files:**
- Create: `src/oppscan/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `ScoringConfig`, the `listing_niche`, `search_hits`, `listing_snapshots`, `reviews`, `seed_counts`, `seed_niche`, `complaint_runs` and `complaints` tables.
- Produces: `top_listings(con, run_id, top_n) -> dict[str, list[int]]` (niche_id → listing ids ordered by best search rank); `NicheMetrics(niche_id, reviews_90d, active_listings, fav_delta, entry_share, gap_per_100, median_price, listing_count, top3_share)`; `compute_metrics(con, run_id, as_of, cfg, prev_run_id) -> list[NicheMetrics]` (sorted by niche_id).

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
from datetime import datetime

import pytest
from helpers import add_listing, add_niche, add_review, assign, make_cfg

from oppscan.metrics import compute_metrics, top_listings

AS_OF = datetime(2026, 10, 1)
RECENT = datetime(2026, 9, 20)
CFG = make_cfg(top_n_per_niche=4)


def build(con):
    add_niche(con, "a", ["s1", "s2"])
    specs = [  # listing_id, seed, rank, price, shop, created
        (1, "s1", 1, 10.0, 1, datetime(2026, 3, 1)),
        (2, "s1", 2, 20.0, 2, datetime(2024, 1, 1)),
        (3, "s2", 1, 30.0, 3, datetime(2024, 1, 1)),
        (4, "s2", 4, 40.0, 4, datetime(2024, 1, 1)),
        (5, "s2", 9, 50.0, 5, datetime(2024, 1, 1)),
    ]
    for lid, seed, rank, price, shop, created in specs:
        add_listing(con, "r1", lid, seed=seed, rank=rank, price=price, shop_id=shop, created=created,
                    favs={1: 15, 2: 25}.get(lid, 0))
        assign(con, "r1", lid, "a")
    for i in range(3):
        add_review(con, 1, RECENT.replace(hour=i))
    add_review(con, 2, datetime(2026, 9, 1))
    add_review(con, 2, datetime(2026, 6, 1))  # outside the 90-day window
    add_review(con, 3, RECENT)
    add_review(con, 4, RECENT)
    for i in range(5):
        add_review(con, 5, RECENT.replace(hour=i))  # listing 5 is outside the top 4
    con.execute("INSERT INTO seed_counts VALUES ('r1', 's1', 500), ('r1', 's2', 800)")


def test_top_listings_orders_by_best_rank_and_limits(con):
    build(con)
    assert top_listings(con, "r1", 4) == {"a": [1, 3, 2, 4]}


def test_compute_metrics(con):
    build(con)
    [m] = compute_metrics(con, "r1", AS_OF, CFG, None)
    assert m.niche_id == "a"
    assert m.reviews_90d == 6
    assert m.active_listings == 4
    assert m.entry_share == pytest.approx(0.5)
    assert m.top3_share == pytest.approx(5 / 6)
    assert m.median_price == 25.0
    assert m.listing_count == 800
    assert m.fav_delta is None
    assert m.gap_per_100 is None


def test_gap_uses_fixable_mentions_over_all_reviews(con):
    build(con)
    con.execute("INSERT INTO complaint_runs VALUES ('r1', 'a', 'ok')")
    con.execute("INSERT INTO complaints VALUES ('r1', 'a', 'confusing', TRUE, 2, ['q'], ['h1', 'h2'])")
    con.execute("INSERT INTO complaints VALUES ('r1', 'a', 'slow seller', FALSE, 5, ['q'], ['h3'])")
    [m] = compute_metrics(con, "r1", AS_OF, CFG, None)
    assert m.gap_per_100 == pytest.approx(200 / 7)


def test_fav_delta_against_previous_run(con):
    build(con)
    add_listing(con, "r0", 1, seed="s1", rank=1, favs=10)
    add_listing(con, "r0", 2, seed="s1", rank=2, favs=30)
    [m] = compute_metrics(con, "r1", AS_OF, CFG, "r0")
    assert m.fav_delta == 5  # +5 on listing 1, the -5 on listing 2 is floored at 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oppscan.metrics'`

- [ ] **Step 3: Implement `src/oppscan/metrics.py`**

```python
"""Raw per-niche metrics for one run, computed over each niche's top search results."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median

from oppscan.config import ScoringConfig


@dataclass(frozen=True)
class NicheMetrics:
    niche_id: str
    reviews_90d: int
    active_listings: int
    fav_delta: int | None
    entry_share: float
    gap_per_100: float | None
    median_price: float | None
    listing_count: int
    top3_share: float


def top_listings(con, run_id: str, top_n: int) -> dict[str, list[int]]:
    rows = con.execute(
        """
        SELECT niche_id, listing_id FROM (
            SELECT ln.niche_id, h.listing_id,
                   ROW_NUMBER() OVER (PARTITION BY ln.niche_id
                                      ORDER BY MIN(h.search_rank), h.listing_id) AS rn
            FROM listing_niche ln
            JOIN search_hits h ON h.run_id = ln.run_id AND h.listing_id = ln.listing_id
            WHERE ln.run_id = ?
            GROUP BY ln.niche_id, h.listing_id
        ) WHERE rn <= ? ORDER BY niche_id, rn
        """,
        [run_id, top_n],
    ).fetchall()
    out: dict[str, list[int]] = {}
    for niche_id, listing_id in rows:
        out.setdefault(niche_id, []).append(listing_id)
    return out


def compute_metrics(con, run_id: str, as_of: datetime, cfg: ScoringConfig,
                    prev_run_id: str | None) -> list[NicheMetrics]:
    tops = top_listings(con, run_id, cfg.top_n_per_niche)
    if not tops:
        return []
    all_ids = sorted({lid for lids in tops.values() for lid in lids})
    window_start = as_of - timedelta(days=cfg.review_window_days)
    new_cutoff = as_of - timedelta(days=cfg.new_listing_days)

    snap = {lid: (price, favs, created, shop) for lid, price, favs, created, shop in con.execute(
        "SELECT listing_id, price_usd, num_favorers, created_at, shop_id FROM listing_snapshots "
        "WHERE run_id = ? AND list_contains(?::BIGINT[], listing_id)", [run_id, all_ids]).fetchall()}
    prev_favs = dict(con.execute(
        "SELECT listing_id, num_favorers FROM listing_snapshots "
        "WHERE run_id = ? AND list_contains(?::BIGINT[], listing_id)", [prev_run_id, all_ids]).fetchall()
    ) if prev_run_id else {}
    recent = dict(con.execute(
        "SELECT listing_id, count(*) FROM reviews WHERE list_contains(?::BIGINT[], listing_id) "
        "AND created_at > ? AND created_at <= ? GROUP BY listing_id",
        [all_ids, window_start, as_of]).fetchall())
    lifetime = dict(con.execute(
        "SELECT listing_id, count(*) FROM reviews WHERE list_contains(?::BIGINT[], listing_id) "
        "AND created_at <= ? GROUP BY listing_id", [all_ids, as_of]).fetchall())
    listing_counts = dict(con.execute(
        """
        SELECT sn.niche_id, max(sc.total_count) FROM seed_counts sc
        JOIN seed_niche sn ON sn.seed = sc.seed
         AND sn.cluster_version = (SELECT any_value(cluster_version) FROM listing_niche WHERE run_id = ?)
        WHERE sc.run_id = ? GROUP BY sn.niche_id
        """, [run_id, run_id]).fetchall())
    status = dict(con.execute("SELECT niche_id, status FROM complaint_runs WHERE run_id = ?", [run_id]).fetchall())
    fixable = dict(con.execute(
        "SELECT niche_id, sum(mentions) FROM complaints WHERE run_id = ? AND fixable GROUP BY niche_id",
        [run_id]).fetchall())

    out = []
    for niche_id, lids in sorted(tops.items()):
        rev = {lid: recent.get(lid, 0) for lid in lids}
        total = sum(rev.values())
        new_reviews = sum(n for lid, n in rev.items() if snap[lid][2] is not None and snap[lid][2] >= new_cutoff)
        by_shop: Counter[int] = Counter()
        for lid, n in rev.items():
            by_shop[snap[lid][3]] += n
        reviewed = sum(lifetime.get(lid, 0) for lid in lids)
        gap = None
        if status.get(niche_id) == "ok" and reviewed:
            gap = 100.0 * (fixable.get(niche_id) or 0) / reviewed
        fav_delta = None
        if prev_run_id:
            fav_delta = sum(max(0, snap[lid][1] - prev_favs[lid]) for lid in lids if lid in prev_favs)
        out.append(NicheMetrics(
            niche_id=niche_id,
            reviews_90d=total,
            active_listings=sum(1 for n in rev.values() if n > 0),
            fav_delta=fav_delta,
            entry_share=new_reviews / total if total else 0.0,
            gap_per_100=gap,
            median_price=median(snap[lid][0] for lid in lids),
            listing_count=int(listing_counts.get(niche_id) or 0),
            top3_share=sum(sorted(by_shop.values(), reverse=True)[:3]) / total if total else 0.0,
        ))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/oppscan/metrics.py tests/test_metrics.py
git commit -m "feat: per-niche demand, entry, gap, price and crowding metrics"
```

---

### Task 9: Complaint mining

**Files:**
- Create: `src/oppscan/complaints.py`
- Test: `tests/test_complaints.py`

**Interfaces:**
- Consumes: `metrics.top_listings`, `LLMClient.call("complaints", ...)`, `LLMOutputError`, `db.cluster_version`.
- Produces: `mine_complaints(con, run_id, llm, top_n) -> list[str]` (failure reasons). Fills `complaints` and `complaint_runs` (status `ok` / `no_reviews` / `failed`).

- [ ] **Step 1: Write the failing tests**

`tests/test_complaints.py`:
```python
from datetime import datetime

from helpers import add_listing, add_niche, add_review, assign

from oppscan.complaints import mine_complaints
from oppscan.fake_llm import FakeLLM
from oppscan.llm import LLMOutputError

D = datetime(2026, 9, 1)


def build(con):
    add_niche(con, "a", ["s1"], name="Budget")
    add_niche(con, "b", ["s2"], name="Notion")
    add_listing(con, "r1", 1, seed="s1", rank=1)
    add_listing(con, "r1", 2, seed="s2", rank=1)
    assign(con, "r1", 1, "a")
    assign(con, "r1", 2, "b")
    neg = add_review(con, 1, D, rating=2, text="Instructions were confusing")
    add_review(con, 1, D.replace(day=2), rating=5, text="Love it")
    wish = add_review(con, 1, D.replace(day=3), rating=5, text="Wish it had dark mode")
    add_review(con, 2, D, rating=5, text="Love it")
    return neg, wish


def statuses(con):
    return dict(con.execute("SELECT niche_id, status FROM complaint_runs WHERE run_id = 'r1'").fetchall())


def test_mines_fixable_complaints(con):
    neg, wish = build(con)
    llm = FakeLLM()
    assert mine_complaints(con, "r1", llm, top_n=20) == []
    rows = con.execute("SELECT niche_id, theme, fixable, mentions, review_hashes FROM complaints").fetchall()
    assert rows == [("a", "Confusing setup instructions", True, 1, [neg])]
    assert statuses(con) == {"a": "ok", "b": "no_reviews"}
    sent = llm.calls[0][1]
    assert sent["niche"] == "Budget"
    assert {r["id"] for r in sent["reviews"]} == {neg, wish}


def test_llm_failure_marks_niche_failed(con):
    build(con)

    class Failing:
        def call(self, task, payload):
            raise LLMOutputError("boom")

    reasons = mine_complaints(con, "r1", Failing(), top_n=20)
    assert reasons == ["complaints a: boom"]
    assert statuses(con)["a"] == "failed"


def test_hallucinated_review_ids_are_dropped(con):
    build(con)

    class Hallucinating:
        def call(self, task, payload):
            return {"themes": [{"theme": "x", "fixable_by_product": True, "review_ids": ["nope"], "quotes": []}]}

    mine_complaints(con, "r1", Hallucinating(), top_n=20)
    assert con.execute("SELECT count(*) FROM complaints").fetchone()[0] == 0
    assert statuses(con)["a"] == "ok"


def test_rerun_replaces_previous_results(con):
    build(con)
    mine_complaints(con, "r1", FakeLLM(), top_n=20)
    mine_complaints(con, "r1", FakeLLM(), top_n=20)
    assert con.execute("SELECT count(*) FROM complaints").fetchone()[0] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_complaints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oppscan.complaints'`

- [ ] **Step 3: Implement `src/oppscan/complaints.py`**

```python
"""Mine complaint and wish themes from reviews of each niche's top listings."""
from __future__ import annotations

import re

from oppscan.db import cluster_version
from oppscan.llm import LLMClient, LLMOutputError
from oppscan.metrics import top_listings

WISH_RE = re.compile(r"\b(wish|would be nice|only issue)\b", re.IGNORECASE)
MAX_REVIEWS = 150


def mine_complaints(con, run_id: str, llm: LLMClient, top_n: int) -> list[str]:
    con.execute("DELETE FROM complaints WHERE run_id = ?", [run_id])
    con.execute("DELETE FROM complaint_runs WHERE run_id = ?", [run_id])
    names = dict(con.execute("SELECT niche_id, name FROM niches WHERE cluster_version = ?",
                             [cluster_version(con, run_id)]).fetchall())
    failures: list[str] = []
    for niche_id, lids in top_listings(con, run_id, top_n).items():
        reviews = _candidate_reviews(con, lids)
        if not reviews:
            _set_status(con, run_id, niche_id, "no_reviews")
            continue
        try:
            result = llm.call("complaints", {"niche": names.get(niche_id, niche_id), "reviews": reviews})
        except LLMOutputError as e:
            _set_status(con, run_id, niche_id, "failed")
            failures.append(f"complaints {niche_id}: {e}")
            continue
        valid = {r["id"] for r in reviews}
        for theme in result["themes"]:
            ids = sorted(set(theme["review_ids"]) & valid)
            if ids:
                con.execute("INSERT INTO complaints VALUES (?, ?, ?, ?, ?, ?, ?)",
                            [run_id, niche_id, theme["theme"], theme["fixable_by_product"],
                             len(ids), theme["quotes"][:3], ids])
        _set_status(con, run_id, niche_id, "ok")
    return failures


def _set_status(con, run_id: str, niche_id: str, status: str) -> None:
    con.execute("INSERT INTO complaint_runs VALUES (?, ?, ?)", [run_id, niche_id, status])


def _candidate_reviews(con, listing_ids: list[int]) -> list[dict]:
    rows = con.execute(
        "SELECT review_hash, rating, text FROM reviews "
        "WHERE list_contains(?::BIGINT[], listing_id) AND text <> '' ORDER BY created_at DESC",
        [listing_ids],
    ).fetchall()
    picked = [{"id": h, "rating": rating, "text": text} for h, rating, text in rows
              if rating <= 4 or WISH_RE.search(text)]
    return picked[:MAX_REVIEWS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_complaints.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/oppscan/complaints.py tests/test_complaints.py
git commit -m "feat: complaint mining per niche"
```

---

### Task 10: Scoring

**Files:**
- Create: `src/oppscan/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `NicheMetrics`, `ScoringConfig`.
- Produces: `percentile_ranks(values) -> list[float]`; `NicheScore(metrics, pct_demand, pct_entry, pct_gap, pct_price, pct_crowding, score, confidence, gap_missing, dropped, dropped_reason, rank)`; `score_niches(metrics, cfg) -> list[NicheScore]` (same order as the input); `sanity_check(scores, seed_niche: dict[str, str], known_big) -> list[str]`; `save_scores(con, run_id, scores)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_scoring.py`:
```python
import pytest
from helpers import make_cfg

from oppscan.metrics import NicheMetrics
from oppscan.scoring import percentile_ranks, sanity_check, save_scores, score_niches


def m(niche_id, reviews=10, active=1, fav=None, entry=0.5, gap=1.0, price=10.0, count=100, top3=0.5):
    return NicheMetrics(niche_id, reviews, active, fav, entry, gap, price, count, top3)


def by_id(scores):
    return {s.metrics.niche_id: s for s in scores}


def test_percentile_ranks():
    assert percentile_ranks([]) == []
    assert percentile_ranks([7]) == [0.5]
    assert percentile_ranks([3, 1, 2]) == [1.0, 0.0, 0.5]
    assert percentile_ranks([1, 1, 2]) == [0.25, 0.25, 1.0]


def test_hand_calculated_scores_floor_and_ranks():
    scores = by_id(score_niches([
        m("a", reviews=10, active=2, entry=0.5, gap=5.0, price=10, count=100, top3=0.5),
        m("b", reviews=50, active=10, entry=0.1, gap=None, price=20, count=1000, top3=0.9),
        m("c", reviews=30, active=9, entry=0.9, gap=1.0, price=5, count=10, top3=0.2),
    ], make_cfg()))
    assert scores["a"].score == pytest.approx(0.325)
    assert scores["b"].score == pytest.approx(0.45)
    assert scores["c"].score == pytest.approx(0.425)
    assert scores["b"].gap_missing and scores["b"].pct_gap == 0.5
    assert scores["a"].dropped and scores["a"].dropped_reason == "demand_floor" and scores["a"].rank is None
    assert (scores["b"].rank, scores["c"].rank) == (1, 2)
    assert scores["b"].confidence == "high" and scores["c"].confidence == "high"
    assert scores["a"].confidence == "low"


def test_high_confidence_ranks_above_low_confidence():
    scores = by_id(score_niches([
        m("x", reviews=100, active=2),
        m("y", reviews=50, active=10),
    ], make_cfg(demand_floor_pct=0.0)))
    assert scores["x"].score > scores["y"].score
    assert (scores["y"].rank, scores["x"].rank) == (1, 2)


def test_demand_blends_favourites_when_every_niche_has_them():
    scores = score_niches([m("p", reviews=10, fav=5), m("q", reviews=20, fav=1)], make_cfg())
    assert [s.pct_demand for s in scores] == [0.5, 0.5]


def test_sanity_check():
    scores = score_niches([m("big", reviews=1), m("other", reviews=50)], make_cfg(demand_floor_pct=0.0))
    problems = sanity_check(scores, {"monthly budget": "big"}, ("monthly budget", "missing seed"))
    assert problems == [
        "known-big niche 'big' has demand percentile 0.00 (below 0.50)",
        "known-big seed 'missing seed' is not in seeds.yaml",
    ]
    assert sanity_check(scores, {"x": "other"}, ("x",)) == []


def test_save_scores_round_trip(con):
    scores = score_niches([m("a", reviews=1), m("b", reviews=40, active=9)], make_cfg())
    save_scores(con, "r1", scores)
    save_scores(con, "r1", scores)
    rows = con.execute("SELECT niche_id, rank, dropped FROM niche_scores ORDER BY niche_id").fetchall()
    assert rows == [("a", None, True), ("b", 1, False)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oppscan.scoring'`

- [ ] **Step 3: Implement `src/oppscan/scoring.py`**

```python
"""Turn niche metrics into percentile scores, confidence labels, drops and ranks."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from oppscan.config import ScoringConfig
from oppscan.metrics import NicheMetrics


@dataclass(frozen=True)
class NicheScore:
    metrics: NicheMetrics
    pct_demand: float
    pct_entry: float
    pct_gap: float
    pct_price: float
    pct_crowding: float
    score: float
    confidence: str
    gap_missing: bool
    dropped: bool
    dropped_reason: str | None
    rank: int | None


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """Average-rank percentiles in [0, 1]; ties share a value; a single value maps to 0.5."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = ((i + j) / 2) / (n - 1)
        i = j + 1
    return out


def _mean(a: list[float], b: list[float]) -> list[float]:
    return [(x + y) / 2 for x, y in zip(a, b)]


def score_niches(metrics: list[NicheMetrics], cfg: ScoringConfig) -> list[NicheScore]:
    if not metrics:
        return []
    w = cfg.weights
    demand = percentile_ranks([m.reviews_90d for m in metrics])
    if all(m.fav_delta is not None for m in metrics):
        demand = _mean(demand, percentile_ranks([m.fav_delta for m in metrics]))
    entry = percentile_ranks([m.entry_share for m in metrics])
    price = percentile_ranks([m.median_price or 0.0 for m in metrics])
    crowding = _mean(percentile_ranks([m.listing_count for m in metrics]),
                     percentile_ranks([m.top3_share for m in metrics]))
    gap = [0.5] * len(metrics)
    known = [i for i, m in enumerate(metrics) if m.gap_per_100 is not None]
    for i, p in zip(known, percentile_ranks([metrics[i].gap_per_100 for i in known])):
        gap[i] = p

    scores = []
    for i, m in enumerate(metrics):
        dropped = demand[i] < cfg.demand_floor_pct
        high = (m.reviews_90d >= cfg.high_conf_min_reviews
                and m.active_listings >= cfg.high_conf_min_listings)
        scores.append(NicheScore(
            metrics=m, pct_demand=demand[i], pct_entry=entry[i], pct_gap=gap[i],
            pct_price=price[i], pct_crowding=crowding[i],
            score=(w.demand * demand[i] + w.entry * entry[i] + w.gap * gap[i]
                   + w.price * price[i] - w.crowding * crowding[i]),
            confidence="high" if high else "low",
            gap_missing=m.gap_per_100 is None,
            dropped=dropped, dropped_reason="demand_floor" if dropped else None, rank=None,
        ))
    ranked = sorted((s for s in scores if not s.dropped),
                    key=lambda s: (s.confidence != "high", -s.score, s.metrics.niche_id))
    rank_of = {s.metrics.niche_id: r for r, s in enumerate(ranked, start=1)}
    return [replace(s, rank=rank_of.get(s.metrics.niche_id)) for s in scores]


def sanity_check(scores: list[NicheScore], seed_niche: dict[str, str],
                 known_big: Sequence[str]) -> list[str]:
    by_id = {s.metrics.niche_id: s for s in scores}
    problems = []
    for seed in known_big:
        niche_id = seed_niche.get(seed)
        if niche_id is None:
            problems.append(f"known-big seed '{seed}' is not in seeds.yaml")
        elif niche_id not in by_id:
            problems.append(f"known-big seed '{seed}' produced no niche data")
        elif by_id[niche_id].pct_demand < 0.5:
            problems.append(f"known-big niche '{niche_id}' has demand percentile "
                            f"{by_id[niche_id].pct_demand:.2f} (below 0.50)")
    return problems


def save_scores(con, run_id: str, scores: list[NicheScore]) -> None:
    con.execute("DELETE FROM niche_scores WHERE run_id = ?", [run_id])
    if not scores:
        return
    con.executemany(
        "INSERT INTO niche_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(run_id, s.metrics.niche_id, s.metrics.reviews_90d, s.metrics.active_listings,
          s.metrics.fav_delta, s.metrics.entry_share, s.metrics.gap_per_100, s.gap_missing,
          s.metrics.median_price, s.metrics.listing_count, s.metrics.top3_share,
          s.pct_demand, s.pct_entry, s.pct_gap, s.pct_price, s.pct_crowding,
          s.score, s.confidence, s.dropped, s.dropped_reason, s.rank) for s in scores],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/oppscan/scoring.py tests/test_scoring.py
git commit -m "feat: percentile scoring, confidence, demand floor and sanity check"
```

---

### Task 11: Briefs, summary and report rendering

**Files:**
- Create: `src/oppscan/briefs.py`, `src/oppscan/report.py`, `src/oppscan/templates/report.html.j2`, `src/oppscan/templates/report.md.j2`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `metrics.top_listings`, `db.get_run`, `db.previous_run`, `db.cluster_version`, `LLMClient`, the `niche_scores`, `complaints` and `niches` tables.
- Produces:
  - `briefs.competitors(con, run_id, niche_id, as_of, cfg, n=3) -> list[dict]` (keys `title`, `price_usd`, `url`, `reviews_90d`)
  - `briefs.brief_payload(con, run_id, niche_id, as_of, cfg) -> dict`
  - `briefs.write_briefs(con, run_id, llm, cfg, as_of) -> list[str]`
  - `briefs.write_summary(con, run_id, llm, cfg) -> list[str]`
  - `report.rank_change(current, previous) -> str`
  - `report.explain(row: dict) -> list[str]`
  - `report.build_context(con, run_id, cfg) -> dict`
  - `report.render_report(con, run_id, cfg, out_dir) -> tuple[Path, Path]` (writes `<run_id>.html` and `<run_id>.md`)

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:
```python
from datetime import datetime

from helpers import add_listing, add_niche, add_review, add_run, assign, make_cfg

from oppscan import db
from oppscan.briefs import competitors, write_briefs, write_summary
from oppscan.fake_llm import FakeLLM
from oppscan.metrics import compute_metrics
from oppscan.report import explain, rank_change, render_report
from oppscan.scoring import save_scores, score_niches

AS_OF = datetime(2026, 10, 1)
CFG = make_cfg(demand_floor_pct=0.0, high_conf_min_reviews=1, high_conf_min_listings=1)


def build(con):
    add_run(con, "r1", AS_OF)
    add_niche(con, "a", ["s1"], name="Budget templates")
    add_niche(con, "b", ["s2"], name="Notion templates")
    add_listing(con, "r1", 1, seed="s1", rank=1, price=12.0, title="Best Budget")
    add_listing(con, "r1", 2, seed="s1", rank=2, price=8.0, title="Other Budget")
    add_listing(con, "r1", 3, seed="s2", rank=1, price=20.0, title="Notion Thing")
    for lid, niche in [(1, "a"), (2, "a"), (3, "b")]:
        assign(con, "r1", lid, niche)
    for i in range(3):
        add_review(con, 1, datetime(2026, 9, 10 + i))
    add_review(con, 2, datetime(2026, 9, 5), rating=2, text="Confusing")
    add_review(con, 3, datetime(2026, 9, 5))
    con.execute("INSERT INTO complaint_runs VALUES ('r1', 'a', 'ok'), ('r1', 'b', 'no_reviews')")
    con.execute("INSERT INTO complaints VALUES ('r1', 'a', 'Confusing setup', TRUE, 1, ['Confusing'], ['h'])")
    save_scores(con, "r1", score_niches(compute_metrics(con, "r1", AS_OF, CFG, None), CFG))
    db.finish_run(con, "r1", "complete", AS_OF)


def test_rank_change():
    assert rank_change(3, None) == "new"
    assert rank_change(2, 5) == "▲3"
    assert rank_change(5, 2) == "▼3"
    assert rank_change(4, 4) == "–"


def test_explain_mentions_each_component():
    row = {"reviews_90d": 40, "active_listings": 9, "fav_delta": None, "pct_demand": 0.8,
           "entry_share": 0.3, "pct_entry": 0.6, "gap_missing": True, "gap_per_100": None, "pct_gap": 0.5,
           "median_price": 9.5, "pct_price": 0.4, "listing_count": 1200, "top3_share": 0.45,
           "pct_crowding": 0.7}
    lines = explain(row)
    assert lines[0].startswith("Demand: 40 reviews in the last 90 days across 9 listings")
    assert "no complaint data" in lines[2]
    assert "$9.50" in lines[3]
    assert "1,200 competing listings" in lines[4]


def test_competitors_ordered_by_recent_reviews(con):
    build(con)
    comps = competitors(con, "r1", "a", AS_OF, CFG)
    assert [c["title"] for c in comps] == ["Best Budget", "Other Budget"]
    assert comps[0]["reviews_90d"] == 3


def test_briefs_summary_and_render(con, tmp_path):
    build(con)
    llm = FakeLLM()
    assert write_briefs(con, "r1", llm, CFG, AS_OF) == []
    assert write_summary(con, "r1", llm, CFG) == []
    brief_payload = next(p for t, p in llm.calls if t == "brief" and p["niche"]["name"] == "Budget templates")
    assert brief_payload["complaints"][0]["theme"] == "Confusing setup"
    assert brief_payload["metrics"]["price_min"] == 8.0

    html_path, md_path = render_report(con, "r1", CFG, tmp_path)
    html = html_path.read_text()
    md = md_path.read_text()
    assert html_path.name == "r1.html" and md_path.name == "r1.md"
    assert "Budget templates" in html and "Notion templates" in html
    assert "Draft brief" in html and "Confusing setup" in html
    assert "Fixture run with 2 ranked niches." in html
    assert "| 1 |" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oppscan.briefs'`

- [ ] **Step 3: Implement `src/oppscan/briefs.py`**

```python
"""Competitor lookup and LLM-written product briefs and run summary."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from oppscan import db
from oppscan.config import ScoringConfig
from oppscan.llm import LLMClient, LLMOutputError
from oppscan.metrics import top_listings


def competitors(con, run_id: str, niche_id: str, as_of: datetime, cfg: ScoringConfig,
                n: int = 3) -> list[dict]:
    lids = top_listings(con, run_id, cfg.top_n_per_niche).get(niche_id, [])
    if not lids:
        return []
    window_start = as_of - timedelta(days=cfg.review_window_days)
    rows = con.execute(
        """
        SELECT l.listing_id, l.title, l.price_usd, l.url,
               count(r.review_hash) FILTER (WHERE r.created_at > ? AND r.created_at <= ?) AS reviews_90d
        FROM listing_snapshots l
        LEFT JOIN reviews r ON r.listing_id = l.listing_id
        WHERE l.run_id = ? AND list_contains(?::BIGINT[], l.listing_id)
        GROUP BY ALL
        ORDER BY reviews_90d DESC, l.listing_id
        LIMIT ?
        """,
        [window_start, as_of, run_id, lids, n],
    ).fetchall()
    return [{"title": t, "price_usd": p, "url": u, "reviews_90d": c} for _, t, p, u, c in rows]


def brief_payload(con, run_id: str, niche_id: str, as_of: datetime, cfg: ScoringConfig) -> dict:
    name, description = con.execute(
        "SELECT name, description FROM niches WHERE niche_id = ? AND cluster_version = ?",
        [niche_id, db.cluster_version(con, run_id)]).fetchone()
    keys = ["reviews_90d", "entry_share", "median_price", "listing_count", "top3_share", "confidence"]
    metrics = dict(zip(keys, con.execute(
        f"SELECT {', '.join(keys)} FROM niche_scores WHERE run_id = ? AND niche_id = ?",
        [run_id, niche_id]).fetchone()))
    lids = top_listings(con, run_id, cfg.top_n_per_niche).get(niche_id, [])
    metrics["price_min"], metrics["price_max"] = con.execute(
        "SELECT min(price_usd), max(price_usd) FROM listing_snapshots "
        "WHERE run_id = ? AND list_contains(?::BIGINT[], listing_id)", [run_id, lids]).fetchone()
    complaints = [{"theme": t, "mentions": k, "quotes": q} for t, k, q in con.execute(
        "SELECT theme, mentions, quotes FROM complaints WHERE run_id = ? AND niche_id = ? AND fixable "
        "ORDER BY mentions DESC, theme LIMIT 5", [run_id, niche_id]).fetchall()]
    return {"niche": {"name": name, "description": description}, "metrics": metrics,
            "complaints": complaints, "competitors": competitors(con, run_id, niche_id, as_of, cfg)}


def write_briefs(con, run_id: str, llm: LLMClient, cfg: ScoringConfig, as_of: datetime) -> list[str]:
    con.execute("DELETE FROM briefs WHERE run_id = ?", [run_id])
    niche_ids = [r[0] for r in con.execute(
        "SELECT niche_id FROM niche_scores WHERE run_id = ? AND rank <= ? ORDER BY rank",
        [run_id, cfg.report_top_k]).fetchall()]
    failures = []
    for niche_id in niche_ids:
        try:
            brief = llm.call("brief", brief_payload(con, run_id, niche_id, as_of, cfg))
        except LLMOutputError as e:
            failures.append(f"brief {niche_id}: {e}")
            continue
        con.execute("INSERT INTO briefs VALUES (?, ?, ?)", [run_id, niche_id, json.dumps(brief)])
    return failures


def write_summary(con, run_id: str, llm: LLMClient, cfg: ScoringConfig) -> list[str]:
    keys = ["rank", "niche", "score", "confidence", "reviews_90d", "median_price"]
    top = [dict(zip(keys, r)) for r in con.execute(
        "SELECT s.rank, n.name, s.score, s.confidence, s.reviews_90d, s.median_price "
        "FROM niche_scores s JOIN niches n ON n.niche_id = s.niche_id AND n.cluster_version = ? "
        "WHERE s.run_id = ? AND s.rank <= ? ORDER BY s.rank",
        [db.cluster_version(con, run_id), run_id, cfg.report_top_k]).fetchall()]
    dropped = con.execute("SELECT count(*) FROM niche_scores WHERE run_id = ? AND dropped",
                          [run_id]).fetchone()[0]
    run = db.get_run(con, run_id)
    payload = {"top": top, "dropped": dropped, "suspect": run["suspect"],
               "status_reasons": run["status_reasons"]}
    try:
        result = llm.call("summary", payload)
    except LLMOutputError as e:
        return [f"summary: {e}"]
    con.execute("INSERT OR REPLACE INTO run_summary VALUES (?, ?)", [run_id, result["summary"]])
    return []
```

- [ ] **Step 4: Implement `src/oppscan/report.py`**

```python
"""Build the report context for a run and render it to HTML and Markdown."""
from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from oppscan import db
from oppscan.briefs import competitors
from oppscan.config import ScoringConfig

_ENV = Environment(loader=PackageLoader("oppscan", "templates"),
                   autoescape=select_autoescape(enabled_extensions=("html.j2",)))


def rank_change(current: int | None, previous: int | None) -> str:
    if previous is None:
        return "new"
    if current is None:
        return ""
    diff = previous - current
    if diff > 0:
        return f"▲{diff}"
    if diff < 0:
        return f"▼{-diff}"
    return "–"


def explain(row: dict) -> list[str]:
    demand = (f"Demand: {row['reviews_90d']} reviews in the last 90 days across "
              f"{row['active_listings']} listings")
    if row["fav_delta"] is not None:
        demand += f", {row['fav_delta']} favourites gained since the last run"
    if row["gap_missing"]:
        gap = "Gap: no complaint data; scored at the median."
    else:
        gap = (f"Gap: {row['gap_per_100']:.1f} fixable complaints per 100 reviews "
               f"(percentile {row['pct_gap']:.0%}).")
    price = (f"Price: median ${row['median_price']:.2f} (percentile {row['pct_price']:.0%})."
             if row["median_price"] is not None else "Price: no price data.")
    return [
        f"{demand} (percentile {row['pct_demand']:.0%}).",
        f"Entry: {row['entry_share']:.0%} of recent reviews went to listings under a year old "
        f"(percentile {row['pct_entry']:.0%}).",
        gap,
        price,
        f"Crowding: {row['listing_count']:,} competing listings; the top 3 shops take "
        f"{row['top3_share']:.0%} of recent reviews (percentile {row['pct_crowding']:.0%}, subtracted).",
    ]


def _rows(con, sql: str, params: list) -> list[dict]:
    cursor = con.execute(sql, params)
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, r)) for r in cursor.fetchall()]


def _drop_reason(row: dict) -> str:
    if row["dropped"]:
        return f"below the demand floor (demand percentile {row['pct_demand']:.0%})"
    return (f"low confidence: {row['reviews_90d']} reviews in 90 days across "
            f"{row['active_listings']} listings")


def build_context(con, run_id: str, cfg: ScoringConfig) -> dict:
    run = db.get_run(con, run_id)
    as_of = run["started_at"]
    prev = db.previous_run(con, run_id)
    prev_ranks = dict(con.execute(
        "SELECT niche_id, rank FROM niche_scores WHERE run_id = ? AND rank IS NOT NULL", [prev]).fetchall()
    ) if prev else {}
    scored = _rows(con,
                   "SELECT s.*, n.name, n.description FROM niche_scores s "
                   "JOIN niches n ON n.niche_id = s.niche_id AND n.cluster_version = ? "
                   "WHERE s.run_id = ? ORDER BY s.rank NULLS LAST, s.score DESC",
                   [db.cluster_version(con, run_id), run_id])
    for row in scored:
        row["change"] = rank_change(row["rank"], prev_ranks.get(row["niche_id"]))
    rows = [r for r in scored if r["rank"] is not None]
    briefs = {nid: json.loads(b) for nid, b in con.execute(
        "SELECT niche_id, brief FROM briefs WHERE run_id = ?", [run_id]).fetchall()}
    cards = [{
        **r,
        "explanations": explain(r),
        "complaints": _rows(con, "SELECT theme, mentions, quotes FROM complaints WHERE run_id = ? "
                                 "AND niche_id = ? AND fixable ORDER BY mentions DESC, theme LIMIT 3",
                            [run_id, r["niche_id"]]),
        "competitors": competitors(con, run_id, r["niche_id"], as_of, cfg),
        "brief": briefs.get(r["niche_id"]),
    } for r in rows[: cfg.report_top_k]]
    summary = con.execute("SELECT summary FROM run_summary WHERE run_id = ?", [run_id]).fetchone()
    counts = {
        "seeds": con.execute("SELECT count(DISTINCT seed) FROM search_hits WHERE run_id = ?", [run_id]).fetchone()[0],
        "listings": con.execute("SELECT count(*) FROM listing_snapshots WHERE run_id = ?", [run_id]).fetchone()[0],
        "reviews": con.execute("SELECT count(*) FROM reviews WHERE listing_id IN "
                               "(SELECT listing_id FROM listing_snapshots WHERE run_id = ?)", [run_id]).fetchone()[0],
    }
    return {
        "run": run, "prev_run_id": prev, "summary": summary[0] if summary else None,
        "rows": rows, "cards": cards, "counts": counts,
        "dropped": [{"name": r["name"], "reason": _drop_reason(r)}
                    for r in scored if r["dropped"] or r["confidence"] != "high"],
        "weights": cfg.weights, "top_n": cfg.top_n_per_niche, "window": cfg.review_window_days,
        "new_days": cfg.new_listing_days, "floor": cfg.demand_floor_pct,
        "blended": bool(scored) and all(r["fav_delta"] is not None for r in scored),
    }


def render_report(con, run_id: str, cfg: ScoringConfig, out_dir: Path) -> tuple[Path, Path]:
    context = build_context(con, run_id, cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{run_id}.html"
    md_path = out_dir / f"{run_id}.md"
    html_path.write_text(_ENV.get_template("report.html.j2").render(**context))
    md_path.write_text(_ENV.get_template("report.md.j2").render(**context))
    return html_path, md_path
```

- [ ] **Step 5: Create `src/oppscan/templates/report.html.j2`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>oppscan {{ run.run_id }}</title>
<style>
:root { --bg:#ffffff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --bar:#3a6ea5; --neg:#b54545; --card:#f7f7f9; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141416; --fg:#ececf0; --muted:#9a9aa2; --line:#2c2c31; --bar:#6c9bd2; --neg:#dd7777; --card:#1d1d21; }
}
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1100px; padding:24px 16px; }
a { color:var(--bar); }
h1 { font-size:1.5rem; margin:0 0 4px; }
h2 { margin-top:2rem; border-bottom:1px solid var(--line); padding-bottom:4px; }
.muted { color:var(--muted); }
.badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:.8rem; border:1px solid var(--line); }
.badge.bad { color:var(--neg); border-color:var(--neg); }
.table-wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.bar { background:var(--line); height:6px; width:60px; border-radius:3px; }
.bar span { display:block; height:6px; background:var(--bar); border-radius:3px; }
.bar.neg span { background:var(--neg); }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; margin:16px 0; }
.card h3 { margin:0 0 8px; }
ul { margin:4px 0 8px; padding-left:20px; }
</style>
</head>
<body>
<h1>Etsy template opportunities</h1>
<p class="muted">Run {{ run.run_id }} · {{ run.started_at.strftime("%Y-%m-%d %H:%M") }} UTC · {{ counts.seeds }} seeds · {{ counts.listings }} listings · {{ counts.reviews }} reviews · {{ run.api_calls }} API calls{% if prev_run_id %} · <a href="{{ prev_run_id }}.html">previous run</a>{% endif %}</p>
<p><span class="badge{% if run.status != 'complete' %} bad{% endif %}">{{ run.status }}</span>
<span class="badge{% if run.suspect %} bad{% endif %}">sanity {{ "suspect" if run.suspect else "ok" }}</span></p>
{% if run.status_reasons %}<ul class="muted">{% for r in run.status_reasons %}<li>{{ r }}</li>{% endfor %}</ul>{% endif %}
{% if summary %}<p>{{ summary }}</p>{% endif %}

<h2>Ranked niches</h2>
<div class="table-wrap"><table>
<tr><th>#</th><th>Niche</th><th>Score</th><th>Demand</th><th>Entry</th><th>Gap</th><th>Price</th><th>Crowding</th><th>Conf.</th><th>Median $</th><th>Δ</th></tr>
{% for r in rows %}
<tr>
<td class="num">{{ r.rank }}</td><td>{{ r.name }}</td><td class="num">{{ "%.2f"|format(r.score) }}</td>
{% for p in [r.pct_demand, r.pct_entry, r.pct_gap, r.pct_price] %}<td><div class="bar"><span style="width:{{ (p * 100)|round|int }}%"></span></div></td>{% endfor %}
<td><div class="bar neg"><span style="width:{{ (r.pct_crowding * 100)|round|int }}%"></span></div></td>
<td>{{ r.confidence }}</td>
<td class="num">{{ "%.2f"|format(r.median_price) if r.median_price is not none else "–" }}</td>
<td>{{ r.change }}</td>
</tr>
{% endfor %}
</table></div>

<h2>Top {{ cards|length }} niches</h2>
{% for c in cards %}
<div class="card">
<h3>{{ c.rank }}. {{ c.name }}</h3>
<p class="muted">{{ c.description }}</p>
<ul>{% for e in c.explanations %}<li>{{ e }}</li>{% endfor %}</ul>
{% if c.complaints %}<strong>Fixable complaints</strong>
<ul>{% for k in c.complaints %}<li>{{ k.theme }} ({{ k.mentions }} reviews){% for q in k.quotes %} — “{{ q }}”{% endfor %}</li>{% endfor %}</ul>{% endif %}
{% if c.competitors %}<strong>Strongest competitors</strong>
<ul>{% for k in c.competitors %}<li><a href="{{ k.url }}">{{ k.title }}</a> · ${{ "%.2f"|format(k.price_usd) }} · {{ k.reviews_90d }} reviews in 90 days</li>{% endfor %}</ul>{% endif %}
{% if c.brief %}<strong>Draft brief</strong>
<ul>
<li>Buyer: {{ c.brief.target_buyer }}</li>
<li>Core features: {{ c.brief.core_features|join("; ") }}</li>
<li>Do better: {{ c.brief.differentiators|join("; ") }}</li>
<li>Suggested price: ${{ "%.2f"|format(c.brief.suggested_price_usd) }}</li>
<li>Title ideas: {{ c.brief.title_ideas|join(" / ") }}</li>
<li>Tags: {{ c.brief.tags|join(", ") }}</li>
</ul>
{% else %}<p class="muted">No brief (generation failed).</p>{% endif %}
</div>
{% endfor %}

<h2>Dropped and low-confidence</h2>
{% if dropped %}<ul>{% for d in dropped %}<li>{{ d.name }} — {{ d.reason }}</li>{% endfor %}</ul>{% else %}<p class="muted">None.</p>{% endif %}

<h2>Method</h2>
<p>score = {{ weights.demand }}·Demand + {{ weights.entry }}·Entry + {{ weights.gap }}·Gap + {{ weights.price }}·Price − {{ weights.crowding }}·Crowding. Each component is a percentile across this run's niches, computed over the top {{ top_n }} search results per niche.</p>
<ul>
<li>Demand: reviews in the last {{ window }} days{% if blended %}, blended 50/50 with favourites gained since the previous run{% endif %}. Niches below the {{ (floor * 100)|int }}th demand percentile are dropped.</li>
<li>Entry: share of recent reviews that went to listings under {{ new_days }} days old.</li>
<li>Gap: complaints a better template would fix, per 100 reviews.</li>
<li>Price: median listing price (USD).</li>
<li>Crowding: total listings for the niche's keywords, and the top 3 shops' share of recent reviews.</li>
</ul>
<p class="muted">These are proxies. Reviews are only a fraction of sales, so this ranks niches against each other; it does not estimate sales.</p>
</body>
</html>
```

- [ ] **Step 6: Create `src/oppscan/templates/report.md.j2`**

```
# Etsy template opportunities — run {{ run.run_id }}

Status: **{{ run.status }}** · sanity: **{{ "suspect" if run.suspect else "ok" }}** · {{ counts.seeds }} seeds · {{ counts.listings }} listings · {{ counts.reviews }} reviews · {{ run.api_calls }} API calls{% if prev_run_id %} · previous run: {{ prev_run_id }}{% endif %}
{% for r in run.status_reasons %}
- {{ r }}
{%- endfor %}

{{ summary or "" }}

## Ranked niches

| # | Niche | Score | Demand | Entry | Gap | Price | Crowding | Conf. | Median $ | Δ |
|---|---|---|---|---|---|---|---|---|---|---|
{% for r in rows -%}
| {{ r.rank }} | {{ r.name }} | {{ "%.2f"|format(r.score) }} | {{ "%.2f"|format(r.pct_demand) }} | {{ "%.2f"|format(r.pct_entry) }} | {{ "%.2f"|format(r.pct_gap) }} | {{ "%.2f"|format(r.pct_price) }} | {{ "%.2f"|format(r.pct_crowding) }} | {{ r.confidence }} | {{ "%.2f"|format(r.median_price) if r.median_price is not none else "–" }} | {{ r.change }} |
{% endfor %}
## Top {{ cards|length }} niches
{% for c in cards %}
### {{ c.rank }}. {{ c.name }}

{% for e in c.explanations -%}
- {{ e }}
{% endfor %}
{% if c.complaints %}**Fixable complaints**
{% for k in c.complaints %}- {{ k.theme }} ({{ k.mentions }} reviews){% for q in k.quotes %} — "{{ q }}"{% endfor %}
{% endfor %}{% endif %}
{% if c.competitors %}**Strongest competitors**
{% for k in c.competitors %}- [{{ k.title }}]({{ k.url }}) · ${{ "%.2f"|format(k.price_usd) }} · {{ k.reviews_90d }} reviews in 90 days
{% endfor %}{% endif %}
{% if c.brief %}**Draft brief**
- Buyer: {{ c.brief.target_buyer }}
- Core features: {{ c.brief.core_features|join("; ") }}
- Do better: {{ c.brief.differentiators|join("; ") }}
- Suggested price: ${{ "%.2f"|format(c.brief.suggested_price_usd) }}
- Title ideas: {{ c.brief.title_ideas|join(" / ") }}
- Tags: {{ c.brief.tags|join(", ") }}
{% else %}_No brief (generation failed)._
{% endif %}
{% endfor %}
## Dropped and low-confidence
{% for d in dropped %}
- {{ d.name }} — {{ d.reason }}
{%- else %}
None.
{%- endfor %}

## Method

score = {{ weights.demand }}·Demand + {{ weights.entry }}·Entry + {{ weights.gap }}·Gap + {{ weights.price }}·Price − {{ weights.crowding }}·Crowding, each a percentile across this run's niches over the top {{ top_n }} search results per niche. These are proxies: the report ranks niches against each other and does not estimate sales.
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: 4 passed

- [ ] **Step 8: Commit**

```bash
git add src/oppscan/briefs.py src/oppscan/report.py src/oppscan/templates tests/test_report.py
git commit -m "feat: briefs, run summary and html/markdown report"
```

---

### Task 12: Pipeline orchestration, CLI and end-to-end test

**Files:**
- Create: `src/oppscan/pipeline.py`, `src/oppscan/cli.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `Paths(db, config_dir, reports_dir, archive_dir)`; `RunResult(run_id, status, reasons, html, md)`; `run_pipeline(paths, *, transport=None, llm=None, api_key=None, resume=None, now=None, qps=None) -> RunResult`; `cli.main(argv=None) -> int` (0 = complete/partial, 2 = paused).

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:
```python
import shutil
from datetime import datetime
from pathlib import Path

import duckdb
import httpx

from oppscan.cli import main
from oppscan.fake_etsy import fake_transport
from oppscan.fake_llm import FakeLLM
from oppscan.pipeline import Paths, run_pipeline

REPO = Path(__file__).resolve().parents[1]
T1 = datetime(2026, 10, 1, 8, 0, 0)
T2 = datetime(2026, 10, 8, 8, 0, 0)
SEEDS = """seeds:
  - budget spreadsheet
  - budget planner
  - notion planner
  - notion dashboard
  - wedding budget
  - wedding planner
  - habit tracker
  - habit spreadsheet
  - monthly budget spreadsheet
"""


def setup_config(tmp_path, quota=100_000):
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    shutil.copy(REPO / "config" / "scoring.yaml", config / "scoring.yaml")
    (config / "seeds.yaml").write_text(SEEDS)
    (config / "etsy.yaml").write_text(
        f"qps: 5\ndaily_quota: {quota}\nsearch_depth: 200\nreviews_for_top: 5\nmax_review_pages: 2\n")
    return Paths(db=tmp_path / "data" / "test.duckdb", config_dir=config,
                 reports_dir=tmp_path / "reports", archive_dir=tmp_path / "data" / "archive")


def run(paths, now, **kw):
    return run_pipeline(paths, transport=kw.pop("transport", fake_transport(now)), llm=FakeLLM(),
                        api_key="fixture", now=now, qps=1e6, **kw)


def test_two_runs_end_to_end(tmp_path):
    paths = setup_config(tmp_path)
    first = run(paths, T1)
    assert first.status == "complete", first.reasons
    html = first.html.read_text()
    assert "Budget templates" in html
    assert "Draft brief" in html

    second = run(paths, T2)
    assert second.status == "complete", second.reasons
    assert f'href="{first.run_id}.html"' in second.html.read_text()
    con = duckdb.connect(str(paths.db), read_only=True)
    with_delta = con.execute("SELECT count(*) FROM niche_scores WHERE run_id = ? AND fav_delta IS NOT NULL",
                             [second.run_id]).fetchone()[0]
    con.close()
    assert with_delta > 0


def test_quota_pause_then_resume(tmp_path):
    paths = setup_config(tmp_path, quota=10)
    calls = []
    inner = fake_transport(T1)

    def counting(request):
        calls.append(str(request.url))
        return inner.handle_request(request)

    paused = run(paths, T1, transport=httpx.MockTransport(counting))
    assert paused.status == "paused"
    assert "--resume" in paused.reasons[0]
    assert paused.html is None
    first_calls = list(calls)
    assert len(first_calls) == 10

    setup_config(tmp_path, quota=100_000)
    resumed = run(paths, T1, transport=httpx.MockTransport(counting), resume=paused.run_id)
    assert resumed.run_id == paused.run_id
    assert resumed.status == "complete", resumed.reasons
    assert not set(first_calls) & set(calls[len(first_calls):])  # nothing re-fetched


def test_cli_fixtures(tmp_path, capsys):
    paths = setup_config(tmp_path)
    code = main(["run", "--fixtures", "--config-dir", str(paths.config_dir),
                 "--data-dir", str(tmp_path / "data"), "--reports-dir", str(tmp_path / "reports")])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "complete" in out
    assert list((tmp_path / "reports" / "fixtures").glob("*.html"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'oppscan.cli'`

- [ ] **Step 3: Implement `src/oppscan/pipeline.py`**

```python
"""Run the whole pipeline once (or resume a paused run) and record its status."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from oppscan import db
from oppscan.briefs import write_briefs, write_summary
from oppscan.clustering import cluster
from oppscan.collector import collect
from oppscan.complaints import mine_complaints
from oppscan.config import load_etsy, load_scoring, load_seeds
from oppscan.etsy import EtsyClient, QuotaExhausted
from oppscan.llm import AnthropicLLM, LLMClient
from oppscan.metrics import compute_metrics
from oppscan.report import render_report
from oppscan.scoring import sanity_check, save_scores, score_niches
from oppscan.staging import stage_run


@dataclass(frozen=True)
class Paths:
    db: Path
    config_dir: Path
    reports_dir: Path
    archive_dir: Path


@dataclass
class RunResult:
    run_id: str
    status: str
    reasons: list[str] = field(default_factory=list)
    html: Path | None = None
    md: Path | None = None


def run_pipeline(paths: Paths, *, transport: httpx.BaseTransport | None = None,
                 llm: LLMClient | None = None, api_key: str | None = None,
                 resume: str | None = None, now: datetime | None = None,
                 qps: float | None = None) -> RunResult:
    con = db.connect(paths.db)
    try:
        return _run(con, paths, transport, llm, api_key, resume, now, qps)
    finally:
        con.close()


def _run(con, paths, transport, llm, api_key, resume, now, qps) -> RunResult:
    seeds = load_seeds(paths.config_dir / "seeds.yaml")
    cfg = load_scoring(paths.config_dir / "scoring.yaml")
    etsy = load_etsy(paths.config_dir / "etsy.yaml", api_key=api_key)

    if resume:
        existing = db.get_run(con, resume)
        if existing is None:
            raise ValueError(f"unknown run {resume}")
        run_id, as_of = resume, existing["started_at"]
        db.set_status(con, run_id, "running")
    else:
        as_of = now or db.utcnow()
        run_id = db.start_run(con, seeds.file_hash, as_of)

    try:
        client = EtsyClient(con, run_id, api_key=etsy.api_key, qps=qps or etsy.qps,
                            daily_quota=etsy.daily_quota, transport=transport)
        try:
            stats = collect(client, seeds, etsy)
        except QuotaExhausted as e:
            reason = f"{e}; resume with: oppscan run --resume {run_id}"
            db.add_reason(con, run_id, reason)
            db.finish_run(con, run_id, "paused", db.utcnow())
            return RunResult(run_id, "paused", [reason])
        finally:
            client.close()

        llm = llm or AnthropicLLM(con)
        reasons: list[str] = []
        if stats.errors:
            reasons.append(f"{len(stats.errors)} Etsy request(s) failed; first: {stats.errors[0]}")
        stage_run(con, run_id)
        cluster(con, run_id, seeds, llm)
        reasons += mine_complaints(con, run_id, llm, cfg.top_n_per_niche)
        scores = score_niches(compute_metrics(con, run_id, as_of, cfg, db.previous_run(con, run_id)), cfg)
        save_scores(con, run_id, scores)

        seed_niche = dict(con.execute("SELECT seed, niche_id FROM seed_niche WHERE cluster_version = ?",
                                      [seeds.file_hash]).fetchall())
        problems = sanity_check(scores, seed_niche, cfg.known_big_seeds)
        if problems:
            db.set_suspect(con, run_id)
            for p in problems:
                db.add_reason(con, run_id, f"sanity: {p}")

        reasons += write_briefs(con, run_id, llm, cfg, as_of)
        reasons += write_summary(con, run_id, llm, cfg)
        for r in reasons:
            db.add_reason(con, run_id, r)
        status = "partial" if reasons else "complete"
        db.finish_run(con, run_id, status, db.utcnow())
        html, md = render_report(con, run_id, cfg, paths.reports_dir)
        db.prune_raw(con, paths.archive_dir)
        return RunResult(run_id, status, reasons, html, md)
    except Exception as e:
        db.add_reason(con, run_id, f"failed: {e!r}")
        db.finish_run(con, run_id, "failed", db.utcnow())
        raise
```

- [ ] **Step 4: Implement `src/oppscan/cli.py`**

```python
"""Command line entry point: `oppscan run`."""
from __future__ import annotations

import argparse
from pathlib import Path

from oppscan import db
from oppscan.fake_etsy import fake_transport
from oppscan.fake_llm import FakeLLM
from oppscan.pipeline import Paths, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oppscan", description="Etsy template opportunity scanner")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="collect, score and write a report")
    run.add_argument("--fixtures", action="store_true",
                     help="use the fake Etsy API and fake LLM (no network, no keys)")
    run.add_argument("--resume", metavar="RUN_ID", help="resume a paused run")
    run.add_argument("--config-dir", type=Path, default=Path("config"))
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    run.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    paths = Paths(
        db=args.data_dir / ("fixtures.duckdb" if args.fixtures else "oppscan.duckdb"),
        config_dir=args.config_dir,
        reports_dir=args.reports_dir / "fixtures" if args.fixtures else args.reports_dir,
        archive_dir=args.data_dir / "archive",
    )
    extra = {}
    if args.fixtures:
        extra = {"transport": fake_transport(db.utcnow()), "llm": FakeLLM(),
                 "api_key": "fixture", "qps": 1e6}
    result = run_pipeline(paths, resume=args.resume, **extra)

    print(f"run {result.run_id}: {result.status}")
    for reason in result.reasons:
        print(f"  - {reason}")
    if result.html:
        print(f"report: {result.html}")
    return 2 if result.status == "paused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the pipeline tests**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 3 passed

- [ ] **Step 6: Run the full suite and a real fixture run**

Run: `uv run pytest -v`
Expected: all tests pass (live tests deselected).

Run: `uv run oppscan run --fixtures`
Expected: prints `run <id>: complete` (or `partial` with listed reasons) and `report: reports/fixtures/<id>.html`. Open the HTML and check the ranked table, the cards and the method section all render.

- [ ] **Step 7: Commit**

```bash
git add src/oppscan/pipeline.py src/oppscan/cli.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestration, cli and end-to-end tests"
```

---

### Task 13: Live API checks and README

**Files:**
- Create: `tests/test_live.py`, `README.md`

**Interfaces:**
- Consumes: `EtsyClient`, `AnthropicLLM`.
- Produces: opt-in checks that confirm the real Etsy response shape matches the fields `staging.py` reads, and that the real Anthropic request shape (structured outputs + `fallbacks`) is accepted.

- [ ] **Step 1: Write the live tests**

`tests/test_live.py`:
```python
"""Opt-in checks against the real APIs. Run with: uv run pytest -m live -v"""
import json
import os
from pathlib import Path

import pytest

from oppscan import db
from oppscan.etsy import EtsyClient
from oppscan.llm import AnthropicLLM

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("ETSY_API_KEY"), reason="ETSY_API_KEY not set")
def test_etsy_response_shapes(con):
    run_id = db.start_run(con, "live", db.utcnow())
    client = EtsyClient(con, run_id, api_key=os.environ["ETSY_API_KEY"], qps=2, daily_quota=50)
    page = client.get("/listings/active", {"keywords": "budget spreadsheet", "limit": 5,
                                           "offset": 0, "sort_on": "score"})
    assert page["count"] > 0
    listing = page["results"][0]
    for key in ("listing_id", "title", "tags", "price", "num_favorers", "shop_id", "listing_type"):
        assert key in listing, key
    assert {"amount", "divisor", "currency_code"} <= set(listing["price"])
    assert any(k in listing for k in ("original_creation_timestamp", "created_timestamp", "creation_timestamp"))

    reviews = client.get(f"/listings/{listing['listing_id']}/reviews", {"limit": 5, "offset": 0})
    assert "results" in reviews
    for review in reviews["results"]:
        assert "rating" in review
        assert "create_timestamp" in review or "created_timestamp" in review

    shop = client.get(f"/shops/{listing['shop_id']}")
    assert "transaction_sold_count" in shop

    sample = Path(__file__).parent / "fixtures" / "etsy_live_sample.json"
    sample.parent.mkdir(exist_ok=True)
    sample.write_text(json.dumps({"search": page, "reviews": reviews, "shop": shop}, indent=2))


@pytest.mark.skipif(not os.environ.get("OPPSCAN_LIVE_LLM"), reason="set OPPSCAN_LIVE_LLM=1 to call Claude")
def test_anthropic_cluster_call(con):
    result = AnthropicLLM(con).call("cluster", {"seeds": [
        {"seed": "budget spreadsheet", "titles": ["Monthly Budget Spreadsheet Google Sheets"]},
        {"seed": "notion planner", "titles": ["Notion Life Planner Template"]},
    ]})
    assert {s for n in result["niches"] for s in n["seeds"]} == {"budget spreadsheet", "notion planner"}
```

- [ ] **Step 2: Check the live tests are deselected by default**

Run: `uv run pytest -v`
Expected: all non-live tests pass; `test_live.py` tests are reported as deselected.

- [ ] **Step 3: Write `README.md`**

````markdown
# oppscan

Scans Etsy for digital spreadsheet and Notion template niches and writes a ranked opportunity
report with draft product briefs. Design: `docs/superpowers/specs/2026-10-01-oppscan-research-agent-design.md`.

## Setup

```bash
uv sync
```

- **Etsy:** create a developer app at etsy.com/developers and wait for approval. Set `ETSY_API_KEY`
  to the value Etsy's docs say goes in the `x-api-key` header (check whether that's the keystring
  alone or `keystring:shared_secret`). Confirm your app's rate limits and update `config/etsy.yaml`.
- **Claude:** set `ANTHROPIC_API_KEY`, or sign in with `ant auth login`.

## Usage

```bash
uv run oppscan run --fixtures     # offline run on fake data (no keys needed)
uv run oppscan run                # live run -> reports/<run_id>.html and .md
uv run oppscan run --resume <id>  # continue a run paused by the Etsy daily quota
```

Edit `config/seeds.yaml` to change what's searched (this re-clusters niches on the next run) and
`config/scoring.yaml` to change weights or thresholds.

## Live checks

Run these once the keys exist, before the first real run:

```bash
ETSY_API_KEY=... uv run pytest -m live -v
OPPSCAN_LIVE_LLM=1 uv run pytest -m live -v -k anthropic
```

The Etsy check confirms the response fields the parser relies on and saves a sample to
`tests/fixtures/etsy_live_sample.json`. If a field differs, update `staging.py`, `fake_etsy.py`
and `tests/test_staging.py` to match.

## Status values

`complete` · `partial` (report written; reasons listed in its header) · `paused` (quota reached;
resume) · `failed`.
````

- [ ] **Step 4: Commit**

```bash
git add tests/test_live.py README.md
git commit -m "docs: readme and opt-in live api checks"
```

- [ ] **Step 5 (needs the keys): Run the live checks and the first real run**

Run: `ETSY_API_KEY=... uv run pytest -m live -v`
Expected: `test_etsy_response_shapes` passes. If it fails on a field name, fix `staging.py` and `fake_etsy.py` to use the real field, update `tests/test_staging.py`, and re-run `uv run pytest`.

Run: `OPPSCAN_LIVE_LLM=1 uv run pytest -m live -v -k anthropic`
Expected: passes. If the SDK rejects `fallbacks=` as an unknown keyword argument, pass it as `extra_body={"fallbacks": "default"}` in `AnthropicLLM._request` instead, update the matching assertion in `tests/test_llm.py`, and re-run.

Run: `uv run oppscan run`
Expected: `complete`, `partial` or `paused` (resume the next day). Read the report's sanity badge before trusting the rankings.
