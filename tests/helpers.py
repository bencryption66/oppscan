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
