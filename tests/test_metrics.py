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


def test_median_price_ignores_null_prices(con):
    build(con)
    con.execute("UPDATE listing_snapshots SET price_usd = NULL WHERE listing_id IN (1, 4)")
    [m] = compute_metrics(con, "r1", AS_OF, CFG, None)
    assert m.median_price == 25.0  # median of 20 and 30


def test_median_price_none_when_all_prices_null(con):
    build(con)
    con.execute("UPDATE listing_snapshots SET price_usd = NULL")
    [m] = compute_metrics(con, "r1", AS_OF, CFG, None)
    assert m.median_price is None
