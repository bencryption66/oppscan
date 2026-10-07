from datetime import datetime, timedelta

import pytest
from helpers import add_listing, add_niche, add_review, assign, make_cfg

from oppscan.metrics import compute_metrics, top_listings, unreviewed_top_listings

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


def test_top_listings_orders_by_favourites_then_rank_and_limits(con):
    build(con)
    # 2 and 1 have favourites (25, 15); 3 and 4 tie on none, so best search rank decides; 5 is cut.
    assert top_listings(con, "r1", 4) == {"a": [2, 1, 3, 4]}


def test_top_listings_prefers_favourited_over_best_search_rank(con):
    add_niche(con, "a", ["s1"])
    for lid, rank, favs in [(1, 1, 0), (2, 2, 1), (3, 3, 500), (4, 4, 500), (5, 5, 90)]:
        add_listing(con, "r1", lid, seed="s1", rank=rank, favs=favs)
        assign(con, "r1", lid, "a")
    assert top_listings(con, "r1", 3) == {"a": [3, 4, 5]}  # 3/4 tie on favourites, rank breaks it


def test_top_listings_ties_fall_back_to_rank_then_listing_id(con):
    add_niche(con, "a", ["s1", "s2"])
    for lid, seed, rank in [(9, "s1", 5), (7, "s1", 2), (8, "s2", 2), (6, "s2", 7)]:
        add_listing(con, "r1", lid, seed=seed, rank=rank, favs=10)
        assign(con, "r1", lid, "a")
    assert top_listings(con, "r1", 4) == {"a": [7, 8, 9, 6]}


def test_compute_metrics(con):
    build(con)
    [m] = compute_metrics(con, "r1", AS_OF, CFG, None)
    assert m.niche_id == "a"
    assert m.reviews_90d == 6
    assert m.active_listings == 4
    # Favourites per month of age: listing 1 is 214 days old with 15 favourites, listing 2 is
    # 1,004 days old with 25; listings 3 and 4 have none. Only listing 1 is under 365 days old.
    r1, r2 = 15 / (214 / 30.44), 25 / (1004 / 30.44)
    assert m.fav_rate == pytest.approx(r1 + r2)
    assert m.engaged_listings == 2
    assert m.entry_share == pytest.approx(r1 / (r1 + r2))
    assert m.top3_share == pytest.approx(1.0)  # only two shops have any favourites
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


def build_favs(con):
    """Ages chosen so age_months is exact: 761 days = 25 months, 1,522 = 50, 2,283 = 75."""
    add_niche(con, "f", ["s"])
    specs = [  # listing_id, shop, days old (None = unknown), favourites -> favourites per month
        (1, 1, 10, 6),       # 10 days floors to 1 month -> 6.0, and it is new
        (2, 1, 761, 50),     # 50 / 25 -> 2.0
        (3, 2, None, 4),     # unknown age counts as 1 month -> 4.0, not new
        (4, 3, 1522, 100),   # 100 / 50 -> 2.0
        (5, 4, 761, 25),     # 25 / 25 -> 1.0
        (6, 5, 2283, 0),     # 0.0
    ]
    for lid, shop, days, favs in specs:
        created = None if days is None else AS_OF - timedelta(days=days)
        add_listing(con, "r1", lid, seed="s", rank=lid, favs=favs, shop_id=shop, created=created)
        assign(con, "r1", lid, "f")
    add_niche(con, "z", ["t"])
    add_listing(con, "r1", 9, seed="t", rank=1, favs=0)
    assign(con, "r1", 9, "z")


def test_favourite_metrics(con):
    build_favs(con)
    f, z = compute_metrics(con, "r1", AS_OF, make_cfg(), None)
    assert f.fav_rate == pytest.approx(15.0)  # 6 + 2 + 4 + 2 + 1 + 0
    assert f.engaged_listings == 4  # listings 1, 2, 4 and 5 have 5+ favourites
    assert f.entry_share == pytest.approx(6 / 15)  # only listing 1 is under 365 days old
    assert f.top3_share == pytest.approx(14 / 15)  # shops 1, 2, 3: 8 + 4 + 2
    assert (z.fav_rate, z.engaged_listings, z.entry_share, z.top3_share) == (0.0, 0, 0.0, 0.0)


def test_engaged_listings_uses_min_favourites(con):
    build_favs(con)
    f, _ = compute_metrics(con, "r1", AS_OF, make_cfg(min_favourites=50), None)
    assert f.engaged_listings == 2  # 50 and 100


def test_unreviewed_top_listings_counts_top_listings_without_fetched_reviews(con):
    build(con)  # top 4 of niche a: listings 1, 3, 2, 4
    for lid in (1, 2, 5):
        con.execute("INSERT INTO raw_api VALUES ('r1', ?, '{}', ?, '{}')", [f"/listings/{lid}/reviews", AS_OF])
    con.execute("INSERT INTO raw_api VALUES ('r0', '/listings/3/reviews', '{}', ?, '{}')", [AS_OF])  # other run
    assert unreviewed_top_listings(con, "r1", 4) == (2, 4)  # listings 3 and 4
