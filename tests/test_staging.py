import json
from datetime import datetime

from oppscan.staging import price_usd, review_hash, stage_run

FX = {"USD": 1.0, "EUR": 0.88511}


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
    usd = {"amount": 1299, "divisor": 100, "currency_code": "USD"}
    assert price_usd(usd, FX) == 12.99
    assert price_usd({"amount": 885, "divisor": 100, "currency_code": "EUR"}, FX) == 10.0
    assert price_usd({"amount": 885, "divisor": 100, "currency_code": "XXX"}, FX) is None
    assert price_usd({"amount": 885, "divisor": 0, "currency_code": "USD"}, FX) is None
    assert price_usd({"amount": 885, "currency_code": "USD"}, FX) is None
    assert price_usd(None, FX) is None


def test_stages_search_results_keeping_downloads_only(con):
    search(con, "r1", "budget", 0, [
        listing(1),
        listing(2, listing_type="physical"),
        listing(3, price={"amount": 500, "divisor": 100, "currency_code": "EUR"}),
    ])
    search(con, "r1", "budget", 100, [listing(4)])
    stage_run(con, "r1", FX)
    hits = con.execute("SELECT seed, listing_id, search_rank FROM search_hits ORDER BY listing_id").fetchall()
    assert hits == [("budget", 1, 1), ("budget", 3, 3), ("budget", 4, 101)]
    assert con.execute("SELECT total_count FROM seed_counts").fetchall() == [(999,)]
    row = con.execute("SELECT price_usd, created_at, shop_id FROM listing_snapshots WHERE listing_id = 1").fetchone()
    assert row == (12.99, datetime(2023, 11, 14, 22, 13, 20), 9)


def test_stages_non_usd_listing_converted_to_usd(con):
    search(con, "r1", "budget", 0, [listing(1, price={"amount": 885, "divisor": 100, "currency_code": "EUR"})])
    stage_run(con, "r1", FX)
    assert con.execute("SELECT price_usd FROM listing_snapshots").fetchone() == (10.0,)


def test_stages_unknown_currency_with_null_price(con):
    search(con, "r1", "budget", 0, [listing(1, price={"amount": 885, "divisor": 100, "currency_code": "XXX"})])
    stage_run(con, "r1", FX)
    assert con.execute("SELECT listing_id, price_usd FROM listing_snapshots").fetchall() == [(1, None)]
    assert con.execute("SELECT listing_id FROM search_hits").fetchall() == [(1,)]


def test_same_listing_in_two_seeds(con):
    search(con, "r1", "a", 0, [listing(1)])
    search(con, "r1", "b", 0, [listing(9), listing(1)])
    stage_run(con, "r1", FX)
    assert con.execute("SELECT seed, search_rank FROM search_hits WHERE listing_id = 1 ORDER BY seed").fetchall() == [("a", 1), ("b", 2)]
    assert con.execute("SELECT count(*) FROM listing_snapshots").fetchone()[0] == 2


def test_stages_shop(con):
    raw(con, "r1", "/shops/9", {}, {"shop_id": 9, "shop_name": "S", "transaction_sold_count": 100,
                                     "review_count": 10, "review_average": 4.8, "create_date": 1600000000})
    stage_run(con, "r1", FX)
    assert con.execute("SELECT shop_name, transaction_sold_count, review_average FROM shop_snapshots").fetchone() == ("S", 100, 4.8)


def test_reviews_dedupe_across_runs(con):
    review = {"rating": 2, "review": "Confusing", "create_timestamp": 1750000000}
    raw(con, "r1", "/listings/1/reviews", {"limit": 100, "offset": 0}, {"count": 1, "results": [review]})
    raw(con, "r2", "/listings/1/reviews", {"limit": 100, "offset": 0}, {"count": 1, "results": [review]})
    stage_run(con, "r1", FX)
    stage_run(con, "r2", FX)
    rows = con.execute("SELECT review_hash, listing_id, rating, text, first_seen_run FROM reviews").fetchall()
    assert rows == [(review_hash(1, 1750000000, "Confusing"), 1, 2, "Confusing", "r1")]


def test_restaging_is_idempotent(con):
    search(con, "r1", "budget", 0, [listing(1)])
    stage_run(con, "r1", FX)
    stage_run(con, "r1", FX)
    assert con.execute("SELECT count(*) FROM search_hits").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM listing_snapshots").fetchone()[0] == 1


def test_language_filter_keeps_matching_and_unknown_languages(con):
    search(con, "r1", "budget", 0, [
        listing(1, language="en-US"),
        listing(2, language="de"),
        listing(3, language="en-GB"),
        listing(4),
        listing(5, language=None),
        listing(6, language="EN"),
    ])
    stage_run(con, "r1", FX, languages=("en",))
    rows = con.execute("SELECT listing_id, language FROM listing_snapshots ORDER BY listing_id").fetchall()
    assert rows == [(1, "en-US"), (3, "en-GB"), (4, None), (5, None), (6, "EN")]
    hits = con.execute("SELECT listing_id, search_rank FROM search_hits ORDER BY listing_id").fetchall()
    assert hits == [(1, 1), (3, 3), (4, 4), (5, 5), (6, 6)]


def test_empty_language_list_keeps_every_language(con):
    search(con, "r1", "budget", 0, [listing(1, language="de"), listing(2, language="fr")])
    stage_run(con, "r1", FX, languages=())
    assert con.execute("SELECT count(*) FROM listing_snapshots").fetchone()[0] == 2
