import json
from dataclasses import replace
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


def test_collect_reviews_top_download_listings_whatever_the_currency(con):
    def listing(lid, currency, listing_type="download"):
        return {"listing_id": lid, "shop_id": lid, "listing_type": listing_type,
                "price": {"amount": 500, "divisor": 100, "currency_code": currency}}

    results = ([listing(1, "EUR", "physical"), listing(2, "EUR"), listing(3, "XXX"), listing(4, "CAD")]
               + [listing(i, "USD") for i in range(5, 10)])

    def handler(request):
        path = request.url.path
        if path.endswith("/listings/active"):
            return httpx.Response(200, json={"count": len(results), "results": results})
        if path.endswith("/reviews"):
            return httpx.Response(200, json={"count": 0, "results": []})
        return httpx.Response(200, json={"shop_id": int(path.rsplit("/", 1)[1])})

    seeds = Seeds(terms=("budget spreadsheet",), file_hash="h")
    collect(client_for(con, httpx.MockTransport(handler)), seeds, SETTINGS)
    reviewed = {int(e.split("/")[2]) for e in endpoints(con) if e.endswith("/reviews")}
    assert reviewed == {2, 3, 4}  # reviews_for_top=3; non-USD ranks above USD, physical skipped
    shops = {int(e.split("/")[2]) for e in endpoints(con) if e.startswith("/shops/")}
    assert shops == {2, 3, 4}


def language_transport():
    def listing(lid, **over):
        return {"listing_id": lid, "shop_id": lid, "listing_type": "download", **over}

    results = [listing(1, language="de"), listing(2, language="en-GB"), listing(3),
               listing(4, language="fr"), listing(5, language="en-US"), listing(6, language="en")]

    def handler(request):
        path = request.url.path
        if path.endswith("/listings/active"):
            return httpx.Response(200, json={"count": len(results), "results": results})
        if path.endswith("/reviews"):
            return httpx.Response(200, json={"count": 0, "results": []})
        return httpx.Response(200, json={"shop_id": int(path.rsplit("/", 1)[1])})

    return httpx.MockTransport(handler)


def reviewed_after_collect(con, languages):
    seeds = Seeds(terms=("budget spreadsheet",), file_hash="h")
    collect(client_for(con, language_transport()), seeds, replace(SETTINGS, languages=languages))
    return {int(e.split("/")[2]) for e in endpoints(con) if e.endswith("/reviews")}


def test_collect_skips_listings_in_other_languages(con):
    assert reviewed_after_collect(con, ("en",)) == {2, 3, 5}  # de and fr skipped; missing language kept


def test_collect_with_no_languages_keeps_every_language(con):
    assert reviewed_after_collect(con, ()) == {1, 2, 3}


def test_fake_listings_are_mostly_english():
    with httpx.Client(transport=fake_transport(T), base_url="https://x/v3/application") as c:
        results = c.get("/listings/active", params={"keywords": "budget spreadsheet", "limit": 100,
                                                    "offset": 0}).json()["results"]
    languages = [r["language"] for r in results]
    assert set(languages) == {"en-US", "de"}
    assert 0.02 <= languages.count("de") / len(languages) <= 0.15
