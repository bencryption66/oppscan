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
    stamps = [r.get("create_timestamp") or r.get("created_timestamp") for r in reviews["results"]]
    assert stamps == sorted(stamps, reverse=True), (
        "reviews are not newest-first; collector._reviews stops paging on the assumption that they are")

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
