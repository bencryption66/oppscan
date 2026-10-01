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


def test_competitor_with_unknown_price_renders_dash(con, tmp_path):
    build(con)
    con.execute("UPDATE listing_snapshots SET price_usd = NULL WHERE listing_id = 1")
    assert write_briefs(con, "r1", FakeLLM(), CFG, AS_OF) == []
    html_path, md_path = render_report(con, "r1", CFG, tmp_path)
    assert " · – · 3 reviews in 90 days" in html_path.read_text()
    assert " · – · 3 reviews in 90 days" in md_path.read_text()
