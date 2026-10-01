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
