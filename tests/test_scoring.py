import pytest
from helpers import make_cfg

from oppscan.metrics import NicheMetrics
from oppscan.scoring import percentile_ranks, sanity_check, save_scores, score_niches


def m(niche_id, reviews=10, active=1, fav=None, entry=0.5, gap=1.0, price=10.0, count=100, top3=0.5,
      fav_rate=10.0, engaged=1):
    return NicheMetrics(niche_id, reviews, active, fav, entry, gap, price, count, top3, fav_rate, engaged)


def by_id(scores):
    return {s.metrics.niche_id: s for s in scores}


def test_percentile_ranks():
    assert percentile_ranks([]) == []
    assert percentile_ranks([7]) == [0.5]
    assert percentile_ranks([3, 1, 2]) == [1.0, 0.0, 0.5]
    assert percentile_ranks([1, 1, 2]) == [0.25, 0.25, 1.0]


def test_hand_calculated_scores_floor_and_ranks():
    # Percentiles (a, b, c):
    #   fav_rate 50, 100, 200 -> 0, .5, 1;  reviews 10, 50, 30 -> 0, 1, .5
    #   demand = mean -> 0, .75, .75 (a is below the .25 floor and dropped)
    #   entry .5, .1, .9 -> .5, 0, 1;  gap 5, None, 1 -> 1, .5 (missing), 0;  price 10, 20, 5 -> .5, 1, 0
    #   crowding = mean(count 100, 1000, 10 -> .5, 1, 0; top3 .5, .9, .2 -> .5, 1, 0) -> .5, 1, 0
    # score = .35 D + .25 E + .20 G + .10 P - .10 C
    #   a = 0 + .125 + .2 + .05 - .05 = .325
    #   b = .2625 + 0 + .1 + .1 - .1 = .3625
    #   c = .2625 + .25 + 0 + 0 - 0 = .5125
    scores = by_id(score_niches([
        m("a", reviews=10, engaged=2, entry=0.5, gap=5.0, price=10, count=100, top3=0.5, fav_rate=50),
        m("b", reviews=50, engaged=10, entry=0.1, gap=None, price=20, count=1000, top3=0.9, fav_rate=100),
        m("c", reviews=30, engaged=9, entry=0.9, gap=1.0, price=5, count=10, top3=0.2, fav_rate=200),
    ], make_cfg()))
    assert [scores[k].pct_demand for k in "abc"] == [0.0, 0.75, 0.75]
    assert scores["a"].score == pytest.approx(0.325)
    assert scores["b"].score == pytest.approx(0.3625)
    assert scores["c"].score == pytest.approx(0.5125)
    assert scores["b"].gap_missing and scores["b"].pct_gap == 0.5
    assert scores["a"].dropped and scores["a"].dropped_reason == "demand_floor" and scores["a"].rank is None
    assert (scores["c"].rank, scores["b"].rank) == (1, 2)
    assert scores["b"].confidence == "high" and scores["c"].confidence == "high"
    assert scores["a"].confidence == "low"


def test_high_confidence_ranks_above_low_confidence():
    scores = by_id(score_niches([
        m("x", reviews=100, engaged=2),
        m("y", reviews=50, engaged=10),
    ], make_cfg(demand_floor_pct=0.0)))
    assert scores["x"].score > scores["y"].score
    assert (scores["y"].rank, scores["x"].rank) == (1, 2)


def test_demand_is_mean_of_favourite_and_review_percentiles():
    scores = score_niches([m("p", reviews=10, fav_rate=30.0), m("q", reviews=20, fav_rate=20.0),
                           m("r", reviews=30, fav_rate=10.0)], make_cfg())
    # fav_rate percentiles 1, .5, 0; review percentiles 0, .5, 1
    assert [s.pct_demand for s in scores] == [0.5, 0.5, 0.5]
    scores = score_niches([m("p", reviews=10, fav_rate=30.0), m("q", reviews=20, fav_rate=40.0),
                           m("r", reviews=30, fav_rate=10.0)], make_cfg())
    # fav_rate .5, 1, 0; reviews 0, .5, 1
    assert [s.pct_demand for s in scores] == [0.25, 0.75, 0.5]


def test_demand_adds_favourite_delta_when_every_niche_has_it():
    scores = score_niches([m("p", reviews=10, fav_rate=40.0, fav=5), m("q", reviews=20, fav_rate=30.0, fav=9),
                           m("r", reviews=30, fav_rate=10.0, fav=1)], make_cfg())
    # fav_rate 1, .5, 0; reviews 0, .5, 1; fav_delta .5, 1, 0
    assert [s.pct_demand for s in scores] == pytest.approx([1.5 / 3, 2 / 3, 1 / 3])
    partial = score_niches([m("p", reviews=10, fav_rate=40.0, fav=5), m("q", reviews=20, fav_rate=30.0)],
                           make_cfg())
    assert [s.pct_demand for s in partial] == [0.5, 0.5]  # fav_delta ignored unless every niche has it


def test_confidence_uses_engaged_listings():
    scores = by_id(score_niches([m("lo", reviews=0, engaged=7), m("hi", reviews=0, engaged=8)],
                                make_cfg(demand_floor_pct=0.0)))
    assert (scores["lo"].confidence, scores["hi"].confidence) == ("low", "high")


def test_sanity_check():
    scores = score_niches([m("big", reviews=1, fav_rate=1.0), m("other", reviews=50, fav_rate=90.0)],
                          make_cfg(demand_floor_pct=0.0))
    problems = sanity_check(scores, {"monthly budget": "big"}, ("monthly budget", "missing seed"))
    assert problems == [
        "known-big niche 'big' has demand percentile 0.00 (below 0.50)",
        "known-big seed 'missing seed' is not in seeds.yaml",
    ]
    assert sanity_check(scores, {"x": "other"}, ("x",)) == []


def test_save_scores_round_trip(con):
    scores = score_niches([m("a", reviews=1, fav_rate=1.0), m("b", reviews=40, fav_rate=12.5, engaged=9)],
                          make_cfg())
    save_scores(con, "r1", scores)
    save_scores(con, "r1", scores)
    rows = con.execute("SELECT niche_id, rank, dropped, fav_rate, engaged_listings FROM niche_scores "
                       "ORDER BY niche_id").fetchall()
    assert rows == [("a", None, True, 1.0, 1), ("b", 1, False, 12.5, 9)]


def test_unknown_median_price_gets_neutral_percentile():
    scores = score_niches([m("a", price=5.0), m("b", price=None), m("c", price=20.0)], make_cfg())
    assert [s.pct_price for s in scores] == [0.0, 0.5, 1.0]
