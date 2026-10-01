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
    assert [r["id"] for r in sent["reviews"]] == ["0", "1"]  # short ids; newest (wish) first
    assert [r["text"] for r in sent["reviews"]] == ["Wish it had dark mode", "Instructions were confusing"]


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
            return {"themes": [{"theme": "x", "fixable_by_product": True, "review_ids": ["nope", "99"], "quotes": []}]}

    assert mine_complaints(con, "r1", Hallucinating(), top_n=20) == []
    assert con.execute("SELECT count(*) FROM complaints").fetchone()[0] == 0
    assert statuses(con)["a"] == "ok"


def test_rerun_replaces_previous_results(con):
    build(con)
    mine_complaints(con, "r1", FakeLLM(), top_n=20)
    mine_complaints(con, "r1", FakeLLM(), top_n=20)
    assert con.execute("SELECT count(*) FROM complaints").fetchone()[0] == 1
