"""Mine complaint and wish themes from reviews of each niche's top listings."""
from __future__ import annotations

import re

from oppscan.db import cluster_version
from oppscan.llm import LLMClient, LLMOutputError
from oppscan.metrics import top_listings

WISH_RE = re.compile(r"\b(wish|would be nice|only issue)\b", re.IGNORECASE)
MAX_REVIEWS = 150


def mine_complaints(con, run_id: str, llm: LLMClient, top_n: int) -> list[str]:
    con.execute("DELETE FROM complaints WHERE run_id = ?", [run_id])
    con.execute("DELETE FROM complaint_runs WHERE run_id = ?", [run_id])
    names = dict(con.execute("SELECT niche_id, name FROM niches WHERE cluster_version = ?",
                             [cluster_version(con, run_id)]).fetchall())
    failures: list[str] = []
    for niche_id, lids in top_listings(con, run_id, top_n).items():
        candidates = _candidate_reviews(con, lids)
        if not candidates:
            _set_status(con, run_id, niche_id, "no_reviews")
            continue
        # Short sequential ids are easier for the model to copy than 16-hex hashes.
        hashes = {str(i): h for i, (h, _, _) in enumerate(candidates)}
        reviews = [{"id": str(i), "rating": rating, "text": text}
                   for i, (_, rating, text) in enumerate(candidates)]
        try:
            result = llm.call("complaints", {"niche": names.get(niche_id, niche_id), "reviews": reviews})
        except LLMOutputError as e:
            _set_status(con, run_id, niche_id, "failed")
            failures.append(f"complaints {niche_id}: {e}")
            continue
        for theme in result["themes"]:
            ids = sorted({hashes[i] for i in theme["review_ids"] if i in hashes})  # unknown ids ignored
            if ids:
                con.execute("INSERT INTO complaints VALUES (?, ?, ?, ?, ?, ?, ?)",
                            [run_id, niche_id, theme["theme"], theme["fixable_by_product"],
                             len(ids), theme["quotes"][:3], ids])
        _set_status(con, run_id, niche_id, "ok")
    return failures


def _set_status(con, run_id: str, niche_id: str, status: str) -> None:
    con.execute("INSERT INTO complaint_runs VALUES (?, ?, ?)", [run_id, niche_id, status])


def _candidate_reviews(con, listing_ids: list[int]) -> list[tuple[str, int, str]]:
    rows = con.execute(
        "SELECT review_hash, rating, text FROM reviews "
        "WHERE list_contains(?::BIGINT[], listing_id) AND text <> '' ORDER BY created_at DESC",
        [listing_ids],
    ).fetchall()
    picked = [(h, rating, text) for h, rating, text in rows if rating <= 4 or WISH_RE.search(text)]
    return picked[:MAX_REVIEWS]
