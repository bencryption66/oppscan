"""Competitor lookup and LLM-written product briefs and run summary."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from oppscan import db
from oppscan.config import ScoringConfig
from oppscan.llm import LLMClient, LLMOutputError
from oppscan.metrics import top_listings


def competitors(con, run_id: str, niche_id: str, as_of: datetime, cfg: ScoringConfig,
                n: int = 3) -> list[dict]:
    lids = top_listings(con, run_id, cfg.top_n_per_niche).get(niche_id, [])
    if not lids:
        return []
    window_start = as_of - timedelta(days=cfg.review_window_days)
    rows = con.execute(
        """
        SELECT l.listing_id, l.title, l.price_usd, l.url,
               count(r.review_hash) FILTER (WHERE r.created_at > ? AND r.created_at <= ?) AS reviews_90d
        FROM listing_snapshots l
        LEFT JOIN reviews r ON r.listing_id = l.listing_id
        WHERE l.run_id = ? AND list_contains(?::BIGINT[], l.listing_id)
        GROUP BY ALL
        ORDER BY reviews_90d DESC, l.listing_id
        LIMIT ?
        """,
        [window_start, as_of, run_id, lids, n],
    ).fetchall()
    return [{"title": t, "price_usd": p, "url": u, "reviews_90d": c} for _, t, p, u, c in rows]


def brief_payload(con, run_id: str, niche_id: str, as_of: datetime, cfg: ScoringConfig) -> dict:
    name, description = con.execute(
        "SELECT name, description FROM niches WHERE niche_id = ? AND cluster_version = ?",
        [niche_id, db.cluster_version(con, run_id)]).fetchone()
    keys = ["fav_rate", "engaged_listings", "reviews_90d", "entry_share", "median_price", "listing_count",
            "top3_share", "confidence"]
    metrics = dict(zip(keys, con.execute(
        f"SELECT {', '.join(keys)} FROM niche_scores WHERE run_id = ? AND niche_id = ?",
        [run_id, niche_id]).fetchone()))
    lids = top_listings(con, run_id, cfg.top_n_per_niche).get(niche_id, [])
    metrics["price_min"], metrics["price_max"] = con.execute(
        "SELECT min(price_usd), max(price_usd) FROM listing_snapshots "
        "WHERE run_id = ? AND list_contains(?::BIGINT[], listing_id)", [run_id, lids]).fetchone()
    complaints = [{"theme": t, "mentions": k, "quotes": q} for t, k, q in con.execute(
        "SELECT theme, mentions, quotes FROM complaints WHERE run_id = ? AND niche_id = ? AND fixable "
        "ORDER BY mentions DESC, theme LIMIT 5", [run_id, niche_id]).fetchall()]
    return {"niche": {"name": name, "description": description}, "metrics": metrics,
            "complaints": complaints, "competitors": competitors(con, run_id, niche_id, as_of, cfg)}


def write_briefs(con, run_id: str, llm: LLMClient, cfg: ScoringConfig, as_of: datetime) -> list[str]:
    con.execute("DELETE FROM briefs WHERE run_id = ?", [run_id])
    niche_ids = [r[0] for r in con.execute(
        "SELECT niche_id FROM niche_scores WHERE run_id = ? AND rank <= ? ORDER BY rank",
        [run_id, cfg.report_top_k]).fetchall()]
    failures = []
    for niche_id in niche_ids:
        try:
            brief = llm.call("brief", brief_payload(con, run_id, niche_id, as_of, cfg))
        except LLMOutputError as e:
            failures.append(f"brief {niche_id}: {e}")
            continue
        con.execute("INSERT INTO briefs VALUES (?, ?, ?)", [run_id, niche_id, json.dumps(brief)])
    return failures


def write_summary(con, run_id: str, llm: LLMClient, cfg: ScoringConfig) -> list[str]:
    keys = ["rank", "niche", "score", "confidence", "reviews_90d", "median_price"]
    top = [dict(zip(keys, r)) for r in con.execute(
        "SELECT s.rank, n.name, s.score, s.confidence, s.reviews_90d, s.median_price "
        "FROM niche_scores s JOIN niches n ON n.niche_id = s.niche_id AND n.cluster_version = ? "
        "WHERE s.run_id = ? AND s.rank <= ? ORDER BY s.rank",
        [db.cluster_version(con, run_id), run_id, cfg.report_top_k]).fetchall()]
    dropped = con.execute("SELECT count(*) FROM niche_scores WHERE run_id = ? AND dropped",
                          [run_id]).fetchone()[0]
    run = db.get_run(con, run_id)
    payload = {"top": top, "dropped": dropped, "suspect": run["suspect"],
               "status_reasons": run["status_reasons"]}
    try:
        result = llm.call("summary", payload)
    except LLMOutputError as e:
        return [f"summary: {e}"]
    con.execute("INSERT OR REPLACE INTO run_summary VALUES (?, ?)", [run_id, result["summary"]])
    return []
