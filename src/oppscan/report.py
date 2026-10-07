"""Build the report context for a run and render it to HTML and Markdown."""
from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from oppscan import db
from oppscan.briefs import competitors
from oppscan.config import ScoringConfig

_ENV = Environment(loader=PackageLoader("oppscan", "templates"),
                   autoescape=select_autoescape(enabled_extensions=("html.j2",)))


def rank_change(current: int | None, previous: int | None) -> str:
    if previous is None:
        return "new"
    if current is None:
        return ""
    diff = previous - current
    if diff > 0:
        return f"▲{diff}"
    if diff < 0:
        return f"▼{-diff}"
    return "–"


def explain(row: dict) -> list[str]:
    demand = (f"Demand: {row['fav_rate']:,.0f} favourites per month across the top listings and "
              f"{row['reviews_90d']} reviews in the last 90 days")
    if row["fav_delta"] is not None:
        demand += f", {row['fav_delta']} favourites gained since the last run"
    if row["gap_missing"]:
        gap = "Gap: no complaint data; scored at the median."
    else:
        gap = (f"Gap: {row['gap_per_100']:.1f} fixable complaints per 100 reviews "
               f"(percentile {row['pct_gap']:.0%}).")
    price = (f"Price: median ${row['median_price']:.2f} (percentile {row['pct_price']:.0%})."
             if row["median_price"] is not None else "Price: no price data.")
    return [
        f"{demand} (percentile {row['pct_demand']:.0%}).",
        f"Entry: {row['entry_share']:.0%} of favourite momentum is on listings under a year old "
        f"(percentile {row['pct_entry']:.0%}).",
        gap,
        price,
        f"Crowding: {row['listing_count']:,} competing listings; the top 3 shops hold "
        f"{row['top3_share']:.0%} of favourite momentum (percentile {row['pct_crowding']:.0%}, subtracted).",
    ]


def _rows(con, sql: str, params: list) -> list[dict]:
    cursor = con.execute(sql, params)
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, r)) for r in cursor.fetchall()]


def _drop_reason(row: dict, min_favourites: int) -> str:
    if row["dropped"]:
        return f"below the demand floor (demand percentile {row['pct_demand']:.0%})"
    return (f"low confidence: only {row['engaged_listings']} of the top listings have "
            f"{min_favourites}+ favourites")


def build_context(con, run_id: str, cfg: ScoringConfig) -> dict:
    run = db.get_run(con, run_id)
    as_of = run["started_at"]
    prev = db.previous_run(con, run_id)
    prev_ranks = dict(con.execute(
        "SELECT niche_id, rank FROM niche_scores WHERE run_id = ? AND rank IS NOT NULL", [prev]).fetchall()
    ) if prev else {}
    scored = _rows(con,
                   "SELECT s.*, n.name, n.description FROM niche_scores s "
                   "JOIN niches n ON n.niche_id = s.niche_id AND n.cluster_version = ? "
                   "WHERE s.run_id = ? ORDER BY s.rank NULLS LAST, s.score DESC",
                   [db.cluster_version(con, run_id), run_id])
    for row in scored:
        row["change"] = rank_change(row["rank"], prev_ranks.get(row["niche_id"]))
    rows = [r for r in scored if r["rank"] is not None]
    briefs = {nid: json.loads(b) for nid, b in con.execute(
        "SELECT niche_id, brief FROM briefs WHERE run_id = ?", [run_id]).fetchall()}
    cards = [{
        **r,
        "explanations": explain(r),
        "complaints": _rows(con, "SELECT theme, mentions, quotes FROM complaints WHERE run_id = ? "
                                 "AND niche_id = ? AND fixable ORDER BY mentions DESC, theme LIMIT 3",
                            [run_id, r["niche_id"]]),
        "competitors": competitors(con, run_id, r["niche_id"], as_of, cfg),
        "brief": briefs.get(r["niche_id"]),
    } for r in rows[: cfg.report_top_k]]
    summary = con.execute("SELECT summary FROM run_summary WHERE run_id = ?", [run_id]).fetchone()
    counts = {
        "seeds": con.execute("SELECT count(DISTINCT seed) FROM search_hits WHERE run_id = ?", [run_id]).fetchone()[0],
        "listings": con.execute("SELECT count(*) FROM listing_snapshots WHERE run_id = ?", [run_id]).fetchone()[0],
        "reviews": con.execute("SELECT count(*) FROM reviews WHERE listing_id IN "
                               "(SELECT listing_id FROM listing_snapshots WHERE run_id = ?)", [run_id]).fetchone()[0],
    }
    return {
        "run": run, "prev_run_id": prev, "summary": summary[0] if summary else None,
        "rows": rows, "cards": cards, "counts": counts,
        "dropped": [{"name": r["name"], "reason": _drop_reason(r, cfg.min_favourites)}
                    for r in scored if r["dropped"] or r["confidence"] != "high"],
        "weights": cfg.weights, "top_n": cfg.top_n_per_niche, "window": cfg.review_window_days,
        "new_days": cfg.new_listing_days, "floor": cfg.demand_floor_pct,
        "min_listings": cfg.high_conf_min_listings, "min_favourites": cfg.min_favourites,
        "blended": bool(scored) and all(r["fav_delta"] is not None for r in scored),
    }


def render_report(con, run_id: str, cfg: ScoringConfig, out_dir: Path) -> tuple[Path, Path]:
    context = build_context(con, run_id, cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{run_id}.html"
    md_path = out_dir / f"{run_id}.md"
    html_path.write_text(_ENV.get_template("report.html.j2").render(**context))
    md_path.write_text(_ENV.get_template("report.md.j2").render(**context))
    return html_path, md_path
