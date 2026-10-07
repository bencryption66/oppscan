"""Raw per-niche metrics for one run, computed over each niche's top search results."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median

from oppscan.config import ScoringConfig


@dataclass(frozen=True)
class NicheMetrics:
    niche_id: str
    reviews_90d: int
    active_listings: int
    fav_delta: int | None
    entry_share: float
    gap_per_100: float | None
    median_price: float | None
    listing_count: int
    top3_share: float
    fav_rate: float
    engaged_listings: int


DAYS_PER_MONTH = 30.44


def favourites_per_month(favs: int, created: datetime | None, as_of: datetime) -> float:
    """Favourites divided by listing age in months; ages under a month, or unknown, count as 1."""
    age_months = 1.0 if created is None else max(1.0, (as_of - created).days / DAYS_PER_MONTH)
    return favs / age_months


def _share(part: float, total: float) -> float:
    return part / total if total else 0.0


def top_listings(con, run_id: str, top_n: int) -> dict[str, list[int]]:
    rows = con.execute(
        """
        SELECT niche_id, listing_id FROM (
            SELECT ln.niche_id, h.listing_id,
                   ROW_NUMBER() OVER (PARTITION BY ln.niche_id
                                      ORDER BY MIN(h.search_rank), h.listing_id) AS rn
            FROM listing_niche ln
            JOIN search_hits h ON h.run_id = ln.run_id AND h.listing_id = ln.listing_id
            WHERE ln.run_id = ?
            GROUP BY ln.niche_id, h.listing_id
        ) WHERE rn <= ? ORDER BY niche_id, rn
        """,
        [run_id, top_n],
    ).fetchall()
    out: dict[str, list[int]] = {}
    for niche_id, listing_id in rows:
        out.setdefault(niche_id, []).append(listing_id)
    return out


def compute_metrics(con, run_id: str, as_of: datetime, cfg: ScoringConfig,
                    prev_run_id: str | None) -> list[NicheMetrics]:
    tops = top_listings(con, run_id, cfg.top_n_per_niche)
    if not tops:
        return []
    all_ids = sorted({lid for lids in tops.values() for lid in lids})
    window_start = as_of - timedelta(days=cfg.review_window_days)
    new_cutoff = as_of - timedelta(days=cfg.new_listing_days)

    snap = {lid: (price, favs, created, shop) for lid, price, favs, created, shop in con.execute(
        "SELECT listing_id, price_usd, num_favorers, created_at, shop_id FROM listing_snapshots "
        "WHERE run_id = ? AND list_contains(?::BIGINT[], listing_id)", [run_id, all_ids]).fetchall()}
    prev_favs = dict(con.execute(
        "SELECT listing_id, num_favorers FROM listing_snapshots "
        "WHERE run_id = ? AND list_contains(?::BIGINT[], listing_id)", [prev_run_id, all_ids]).fetchall()
    ) if prev_run_id else {}
    recent = dict(con.execute(
        "SELECT listing_id, count(*) FROM reviews WHERE list_contains(?::BIGINT[], listing_id) "
        "AND created_at > ? AND created_at <= ? GROUP BY listing_id",
        [all_ids, window_start, as_of]).fetchall())
    lifetime = dict(con.execute(
        "SELECT listing_id, count(*) FROM reviews WHERE list_contains(?::BIGINT[], listing_id) "
        "AND created_at <= ? GROUP BY listing_id", [all_ids, as_of]).fetchall())
    listing_counts = dict(con.execute(
        """
        SELECT sn.niche_id, max(sc.total_count) FROM seed_counts sc
        JOIN seed_niche sn ON sn.seed = sc.seed
         AND sn.cluster_version = (SELECT any_value(cluster_version) FROM listing_niche WHERE run_id = ?)
        WHERE sc.run_id = ? GROUP BY sn.niche_id
        """, [run_id, run_id]).fetchall())
    status = dict(con.execute("SELECT niche_id, status FROM complaint_runs WHERE run_id = ?", [run_id]).fetchall())
    fixable = dict(con.execute(
        "SELECT niche_id, sum(mentions) FROM complaints WHERE run_id = ? AND fixable GROUP BY niche_id",
        [run_id]).fetchall())

    out = []
    for niche_id, lids in sorted(tops.items()):
        rev = {lid: recent.get(lid, 0) for lid in lids}
        rate = {lid: favourites_per_month(snap[lid][1], snap[lid][2], as_of) for lid in lids}
        fav_rate = sum(rate.values())
        new_rate = sum(r for lid, r in rate.items() if snap[lid][2] is not None and snap[lid][2] >= new_cutoff)
        by_shop: Counter[int] = Counter()
        for lid, r in rate.items():
            by_shop[snap[lid][3]] += r
        reviewed = sum(lifetime.get(lid, 0) for lid in lids)
        gap = None
        if status.get(niche_id) == "ok" and reviewed:
            gap = 100.0 * (fixable.get(niche_id) or 0) / reviewed
        fav_delta = None
        if prev_run_id:
            fav_delta = sum(max(0, snap[lid][1] - prev_favs[lid]) for lid in lids if lid in prev_favs)
        prices = [snap[lid][0] for lid in lids if snap[lid][0] is not None]
        out.append(NicheMetrics(
            niche_id=niche_id,
            reviews_90d=sum(rev.values()),
            active_listings=sum(1 for n in rev.values() if n > 0),
            fav_delta=fav_delta,
            entry_share=_share(new_rate, fav_rate),
            gap_per_100=gap,
            median_price=median(prices) if prices else None,
            listing_count=int(listing_counts.get(niche_id) or 0),
            top3_share=_share(sum(sorted(by_shop.values(), reverse=True)[:3]), fav_rate),
            fav_rate=fav_rate,
            engaged_listings=sum(1 for lid in lids if snap[lid][1] >= cfg.min_favourites),
        ))
    return out
