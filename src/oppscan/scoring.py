"""Turn niche metrics into percentile scores, confidence labels, drops and ranks."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from oppscan.config import ScoringConfig
from oppscan.metrics import NicheMetrics


@dataclass(frozen=True)
class NicheScore:
    metrics: NicheMetrics
    pct_demand: float
    pct_entry: float
    pct_gap: float
    pct_price: float
    pct_crowding: float
    score: float
    confidence: str
    gap_missing: bool
    dropped: bool
    dropped_reason: str | None
    rank: int | None


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """Average-rank percentiles in [0, 1]; ties share a value; a single value maps to 0.5."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = ((i + j) / 2) / (n - 1)
        i = j + 1
    return out


def _mean(*columns: list[float]) -> list[float]:
    return [sum(values) / len(values) for values in zip(*columns)]


def score_niches(metrics: list[NicheMetrics], cfg: ScoringConfig) -> list[NicheScore]:
    if not metrics:
        return []
    w = cfg.weights
    demand_parts = [percentile_ranks([m.fav_rate for m in metrics]),
                    percentile_ranks([m.reviews_90d for m in metrics])]
    if all(m.fav_delta is not None for m in metrics):
        demand_parts.append(percentile_ranks([m.fav_delta for m in metrics]))
    demand = _mean(*demand_parts)
    entry = percentile_ranks([m.entry_share for m in metrics])
    crowding = _mean(percentile_ranks([m.listing_count for m in metrics]),
                     percentile_ranks([m.top3_share for m in metrics]))
    gap = [0.5] * len(metrics)
    known = [i for i, m in enumerate(metrics) if m.gap_per_100 is not None]
    for i, p in zip(known, percentile_ranks([metrics[i].gap_per_100 for i in known])):
        gap[i] = p

    price = [0.5] * len(metrics)
    priced = [i for i, m in enumerate(metrics) if m.median_price is not None]
    for i, p in zip(priced, percentile_ranks([metrics[i].median_price for i in priced])):
        price[i] = p

    scores = []
    for i, m in enumerate(metrics):
        dropped = demand[i] < cfg.demand_floor_pct
        high = m.engaged_listings >= cfg.high_conf_min_listings
        scores.append(NicheScore(
            metrics=m, pct_demand=demand[i], pct_entry=entry[i], pct_gap=gap[i],
            pct_price=price[i], pct_crowding=crowding[i],
            score=(w.demand * demand[i] + w.entry * entry[i] + w.gap * gap[i]
                   + w.price * price[i] - w.crowding * crowding[i]),
            confidence="high" if high else "low",
            gap_missing=m.gap_per_100 is None,
            dropped=dropped, dropped_reason="demand_floor" if dropped else None, rank=None,
        ))
    ranked = sorted((s for s in scores if not s.dropped),
                    key=lambda s: (s.confidence != "high", -s.score, s.metrics.niche_id))
    rank_of = {s.metrics.niche_id: r for r, s in enumerate(ranked, start=1)}
    return [replace(s, rank=rank_of.get(s.metrics.niche_id)) for s in scores]


def sanity_check(scores: list[NicheScore], seed_niche: dict[str, str],
                 known_big: Sequence[str]) -> list[str]:
    by_id = {s.metrics.niche_id: s for s in scores}
    problems = []
    for seed in known_big:
        niche_id = seed_niche.get(seed)
        if niche_id is None:
            problems.append(f"known-big seed '{seed}' is not in seeds.yaml")
        elif niche_id not in by_id:
            problems.append(f"known-big seed '{seed}' produced no niche data")
        elif by_id[niche_id].pct_demand < 0.5:
            problems.append(f"known-big niche '{niche_id}' has demand percentile "
                            f"{by_id[niche_id].pct_demand:.2f} (below 0.50)")
    return problems


SCORE_COLUMNS = ["run_id", "niche_id", "reviews_90d", "active_listings", "fav_delta", "entry_share",
                 "gap_per_100", "gap_missing", "median_price", "listing_count", "top3_share",
                 "pct_demand", "pct_entry", "pct_gap", "pct_price", "pct_crowding",
                 "score", "confidence", "dropped", "dropped_reason", "rank", "fav_rate", "engaged_listings"]


def save_scores(con, run_id: str, scores: list[NicheScore]) -> None:
    con.execute("DELETE FROM niche_scores WHERE run_id = ?", [run_id])
    if not scores:
        return
    con.executemany(
        f"INSERT INTO niche_scores ({', '.join(SCORE_COLUMNS)}) VALUES ({', '.join('?' * len(SCORE_COLUMNS))})",
        [(run_id, s.metrics.niche_id, s.metrics.reviews_90d, s.metrics.active_listings,
          s.metrics.fav_delta, s.metrics.entry_share, s.metrics.gap_per_100, s.gap_missing,
          s.metrics.median_price, s.metrics.listing_count, s.metrics.top3_share,
          s.pct_demand, s.pct_entry, s.pct_gap, s.pct_price, s.pct_crowding,
          s.score, s.confidence, s.dropped, s.dropped_reason, s.rank,
          s.metrics.fav_rate, s.metrics.engaged_listings) for s in scores],
    )
