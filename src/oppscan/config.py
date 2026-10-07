"""Load oppscan's YAML configuration files."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Seeds:
    terms: tuple[str, ...]
    file_hash: str


@dataclass(frozen=True)
class Weights:
    demand: float
    entry: float
    gap: float
    price: float
    crowding: float


@dataclass(frozen=True)
class ScoringConfig:
    weights: Weights
    demand_floor_pct: float
    high_conf_min_reviews: int
    high_conf_min_listings: int
    top_n_per_niche: int
    review_window_days: int
    new_listing_days: int
    report_top_k: int
    known_big_seeds: tuple[str, ...]


@dataclass(frozen=True)
class EtsySettings:
    api_key: str
    qps: float
    daily_quota: int
    search_depth: int
    reviews_for_top: int
    max_review_pages: int
    languages: tuple[str, ...] = ()  # listing-language prefixes to keep; empty keeps all


def language_ok(language: str | None, languages: tuple[str, ...]) -> bool:
    """True if a listing in `language` should be kept: no filter, unknown language, or a prefix match."""
    if not languages or not language:
        return True
    return language.lower().startswith(tuple(p.lower() for p in languages))


def normalise(term: str) -> str:
    return " ".join(str(term).lower().split())


def load_seeds(path: Path) -> Seeds:
    data = yaml.safe_load(Path(path).read_text()) or {}
    terms = tuple(dict.fromkeys(t for t in (normalise(s) for s in data.get("seeds", [])) if t))
    if not terms:
        raise ValueError(f"{path}: no seeds defined")
    # Hash the normalised terms, so comment or whitespace edits don't trigger re-clustering.
    return Seeds(terms=terms, file_hash=hashlib.sha256("\n".join(terms).encode()).hexdigest()[:16])


def load_scoring(path: Path) -> ScoringConfig:
    d = yaml.safe_load(Path(path).read_text())
    hc = d["high_confidence"]
    return ScoringConfig(
        weights=Weights(**{k: float(v) for k, v in d["weights"].items()}),
        demand_floor_pct=float(d["demand_floor_pct"]),
        high_conf_min_reviews=int(hc["min_reviews_90d"]),
        high_conf_min_listings=int(hc["min_listings"]),
        top_n_per_niche=int(d["top_n_per_niche"]),
        review_window_days=int(d["review_window_days"]),
        new_listing_days=int(d["new_listing_days"]),
        report_top_k=int(d["report_top_k"]),
        known_big_seeds=tuple(normalise(s) for s in d.get("known_big_seeds", [])),
    )


def load_etsy(path: Path, api_key: str | None = None) -> EtsySettings:
    d = yaml.safe_load(Path(path).read_text())
    key = api_key if api_key is not None else os.environ.get("ETSY_API_KEY", "")
    if not key:
        raise ValueError("ETSY_API_KEY is not set")
    return EtsySettings(
        api_key=key,
        qps=float(d["qps"]),
        daily_quota=int(d["daily_quota"]),
        search_depth=int(d["search_depth"]),
        reviews_for_top=int(d["reviews_for_top"]),
        max_review_pages=int(d["max_review_pages"]),
        languages=tuple(d.get("languages", [])),
    )


def load_fx(path: Path) -> dict[str, float]:
    """Units of each currency per 1 USD, keyed by uppercase currency code."""
    d = yaml.safe_load(Path(path).read_text()) or {}
    rates = {str(code).upper(): float(rate) for code, rate in (d.get("per_usd") or {}).items()}
    if "USD" not in rates:
        raise ValueError(f"{path}: per_usd must include USD")
    for code, rate in rates.items():
        if rate <= 0:
            raise ValueError(f"{path}: rate for {code} must be positive, got {rate}")
    return rates
