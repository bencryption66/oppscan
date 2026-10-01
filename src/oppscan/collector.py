"""Fetch search results, shops and reviews for every seed. Responses land in raw_api via EtsyClient."""
from __future__ import annotations

from dataclasses import dataclass, field

from oppscan.config import EtsySettings, Seeds
from oppscan.etsy import EtsyClient, EtsyError

PAGE = 100


@dataclass
class CollectStats:
    seeds: int = 0
    listings_seen: int = 0
    errors: list[str] = field(default_factory=list)


def collect(client: EtsyClient, seeds: Seeds, settings: EtsySettings) -> CollectStats:
    stats = CollectStats()
    for seed in seeds.terms:
        try:
            listings = _search(client, seed, settings.search_depth)
        except EtsyError as e:
            stats.errors.append(f"search '{seed}': {e}")
            continue
        stats.seeds += 1
        downloads = [l for l in listings if l.get("listing_type") == "download"]
        stats.listings_seen += len(downloads)
        top = downloads[: settings.reviews_for_top]
        for shop_id in dict.fromkeys(l["shop_id"] for l in top):
            try:
                client.get(f"/shops/{shop_id}")
            except EtsyError as e:
                stats.errors.append(f"shop {shop_id}: {e}")
        for listing in top:
            try:
                _reviews(client, listing["listing_id"], settings.max_review_pages)
            except EtsyError as e:
                stats.errors.append(f"reviews {listing['listing_id']}: {e}")
    return stats


def _search(client: EtsyClient, seed: str, depth: int) -> list[dict]:
    out: list[dict] = []
    for offset in range(0, depth, PAGE):
        page = client.get("/listings/active",
                          {"keywords": seed, "limit": PAGE, "offset": offset, "sort_on": "score"})
        results = page.get("results", [])
        out.extend(results)
        if len(results) < PAGE or offset + PAGE >= page.get("count", 0):
            break
    return out


def _reviews(client: EtsyClient, listing_id: int, max_pages: int) -> None:
    for page_no in range(max_pages):
        page = client.get(f"/listings/{listing_id}/reviews", {"limit": PAGE, "offset": page_no * PAGE})
        if len(page.get("results", [])) < PAGE:
            break
