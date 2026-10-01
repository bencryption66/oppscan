"""Parse a run's raw_api JSON into the staged tables."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime

SHOP_RE = re.compile(r"^/shops/(\d+)$")
REVIEWS_RE = re.compile(r"^/listings/(\d+)/reviews$")
STAGED_TABLES = ("search_hits", "seed_counts", "listing_snapshots", "shop_snapshots")


def _ts(value) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value), UTC).replace(tzinfo=None)


def review_hash(listing_id: int, created_ts: int, text: str) -> str:
    return hashlib.sha256(f"{listing_id}|{created_ts}|{text}".encode()).hexdigest()[:16]


def price_usd(price: dict | None) -> float | None:
    if not price or price.get("currency_code") != "USD":
        return None
    return price["amount"] / price["divisor"]


def stage_run(con, run_id: str) -> None:
    for table in STAGED_TABLES:
        con.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_id])

    hits: dict[tuple[str, int], int] = {}
    counts: dict[str, int] = {}
    listings: dict[int, tuple] = {}
    shops: dict[int, tuple] = {}
    reviews: dict[str, tuple] = {}

    rows = con.execute("SELECT endpoint, request_key, payload FROM raw_api WHERE run_id = ?", [run_id]).fetchall()
    for endpoint, request_key, payload_json in rows:
        payload = json.loads(payload_json)
        params = json.loads(request_key)
        if endpoint == "/listings/active":
            seed, offset = params["keywords"], int(params["offset"])
            if offset == 0:
                counts[seed] = int(payload.get("count", 0))
            for i, item in enumerate(payload.get("results", [])):
                usd = price_usd(item.get("price"))
                if item.get("listing_type") != "download" or usd is None:
                    continue
                lid = int(item["listing_id"])
                rank = offset + i + 1
                hits[(seed, lid)] = min(rank, hits.get((seed, lid), rank))
                created = (item.get("original_creation_timestamp") or item.get("created_timestamp")
                           or item.get("creation_timestamp"))
                listings[lid] = (run_id, lid, item.get("title", ""), item.get("tags") or [], usd,
                                 int(item.get("num_favorers") or 0), int(item.get("views") or 0),
                                 _ts(created), int(item["shop_id"]), item["listing_type"], item.get("url"))
        elif SHOP_RE.match(endpoint):
            shop_id = int(payload["shop_id"])
            shops[shop_id] = (run_id, shop_id, payload.get("shop_name"),
                              int(payload.get("transaction_sold_count") or 0),
                              int(payload.get("review_count") or 0),
                              payload.get("review_average"), _ts(payload.get("create_date")))
        elif m := REVIEWS_RE.match(endpoint):
            lid = int(m.group(1))
            for r in payload.get("results", []):
                created = r.get("create_timestamp") or r.get("created_timestamp")
                text = r.get("review") or ""
                h = review_hash(lid, created, text)
                reviews[h] = (h, lid, int(r.get("rating") or 0), text, _ts(created), run_id)

    if hits:
        con.executemany("INSERT INTO search_hits VALUES (?, ?, ?, ?)",
                        [(run_id, seed, lid, rank) for (seed, lid), rank in hits.items()])
    if counts:
        con.executemany("INSERT INTO seed_counts VALUES (?, ?, ?)",
                        [(run_id, seed, n) for seed, n in counts.items()])
    if listings:
        con.executemany("INSERT INTO listing_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        list(listings.values()))
    if shops:
        con.executemany("INSERT INTO shop_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)", list(shops.values()))
    if reviews:
        con.executemany("INSERT OR IGNORE INTO reviews VALUES (?, ?, ?, ?, ?, ?)", list(reviews.values()))
