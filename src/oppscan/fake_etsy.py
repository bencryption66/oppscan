"""Deterministic fake of the Etsy endpoints oppscan uses (tests and `oppscan run --fixtures`).

Listings belong to a "family" (the first word of the search keywords), so related seeds
share listings. Listings and reviews are fixed relative to ANCHOR. Requests only see
reviews dated on or before `now`, so later runs see more reviews and favourites.
"""
from __future__ import annotations

import hashlib
import random
import re
from datetime import UTC, datetime, timedelta

import httpx

ANCHOR = datetime(2026, 9, 1)
FUTURE = ANCHOR + timedelta(days=120)
POOL_SIZE = 500
ADJECTIVES = ["Simple", "Ultimate", "Aesthetic", "Minimal", "Editable", "Printable", "Digital"]
NOUNS = ["Spreadsheet", "Planner", "Tracker", "Dashboard", "Template", "Workbook"]
POSITIVE = ["Love it, easy to use", "Exactly what I needed", "Beautiful and practical",
            "Wish it had a dark mode, otherwise great"]
NEGATIVE = ["Instructions were confusing", "Does not work in Google Sheets on my phone",
            "Seller never replied to my message", "Formulas broke when I added rows"]


def _rng(*parts) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _unix(dt: datetime) -> int:
    return int(dt.replace(tzinfo=UTC).timestamp())


def _family(keywords: str) -> str:
    words = keywords.split()
    return words[0] if words else "misc"


def _family_base(family: str) -> int:
    return 1_000_000 + int(hashlib.sha256(family.encode()).hexdigest()[:6], 16) * 1000


def _listing_attrs(listing_id: int) -> dict:
    rng = _rng("listing", listing_id)
    return {
        "listing_type": rng.choices(["download", "physical", "both"], [85, 10, 5])[0],
        "currency": rng.choices(["USD", "EUR", "GBP", "CAD", "XXX"], [70, 10, 8, 7, 5])[0],
        "amount": rng.randint(299, 2999),
        "created": ANCHOR - timedelta(days=rng.randint(1, 1200)),
        "popularity": rng.paretovariate(1.5),
        "shop_id": 50_000 + rng.randint(0, 120),
        "adjective": rng.choice(ADJECTIVES),
        "noun": rng.choice(NOUNS),
        "language": rng.choices(["en-US", "de"], [92, 8])[0],  # drawn last so other attributes are unchanged
    }


def _listing(listing_id: int, family: str, now: datetime) -> dict:
    a = _listing_attrs(listing_id)
    age_days = max(0, (now - a["created"]).days)
    return {
        "listing_id": listing_id,
        "title": f"{a['adjective']} {family.title()} {a['noun']}",
        "tags": [family, a["noun"].lower(), "digital download"],
        "price": {"amount": a["amount"], "divisor": 100, "currency_code": a["currency"]},
        "num_favorers": int(a["popularity"] * 50 + age_days * a["popularity"] * 0.5),
        "views": int(a["popularity"] * 500 + age_days * a["popularity"] * 5),
        "original_creation_timestamp": _unix(a["created"]),
        "created_timestamp": _unix(a["created"]),
        "shop_id": a["shop_id"],
        "listing_type": a["listing_type"],
        "url": f"https://www.etsy.com/listing/{listing_id}",
        "language": a["language"],
    }


def _reviews(listing_id: int, now: datetime) -> list[dict]:
    a = _listing_attrs(listing_id)
    rng = _rng("reviews", listing_id)
    span = max(1, (FUTURE - a["created"]).days)
    out = []
    for _ in range(min(400, int(a["popularity"] * 15))):
        created = a["created"] + timedelta(days=rng.randint(0, span), seconds=rng.randint(0, 86_399))
        rating = rng.choices([5, 4, 3, 2, 1], [75, 12, 7, 3, 3])[0]
        text = rng.choice(POSITIVE if rating == 5 else NEGATIVE)
        if created <= now:
            out.append({"listing_id": listing_id, "rating": rating, "review": text,
                        "create_timestamp": _unix(created), "created_timestamp": _unix(created)})
    out.sort(key=lambda r: r["create_timestamp"], reverse=True)
    return out


def fake_transport(now: datetime) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v3/application")
        params = request.url.params
        limit = int(params.get("limit", 25))
        offset = int(params.get("offset", 0))

        if path == "/listings/active":
            keywords = params.get("keywords", "")
            family = _family(keywords)
            base = _family_base(family)
            order = list(range(POOL_SIZE))
            _rng("order", keywords).shuffle(order)
            count = 150 + _rng("count", keywords).randint(0, 2000)
            available = order[: min(count, POOL_SIZE)]
            page = available[offset: offset + limit]
            return httpx.Response(200, json={
                "count": count,
                "results": [_listing(base + i, family, now) for i in page],
            })

        if m := re.fullmatch(r"/shops/(\d+)", path):
            shop_id = int(m.group(1))
            rng = _rng("shop", shop_id)
            return httpx.Response(200, json={
                "shop_id": shop_id,
                "shop_name": f"Shop{shop_id}",
                "transaction_sold_count": rng.randint(10, 50_000),
                "review_count": rng.randint(1, 5_000),
                "review_average": round(rng.uniform(4.0, 5.0), 2),
                "create_date": _unix(ANCHOR - timedelta(days=rng.randint(100, 3000))),
            })

        if m := re.fullmatch(r"/listings/(\d+)/reviews", path):
            reviews = _reviews(int(m.group(1)), now)
            return httpx.Response(200, json={"count": len(reviews),
                                             "results": reviews[offset: offset + limit]})

        return httpx.Response(404, json={"error": f"no fake for {path}"})

    return httpx.MockTransport(handler)
