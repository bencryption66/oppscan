"""Deterministic stand-in for AnthropicLLM, used by tests and `oppscan run --fixtures`."""
from __future__ import annotations

from oppscan.llm import validate


class FakeLLM:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def call(self, task: str, payload: dict) -> dict:
        self.calls.append((task, payload))
        return validate(task, getattr(self, f"_{task}")(payload))

    def _cluster(self, payload: dict) -> dict:
        groups: dict[str, list[str]] = {}
        for item in payload["seeds"]:
            groups.setdefault(item["seed"].split()[0], []).append(item["seed"])
        return {"niches": [{"name": f"{word.title()} templates",
                            "description": f"Templates found by '{word}' searches",
                            "seeds": seeds} for word, seeds in sorted(groups.items())]}

    def _complaints(self, payload: dict) -> dict:
        negative = [r for r in payload["reviews"] if r["rating"] <= 4]
        if not negative:
            return {"themes": []}
        return {"themes": [{"theme": "Confusing setup instructions", "fixable_by_product": True,
                            "review_ids": [r["id"] for r in negative],
                            "quotes": [negative[0]["text"][:80]]}]}

    def _brief(self, payload: dict) -> dict:
        name = payload["niche"]["name"]
        fixes = [f"Fix: {c['theme']}" for c in payload["complaints"]][:3]
        return {"target_buyer": f"Buyers searching for {name.lower()}",
                "core_features": ["Clear setup guide", "Works in Excel and Google Sheets"],
                "differentiators": fixes or ["Cleaner design than current listings"],
                "suggested_price_usd": round(payload["metrics"].get("median_price") or 9.99, 2),
                "title_ideas": [f"{name} | Editable Digital Template"],
                "tags": ["template", "spreadsheet"]}

    def _summary(self, payload: dict) -> dict:
        return {"summary": f"Fixture run with {len(payload['top'])} ranked niches."}
