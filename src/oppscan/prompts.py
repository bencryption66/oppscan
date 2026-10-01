"""JSON schemas and prompt text for each LLM task."""
from __future__ import annotations

import json


def _obj(**properties) -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def _arr(items: dict) -> dict:
    return {"type": "array", "items": items}


STR = {"type": "string"}

SCHEMAS: dict[str, dict] = {
    "cluster": _obj(niches=_arr(_obj(name=STR, description=STR, seeds=_arr(STR)))),
    "complaints": _obj(themes=_arr(_obj(theme=STR, fixable_by_product={"type": "boolean"},
                                        review_ids=_arr(STR), quotes=_arr(STR)))),
    "brief": _obj(target_buyer=STR, core_features=_arr(STR), differentiators=_arr(STR),
                  suggested_price_usd={"type": "number"}, title_ideas=_arr(STR), tags=_arr(STR)),
    "summary": _obj(summary=STR),
}

SYSTEM = ("You analyse Etsy marketplace data for a seller of digital spreadsheet and Notion "
          "templates. Base every statement on the data provided and be specific.")

INSTRUCTIONS: dict[str, str] = {
    "cluster": (
        "Group these search seeds into product niches. Each seed comes with the titles of its "
        "top-ranked listings. A niche is a specific product a buyer searches for, for example "
        "'ADHD-friendly Notion life planner' rather than 'planner'. Put every seed in exactly one "
        "niche and copy each seed string exactly as given. Aim for roughly one niche per two or "
        "three seeds; a seed may be a niche on its own."
    ),
    "complaints": (
        "These are reviews of the top listings in one niche. Identify recurring complaint or wish "
        "themes. For each theme give the ids of the reviews that mention it, up to three short "
        "verbatim quotes (under 20 words each), and set fixable_by_product to true only if a "
        "better-designed template would fix it (for example confusing instructions, a missing "
        "feature, breaking on mobile). Seller-service issues (slow replies, delivery, refunds) are "
        "false. Ignore pure praise. Return an empty list if there are no complaints."
    ),
    "brief": (
        "Write a product brief for a new digital template in this niche that would beat the "
        "current top listings. Use the complaints to decide what to do better. Keep the price "
        "within the observed price band unless the data justifies otherwise. title_ideas: three "
        "Etsy titles under 140 characters. tags: up to 13 Etsy tags, each under 20 characters."
    ),
    "summary": (
        "Write a three to five sentence plain summary of this research run for the seller: which "
        "niches look most promising and why, and any caveats about data quality. Factual tone."
    ),
}


def render(task: str, payload: dict) -> tuple[str, str]:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return SYSTEM, f"{INSTRUCTIONS[task]}\n\n<data>\n{data}\n</data>"
