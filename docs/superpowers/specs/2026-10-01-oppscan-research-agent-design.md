# oppscan — Market Research Agent Design

**Date:** 2026-10-01
**Status:** Approved design, pending implementation plan

## Purpose

First stage of a larger "AI-produced digital products" pipeline. oppscan scans Etsy for
digital spreadsheet and Notion templates and produces a **ranked opportunity report**: the
niches with the best mix of real demand, room for new entrants and fixable quality gaps, each
with a draft product brief.

It ranks niches **relative to each other**. It does not estimate absolute sales. Real
validation is a small listing test of the top-ranked niches; oppscan decides which niches to
test.

## Scope

**In v1**
- Source: Etsy Open API v3 only (official, free developer key), `listing_type = download`.
- Niche: spreadsheet (Excel / Google Sheets) and Notion templates.
- Output: HTML + Markdown report per run.
- Manual invocation: `oppscan run`.

**Out of v1**
- Gumroad, Notion template gallery, other marketplaces (no compliant search API).
- Scraping of any kind.
- Agentic drill-down (approach C). The design leaves room for it as a later stage.
- Scheduling, live exchange rates (prices convert with a static table, `config/fx.yaml`).
- Paid data tools (eRank, EverBee).

## Approach

A deterministic Python pipeline with LLM calls only at judgement steps (clustering, complaint
mining, brief writing). Chosen over a fully agentic design because consistent scoring between
runs is what makes the ranking meaningful, and because it is 10–50× cheaper.

## Architecture

```
seeds.yaml ─► 1 Collector ─► DuckDB raw ─► 2 Metrics (SQL) ─► 3 Niche clustering (LLM)
                                                                       │
  report.html/.md ◄─ 6 Report writer ◄─ 5 Scoring (SQL) ◄─ 4 Complaint mining (LLM)
```

| # | Component | Responsibility | Depends on |
|---|-----------|----------------|------------|
| 1 | Collector | For each seed term: `findAllListingsActive` (top ~300, download only); `getShop` for each new shop; `getReviewsByListing` for the top 20 listings per seed. Stores raw JSON with `run_id`. Rate-limit pacing and caching. | Etsy API key, `seeds.yaml` |
| 2 | Metrics | SQL over staged tables: per-listing and per-niche metrics (below). | DuckDB |
| 3 | Niche clustering | One batched Claude call groups seeds and listing titles/tags into ~30–80 niches. Cached by a hash of the inputs. | Anthropic API (Sonnet 5.5) |
| 4 | Complaint mining | Pulls structured complaints from reviews (≤4★, plus 5★ text containing "wish", "would be nice", "only issue"). Tags each theme `fixable_by_product` or `not_fixable`. | Anthropic API (Sonnet 5.5) |
| 5 | Scoring | Percentile-based weighted score per niche. Weights in `scoring.yaml`. | Metrics, complaints |
| 6 | Report writer | Jinja HTML/MD report; Claude writes the summary and the top-10 briefs. | Anthropic API (Opus 5.5) |

**Stack:** Python 3.12, uv, httpx, DuckDB, Anthropic SDK, Jinja2, pytest.
**Location:** `~/Projects/AI & MCP/oppscan`.
**Seeds:** 50–150 terms in `seeds.yaml` (e.g. "budget spreadsheet", "notion planner",
"wedding budget google sheets").

## Data model

DuckDB file `data/oppscan.duckdb`. Every observation is a snapshot keyed by `run_id`, so
later runs can compute real deltas between runs.

**Raw (append-only)**

| table | key | contents |
|---|---|---|
| `runs` | `run_id` | started_at, finished_at, seed file hash, API calls used, status (`complete` / `partial` / `failed`), status reasons |
| `raw_api` | `run_id, endpoint, request_key` | full JSON response |

**Staged (rebuilt from raw by SQL)**

| table | grain | columns |
|---|---|---|
| `search_hits` | run × seed × listing | search_rank |
| `listing_snapshots` | run × listing | title, tags[], price_usd, num_favorers, views, created_at, shop_id, listing_type, url, language |
| `shop_snapshots` | run × shop | transaction_sold_count, review_count, review_average, created_at |
| `reviews` | review | review_hash = hash(listing_id, created_at, text); listing_id, rating, text, created_at. Deduplicated across runs. |

**Derived**

| table | grain | contents |
|---|---|---|
| `niches` | niche | niche_id, name, description, cluster_version |
| `listing_niche` | listing × cluster_version | niche assignment |
| `complaints` | niche × theme | theme, fixable flag, up to 3 quotes, mention count, source review hashes |
| `niche_scores` | run × niche | component metrics, percentiles, score, rank, confidence, dropped flag + reason |
| `briefs` | run × niche | generated brief text (top 10) |

**Prices:** `price.amount / price.divisor`, converted to USD with the static rates in `config/fx.yaml` (units per 1 USD, rounded to 2 decimals). Every `download` listing is kept whatever its currency; a currency missing from the table gives `price_usd = NULL`, and the median price skips NULLs.
**Retention:** raw JSON is kept for the last 8 runs; older raw JSON is compressed.

## Scoring

All components are computed over the **20 most favourited listings per niche** (from up to 300 search
results per seed), then converted to
percentiles (0–1) across all niches in the run. Only `download` listings whose `language` matches
`languages` in `etsy.yaml` (default `[en]`, prefix match, missing language kept) are scored.
Search order (`sort_on=score`) is relevance, not popularity: on the first live run a niche's top 20 by search
position had 56 favourites in total against 2,891 for its 20 most favourited, which understated demand.
Ties fall back to best search rank, then `listing_id`.

Listing reviews proved too sparse to carry the score on their own (first live run: about 80% of
top listings had no review in the last year), so favourites are the main demand signal.
**Favourite momentum** (`fav_rate`) is the sum over the top listings of
`num_favorers / age_months`, where `age_months = max(1, days since created / 30.44)`; an unknown
creation date counts as one month.

| Component | Measure | Weight |
|---|---|---|
| Demand | Mean of the percentiles of `fav_rate` and of reviews dated in the last 90 days. From run 2, when every niche has it, the percentile of favourites gained since the previous run is a third term. | +0.35 |
| Entry | Share of `fav_rate` from listings created < 12 months ago (0 when `fav_rate` is 0) | +0.25 |
| Gap | `fixable_by_product` complaint mentions per 100 reviews | +0.20 |
| Price | Median price (USD) | +0.10 |
| Crowding | Average of the percentile of total listing count (API `count`) and the percentile of the top-3 shops' share of `fav_rate` | −0.10 |

`score = 0.35·D + 0.25·E + 0.20·G + 0.10·P − 0.10·C`

**Guard rails**
- **Demand floor:** niches below the 25th percentile of Demand are dropped (`dropped_reason = 'demand_floor'`).
- **Confidence:** `high` if ≥ 8 of the top listings have ≥ 5 favourites (`high_confidence.min_listings` and `min_favourites` in `scoring.yaml`); otherwise `low`. All `high` niches rank above all `low` niches.
- **Missing complaint data:** Gap is set to the median percentile (0.5) and the niche is flagged.
- **Re-scoring:** `oppscan rescore <run_id>` re-runs everything after collection for a complete, partial or failed run from its stored `raw_api` rows, with no Etsy calls, so scoring changes can be applied to past runs.
- **Sanity check:** `scoring.yaml` lists known-big niches (e.g. "monthly budget spreadsheet"). If any lands below the 50th percentile of Demand, the run is flagged `suspect` in the report header.

## Report

`reports/<run_id>.html` and `reports/<run_id>.md`. Can optionally be published as a private
artifact.

1. **Header:** run date; counts of seeds, listings, reviews and API calls; run status; sanity
   badge; link to the previous report.
2. **Ranked table:** rank, niche, score, five component bars, confidence, median price, rank
   change vs the previous run (▲ / ▼ / new).
3. **Top-10 niche cards:**
   - one sentence per component explaining the rank;
   - top 3 fixable complaints with short quotes;
   - the 3 strongest competitor listings by favourites per month, then 90-day reviews (title, price, favourites/month, 90-day reviews, link);
   - draft brief: target buyer, core features, what to do better than competitors,
     suggested price, title and tag ideas.
4. **Dropped and borderline:** one line per dropped or low-confidence niche, with the reason.
5. **Method appendix:** weights, proxy definitions, the relative-ranking caveat.

## Error handling

- **Etsy API:** exponential backoff on 429/5xx. When the daily quota is used up, the run pauses
  and resumes from the `raw_api` cache (`oppscan run --resume <run_id>`). A request already
  stored for the run is never sent again.
- **Rate limits:** read from config. Verify the actual limits on the approved key before the
  first live run.
- **LLM steps:** every response is checked against a JSON schema. Invalid output is retried
  once, then the batch is marked failed and the run continues with status `partial`.
- **Run status** and its reasons are recorded in `runs` and shown in the report header. A
  partial run still produces a report.

## Testing

- **Unit (pytest):** metric SQL and scoring against fixture tables with hand-calculated
  expected scores; percentile, demand-floor and confidence logic; price normalisation;
  review-hash deduplication.
- **Collector:** recorded Etsy responses committed as fixtures. Tests make no network calls.
- **LLM steps:** schema validation over recorded responses. Optional live smoke test behind
  `--live`.
- **End to end:** `oppscan run --fixtures` produces a full report from fixtures, so the pipeline
  can be proved before the Etsy key is approved.

## Prerequisites

- Etsy developer account and approved app key (approval may take days).
- `ANTHROPIC_API_KEY`.

## Future stages (not in this spec)

- Approach C: agentic drill-down into the top 5 niches.
- More sources (Gumroad, the Notion gallery) once a compliant data path exists.
- Scheduled weekly runs.
- Feeding briefs into producer agents.
