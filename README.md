# oppscan

Scans Etsy for digital spreadsheet and Notion template niches and writes a ranked opportunity
report with draft product briefs. Design: `docs/superpowers/specs/2026-10-01-oppscan-research-agent-design.md`.

## Setup

```bash
uv sync
```

- **Etsy:** create a developer app at etsy.com/developers and wait for approval. Set `ETSY_API_KEY`
  to `<keystring>:<shared_secret>` (both values from the Your Apps page, joined by a colon; Etsy's
  v3 `x-api-key` header requires both). Check your app's rate limits in the Developer Portal and
  update `qps` and `daily_quota` in `config/etsy.yaml` to match.
- **Claude:** set `ANTHROPIC_API_KEY`, or sign in with `ant auth login`.

## Usage

```bash
uv run oppscan run --fixtures     # offline run on fake data (no keys needed)
uv run oppscan run                # live run -> reports/<run_id>.html and .md
uv run oppscan run --resume <id>  # continue a paused, failed or interrupted run
```

A paused run (Etsy daily quota or persistent rate limiting) exits with code 2 and prints the
`--resume` command. `--resume` re-uses the responses already fetched, so it spends no quota on
them. Resuming a complete or partial run, or an unknown id, prints `error: ...` and exits with 1.

For the first live run, set `max_review_pages: 1` and use a smaller seed list to stay inside the
daily quota.

Edit `config/seeds.yaml` to change what's searched (this re-clusters niches on the next run) and
`config/scoring.yaml` to change weights or thresholds. `config/fx.yaml` holds the static exchange
rates (units per 1 USD) used to convert non-USD listing prices; update it by hand now and then. A
listing in a currency missing from the table is kept with an unknown price.

## Live checks

Run these once the keys exist, before the first real run:

```bash
ETSY_API_KEY=... uv run pytest -m live -v
OPPSCAN_LIVE_LLM=1 uv run pytest -m live -v -k anthropic
```

The Etsy check confirms the response fields the parser relies on and saves a sample to
`tests/fixtures/etsy_live_sample.json`. If a field differs, update `staging.py`, `fake_etsy.py`
and `tests/test_staging.py` to match.

## Status values

`complete` · `partial` (report written; reasons listed in its header) · `paused` (quota or rate
limit reached; resume) · `failed` (resume after fixing the cause). A run left `running` was
interrupted and can also be resumed.
