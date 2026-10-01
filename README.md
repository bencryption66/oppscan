# oppscan

Scans Etsy for digital spreadsheet and Notion template niches and writes a ranked opportunity
report with draft product briefs. Design: `docs/superpowers/specs/2026-10-01-oppscan-research-agent-design.md`.

## Setup

```bash
uv sync
```

- **Etsy:** create a developer app at etsy.com/developers and wait for approval. Set `ETSY_API_KEY`
  to the value Etsy's docs say goes in the `x-api-key` header (check whether that's the keystring
  alone or `keystring:shared_secret`). Confirm your app's rate limits and update `config/etsy.yaml`.
- **Claude:** set `ANTHROPIC_API_KEY`, or sign in with `ant auth login`.

## Usage

```bash
uv run oppscan run --fixtures     # offline run on fake data (no keys needed)
uv run oppscan run                # live run -> reports/<run_id>.html and .md
uv run oppscan run --resume <id>  # continue a run paused by the Etsy daily quota
```

Edit `config/seeds.yaml` to change what's searched (this re-clusters niches on the next run) and
`config/scoring.yaml` to change weights or thresholds.

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

`complete` · `partial` (report written; reasons listed in its header) · `paused` (quota reached;
resume) · `failed`.
