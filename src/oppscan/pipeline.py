"""Run the whole pipeline once, resume an unfinished run, or re-score a finished one; record its status."""
from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from oppscan import db
from oppscan.briefs import write_briefs, write_summary
from oppscan.clustering import cluster
from oppscan.collector import collect
from oppscan.complaints import mine_complaints
from oppscan.config import EtsySettings, ScoringConfig, Seeds, load_etsy, load_fx, load_scoring, load_seeds
from oppscan.etsy import EtsyClient, QuotaExhausted
from oppscan.llm import AnthropicLLM, LLMClient
from oppscan.metrics import compute_metrics
from oppscan.report import render_report
from oppscan.scoring import sanity_check, save_scores, score_niches
from oppscan.staging import stage_run

RESUMABLE = ("paused", "failed", "running")  # running = interrupted (e.g. Ctrl-C)
RESCORABLE = ("complete", "partial")
ETSY_FAILURES_RE = re.compile(r"^\d+ Etsy request\(s\) failed; first: ")


@dataclass(frozen=True)
class Paths:
    db: Path
    config_dir: Path
    reports_dir: Path
    archive_dir: Path


@dataclass(frozen=True)
class Settings:
    seeds: Seeds
    cfg: ScoringConfig
    etsy: EtsySettings
    fx: dict[str, float]


def load_settings(config_dir: Path, api_key: str | None = None, require_key: bool = True) -> Settings:
    return Settings(seeds=load_seeds(config_dir / "seeds.yaml"),
                    cfg=load_scoring(config_dir / "scoring.yaml"),
                    etsy=load_etsy(config_dir / "etsy.yaml", api_key=api_key, require_key=require_key),
                    fx=load_fx(config_dir / "fx.yaml"))


@dataclass
class RunResult:
    run_id: str
    status: str
    reasons: list[str] = field(default_factory=list)
    html: Path | None = None
    md: Path | None = None


def run_pipeline(paths: Paths, *, transport: httpx.BaseTransport | None = None,
                 llm: LLMClient | None = None, api_key: str | None = None,
                 resume: str | None = None, now: datetime | None = None,
                 qps: float | None = None, daily_quota: int | None = None) -> RunResult:
    con = db.connect(paths.db)
    try:
        return _run(con, paths, transport, llm, api_key, resume, now, qps, daily_quota)
    finally:
        con.close()


def rescore_pipeline(paths: Paths, run_id: str, *, llm: LLMClient | None = None) -> RunResult:
    """Re-run everything after collection for a finished run, from its stored API responses."""
    con = db.connect(paths.db)
    try:
        return _rescore(con, paths, run_id, llm)
    finally:
        con.close()


def _run(con, paths, transport, llm, api_key, resume, now, qps, daily_quota) -> RunResult:
    s = load_settings(paths.config_dir, api_key=api_key)
    # Fail on missing credentials before creating a run row or spending Etsy quota.
    llm = llm or AnthropicLLM(con)

    if resume:
        existing = db.get_run(con, resume)
        if existing is None:
            raise ValueError(f"unknown run {resume}")
        if existing["status"] not in RESUMABLE:
            raise ValueError(f"run {resume} is {existing['status']}; "
                             "only a paused, failed or interrupted run can be resumed")
        run_id, as_of = resume, existing["started_at"]
        db.reopen_run(con, run_id)
    else:
        as_of = now or db.utcnow()
        run_id = db.start_run(con, s.seeds.file_hash, as_of)

    with _fail_run_on_error(con, run_id):
        client = EtsyClient(con, run_id, api_key=s.etsy.api_key, qps=qps or s.etsy.qps,
                            daily_quota=daily_quota or s.etsy.daily_quota, transport=transport)
        try:
            stats = collect(client, s.seeds, s.etsy)
        except QuotaExhausted as e:
            reason = f"{e}; resume with: oppscan run --resume {run_id}"
            db.add_reason(con, run_id, reason)
            db.finish_run(con, run_id, "paused", db.utcnow())
            return RunResult(run_id, "paused", [reason])
        finally:
            client.close()

        reasons: list[str] = []
        if stats.errors:
            reasons.append(f"{len(stats.errors)} Etsy request(s) failed; first: {stats.errors[0]}")
        return _analyse(con, paths, s, llm, run_id, as_of, reasons)


def _rescore(con, paths, run_id, llm) -> RunResult:
    s = load_settings(paths.config_dir, require_key=False)  # no Etsy calls, so no Etsy key needed
    llm = llm or AnthropicLLM(con)  # fail on missing credentials before touching the run

    existing = db.get_run(con, run_id)
    if existing is None:
        raise ValueError(f"unknown run {run_id}")
    if existing["status"] not in RESCORABLE:
        raise ValueError(f"run {run_id} is {existing['status']}; only a complete or partial run can be "
                         "re-scored (resume it with: oppscan run --resume)")
    if existing["seeds_hash"] != s.seeds.file_hash:
        raise ValueError(f"run {run_id} used a different seeds file than {paths.config_dir / 'seeds.yaml'}; "
                         "pass the --config-dir it ran with")
    if not con.execute("SELECT count(*) FROM raw_api WHERE run_id = ?", [run_id]).fetchone()[0]:
        raise ValueError(f"run {run_id} has no stored API responses (archived after 8 newer runs)")

    # Collection errors can't be recomputed without Etsy, so keep that reason from the original run.
    reasons = [r for r in existing["status_reasons"] if ETSY_FAILURES_RE.match(r)]
    db.reopen_run(con, run_id)
    with _fail_run_on_error(con, run_id):
        return _analyse(con, paths, s, llm, run_id, existing["started_at"], reasons)


@contextmanager
def _fail_run_on_error(con, run_id: str) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        try:
            db.add_reason(con, run_id, f"failed: {e!r}")
            db.finish_run(con, run_id, "failed", db.utcnow())
        except Exception as bookkeeping:  # never mask the original error
            e.add_note(f"also failed to record the failure: {bookkeeping!r}")
        raise


def _analyse(con, paths: Paths, s: Settings, llm: LLMClient, run_id: str, as_of: datetime,
             reasons: list[str]) -> RunResult:
    """Everything after collection: stage, cluster, mine, score, check, brief, summarise, report."""
    reasons = list(reasons)
    for r in reasons:
        db.add_reason(con, run_id, r)
    cfg = s.cfg
    stage_run(con, run_id, s.fx, s.etsy.languages)
    cluster(con, run_id, s.seeds, llm)
    complaint_reasons = mine_complaints(con, run_id, llm, cfg.top_n_per_niche)
    for r in complaint_reasons:
        db.add_reason(con, run_id, r)
    reasons += complaint_reasons
    scores = score_niches(compute_metrics(con, run_id, as_of, cfg, db.previous_run(con, run_id)), cfg)
    save_scores(con, run_id, scores)

    seed_niche = dict(con.execute("SELECT seed, niche_id FROM seed_niche WHERE cluster_version = ?",
                                  [s.seeds.file_hash]).fetchall())
    problems = sanity_check(scores, seed_niche, cfg.known_big_seeds)
    if problems:
        db.set_suspect(con, run_id)
        for p in problems:
            db.add_reason(con, run_id, f"sanity: {p}")

    brief_reasons = write_briefs(con, run_id, llm, cfg, as_of)
    for r in brief_reasons:
        db.add_reason(con, run_id, r)
    reasons += brief_reasons
    for r in write_summary(con, run_id, llm, cfg):
        db.add_reason(con, run_id, r)
        reasons.append(r)
    status = "partial" if reasons else "complete"
    db.finish_run(con, run_id, status, db.utcnow())
    html, md = render_report(con, run_id, cfg, paths.reports_dir)
    db.prune_raw(con, paths.archive_dir)
    return RunResult(run_id, status, reasons, html, md)
