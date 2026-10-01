"""Run the whole pipeline once (or resume an unfinished run) and record its status."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from oppscan import db
from oppscan.briefs import write_briefs, write_summary
from oppscan.clustering import cluster
from oppscan.collector import collect
from oppscan.complaints import mine_complaints
from oppscan.config import load_etsy, load_scoring, load_seeds
from oppscan.etsy import EtsyClient, QuotaExhausted
from oppscan.llm import AnthropicLLM, LLMClient
from oppscan.metrics import compute_metrics
from oppscan.report import render_report
from oppscan.scoring import sanity_check, save_scores, score_niches
from oppscan.staging import stage_run

RESUMABLE = ("paused", "failed", "running")  # running = interrupted (e.g. Ctrl-C)


@dataclass(frozen=True)
class Paths:
    db: Path
    config_dir: Path
    reports_dir: Path
    archive_dir: Path


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
                 qps: float | None = None) -> RunResult:
    con = db.connect(paths.db)
    try:
        return _run(con, paths, transport, llm, api_key, resume, now, qps)
    finally:
        con.close()


def _run(con, paths, transport, llm, api_key, resume, now, qps) -> RunResult:
    seeds = load_seeds(paths.config_dir / "seeds.yaml")
    cfg = load_scoring(paths.config_dir / "scoring.yaml")
    etsy = load_etsy(paths.config_dir / "etsy.yaml", api_key=api_key)

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
        run_id = db.start_run(con, seeds.file_hash, as_of)

    try:
        llm = llm or AnthropicLLM(con)  # fail on missing credentials before spending Etsy quota
        client = EtsyClient(con, run_id, api_key=etsy.api_key, qps=qps or etsy.qps,
                            daily_quota=etsy.daily_quota, transport=transport)
        try:
            stats = collect(client, seeds, etsy)
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
            db.add_reason(con, run_id, reasons[-1])
        stage_run(con, run_id)
        cluster(con, run_id, seeds, llm)
        complaint_reasons = mine_complaints(con, run_id, llm, cfg.top_n_per_niche)
        for r in complaint_reasons:
            db.add_reason(con, run_id, r)
        reasons += complaint_reasons
        scores = score_niches(compute_metrics(con, run_id, as_of, cfg, db.previous_run(con, run_id)), cfg)
        save_scores(con, run_id, scores)

        seed_niche = dict(con.execute("SELECT seed, niche_id FROM seed_niche WHERE cluster_version = ?",
                                      [seeds.file_hash]).fetchall())
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
    except Exception as e:
        try:
            db.add_reason(con, run_id, f"failed: {e!r}")
            db.finish_run(con, run_id, "failed", db.utcnow())
        except Exception as bookkeeping:  # never mask the original error
            e.add_note(f"also failed to record the failure: {bookkeeping!r}")
        raise
