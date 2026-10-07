import os
import shutil
from datetime import datetime
from pathlib import Path

import duckdb
import httpx
import pytest

from oppscan import db
from oppscan.cli import main
from oppscan.fake_etsy import fake_transport
from oppscan.fake_llm import FakeLLM
from oppscan.pipeline import Paths, run_pipeline

REPO = Path(__file__).resolve().parents[1]
T1 = datetime(2026, 10, 1, 8, 0, 0)
T2 = datetime(2026, 10, 8, 8, 0, 0)
SEEDS = """seeds:
  - budget spreadsheet
  - budget planner
  - notion planner
  - notion dashboard
  - wedding budget
  - wedding planner
  - habit tracker
  - habit spreadsheet
  - monthly budget spreadsheet
"""


def setup_config(tmp_path, quota=100_000):
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    shutil.copy(REPO / "config" / "scoring.yaml", config / "scoring.yaml")
    shutil.copy(REPO / "config" / "fx.yaml", config / "fx.yaml")
    (config / "seeds.yaml").write_text(SEEDS)
    (config / "etsy.yaml").write_text(
        f"qps: 5\ndaily_quota: {quota}\nsearch_depth: 200\nreviews_for_top: 5\nmax_review_pages: 2\n"
        "languages: [en]\n")
    return Paths(db=tmp_path / "data" / "test.duckdb", config_dir=config,
                 reports_dir=tmp_path / "reports", archive_dir=tmp_path / "data" / "archive")


def run(paths, now, **kw):
    return run_pipeline(paths, transport=kw.pop("transport", fake_transport(now)), llm=FakeLLM(),
                        api_key="fixture", now=now, qps=1e6, **kw)


def test_two_runs_end_to_end(tmp_path):
    paths = setup_config(tmp_path)
    first = run(paths, T1)
    assert first.status == "complete", first.reasons
    html = first.html.read_text()
    assert "Budget templates" in html
    assert "Draft brief" in html

    second = run(paths, T2)
    assert second.status == "complete", second.reasons
    assert f'href="{first.run_id}.html"' in second.html.read_text()
    con = duckdb.connect(str(paths.db), read_only=True)
    with_delta = con.execute("SELECT count(*) FROM niche_scores WHERE run_id = ? AND fav_delta IS NOT NULL",
                             [second.run_id]).fetchone()[0]
    languages = {r[0] for r in con.execute("SELECT DISTINCT language FROM listing_snapshots").fetchall()}
    con.close()
    assert with_delta > 0
    assert languages == {"en-US"}  # the fake API's "de" listings are filtered out


def test_quota_pause_then_resume(tmp_path):
    paths = setup_config(tmp_path, quota=10)
    calls = []
    inner = fake_transport(T1)

    def counting(request):
        calls.append(str(request.url))
        return inner.handle_request(request)

    paused = run(paths, T1, transport=httpx.MockTransport(counting))
    assert paused.status == "paused"
    assert "--resume" in paused.reasons[0]
    assert paused.html is None
    first_calls = list(calls)
    assert len(first_calls) == 10

    setup_config(tmp_path, quota=100_000)
    resumed = run(paths, T1, transport=httpx.MockTransport(counting), resume=paused.run_id)
    assert resumed.run_id == paused.run_id
    assert resumed.status == "complete", resumed.reasons
    assert not set(first_calls) & set(calls[len(first_calls):])  # nothing re-fetched
    con = duckdb.connect(str(paths.db), read_only=True)
    stored = con.execute("SELECT status_reasons FROM runs WHERE run_id = ?", [resumed.run_id]).fetchone()[0]
    con.close()
    assert not any("--resume" in r for r in (stored or []))


def test_resume_rejects_completed_run(tmp_path):
    paths = setup_config(tmp_path)
    first = run(paths, T1)
    assert first.status == "complete", first.reasons
    with pytest.raises(ValueError, match=f"{first.run_id}.*complete"):
        run(paths, T1, resume=first.run_id)


def counting_transport(now, calls):
    inner = fake_transport(now)

    def handler(request):
        calls.append(str(request.url))
        return inner.handle_request(request)

    return httpx.MockTransport(handler)


def test_resume_failed_run(tmp_path):
    paths = setup_config(tmp_path)
    calls = []

    class Broken:
        def call(self, task, payload):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(paths, transport=counting_transport(T1, calls), llm=Broken(),
                     api_key="fixture", now=T1, qps=1e6)
    con = duckdb.connect(str(paths.db), read_only=True)
    run_id, status = con.execute("SELECT run_id, status FROM runs").fetchone()
    con.close()
    assert status == "failed"
    first_calls = len(calls)
    assert first_calls > 0

    resumed = run(paths, T1, transport=counting_transport(T1, calls), resume=run_id)
    assert resumed.run_id == run_id
    assert resumed.status == "complete", resumed.reasons
    assert len(calls) == first_calls  # nothing re-fetched


def test_failure_bookkeeping_errors_do_not_mask_original(tmp_path, monkeypatch):
    paths = setup_config(tmp_path)

    class Broken:
        def call(self, task, payload):
            raise RuntimeError("boom")

    def broken_db(*args):
        raise OSError("db gone")

    monkeypatch.setattr("oppscan.db.add_reason", broken_db)
    monkeypatch.setattr("oppscan.db.finish_run", broken_db)
    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(paths, transport=fake_transport(T1), llm=Broken(), api_key="fixture", now=T1, qps=1e6)


def test_missing_llm_credentials_fail_before_etsy_calls(tmp_path, monkeypatch):
    paths = setup_config(tmp_path)
    calls = []

    for name in list(os.environ):
        if name.startswith("ANTHROPIC_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))  # real SDK client, no credentials anywhere
    with pytest.raises(ValueError, match="No Anthropic credentials"):
        run_pipeline(paths, transport=counting_transport(T1, calls), api_key="fixture", now=T1, qps=1e6)
    assert calls == []
    con = duckdb.connect(str(paths.db), read_only=True)
    assert con.execute("SELECT count(*) FROM runs").fetchone()[0] == 0  # no empty failed run left behind
    con.close()


def test_resume_rejects_unknown_run(tmp_path):
    paths = setup_config(tmp_path)
    with pytest.raises(ValueError, match="nope"):
        run(paths, T1, resume="nope")


def test_cli_fixtures(tmp_path, capsys):
    paths = setup_config(tmp_path)
    code = main(["run", "--fixtures", "--config-dir", str(paths.config_dir),
                 "--data-dir", str(tmp_path / "data"), "--reports-dir", str(tmp_path / "reports")])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "complete" in out
    assert list((tmp_path / "reports" / "fixtures").glob("*.html"))


def test_cli_fixtures_ignores_daily_quota(tmp_path, capsys):
    paths = setup_config(tmp_path, quota=5000)
    (tmp_path / "data").mkdir()
    con = db.connect(tmp_path / "data" / "fixtures.duckdb")
    now = db.utcnow()
    con.executemany("INSERT INTO raw_api VALUES (?, ?, ?, ?, ?)",
                    [("old", f"/shops/{i}", "{}", now, "{}") for i in range(5000)])
    con.close()
    code = main(["run", "--fixtures", "--config-dir", str(paths.config_dir),
                 "--data-dir", str(tmp_path / "data"), "--reports-dir", str(tmp_path / "reports")])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "complete" in out
