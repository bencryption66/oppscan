import shutil
from datetime import datetime
from pathlib import Path

import duckdb
import httpx

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
    (config / "seeds.yaml").write_text(SEEDS)
    (config / "etsy.yaml").write_text(
        f"qps: 5\ndaily_quota: {quota}\nsearch_depth: 200\nreviews_for_top: 5\nmax_review_pages: 2\n")
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
    con.close()
    assert with_delta > 0


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


def test_cli_fixtures(tmp_path, capsys):
    paths = setup_config(tmp_path)
    code = main(["run", "--fixtures", "--config-dir", str(paths.config_dir),
                 "--data-dir", str(tmp_path / "data"), "--reports-dir", str(tmp_path / "reports")])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "complete" in out
    assert list((tmp_path / "reports" / "fixtures").glob("*.html"))
