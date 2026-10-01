from pathlib import Path

import pytest

from oppscan.config import load_etsy, load_scoring, load_seeds

REPO = Path(__file__).resolve().parents[1]


def test_load_seeds_normalises_and_dedupes(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("seeds:\n  - Budget  Spreadsheet\n  - budget spreadsheet\n  - '  '\n  - notion planner\n")
    seeds = load_seeds(path)
    assert seeds.terms == ("budget spreadsheet", "notion planner")
    assert len(seeds.file_hash) == 16


def test_load_seeds_hash_changes_with_content(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("seeds: [a]\n")
    first = load_seeds(path).file_hash
    path.write_text("seeds: [a, b]\n")
    assert load_seeds(path).file_hash != first


def test_load_seeds_rejects_empty(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("seeds: []\n")
    with pytest.raises(ValueError):
        load_seeds(path)


def test_repo_scoring_config_loads():
    cfg = load_scoring(REPO / "config" / "scoring.yaml")
    assert cfg.weights.demand == 0.35
    assert cfg.weights.crowding == 0.10
    assert cfg.demand_floor_pct == 0.25
    assert cfg.high_conf_min_reviews == 30
    assert cfg.high_conf_min_listings == 8
    assert cfg.top_n_per_niche == 20
    assert cfg.known_big_seeds == ("monthly budget spreadsheet",)


def test_repo_seeds_include_known_big_seeds():
    seeds = load_seeds(REPO / "config" / "seeds.yaml")
    cfg = load_scoring(REPO / "config" / "scoring.yaml")
    assert 50 <= len(seeds.terms) <= 150
    assert set(cfg.known_big_seeds) <= set(seeds.terms)


def test_load_etsy_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ETSY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ETSY_API_KEY"):
        load_etsy(REPO / "config" / "etsy.yaml")


def test_load_etsy_reads_env_key(monkeypatch):
    monkeypatch.setenv("ETSY_API_KEY", "abc")
    settings = load_etsy(REPO / "config" / "etsy.yaml")
    assert settings.api_key == "abc"
    assert settings.search_depth == 300
    assert settings.reviews_for_top == 20


def test_load_etsy_explicit_key_wins(monkeypatch):
    monkeypatch.setenv("ETSY_API_KEY", "env")
    assert load_etsy(REPO / "config" / "etsy.yaml", api_key="explicit").api_key == "explicit"
