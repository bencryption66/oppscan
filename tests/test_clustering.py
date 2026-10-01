from helpers import add_listing

from oppscan.clustering import cluster, slugify
from oppscan.config import Seeds
from oppscan.fake_llm import FakeLLM

SEEDS = Seeds(terms=("budget spreadsheet", "budget planner", "notion planner"), file_hash="v1")


class StubLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def call(self, task, payload):
        self.calls.append((task, payload))
        return self.response


def seed_data(con):
    add_listing(con, "r1", 1, seed="budget spreadsheet", rank=1, title="Budget Sheet")
    add_listing(con, "r1", 2, seed="budget spreadsheet", rank=2)
    add_listing(con, "r1", 2, seed="budget planner", rank=1)
    add_listing(con, "r1", 4, seed="notion planner", rank=1)
    add_listing(con, "r1", 5, seed="budget planner", rank=7)
    add_listing(con, "r1", 5, seed="notion planner", rank=2)


def niche_of(con, listing_id):
    return con.execute("SELECT niche_id FROM listing_niche WHERE run_id = 'r1' AND listing_id = ?",
                       [listing_id]).fetchone()[0]


def test_slugify():
    assert slugify("ADHD-friendly Notion Planner!") == "adhd-friendly-notion-planner"
    assert slugify("!!!") == "niche"


def test_cluster_groups_seeds_and_assigns_listings(con):
    seed_data(con)
    llm = FakeLLM()
    assert cluster(con, "r1", SEEDS, llm) == "v1"
    seed_map = dict(con.execute("SELECT seed, niche_id FROM seed_niche WHERE cluster_version = 'v1'").fetchall())
    assert seed_map == {"budget spreadsheet": "budget-templates", "budget planner": "budget-templates",
                        "notion planner": "notion-templates"}
    assert niche_of(con, 1) == "budget-templates"
    assert niche_of(con, 4) == "notion-templates"
    assert niche_of(con, 5) == "notion-templates"  # best rank (2) is under a notion seed
    payload = llm.calls[0][1]
    assert payload["seeds"][0] == {"seed": "budget spreadsheet", "titles": ["Budget Sheet", "Listing 2"]}


def test_cluster_is_cached_per_seed_file(con):
    seed_data(con)
    llm = FakeLLM()
    cluster(con, "r1", SEEDS, llm)
    cluster(con, "r1", SEEDS, llm)
    assert len(llm.calls) == 1


def test_unassigned_seeds_get_their_own_niche_and_unknown_seeds_are_ignored(con):
    seed_data(con)
    llm = StubLLM({"niches": [{"name": "Budget", "description": "d",
                               "seeds": ["Budget  Spreadsheet", "not a seed"]}]})
    cluster(con, "r1", SEEDS, llm)
    seed_map = dict(con.execute("SELECT seed, niche_id FROM seed_niche").fetchall())
    assert seed_map == {"budget spreadsheet": "budget", "budget planner": "budget-planner",
                        "notion planner": "notion-planner"}


def test_duplicate_niche_names_get_unique_ids(con):
    seed_data(con)
    llm = StubLLM({"niches": [
        {"name": "Budget", "description": "a", "seeds": ["budget spreadsheet"]},
        {"name": "Budget", "description": "b", "seeds": ["budget planner"]},
        {"name": "Notion", "description": "c", "seeds": ["notion planner"]},
    ]})
    cluster(con, "r1", SEEDS, llm)
    ids = sorted(r[0] for r in con.execute("SELECT niche_id FROM niches").fetchall())
    assert ids == ["budget", "budget-2", "notion"]
