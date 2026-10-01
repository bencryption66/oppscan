"""Group seeds into niches (one LLM call, cached per seed file) and assign listings to niches."""
from __future__ import annotations

import re

from oppscan.config import Seeds, normalise
from oppscan.llm import LLMClient

TITLES_PER_SEED = 10


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:48] or "niche"


def cluster(con, run_id: str, seeds: Seeds, llm: LLMClient) -> str:
    version = seeds.file_hash
    exists = con.execute("SELECT count(*) FROM seed_niche WHERE cluster_version = ?", [version]).fetchone()[0]
    if not exists:
        _cluster_seeds(con, run_id, seeds, llm, version)
    _assign_listings(con, run_id, version)
    return version


def _cluster_seeds(con, run_id: str, seeds: Seeds, llm: LLMClient, version: str) -> None:
    payload = {"seeds": [{"seed": s, "titles": _top_titles(con, run_id, s)} for s in seeds.terms]}
    result = llm.call("cluster", payload)
    known = set(seeds.terms)
    assigned: dict[str, str] = {}
    niches: dict[str, tuple[str, str]] = {}
    for niche in result["niches"]:
        members = [s for s in (normalise(x) for x in niche["seeds"]) if s in known and s not in assigned]
        if not members:
            continue
        niche_id = _unique(slugify(niche["name"]), niches)
        niches[niche_id] = (niche["name"], niche["description"])
        for seed in members:
            assigned[seed] = niche_id
    for seed in seeds.terms:
        if seed not in assigned:
            niche_id = _unique(slugify(seed), niches)
            niches[niche_id] = (seed, f"Unclustered seed '{seed}'")
            assigned[seed] = niche_id
    con.executemany("INSERT INTO niches VALUES (?, ?, ?, ?)",
                    [(nid, version, name, desc) for nid, (name, desc) in niches.items()])
    con.executemany("INSERT INTO seed_niche VALUES (?, ?, ?)",
                    [(version, seed, nid) for seed, nid in assigned.items()])


def _unique(base: str, taken: dict) -> str:
    niche_id, i = base, 2
    while niche_id in taken:
        niche_id = f"{base}-{i}"
        i += 1
    return niche_id


def _top_titles(con, run_id: str, seed: str) -> list[str]:
    rows = con.execute(
        "SELECT l.title FROM search_hits h JOIN listing_snapshots l USING (run_id, listing_id) "
        "WHERE h.run_id = ? AND h.seed = ? ORDER BY h.search_rank LIMIT ?",
        [run_id, seed, TITLES_PER_SEED],
    ).fetchall()
    return [r[0] for r in rows]


def _assign_listings(con, run_id: str, version: str) -> None:
    con.execute("DELETE FROM listing_niche WHERE run_id = ?", [run_id])
    con.execute(
        """
        INSERT INTO listing_niche
        SELECT run_id, listing_id, niche_id, ? FROM (
            SELECT h.run_id, h.listing_id, sn.niche_id,
                   ROW_NUMBER() OVER (PARTITION BY h.listing_id ORDER BY h.search_rank, h.seed) AS rn
            FROM search_hits h
            JOIN seed_niche sn ON sn.seed = h.seed AND sn.cluster_version = ?
            WHERE h.run_id = ?
        ) WHERE rn = 1
        """,
        [version, version, run_id],
    )
