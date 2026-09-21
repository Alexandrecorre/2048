"""Jeu de données d'expérience persistant : matérialise les transitions
(état, action, récompense) des meilleures parties de chaque expérience,
taguées par l'agent qui les a produites.

Ne duplique pas le stockage existant : les replays (seed + coups,
chapitre 3) et `replay_trajectory` (chapitre 9, viewer) suffisent déjà à
reconstruire l'état complet à chaque pas — ce module se contente de
sélectionner les meilleures parties de chaque expérience et de les
matérialiser dans un format exploitable pour l'apprentissage (ex: buffer
de replay du DQN, chapitre 8), indépendamment de l'agent qui les a
produites : même un agent faible a de bonnes parties qui portent de
l'information sur ce qui fonctionne.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import polars as pl

from ._g2048 import replay_trajectory

TOP_K_GAMES = 10
EXPERIENCE_PATH = Path("results") / "experience" / "dataset.parquet"


def _read_config_agent(exp_dir: Path) -> str:
    config = tomllib.loads((exp_dir / "config.toml").read_text(encoding="utf-8"))
    return config.get("agent", "unknown")


def extract_experience(exp_dir: Path, top_k: int = TOP_K_GAMES) -> pl.DataFrame:
    """Matérialise les transitions des `top_k` meilleures parties (par
    score final) d'une expérience, taguées par l'agent qui les a jouées."""
    results_path = exp_dir / "results.parquet"
    replays_path = exp_dir / "replays.jsonl"
    if not results_path.exists() or not replays_path.exists():
        return pl.DataFrame([])

    agent = _read_config_agent(exp_dir)
    best = pl.read_parquet(results_path).sort("score", descending=True).head(top_k)
    best_seeds = set(best["seed"].to_list())

    moves_by_seed: dict[int, list[int]] = {}
    for line in replays_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["seed"] in best_seeds:
            moves_by_seed[entry["seed"]] = entry["moves"]

    rows = []
    for seed, moves in moves_by_seed.items():
        steps = replay_trajectory(seed, moves)
        num_steps = len(steps)
        for step_index, step in enumerate(steps):
            rows.append(
                {
                    "source_agent": agent,
                    "source_experiment": exp_dir.name,
                    "seed": seed,
                    "step_index": step_index,
                    "board_before": step["board_before"],
                    "chosen_direction": step["chosen_direction"],
                    "gained": step["gained"],
                    "score_after": step["score_after"],
                    "legal": [opt["legal"] for opt in step["move_options"]],
                    "terminal": step_index == num_steps - 1,
                }
            )
    return pl.DataFrame(rows)


def append_experience(
    exp_dir: Path, top_k: int = TOP_K_GAMES, dataset_path: Path = EXPERIENCE_PATH
) -> int:
    """Ajoute les transitions des meilleures parties de `exp_dir` au jeu
    de données persistant. Retourne le nombre de transitions ajoutées.
    Idempotent : relancer sur la même expérience remplace ses lignes
    plutôt que de les dupliquer."""
    new_rows = extract_experience(exp_dir, top_k=top_k)
    if new_rows.height == 0:
        return 0

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists():
        existing = pl.read_parquet(dataset_path)
        existing = existing.filter(pl.col("source_experiment") != exp_dir.name)
        combined = pl.concat([existing, new_rows])
    else:
        combined = new_rows
    combined.write_parquet(dataset_path)
    return new_rows.height


def load_experience(dataset_path: Path = EXPERIENCE_PATH) -> pl.DataFrame | None:
    if not dataset_path.exists():
        return None
    return pl.read_parquet(dataset_path)
