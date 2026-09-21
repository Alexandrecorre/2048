"""Chapitre 7 — expérience clé : profondeur de recherche × qualité de
l'évaluation. À partir de quelle profondeur une évaluation pauvre
rattrape-t-elle (en score) une évaluation riche à faible profondeur ?

Compare deux évaluateurs (mêmes features, poids différents) à travers
plusieurs profondeurs d'expectimax, et mesure score et temps par coup.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ._g2048 import run_expectimax, train_td_features
from .features import DEFAULT_FEATURES as ALL_FEATURES

# "Pauvre" : une seule piste faible (chapitre 6 : proche de la ligne de
# base "aucune piste"). "Riche" : poids appris par TD sur les pistes par
# défaut (chapitre 6 : snake_weighted exclue, voir features.py).
POOR_FEATURES = ["empty_cells"]
POOR_WEIGHTS = [1.0]

# Moins de parties aux profondeurs élevées pour rester dans un temps
# raisonnable (le facteur de branchement de l'expectimax croît vite).
GAMES_PER_DEPTH = {1: 20, 2: 20, 3: 15, 4: 6, 5: 3}
MAX_MOVES = 800


def train_rich_weights(*, seed: int = 0) -> list[float]:
    result = train_td_features(
        ALL_FEATURES, games=2000, max_moves=3000, alpha=0.001, seed=seed, eval_every=2000, eval_games=1
    )
    return result["weights"]


def run_depth_sweep(
    label: str, enabled: list[str], weights: list[float], *, seed_offset: int = 20_000
) -> pl.DataFrame:
    rows = []
    for depth, num_games in GAMES_PER_DEPTH.items():
        results = run_expectimax(enabled, weights, depth, num_games, MAX_MOVES, seed_offset)
        mean_score = sum(r["score"] for r in results) / len(results)
        mean_ms_per_move = sum(r["duration_ms"] / max(r["num_moves"], 1) for r in results) / len(results)
        rows.append(
            {
                "evaluation": label,
                "depth": depth,
                "num_games": num_games,
                "mean_score": mean_score,
                "mean_ms_per_move": mean_ms_per_move,
            }
        )
    return pl.DataFrame(rows)


def run_experiment(*, seed: int = 0) -> pl.DataFrame:
    rich_weights = train_rich_weights(seed=seed)
    poor_df = run_depth_sweep("pauvre (empty_cells seule)", POOR_FEATURES, POOR_WEIGHTS)
    rich_df = run_depth_sweep("riche (TD, 6 pistes)", ALL_FEATURES, rich_weights)
    return pl.concat([poor_df, rich_df])


def save(df: pl.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "depth_vs_quality.parquet")
