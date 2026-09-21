"""Protocole d'ablation (chapitre 6) : compare les pistes du chapitre 5 et
les méthodes d'apprentissage du chapitre 6 (évolution, TD sur features,
TD sur témoin n-tuple) en tenant la profondeur de recherche fixée à 1.

Produit une matrice "pistes × méthode d'apprentissage × performance" avec
intervalles de confiance, et les courbes d'apprentissage de chaque run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ._g2048 import (
    scores_of_feature_weights,
    scores_of_ntuple_weights,
    train_evolution,
    train_td_features,
    train_td_ntuple,
)
from .features import ALL_FEATURES


def ablation_feature_sets() -> dict[str, list[str]]:
    """Protocole d'ablation : toutes les pistes, toutes sauf une, chacune
    seule, puis aucune."""
    sets: dict[str, list[str]] = {"all": list(ALL_FEATURES)}
    for f in ALL_FEATURES:
        sets[f"all_but_{f}"] = [g for g in ALL_FEATURES if g != f]
    for f in ALL_FEATURES:
        sets[f"only_{f}"] = [f]
    sets["none"] = []
    return sets


def _confidence_interval_95(scores: list[int]) -> tuple[float, float, float]:
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
    se = math.sqrt(variance / n)
    half_width = 1.96 * se
    return mean, mean - half_width, mean + half_width


@dataclass
class RunResult:
    feature_set: str
    method: str
    mean_score: float
    ci_low: float
    ci_high: float
    history: list[dict] = field(default_factory=list)


def run_evolution(
    feature_set: str,
    enabled: list[str],
    *,
    seed: int = 0,
    population_size: int = 24,
    generations: int = 15,
    games_per_eval: int = 6,
    max_moves: int = 1500,
) -> RunResult:
    if not enabled:
        # Rien à optimiser : évaluation constante (=0), la politique se
        # réduit au score immédiat des fusions (équivalent à l'agent
        # glouton du chapitre 4) — c'est la ligne de base "aucune piste".
        scores = scores_of_feature_weights([], [], num_games=100, max_moves=3000, seed_offset=10_000)
        mean, lo, hi = _confidence_interval_95(scores)
        return RunResult(feature_set, "evolution", mean, lo, hi, [])

    result = train_evolution(
        enabled,
        population_size=population_size,
        generations=generations,
        games_per_eval=games_per_eval,
        max_moves=max_moves,
        mutation_std=0.3,
        elite_fraction=0.25,
        seed=seed,
    )
    scores = scores_of_feature_weights(
        enabled, result["best_weights"], num_games=100, max_moves=3000, seed_offset=10_000
    )
    mean, lo, hi = _confidence_interval_95(scores)
    return RunResult(feature_set, "evolution", mean, lo, hi, result["history"])


def run_td_features(
    feature_set: str,
    enabled: list[str],
    *,
    seed: int = 0,
    games: int = 2000,
    max_moves: int = 3000,
    alpha: float = 0.001,
    eval_every: int = 200,
    eval_games: int = 20,
) -> RunResult:
    if not enabled:
        # Idem : rien à apprendre sans piste, ligne de base "aucune piste".
        scores = scores_of_feature_weights([], [], num_games=100, max_moves=3000, seed_offset=10_000)
        mean, lo, hi = _confidence_interval_95(scores)
        return RunResult(feature_set, "td", mean, lo, hi, [])

    result = train_td_features(
        enabled,
        games=games,
        max_moves=max_moves,
        alpha=alpha,
        seed=seed,
        eval_every=eval_every,
        eval_games=eval_games,
    )
    scores = scores_of_feature_weights(
        enabled, result["weights"], num_games=100, max_moves=3000, seed_offset=10_000
    )
    mean, lo, hi = _confidence_interval_95(scores)
    return RunResult(feature_set, "td", mean, lo, hi, result["history"])


def run_td_ntuple(*, seed: int = 0) -> RunResult:
    """6c. Témoin sans pistes : un seul run, indépendant du protocole
    d'ablation puisqu'il n'y a pas de features à retirer."""
    result = train_td_ntuple(
        games=2000, max_moves=3000, alpha=0.01, seed=seed, eval_every=200, eval_games=20
    )
    scores = scores_of_ntuple_weights(result["weights"], num_games=100, max_moves=3000, seed_offset=10_000)
    mean, lo, hi = _confidence_interval_95(scores)
    return RunResult("none (grille brute)", "td_ntuple", mean, lo, hi, result["history"])


def run_full_ablation(*, seed: int = 0) -> list[RunResult]:
    results = []
    for feature_set, enabled in ablation_feature_sets().items():
        results.append(run_evolution(feature_set, enabled, seed=seed))
        results.append(run_td_features(feature_set, enabled, seed=seed))
    results.append(run_td_ntuple(seed=seed))
    return results


def results_to_matrix(results: list[RunResult]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "feature_set": r.feature_set,
                "method": r.method,
                "mean_score": r.mean_score,
                "ci_low_95": r.ci_low,
                "ci_high_95": r.ci_high,
            }
            for r in results
        ]
    )


def save_learning_curves(results: list[RunResult], out_path: Path) -> None:
    rows = [
        {"feature_set": r.feature_set, "method": r.method, "steps": p["steps"], "mean_score": p["mean_score"]}
        for r in results
        for p in r.history
    ]
    pl.DataFrame(rows).write_parquet(out_path)
