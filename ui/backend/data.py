"""Lecture des dossiers results/ (chapitre 3) et des artefacts des
chapitres 6/7/8, avec DuckDB pour les requêtes sur les fichiers Parquet."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"

MILESTONES = [128, 256, 512, 1024, 2048, 4096]


def _is_experiment_dir(path: Path) -> bool:
    return (path / "results.parquet").exists() and (path / "config.toml").exists()


def list_experiments() -> list[dict[str, Any]]:
    if not RESULTS_ROOT.exists():
        return []
    experiments = []
    for path in sorted(RESULTS_ROOT.iterdir()):
        if not path.is_dir() or not _is_experiment_dir(path):
            continue
        config = tomllib.loads((path / "config.toml").read_text(encoding="utf-8"))
        metadata = {}
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        experiments.append(
            {
                "dir": path.name,
                "name": config.get("name", path.name),
                "agent": config.get("agent"),
                "num_games": config.get("num_games"),
                "seed": config.get("seed"),
                "timestamp": metadata.get("timestamp"),
                "commit": metadata.get("commit"),
            }
        )
    return experiments


def _experiment_path(name: str) -> Path:
    path = RESULTS_ROOT / name
    if not _is_experiment_dir(path):
        raise FileNotFoundError(f"expérience inconnue: {name}")
    return path


def experiment_summary(name: str) -> dict[str, Any]:
    path = _experiment_path(name)
    parquet_path = str(path / "results.parquet")

    row = duckdb.sql(
        f"""
        select
            count(*) as num_games,
            avg(score) as mean_score,
            avg(num_moves) as mean_survival_moves,
            avg(duration_ms / greatest(num_moves, 1)) as mean_ms_per_move,
            max(max_tile) as max_tile_reached
        from read_parquet('{parquet_path}')
        """
    ).fetchone()

    columns = ["num_games", "mean_score", "mean_survival_moves", "mean_ms_per_move", "max_tile_reached"]
    summary = dict(zip(columns, row))

    for milestone in MILESTONES:
        rate = duckdb.sql(
            f"""
            select avg(case when max_tile >= {milestone} then 1.0 else 0.0 end)
            from read_parquet('{parquet_path}')
            """
        ).fetchone()[0]
        summary[f"success_rate_{milestone}"] = rate

    return summary


def experiment_scores(name: str) -> list[dict[str, Any]]:
    path = _experiment_path(name)
    parquet_path = str(path / "results.parquet")
    rows = duckdb.sql(
        f"select seed, score, max_tile, num_moves, duration_ms from read_parquet('{parquet_path}') order by seed"
    ).fetchall()
    return [
        {"seed": r[0], "score": r[1], "max_tile": r[2], "num_moves": r[3], "duration_ms": r[4]} for r in rows
    ]


def experiment_replays(name: str) -> list[dict[str, Any]]:
    path = _experiment_path(name)
    replays_path = path / "replays.jsonl"
    if not replays_path.exists():
        return []
    out = []
    for line in replays_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        out.append({"seed": entry["seed"], "num_moves": len(entry["moves"])})
    return out


def experiment_replay_moves(name: str, seed: int) -> list[int]:
    path = _experiment_path(name)
    replays_path = path / "replays.jsonl"
    for line in replays_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["seed"] == seed:
            return entry["moves"]
    raise FileNotFoundError(f"aucun replay pour la seed {seed} dans {name}")


def _read_parquet_if_exists(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    rows = duckdb.sql(f"select * from read_parquet('{path}')").fetchall()
    columns = [c[0] for c in duckdb.sql(f"describe select * from read_parquet('{path}')").fetchall()]
    return [dict(zip(columns, r)) for r in rows]


def ablation_matrix() -> list[dict[str, Any]] | None:
    return _read_parquet_if_exists(RESULTS_ROOT / "ablation_chapter6" / "matrix.parquet")


def ablation_learning_curves() -> list[dict[str, Any]] | None:
    return _read_parquet_if_exists(RESULTS_ROOT / "ablation_chapter6" / "learning_curves.parquet")


def search_experiment_results() -> list[dict[str, Any]] | None:
    return _read_parquet_if_exists(RESULTS_ROOT / "search_experiment" / "depth_vs_quality.parquet")


def dqn_history(representation: str) -> list[dict[str, Any]] | None:
    return _read_parquet_if_exists(
        RESULTS_ROOT / f"dqn_{representation}" / f"dqn_{representation}_history.parquet"
    )
