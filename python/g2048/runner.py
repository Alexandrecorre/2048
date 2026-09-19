"""Harnais d'expérimentation (chapitre 3) : charge une config TOML, lance le
lot de parties via le moteur Rust, et écrit un dossier de résultats
autosuffisant (config copiée, Parquet, replays JSONL, métadonnées)."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from ._g2048 import run_batch as _run_batch


@dataclass
class ExperimentConfig:
    name: str
    raw_toml: str


def load_config(path: Path) -> ExperimentConfig:
    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    if "name" not in data:
        raise ValueError(f"la config {path} doit définir un champ 'name'")
    return ExperimentConfig(name=data["name"], raw_toml=text)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def run_experiment(config_path: Path, results_root: Path = Path("results")) -> Path:
    """Exécute l'expérience décrite par `config_path` et écrit
    `results/<name>_<date>/`. Retourne le chemin du dossier créé."""
    config = load_config(config_path)

    start = time.time()
    games = _run_batch(config.raw_toml)
    duration_s = time.time() - start

    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_root / f"{config.name}_{date_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, out_dir / "config.toml")

    rows = [
        {
            "seed": g["seed"],
            "score": g["score"],
            "max_tile": g["max_tile"],
            "num_moves": g["num_moves"],
            "duration_ms": g["duration_ms"],
        }
        for g in games
    ]
    pl.DataFrame(rows).write_parquet(out_dir / "results.parquet")

    with (out_dir / "replays.jsonl").open("w", encoding="utf-8") as f:
        for g in games:
            f.write(json.dumps({"seed": g["seed"], "moves": g["moves"]}) + "\n")

    metadata = {
        "commit": _git_commit(),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "num_games": len(games),
        "duration_s": duration_s,
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return out_dir
