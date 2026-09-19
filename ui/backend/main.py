"""API FastAPI (chapitre 9) : sert les données des dossiers results/ au
dashboard et au viewer React. Lecture des Parquet via DuckDB, replays
rejoués à la demande via le moteur Rust (g2048.replay_trajectory)."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from g2048 import replay_trajectory  # noqa: E402

from . import data  # noqa: E402

app = FastAPI(title="g2048 dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/experiments")
def get_experiments():
    return data.list_experiments()


@app.get("/api/experiments/{name}/summary")
def get_experiment_summary(name: str):
    try:
        return data.experiment_summary(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/experiments/{name}/scores")
def get_experiment_scores(name: str):
    try:
        return data.experiment_scores(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/experiments/{name}/replays")
def get_experiment_replays(name: str):
    try:
        return data.experiment_replays(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/experiments/{name}/replay/{seed}")
def get_experiment_replay(name: str, seed: int):
    try:
        moves = data.experiment_replay_moves(name, seed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return replay_trajectory(seed, moves)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/ablation/matrix")
def get_ablation_matrix():
    result = data.ablation_matrix()
    if result is None:
        raise HTTPException(status_code=404, detail="pas de résultats d'ablation (chapitre 6)")
    return result


@app.get("/api/ablation/learning-curves")
def get_ablation_learning_curves():
    result = data.ablation_learning_curves()
    if result is None:
        raise HTTPException(status_code=404, detail="pas de résultats d'ablation (chapitre 6)")
    return result


@app.get("/api/search-experiment")
def get_search_experiment():
    result = data.search_experiment_results()
    if result is None:
        raise HTTPException(status_code=404, detail="pas de résultats de recherche (chapitre 7)")
    return result


@app.get("/api/dqn/{representation}")
def get_dqn_history(representation: str):
    result = data.dqn_history(representation)
    if result is None:
        raise HTTPException(status_code=404, detail=f"pas d'historique DQN pour '{representation}'")
    return result
