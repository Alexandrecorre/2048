import json

import polars as pl

from g2048 import Env, replay_raw, run_batch_raw, run_experiment

BASELINE_CONFIG = "experiments/baseline_random.toml"


def test_run_batch_raw_returns_one_result_per_game():
    toml_text = """
    name = "t"
    agent = "random"
    num_games = 5
    seed = 1
    max_moves = 100
    """
    games = run_batch_raw(toml_text)
    assert len(games) == 5
    for g in games:
        assert set(g.keys()) == {"seed", "score", "max_tile", "num_moves", "duration_ms", "moves"}


def test_replay_reproduces_the_same_outcome():
    toml_text = """
    name = "t"
    agent = "random"
    num_games = 3
    seed = 10
    max_moves = 200
    """
    games = run_batch_raw(toml_text)
    for g in games:
        replayed = replay_raw(g["seed"], g["moves"])
        assert replayed["score"] == g["score"]
        assert replayed["max_tile"] == g["max_tile"]
        assert replayed["num_moves"] == g["num_moves"]


def test_env_gymnasium_like_interface():
    env = Env(1)
    obs = env.reset(1)
    assert len(obs) == 16
    legal = env.legal_moves()
    assert len(legal) > 0
    obs, reward, terminated, truncated, info = env.step(legal[0])
    assert len(obs) == 16
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "invalid_move" in info


def test_run_experiment_writes_a_self_contained_reproducible_results_dir(tmp_path):
    out_dir = run_experiment(_config_path(), results_root=tmp_path)

    assert (out_dir / "config.toml").exists()
    assert (out_dir / "results.parquet").exists()
    assert (out_dir / "replays.jsonl").exists()
    assert (out_dir / "metadata.json").exists()

    df = pl.read_parquet(out_dir / "results.parquet")
    replays = [json.loads(line) for line in (out_dir / "replays.jsonl").read_text().splitlines()]
    assert len(df) == len(replays)

    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert "commit" in metadata and "machine" in metadata

    # Critère de sortie du chapitre 3 : l'expérience est rejouable à
    # l'identique à partir de son seul dossier de résultats.
    row = df.row(0, named=True)
    replay_entry = next(r for r in replays if r["seed"] == row["seed"])
    replayed = replay_raw(replay_entry["seed"], replay_entry["moves"])
    assert replayed["score"] == row["score"]
    assert replayed["max_tile"] == row["max_tile"]
    assert replayed["num_moves"] == row["num_moves"]


def _config_path():
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / BASELINE_CONFIG
