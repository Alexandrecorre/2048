from collections import deque
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from g2048.deep_rl import DqnConfig, evaluate_dqn, load_experience_into_buffer, train_dqn  # noqa: E402
from g2048.runner import run_experiment  # noqa: E402

BASELINE_CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "baseline_random.toml"


@pytest.mark.parametrize("representation", ["onehot", "features"])
def test_train_dqn_runs_end_to_end_at_tiny_scale(representation):
    config = DqnConfig(
        representation=representation,
        episodes=3,
        max_moves=30,
        batch_size=4,
        buffer_size=100,
        eval_every=3,
        eval_games=2,
        target_update_every=3,
    )
    net, history = train_dqn(config)
    assert len(history) == 1
    assert history[0]["episode"] == 3
    assert isinstance(history[0]["mean_score"], float)


def test_evaluate_dqn_returns_a_float_mean_score():
    config = DqnConfig(representation="onehot", max_moves=20)
    net, _ = train_dqn(
        DqnConfig(representation="onehot", episodes=2, max_moves=20, batch_size=4, buffer_size=50, eval_every=2, eval_games=1)
    )
    mean_score = evaluate_dqn(net, config, num_games=2)
    assert mean_score >= 0


def test_load_experience_into_buffer_fills_transitions_from_a_dataset(tmp_path):
    out_dir = run_experiment(BASELINE_CONFIG, results_root=tmp_path)
    dataset_path = tmp_path / "experience" / "dataset.parquet"
    assert dataset_path.exists()

    config = DqnConfig(representation="onehot")
    buffer = deque(maxlen=10_000)
    added = load_experience_into_buffer(buffer, config, dataset_path=dataset_path)

    assert added > 0
    assert len(buffer) == added
    sample = buffer[0]
    assert sample.state.shape[0] == 256


def test_train_dqn_warm_start_does_not_crash(tmp_path, monkeypatch):
    # train_dqn(warm_start=True) lit le chemin par défaut ("results/experience/..."),
    # relatif au cwd : on écrit donc l'expérience sous tmp_path/results puis on
    # se place dans tmp_path pour que les chemins relatifs coïncident.
    run_experiment(BASELINE_CONFIG, results_root=tmp_path / "results")
    monkeypatch.chdir(tmp_path)

    config = DqnConfig(
        representation="onehot",
        episodes=2,
        max_moves=20,
        batch_size=4,
        buffer_size=100,
        eval_every=2,
        eval_games=1,
        warm_start_from_experience=True,
    )
    net, history = train_dqn(config)
    assert len(history) == 1
