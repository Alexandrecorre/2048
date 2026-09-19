import pytest

torch = pytest.importorskip("torch")

from g2048.deep_rl import DqnConfig, evaluate_dqn, train_dqn  # noqa: E402


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
