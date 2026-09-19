from g2048._g2048 import scores_of_feature_weights, train_evolution, train_td_features, train_td_ntuple
from g2048.ablation import (
    _confidence_interval_95,
    ablation_feature_sets,
    results_to_matrix,
    run_evolution,
    run_td_features,
    run_td_ntuple,
)


def test_ablation_feature_sets_covers_all_none_and_each_alone_and_all_but_one():
    sets = ablation_feature_sets()
    assert len(sets["all"]) == 6
    assert sets["none"] == []
    assert len(sets["only_empty_cells"]) == 1
    assert len(sets["all_but_empty_cells"]) == 5


def test_train_evolution_binding_returns_weights_and_history():
    result = train_evolution(
        ["empty_cells", "max_tile_in_corner"],
        population_size=4,
        generations=2,
        games_per_eval=2,
        max_moves=100,
        mutation_std=0.3,
        elite_fraction=0.5,
        seed=1,
    )
    assert len(result["best_weights"]) == 2
    assert len(result["history"]) == 2


def test_train_td_features_binding_returns_weights_and_history():
    result = train_td_features(
        ["empty_cells", "max_tile_in_corner"],
        games=6,
        max_moves=100,
        alpha=0.01,
        seed=1,
        eval_every=3,
        eval_games=2,
    )
    assert len(result["weights"]) == 2
    assert len(result["history"]) == 2


def test_train_td_ntuple_binding_returns_256_weights():
    result = train_td_ntuple(games=4, max_moves=100, alpha=0.01, seed=1, eval_every=2, eval_games=2)
    assert len(result["weights"]) == 256


def test_scores_of_feature_weights_returns_one_score_per_game():
    scores = scores_of_feature_weights(["empty_cells"], [1.0], num_games=5, max_moves=100, seed_offset=0)
    assert len(scores) == 5


def test_confidence_interval_brackets_the_mean():
    mean, lo, hi = _confidence_interval_95([10, 20, 30, 40, 50])
    assert lo <= mean <= hi


def test_run_evolution_and_td_at_tiny_scale_produce_a_row(monkeypatch):
    import g2048.ablation as ablation

    monkeypatch.setattr(
        ablation,
        "train_evolution",
        lambda enabled, **kw: {"best_weights": [0.1] * len(enabled), "history": [{"steps": 0, "mean_score": 1.0}]},
    )
    monkeypatch.setattr(
        ablation,
        "train_td_features",
        lambda enabled, **kw: {"weights": [0.1] * len(enabled), "history": [{"steps": 0, "mean_score": 1.0}]},
    )
    monkeypatch.setattr(
        ablation,
        "scores_of_feature_weights",
        lambda enabled, weights, **kw: [100, 200, 150],
    )

    r1 = run_evolution("only_empty_cells", ["empty_cells"])
    r2 = run_td_features("only_empty_cells", ["empty_cells"])
    matrix = results_to_matrix([r1, r2])
    assert matrix.height == 2
    assert set(matrix.columns) == {"feature_set", "method", "mean_score", "ci_low_95", "ci_high_95"}


def test_run_td_ntuple_at_tiny_scale(monkeypatch):
    import g2048.ablation as ablation

    monkeypatch.setattr(
        ablation, "train_td_ntuple", lambda **kw: {"weights": [0.0] * 256, "history": [{"steps": 0, "mean_score": 1.0}]}
    )
    monkeypatch.setattr(ablation, "scores_of_ntuple_weights", lambda weights, **kw: [100, 200])

    result = run_td_ntuple()
    assert result.method == "td_ntuple"
