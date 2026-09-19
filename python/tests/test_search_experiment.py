from g2048._g2048 import run_expectimax
from g2048.search_experiment import run_depth_sweep


def test_run_expectimax_binding_returns_one_result_per_game():
    results = run_expectimax(["empty_cells"], [1.0], 2, 3, 50, 0)
    assert len(results) == 3
    for r in results:
        assert set(r.keys()) == {"seed", "score", "max_tile", "num_moves", "duration_ms"}


def test_run_depth_sweep_at_tiny_scale(monkeypatch):
    import g2048.search_experiment as se

    monkeypatch.setattr(se, "GAMES_PER_DEPTH", {1: 2, 2: 2})
    monkeypatch.setattr(se, "MAX_MOVES", 50)

    df = run_depth_sweep("test", ["empty_cells"], [1.0])
    assert df.height == 2
    assert set(df.columns) == {"evaluation", "depth", "num_games", "mean_score", "mean_ms_per_move"}
