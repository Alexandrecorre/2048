from pathlib import Path

from g2048 import run_experiment
from g2048.analysis import summarize, summarize_many

BASELINE_CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "baseline_random.toml"


def test_summarize_computes_expected_columns(tmp_path):
    out_dir = run_experiment(BASELINE_CONFIG, results_root=tmp_path)
    df = summarize(out_dir)

    assert df.height == 1
    for col in ["experiment", "num_games", "mean_score", "mean_survival_moves", "mean_ms_per_move", "success_rate_128"]:
        assert col in df.columns


def test_summarize_many_concatenates_rows(tmp_path):
    out_dir = run_experiment(BASELINE_CONFIG, results_root=tmp_path)
    df = summarize_many([out_dir, out_dir])
    assert df.height == 2
