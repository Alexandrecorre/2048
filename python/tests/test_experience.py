from pathlib import Path

from g2048.experience import append_experience, extract_experience, load_experience
from g2048.runner import run_experiment

BASELINE_CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "baseline_random.toml"


def test_extract_experience_keeps_only_top_k_games_and_tags_the_agent(tmp_path):
    out_dir = run_experiment(BASELINE_CONFIG, results_root=tmp_path)
    df = extract_experience(out_dir, top_k=3)

    assert df.height > 0
    assert set(df["source_agent"].unique().to_list()) == {"random"}
    assert df["seed"].n_unique() == 3
    # Une seule transition marquée terminale par partie.
    assert df.filter(df["terminal"]).height == 3


def test_append_experience_builds_and_updates_the_dataset(tmp_path):
    dataset_path = tmp_path / "experience" / "dataset.parquet"
    out_dir = run_experiment(BASELINE_CONFIG, results_root=tmp_path / "results")

    added = append_experience(out_dir, top_k=2, dataset_path=dataset_path)
    assert added > 0

    df = load_experience(dataset_path)
    assert df is not None
    assert df.height == added

    # Rejouer la même expérience remplace ses lignes, ne les duplique pas.
    added_again = append_experience(out_dir, top_k=2, dataset_path=dataset_path)
    df_again = load_experience(dataset_path)
    assert df_again.height == added_again == added


def test_run_experiment_auto_populates_the_experience_dataset(tmp_path):
    out_dir = run_experiment(BASELINE_CONFIG, results_root=tmp_path)
    dataset_path = tmp_path / "experience" / "dataset.parquet"
    assert dataset_path.exists()

    df = load_experience(dataset_path)
    assert df is not None
    assert (df["source_experiment"] == out_dir.name).all()
