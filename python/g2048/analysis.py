"""Script d'analyse minimal (chapitre 4) : calcule les métriques du
chapitre 0 (taux de réussite par palier, survie, score, temps par coup) à
partir d'un ou plusieurs dossiers de résultats."""

from __future__ import annotations

from pathlib import Path

import polars as pl

MILESTONES = [128, 256, 512, 1024, 2048, 4096]


def summarize(results_dir: Path) -> pl.DataFrame:
    df = pl.read_parquet(results_dir / "results.parquet")

    metrics = df.select(
        pl.col("score").mean().alias("mean_score"),
        pl.col("num_moves").mean().alias("mean_survival_moves"),
        (
            pl.col("duration_ms")
            / pl.when(pl.col("num_moves") > 0).then(pl.col("num_moves")).otherwise(1)
        )
        .mean()
        .alias("mean_ms_per_move"),
        pl.col("max_tile").max().alias("max_tile_reached"),
    ).row(0, named=True)

    row = {"experiment": results_dir.name, "num_games": len(df), **metrics}
    for milestone in MILESTONES:
        row[f"success_rate_{milestone}"] = float((df["max_tile"] >= milestone).mean())

    return pl.DataFrame([row])


def summarize_many(results_dirs: list[Path]) -> pl.DataFrame:
    return pl.concat([summarize(d) for d in results_dirs])
