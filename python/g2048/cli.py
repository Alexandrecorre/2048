"""CLI : `python -m g2048 run experiments/xxx.toml`."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from .analysis import summarize_many
from .runner import run_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="g2048")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Lance une expérience à partir d'une config TOML")
    run_parser.add_argument("config", type=Path)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Résume un ou plusieurs dossiers de résultats (results/<nom>_<date>/)"
    )
    analyze_parser.add_argument("results_dirs", type=Path, nargs="+")

    args = parser.parse_args(argv)

    if args.command == "run":
        out_dir = run_experiment(args.config)
        print(f"Résultats écrits dans {out_dir}")
    elif args.command == "analyze":
        df = summarize_many(args.results_dirs)
        with pl.Config(tbl_cols=-1, tbl_width_chars=240):
            print(df)


if __name__ == "__main__":
    main()
