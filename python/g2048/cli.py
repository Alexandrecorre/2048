"""CLI : `python -m g2048 run experiments/xxx.toml`."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="g2048")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Lance une expérience à partir d'une config TOML")
    run_parser.add_argument("config", type=Path)

    args = parser.parse_args(argv)

    if args.command == "run":
        out_dir = run_experiment(args.config)
        print(f"Résultats écrits dans {out_dir}")


if __name__ == "__main__":
    main()
