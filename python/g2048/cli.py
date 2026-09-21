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

    ablation_parser = subparsers.add_parser(
        "ablation", help="Protocole d'ablation du chapitre 6 (pistes × méthode d'apprentissage)"
    )
    ablation_parser.add_argument("out_dir", type=Path)
    ablation_parser.add_argument("--seed", type=int, default=0)

    search_parser = subparsers.add_parser(
        "search-experiment",
        help="Chapitre 7 : profondeur d'expectimax × qualité de l'évaluation",
    )
    search_parser.add_argument("out_dir", type=Path)
    search_parser.add_argument("--seed", type=int, default=0)

    dqn_parser = subparsers.add_parser(
        "train-dqn", help="Chapitre 8 (optionnel) : entraîne un DQN (grille brute ou features)"
    )
    dqn_parser.add_argument("out_dir", type=Path)
    dqn_parser.add_argument("--representation", choices=["onehot", "features"], default="onehot")
    dqn_parser.add_argument("--episodes", type=int, default=500)
    dqn_parser.add_argument("--max-moves", type=int, default=1000)
    dqn_parser.add_argument("--seed", type=int, default=0)
    dqn_parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Précharge le buffer avec le jeu de données d'expérience persistant",
    )

    experience_parser = subparsers.add_parser(
        "build-experience",
        help="Reconstruit le jeu de données d'expérience à partir de results/*/ existants",
    )
    experience_parser.add_argument("results_root", type=Path, nargs="?", default=Path("results"))

    args = parser.parse_args(argv)

    if args.command == "run":
        out_dir = run_experiment(args.config)
        print(f"Résultats écrits dans {out_dir}")
    elif args.command == "analyze":
        df = summarize_many(args.results_dirs)
        with pl.Config(tbl_cols=-1, tbl_width_chars=240):
            print(df)
    elif args.command == "ablation":
        from .ablation import results_to_matrix, run_full_ablation, save_learning_curves

        args.out_dir.mkdir(parents=True, exist_ok=True)
        results = run_full_ablation(seed=args.seed)
        matrix = results_to_matrix(results)
        matrix.write_parquet(args.out_dir / "matrix.parquet")
        save_learning_curves(results, args.out_dir / "learning_curves.parquet")
        with pl.Config(tbl_cols=-1, tbl_width_chars=240, tbl_rows=-1):
            print(matrix.sort("mean_score", descending=True))
        print(f"Matrice et courbes écrites dans {args.out_dir}")
    elif args.command == "search-experiment":
        from .search_experiment import run_experiment, save

        df = run_experiment(seed=args.seed)
        save(df, args.out_dir)
        with pl.Config(tbl_cols=-1, tbl_width_chars=240, tbl_rows=-1):
            print(df.sort(["evaluation", "depth"]))
        print(f"Résultats écrits dans {args.out_dir}")
    elif args.command == "train-dqn":
        from .deep_rl import DqnConfig, save_weights, train_dqn

        args.out_dir.mkdir(parents=True, exist_ok=True)
        config = DqnConfig(
            representation=args.representation,
            episodes=args.episodes,
            max_moves=args.max_moves,
            seed=args.seed,
            warm_start_from_experience=args.warm_start,
        )
        net, history = train_dqn(config)
        save_weights(net, args.out_dir / f"dqn_{args.representation}.pt")
        pl.DataFrame(history).write_parquet(args.out_dir / f"dqn_{args.representation}_history.parquet")
        print(f"Historique (dernier point) : {history[-1] if history else 'aucun'}")
        print(f"Poids et historique écrits dans {args.out_dir}")
    elif args.command == "build-experience":
        from .experience import TOP_K_GAMES, append_experience

        total = 0
        for exp_dir in sorted(args.results_root.iterdir()):
            if not exp_dir.is_dir() or not (exp_dir / "results.parquet").exists():
                continue
            added = append_experience(exp_dir, top_k=TOP_K_GAMES)
            if added:
                print(f"  {exp_dir.name}: +{added} transitions")
            total += added
        print(f"Total : {total} transitions dans {args.results_root / 'experience' / 'dataset.parquet'}")


if __name__ == "__main__":
    main()
