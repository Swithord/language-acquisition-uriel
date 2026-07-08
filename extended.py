#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from langsim.fitting.stex_extended import run_stex_extended_analysis
from langsim.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the STEX-only extended confounding analysis."
    )

    parser.add_argument(
        "--stex_base",
        default="data/stex_base.csv",
        help="Path to stex_base.csv.",
    )
    parser.add_argument(
        "--stex_extended",
        default="data/stex_extended.csv",
        help="Path to stex_extended.csv.",
    )
    parser.add_argument(
        "--outdir",
        default="artifacts/stex_extended",
        help="Output directory.",
    )
    parser.add_argument(
        "--outcome_col",
        default="speaking_score",
        help="Outcome column in stex_base.csv.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Grouped CV folds.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    run_stex_extended_analysis(
        stex_base_path=args.stex_base,
        stex_extended_path=args.stex_extended,
        outdir=outdir,
        outcome_col=args.outcome_col,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    print(f"Finished STEX extended analysis. Outputs written to {outdir}")


if __name__ == "__main__":
    main()