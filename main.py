#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from langsim.config import DISTANCE_FAMILIES, OUTCOME_COL
from langsim.dataloader.core import prepare_core_data
from langsim.evaluate.plotting import make_core_plots
from langsim.fitting.agreement import coverage_table, run_agreement_analysis
from langsim.fitting.inference import (
    run_family_joint_effects,
    run_modality_effects,
    run_single_distance_effects,
)
from langsim.fitting.prediction import run_predictive_analysis
from langsim.utils.io import ensure_dir, write_csv, write_json
from langsim.utils.preprocess import available_distance_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the core language-distance analysis."
    )

    parser.add_argument(
        "--csv",
        nargs="+",
        default=["data/toefl_base.csv", "data/stex_base.csv"],
        help="Core-format CSV files. Each must contain the shared base columns.",
    )
    parser.add_argument(
        "--outdir",
        default="artifacts/core",
        help="Output directory.",
    )
    parser.add_argument(
        "--outcome_col",
        default=OUTCOME_COL,
        help="Outcome column.",
    )
    parser.add_argument(
        "--dataset_col",
        default="dataset",
        help="Dataset column.",
    )
    parser.add_argument(
        "--native_col",
        default="native_code",
        help="Native-language code column.",
    )
    parser.add_argument(
        "--target_col",
        default="target_code",
        help="Target-language code column.",
    )
    parser.add_argument(
        "--aggregate_pairs",
        action="store_true",
        default=True,
        help="Aggregate repeated native-target observations to language-pair means.",
    )
    parser.add_argument(
        "--no_aggregate_pairs",
        action="store_false",
        dest="aggregate_pairs",
        help="Do not aggregate repeated native-target observations.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of grouped CV folds.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--make_plots",
        action="store_true",
        help="Write plots.",
    )

    return parser.parse_args()


def _targets_by_dataset(
    df,
    dataset_col: str,
    target_col: str,
) -> dict[str, list[str]]:
    if dataset_col not in df.columns or target_col not in df.columns:
        return {}

    targets: dict[str, list[str]] = {}

    for dataset, sub in df.groupby(dataset_col, dropna=False):
        values = (
            sub[target_col]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
        targets[str(dataset)] = values

    return targets


def _n_pairs_by_dataset(
    df,
    dataset_col: str,
) -> dict[str, int]:
    if dataset_col not in df.columns or "pair_id" not in df.columns:
        return {}

    counts: dict[str, int] = {}

    for dataset, sub in df.groupby(dataset_col, dropna=False):
        counts[str(dataset)] = int(sub["pair_id"].nunique())

    return counts


def _n_rows_by_dataset(
    df,
    dataset_col: str,
) -> dict[str, int]:
    if dataset_col not in df.columns:
        return {}

    return {
        str(dataset): int(sub.shape[0])
        for dataset, sub in df.groupby(dataset_col, dropna=False)
    }


def _n_original_observations_by_dataset(
    df,
    dataset_col: str,
) -> dict[str, int]:
    if dataset_col not in df.columns or "n_observations" not in df.columns:
        return {}

    return {
        str(dataset): int(sub["n_observations"].sum())
        for dataset, sub in df.groupby(dataset_col, dropna=False)
    }


def _make_metadata(
    args: argparse.Namespace,
    df,
    outdir: Path,
    distance_cols: list[str],
) -> dict[str, Any]:
    return {
        "csv": args.csv,
        "outdir": str(outdir),
        "outcome_col": args.outcome_col,
        "dataset_col": args.dataset_col,
        "native_col": args.native_col,
        "target_col": args.target_col,
        "aggregate_pairs": args.aggregate_pairs,
        "n_rows": int(df.shape[0]),
        "n_rows_by_dataset": _n_rows_by_dataset(
            df=df,
            dataset_col=args.dataset_col,
        ),
        "n_pairs": int(df["pair_id"].nunique()) if "pair_id" in df.columns else None,
        "n_pairs_by_dataset": _n_pairs_by_dataset(
            df=df,
            dataset_col=args.dataset_col,
        ),
        "n_original_observations_by_dataset": _n_original_observations_by_dataset(
            df=df,
            dataset_col=args.dataset_col,
        ),
        "targets_by_dataset": _targets_by_dataset(
            df=df,
            dataset_col=args.dataset_col,
            target_col=args.target_col,
        ),
        "distance_columns": distance_cols,
        "distance_families": {
            family: [col for col in cols if col in df.columns]
            for family, cols in DISTANCE_FAMILIES.items()
        },
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = prepare_core_data(
        paths=args.csv,
        outcome_col=args.outcome_col,
        dataset_col=args.dataset_col,
        native_col=args.native_col,
        target_col=args.target_col,
        aggregate_pairs=args.aggregate_pairs,
    )

    distance_cols = available_distance_columns(df, DISTANCE_FAMILIES)

    if len(distance_cols) == 0:
        raise ValueError("No configured distance columns were found in the input data.")

    write_csv(df, outdir / "analysis_data.csv")

    metadata = _make_metadata(
        args=args,
        df=df,
        outdir=outdir,
        distance_cols=distance_cols,
    )
    write_json(metadata, outdir / "metadata.json")

    coverage = coverage_table(
        df=df,
        distance_cols=distance_cols,
        dataset_col=args.dataset_col,
    )
    write_csv(coverage, outdir / "coverage.csv")

    agreement = run_agreement_analysis(
        df=df,
        families=DISTANCE_FAMILIES,
        dataset_col=args.dataset_col,
    )
    write_csv(agreement, outdir / "distance_agreement.csv")

    single_effects = run_single_distance_effects(
        df=df,
        distance_cols=distance_cols,
        outcome_col="outcome_z",
        dataset_col=args.dataset_col,
    )
    write_csv(single_effects, outdir / "single_distance_effects.csv")

    family_effects = run_family_joint_effects(
        df=df,
        families=DISTANCE_FAMILIES,
        outcome_col="outcome_z",
        dataset_col=args.dataset_col,
    )
    write_csv(family_effects, outdir / "family_joint_effects.csv")

    modality_effects = run_modality_effects(
        df=df,
        families=DISTANCE_FAMILIES,
        outcome_col="outcome_z",
        dataset_col=args.dataset_col,
    )
    write_csv(modality_effects, outdir / "modality_effects.csv")

    per_fold, predictive_summary = run_predictive_analysis(
        df=df,
        distance_cols=distance_cols,
        families=DISTANCE_FAMILIES,
        outcome_col="outcome_z",
        group_col="pair_id",
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    write_csv(per_fold, outdir / "predictive_per_fold.csv")
    write_csv(predictive_summary, outdir / "predictive_summary.csv")

    if args.make_plots:
        plot_dir = outdir / "plots"
        ensure_dir(plot_dir)

        make_core_plots(
            df=df,
            distance_cols=distance_cols,
            single_effects=single_effects,
            family_effects=family_effects,
            modality_effects=modality_effects,
            predictive_summary=predictive_summary,
            predictive_per_fold=per_fold,
            outdir=plot_dir,
            dataset_col=args.dataset_col,
            families=DISTANCE_FAMILIES,
            agreement=agreement,
        )

    print(f"Finished core analysis. Outputs written to {outdir}")


if __name__ == "__main__":
    main()