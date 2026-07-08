#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from langsim.config import DISTANCE_FAMILIES, OUTCOME_COL
from langsim.dataloader.core import prepare_core_data
from langsim.evaluate.plotting import (
    plot_distance_correlation_heatmap,
    plot_prediction_summary,
    plot_single_effects_forest,
)
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
        help="Write diagnostic plots.",
    )

    return parser.parse_args()


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

    metadata = {
        "csv": args.csv,
        "outdir": str(outdir),
        "outcome_col": args.outcome_col,
        "dataset_col": args.dataset_col,
        "native_col": args.native_col,
        "target_col": args.target_col,
        "aggregate_pairs": args.aggregate_pairs,
        "n_rows": int(df.shape[0]),
        "distance_columns": distance_cols,
        "distance_families": {
            family: [col for col in cols if col in df.columns]
            for family, cols in DISTANCE_FAMILIES.items()
        },
    }
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

        plot_distance_correlation_heatmap(
            df=df,
            distance_cols=distance_cols,
            outpath=plot_dir / "distance_spearman_heatmap.png",
        )
        plot_single_effects_forest(
            effects=single_effects,
            outpath=plot_dir / "single_effects_forest.png",
        )
        plot_prediction_summary(
            summary=predictive_summary,
            outpath=plot_dir / "predictive_summary.png",
        )

    print(f"Finished core analysis. Outputs written to {outdir}")


if __name__ == "__main__":
    main()