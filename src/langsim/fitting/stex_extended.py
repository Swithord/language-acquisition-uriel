from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from langsim.config import (
    AUXILIARY_TARGET_PREFIX,
    CATEGORICAL_CONFOUNDER_COLS,
    COUNTRY_CONFOUNDER_COLS,
    DISTANCE_FAMILIES,
    LEARNER_CONFOUNDER_COLS,
    NATIVE_AUXILIARY_PREFIX,
)
from langsim.dataloader.extended import load_stex_extended
from langsim.fitting.prediction import run_predictive_analysis
from langsim.models.base import RidgePredictor
from langsim.utils.io import write_csv, write_json
from langsim.utils.preprocess import (
    available_columns,
    available_distance_columns,
    block_pca_scores,
    numeric_frame,
    standardise_numeric_frame,
    zscore,
)
from langsim.utils.stats import confidence_interval, fit_ols, partial_r2


def _prefixed_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [col for col in df.columns if col.startswith(prefix)]


def _make_adjustment_frame(df: pd.DataFrame) -> pd.DataFrame:
    learner_cols = available_columns(df, LEARNER_CONFOUNDER_COLS)
    country_cols = available_columns(df, COUNTRY_CONFOUNDER_COLS)
    categorical_cols = available_columns(df, CATEGORICAL_CONFOUNDER_COLS)

    aux_target_cols = _prefixed_columns(df, AUXILIARY_TARGET_PREFIX)
    native_aux_cols = _prefixed_columns(df, NATIVE_AUXILIARY_PREFIX)

    learner = standardise_numeric_frame(numeric_frame(df, learner_cols))
    country = block_pca_scores(
        df=df,
        columns=country_cols,
        prefix="country_opportunity",
        n_components=2,
    )
    aux_target = block_pca_scores(
        df=df,
        columns=aux_target_cols,
        prefix="auxiliary_target",
        n_components=2,
    )
    native_aux = block_pca_scores(
        df=df,
        columns=native_aux_cols,
        prefix="native_auxiliary",
        n_components=2,
    )

    categorical_pieces = []

    for col in categorical_cols:
        dummies = pd.get_dummies(
            df[col].astype("object").fillna("missing"),
            prefix=col,
            drop_first=True,
            dtype=float,
        )
        categorical_pieces.append(dummies)

    simple = pd.DataFrame(index=df.index)

    if "is_monolingual" in df.columns:
        simple["is_monolingual"] = pd.to_numeric(
            df["is_monolingual"],
            errors="coerce",
        )

    pieces = [learner, country, aux_target, native_aux, simple] + categorical_pieces
    pieces = [piece for piece in pieces if piece.shape[1] > 0]

    if not pieces:
        return pd.DataFrame(index=df.index)

    adjustment = pd.concat(pieces, axis=1)

    for col in adjustment.columns:
        adjustment[col] = pd.to_numeric(adjustment[col], errors="coerce")
        mean = adjustment[col].mean(skipna=True)
        adjustment[col] = adjustment[col].fillna(mean)

    return adjustment


def _fit_distance_model(
    df: pd.DataFrame,
    distance_col: str,
    outcome_col: str,
    adjustment: pd.DataFrame | None,
    cluster_col: str,
) -> dict:
    cols = [outcome_col, distance_col, cluster_col]
    model_df = df[cols].copy()

    model_df[outcome_col] = pd.to_numeric(model_df[outcome_col], errors="coerce")
    model_df[distance_col] = pd.to_numeric(model_df[distance_col], errors="coerce")
    model_df = model_df.dropna(subset=[outcome_col, distance_col, cluster_col])

    if model_df.shape[0] < 10:
        return {
            "measure": distance_col,
            "n": int(model_df.shape[0]),
            "beta": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "partial_r2": np.nan,
        }

    x_distance = pd.DataFrame(
        {distance_col: zscore(model_df[distance_col])},
        index=model_df.index,
    )

    if adjustment is None or adjustment.shape[1] == 0:
        x = x_distance
        reduced_x = pd.DataFrame(index=model_df.index)
    else:
        adj = adjustment.loc[model_df.index].copy()
        x = pd.concat([x_distance, adj], axis=1)
        reduced_x = adj

    y = model_df[outcome_col]
    groups = model_df[cluster_col]

    result = fit_ols(
        y=y,
        x=x,
        cov_type="cluster",
        groups=groups,
    )

    pr2 = partial_r2(
        y=y,
        full_x=x,
        reduced_x=reduced_x,
    )

    ci_low, ci_high = confidence_interval(result, distance_col)

    return {
        "measure": distance_col,
        "n": int(model_df.shape[0]),
        "beta": float(result.params.get(distance_col, np.nan)),
        "se": float(result.bse.get(distance_col, np.nan)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(result.pvalues.get(distance_col, np.nan)),
        "partial_r2": pr2,
    }


def _run_unadjusted_adjusted_effects(
    df: pd.DataFrame,
    distance_cols: Sequence[str],
    outcome_col: str,
    cluster_col: str,
) -> pd.DataFrame:
    adjustment = _make_adjustment_frame(df)

    rows = []

    for distance_col in distance_cols:
        unadjusted = _fit_distance_model(
            df=df,
            distance_col=distance_col,
            outcome_col=outcome_col,
            adjustment=None,
            cluster_col=cluster_col,
        )
        adjusted = _fit_distance_model(
            df=df,
            distance_col=distance_col,
            outcome_col=outcome_col,
            adjustment=adjustment,
            cluster_col=cluster_col,
        )

        beta_unadj = unadjusted["beta"]
        beta_adj = adjusted["beta"]

        if np.isfinite(beta_unadj) and beta_unadj != 0 and np.isfinite(beta_adj):
            attenuation = 1.0 - beta_adj / beta_unadj
        else:
            attenuation = np.nan

        rows.append(
            {
                "measure": distance_col,
                "n_unadjusted": unadjusted["n"],
                "beta_unadjusted": beta_unadj,
                "se_unadjusted": unadjusted["se"],
                "p_unadjusted": unadjusted["p_value"],
                "partial_r2_unadjusted": unadjusted["partial_r2"],
                "n_adjusted": adjusted["n"],
                "beta_adjusted": beta_adj,
                "se_adjusted": adjusted["se"],
                "p_adjusted": adjusted["p_value"],
                "partial_r2_adjusted": adjusted["partial_r2"],
                "attenuation": attenuation,
            }
        )

    return pd.DataFrame(rows)


def _extended_predictive_tables(
    df: pd.DataFrame,
    distance_cols: Sequence[str],
    outcome_col: str,
    group_col: str,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adjustment = _make_adjustment_frame(df)
    adjustment_cols = list(adjustment.columns)

    pred_df = pd.concat(
        [
            df[[outcome_col, group_col] + list(distance_cols)].copy(),
            adjustment,
        ],
        axis=1,
    )

    families = {
        "distances": list(distance_cols),
        "confounders": adjustment_cols,
    }

    all_rows = []
    all_summaries = []

    model_sets = {
        "distances_only": list(distance_cols),
        "confounders_only": adjustment_cols,
        "confounders_plus_distances": adjustment_cols + list(distance_cols),
    }

    for model_name, features in model_sets.items():
        per_fold, summary = run_predictive_analysis(
            df=pred_df,
            distance_cols=features,
            families=families,
            outcome_col=outcome_col,
            group_col=group_col,
            n_splits=n_splits,
            random_state=random_state,
        )

        per_fold = per_fold[per_fold["model"].isin(["baseline_mean", "all_distances_ridge"])].copy()
        per_fold["extended_model"] = per_fold["model"].replace(
            {"all_distances_ridge": model_name}
        )

        summary = (
            per_fold.groupby("extended_model", as_index=False)[
                ["rmse", "mae", "r2", "spearman"]
            ]
            .agg(["mean", "std"])
        )
        summary.columns = [
            "_".join(col).strip("_") for col in summary.columns.to_flat_index()
        ]

        all_rows.append(per_fold)
        all_summaries.append(summary)

    per_fold_out = pd.concat(all_rows, ignore_index=True)
    summary_out = pd.concat(all_summaries, ignore_index=True)

    summary_out = summary_out.drop_duplicates(subset=["extended_model"])

    return per_fold_out, summary_out


def run_stex_extended_analysis(
    stex_base_path: str | Path,
    stex_extended_path: str | Path,
    outdir: str | Path,
    outcome_col: str,
    n_splits: int,
    random_state: int,
) -> None:
    outdir = Path(outdir)

    df = load_stex_extended(
        stex_base_path=stex_base_path,
        stex_extended_path=stex_extended_path,
        outcome_col=outcome_col,
    )

    distance_cols = available_distance_columns(df, DISTANCE_FAMILIES)

    metadata = {
        "stex_base_path": str(stex_base_path),
        "stex_extended_path": str(stex_extended_path),
        "outdir": str(outdir),
        "outcome_col": outcome_col,
        "n_rows": int(df.shape[0]),
        "n_pairs": int(df["pair_id"].nunique()),
        "distance_columns": distance_cols,
    }
    write_json(metadata, outdir / "metadata.json")
    write_csv(df, outdir / "analysis_data.csv")

    effects = _run_unadjusted_adjusted_effects(
        df=df,
        distance_cols=distance_cols,
        outcome_col="outcome_z",
        cluster_col="pair_id",
    )
    write_csv(effects, outdir / "distance_effect_attenuation.csv")

    per_fold, summary = _extended_predictive_tables(
        df=df,
        distance_cols=distance_cols,
        outcome_col="outcome_z",
        group_col="pair_id",
        n_splits=n_splits,
        random_state=random_state,
    )
    write_csv(per_fold, outdir / "predictive_per_fold.csv")
    write_csv(summary, outdir / "predictive_summary.csv")