from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from langsim.utils.preprocess import (
    available_columns,
    make_modality_scores,
    standardise_numeric_frame,
    zscore,
)
from langsim.utils.stats import confidence_interval, fit_ols, partial_r2, wald_joint_pvalue


def _dataset_dummies(df: pd.DataFrame, dataset_col: str) -> pd.DataFrame:
    if dataset_col not in df.columns:
        return pd.DataFrame(index=df.index)

    dummies = pd.get_dummies(
        df[dataset_col].astype(str),
        prefix="dataset",
        drop_first=True,
        dtype=float,
    )
    return dummies


def _single_effect_for_scope(
    df: pd.DataFrame,
    measure: str,
    outcome_col: str,
    dataset_col: str | None,
    scope: str,
) -> dict:
    cols = [outcome_col, measure]
    if dataset_col is not None and dataset_col in df.columns:
        cols.append(dataset_col)

    model_df = df[cols].copy()
    model_df[measure] = pd.to_numeric(model_df[measure], errors="coerce")
    model_df[outcome_col] = pd.to_numeric(model_df[outcome_col], errors="coerce")
    model_df = model_df.dropna(subset=[outcome_col, measure])

    if model_df.shape[0] < 5:
        return {
            "scope": scope,
            "measure": measure,
            "n": int(model_df.shape[0]),
            "beta": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "partial_r2": np.nan,
        }

    model_df[measure] = zscore(model_df[measure])

    x = pd.DataFrame({measure: model_df[measure]}, index=model_df.index)

    if dataset_col is not None and dataset_col in model_df.columns:
        x = pd.concat([x, _dataset_dummies(model_df, dataset_col)], axis=1)

    y = model_df[outcome_col]

    result = fit_ols(y=y, x=x, cov_type="HC3")

    reduced_x = x.drop(columns=[measure])
    pr2 = partial_r2(y=y, full_x=x, reduced_x=reduced_x)

    ci_low, ci_high = confidence_interval(result, measure)

    return {
        "scope": scope,
        "measure": measure,
        "n": int(model_df.shape[0]),
        "beta": float(result.params.get(measure, np.nan)),
        "se": float(result.bse.get(measure, np.nan)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(result.pvalues.get(measure, np.nan)),
        "partial_r2": pr2,
    }


def run_single_distance_effects(
    df: pd.DataFrame,
    distance_cols: Sequence[str],
    outcome_col: str,
    dataset_col: str,
) -> pd.DataFrame:
    rows = []

    for measure in distance_cols:
        rows.append(
            _single_effect_for_scope(
                df=df,
                measure=measure,
                outcome_col=outcome_col,
                dataset_col=dataset_col,
                scope="pooled",
            )
        )

        for dataset, sub in df.groupby(dataset_col, dropna=False):
            rows.append(
                _single_effect_for_scope(
                    df=sub,
                    measure=measure,
                    outcome_col=outcome_col,
                    dataset_col=None,
                    scope=str(dataset),
                )
            )

    return pd.DataFrame(rows)


def run_family_joint_effects(
    df: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
    outcome_col: str,
    dataset_col: str,
) -> pd.DataFrame:
    rows = []

    for family, cols in families.items():
        active_cols = available_columns(df, cols)

        active_cols = [
            col
            for col in active_cols
            if pd.to_numeric(df[col], errors="coerce").notna().sum() >= 5
        ]

        if not active_cols:
            rows.append(
                {
                    "family": family,
                    "n_measures": 0,
                    "measures": "",
                    "n": 0,
                    "joint_p_value": np.nan,
                    "partial_r2": np.nan,
                    "direction_coherence_negative": np.nan,
                }
            )
            continue

        cols_needed = [outcome_col, dataset_col] + active_cols
        model_df = df[cols_needed].copy()

        for col in active_cols:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

        model_df[outcome_col] = pd.to_numeric(model_df[outcome_col], errors="coerce")
        model_df = model_df.dropna(subset=[outcome_col] + active_cols)

        if model_df.shape[0] < len(active_cols) + 5:
            rows.append(
                {
                    "family": family,
                    "n_measures": len(active_cols),
                    "measures": " ".join(active_cols),
                    "n": int(model_df.shape[0]),
                    "joint_p_value": np.nan,
                    "partial_r2": np.nan,
                    "direction_coherence_negative": np.nan,
                }
            )
            continue

        x_family = standardise_numeric_frame(model_df[active_cols])
        x_dataset = _dataset_dummies(model_df, dataset_col)
        x = pd.concat([x_family, x_dataset], axis=1)
        y = model_df[outcome_col]

        result = fit_ols(y=y, x=x, cov_type="HC3")

        reduced_x = x_dataset
        pr2 = partial_r2(y=y, full_x=x, reduced_x=reduced_x)

        betas = result.params.reindex(active_cols)
        direction_coherence = float((betas < 0).mean())

        rows.append(
            {
                "family": family,
                "n_measures": len(active_cols),
                "measures": " ".join(active_cols),
                "n": int(model_df.shape[0]),
                "joint_p_value": wald_joint_pvalue(result, active_cols),
                "partial_r2": pr2,
                "direction_coherence_negative": direction_coherence,
            }
        )

    return pd.DataFrame(rows)


def run_modality_effects(
    df: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
    outcome_col: str,
    dataset_col: str,
) -> pd.DataFrame:
    scores = make_modality_scores(df, families)

    if scores.shape[1] == 0:
        return pd.DataFrame()

    model_df = pd.concat(
        [df[[outcome_col, dataset_col]], scores],
        axis=1,
    ).dropna(subset=[outcome_col])

    score_cols = list(scores.columns)

    x_scores = model_df[score_cols]
    x_dataset = _dataset_dummies(model_df, dataset_col)
    x = pd.concat([x_scores, x_dataset], axis=1)
    y = pd.to_numeric(model_df[outcome_col], errors="coerce")

    keep = y.notna()
    x = x.loc[keep]
    y = y.loc[keep]

    result = fit_ols(y=y, x=x, cov_type="HC3")

    rows = []

    for score_col in score_cols:
        reduced_x = x.drop(columns=[score_col])
        pr2 = partial_r2(y=y, full_x=x, reduced_x=reduced_x)
        ci_low, ci_high = confidence_interval(result, score_col)

        rows.append(
            {
                "modality_score": score_col,
                "n": int(y.shape[0]),
                "beta": float(result.params.get(score_col, np.nan)),
                "se": float(result.bse.get(score_col, np.nan)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": float(result.pvalues.get(score_col, np.nan)),
                "drop_one_partial_r2": pr2,
            }
        )

    return pd.DataFrame(rows)