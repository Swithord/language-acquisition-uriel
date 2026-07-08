from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from langsim.evaluate.metrics import regression_metrics
from langsim.models.base import (
    BestSingleDistancePredictor,
    MeanPredictor,
    ModalityRidgePredictor,
    RidgePredictor,
)


def _make_splits(
    df: pd.DataFrame,
    group_col: str,
    n_splits: int,
    random_state: int,
):
    groups = df[group_col].astype(str)

    n_groups = groups.nunique()
    n_splits = min(n_splits, n_groups)

    if n_splits >= 2:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(df, groups=groups))

    splitter = KFold(
        n_splits=min(5, len(df)),
        shuffle=True,
        random_state=random_state,
    )
    return list(splitter.split(df))


def run_predictive_analysis(
    df: pd.DataFrame,
    distance_cols: Sequence[str],
    families: Mapping[str, Sequence[str]],
    outcome_col: str,
    group_col: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_df = df.dropna(subset=[outcome_col, group_col]).copy()

    if analysis_df.shape[0] < 5:
        raise ValueError("Not enough rows for predictive analysis.")

    splits = _make_splits(
        df=analysis_df,
        group_col=group_col,
        n_splits=n_splits,
        random_state=random_state,
    )

    model_factories = {
        "baseline_mean": lambda: MeanPredictor(),
        "best_single_distance": lambda: BestSingleDistancePredictor(distance_cols),
        "modality_scores_ridge": lambda: ModalityRidgePredictor(families),
        "all_distances_ridge": lambda: RidgePredictor(distance_cols),
    }

    rows = []

    for fold_id, (train_idx, test_idx) in enumerate(splits):
        train = analysis_df.iloc[train_idx].copy()
        test = analysis_df.iloc[test_idx].copy()

        y_train = pd.to_numeric(train[outcome_col], errors="coerce")
        y_test = pd.to_numeric(test[outcome_col], errors="coerce")

        for model_name, factory in model_factories.items():
            model = factory()
            model.fit(train, y_train)
            pred = model.predict(test)

            metrics = regression_metrics(y_true=y_test, y_pred=pred)

            rows.append(
                {
                    "fold": fold_id,
                    "model": model_name,
                    "n_train": int(train.shape[0]),
                    "n_test": int(test.shape[0]),
                    **metrics,
                }
            )

    per_fold = pd.DataFrame(rows)

    metric_cols = ["rmse", "mae", "r2", "spearman"]
    summary = (
        per_fold.groupby("model", as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )

    summary.columns = [
        "_".join(col).strip("_") for col in summary.columns.to_flat_index()
    ]

    return per_fold, summary