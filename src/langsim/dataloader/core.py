from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from langsim.config import DISTANCE_FAMILIES
from langsim.utils.preprocess import add_pair_id, available_distance_columns, zscore_by_group


def read_core_csvs(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = []

    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)

        if "dataset" not in frame.columns:
            frame["dataset"] = path.stem.replace("_base", "")

        frames.append(frame)

    if not frames:
        raise ValueError("No CSV paths were provided.")

    return pd.concat(frames, ignore_index=True)


def validate_core_columns(
    df: pd.DataFrame,
    outcome_col: str,
    dataset_col: str,
    native_col: str,
    target_col: str,
) -> None:
    required = [outcome_col, dataset_col, native_col, target_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required core columns: {missing}")


def aggregate_to_language_pairs(
    df: pd.DataFrame,
    outcome_col: str,
    dataset_col: str,
    native_col: str,
    target_col: str,
) -> pd.DataFrame:
    distance_cols = available_distance_columns(df, DISTANCE_FAMILIES)

    group_cols = [dataset_col, native_col, target_col]
    optional_label_cols = [
        col
        for col in ["native_language", "target_language"]
        if col in df.columns
    ]

    agg = {
        outcome_col: "mean",
        "observation": "count" if "observation" in df.columns else "size",
    }

    for col in optional_label_cols:
        agg[col] = "first"

    for col in distance_cols:
        agg[col] = "first"

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(agg)
        .reset_index()
        .rename(columns={"observation": "n_observations"})
    )

    return out


def prepare_core_data(
    paths: Iterable[str | Path],
    outcome_col: str,
    dataset_col: str,
    native_col: str,
    target_col: str,
    aggregate_pairs: bool = True,
) -> pd.DataFrame:
    df = read_core_csvs(paths)

    validate_core_columns(
        df=df,
        outcome_col=outcome_col,
        dataset_col=dataset_col,
        native_col=native_col,
        target_col=target_col,
    )

    if aggregate_pairs:
        df = aggregate_to_language_pairs(
            df=df,
            outcome_col=outcome_col,
            dataset_col=dataset_col,
            native_col=native_col,
            target_col=target_col,
        )
    else:
        if "n_observations" not in df.columns:
            df["n_observations"] = 1

    df = add_pair_id(
        df=df,
        dataset_col=dataset_col,
        native_col=native_col,
        target_col=target_col,
    )

    df["outcome"] = pd.to_numeric(df[outcome_col], errors="coerce")
    df["outcome_z"] = zscore_by_group(
        df=df,
        value_col="outcome",
        group_col=dataset_col,
    )

    return df