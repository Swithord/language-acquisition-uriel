from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def available_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def unique_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)

    return out


def family_lookup(
    families: Mapping[str, Sequence[str]] | None,
) -> dict[str, str]:
    if families is None:
        return {}

    lookup: dict[str, str] = {}

    for family, cols in families.items():
        for col in cols:
            lookup[col] = family

    return lookup


def columns_ordered_by_family(
    columns: Sequence[str],
    families: Mapping[str, Sequence[str]] | None,
) -> list[str]:
    columns = unique_preserve_order(list(columns))

    if families is None:
        return columns

    ordered: list[str] = []
    seen: set[str] = set()

    for _, family_cols in families.items():
        for col in family_cols:
            if col in columns and col not in seen:
                ordered.append(col)
                seen.add(col)

    for col in columns:
        if col not in seen:
            ordered.append(col)
            seen.add(col)

    return ordered


def family_boundaries(
    ordered_columns: Sequence[str],
    families: Mapping[str, Sequence[str]] | None,
) -> list[int]:
    if families is None:
        return []

    lookup = family_lookup(families)
    boundaries: list[int] = []

    previous_family: str | None = None

    for idx, col in enumerate(ordered_columns):
        current_family = lookup.get(col, "other")

        if previous_family is not None and current_family != previous_family:
            boundaries.append(idx)

        previous_family = current_family

    return boundaries


def available_distance_columns(
    df: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
) -> list[str]:
    cols: list[str] = []

    for family_cols in families.values():
        cols.extend([col for col in family_cols if col in df.columns])

    return unique_preserve_order(cols)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean(skipna=True)
    std = values.std(skipna=True, ddof=1)

    if pd.isna(std) or std == 0:
        return values * np.nan

    return (values - mean) / std


def zscore_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
) -> pd.Series:
    return df.groupby(group_col, dropna=False)[value_col].transform(zscore)


def add_pair_id(
    df: pd.DataFrame,
    dataset_col: str,
    native_col: str,
    target_col: str,
) -> pd.DataFrame:
    out = df.copy()

    out["pair_id"] = (
        out[dataset_col].astype(str)
        + "::"
        + out[native_col].astype(str)
        + "->"
        + out[target_col].astype(str)
    )

    return out


def numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    cols = available_columns(df, columns)
    out = pd.DataFrame(index=df.index)

    for col in cols:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    return out


def standardise_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        out[col] = zscore(df[col])

    return out


def block_pca_scores(
    df: pd.DataFrame,
    columns: Sequence[str],
    prefix: str,
    n_components: int = 1,
    min_nonmissing: int = 3,
) -> pd.DataFrame:
    cols = [
        col
        for col in columns
        if col in df.columns
        and pd.to_numeric(df[col], errors="coerce").notna().sum() >= min_nonmissing
    ]

    if not cols:
        return pd.DataFrame(index=df.index)

    x = numeric_frame(df, cols)

    if len(cols) == 1:
        score = zscore(x[cols[0]])
        return pd.DataFrame({f"{prefix}_pc1": score}, index=df.index)

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    pca = PCA(n_components=min(n_components, len(cols)))

    x_imp = imputer.fit_transform(x)
    x_std = scaler.fit_transform(x_imp)
    scores = pca.fit_transform(x_std)

    mean_distance = np.nanmean(x_std, axis=1)

    for j in range(scores.shape[1]):
        corr = np.corrcoef(scores[:, j], mean_distance)[0, 1]
        if np.isfinite(corr) and corr < 0:
            scores[:, j] *= -1

    return pd.DataFrame(
        {
            f"{prefix}_pc{j + 1}": scores[:, j]
            for j in range(scores.shape[1])
        },
        index=df.index,
    )


def make_modality_scores(
    df: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    pieces = []

    for family, cols in families.items():
        score = block_pca_scores(
            df=df,
            columns=cols,
            prefix=f"modality_{family}",
            n_components=1,
        )
        pieces.append(score)

    if not pieces:
        return pd.DataFrame(index=df.index)

    return pd.concat(pieces, axis=1)