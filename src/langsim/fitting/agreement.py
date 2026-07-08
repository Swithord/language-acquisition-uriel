from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from langsim.utils.preprocess import available_columns


def coverage_table(
    df: pd.DataFrame,
    distance_cols: Sequence[str],
    dataset_col: str,
) -> pd.DataFrame:
    rows = []

    scopes = [("pooled", df)]

    for dataset, sub in df.groupby(dataset_col, dropna=False):
        scopes.append((str(dataset), sub))

    for scope, sub in scopes:
        for col in distance_cols:
            n = int(pd.to_numeric(sub[col], errors="coerce").notna().sum())
            rows.append(
                {
                    "scope": scope,
                    "measure": col,
                    "n_nonmissing": n,
                    "n_total": int(sub.shape[0]),
                    "coverage": n / sub.shape[0] if sub.shape[0] else np.nan,
                }
            )

    return pd.DataFrame(rows)


def _mean_abs_offdiag(corr: pd.DataFrame) -> float:
    if corr.shape[0] < 2:
        return np.nan

    values = corr.to_numpy(dtype=float)
    mask = ~np.eye(values.shape[0], dtype=bool)
    vals = values[mask]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.nan

    return float(np.mean(np.abs(vals)))


def _pc1_variance_explained(x: pd.DataFrame) -> float:
    if x.shape[1] == 0:
        return np.nan

    if x.shape[1] == 1:
        return 1.0

    values = x.apply(pd.to_numeric, errors="coerce")

    if values.notna().sum().sum() == 0:
        return np.nan

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    pca = PCA(n_components=1)

    transformed = scaler.fit_transform(imputer.fit_transform(values))
    pca.fit(transformed)

    return float(pca.explained_variance_ratio_[0])


def run_agreement_analysis(
    df: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
    dataset_col: str,
) -> pd.DataFrame:
    rows = []

    scopes = [("pooled", df)]

    for dataset, sub in df.groupby(dataset_col, dropna=False):
        scopes.append((str(dataset), sub))

    for scope, sub in scopes:
        for family, cols in families.items():
            active_cols = available_columns(sub, cols)

            active_cols = [
                col
                for col in active_cols
                if pd.to_numeric(sub[col], errors="coerce").notna().sum() >= 3
            ]

            if not active_cols:
                rows.append(
                    {
                        "scope": scope,
                        "family": family,
                        "n_measures": 0,
                        "measures": "",
                        "mean_abs_spearman": np.nan,
                        "pc1_variance_explained": np.nan,
                    }
                )
                continue

            x = sub[active_cols].apply(pd.to_numeric, errors="coerce")
            corr = x.corr(method="spearman", min_periods=3)

            rows.append(
                {
                    "scope": scope,
                    "family": family,
                    "n_measures": len(active_cols),
                    "measures": " ".join(active_cols),
                    "mean_abs_spearman": _mean_abs_offdiag(corr),
                    "pc1_variance_explained": _pc1_variance_explained(x),
                }
            )

    return pd.DataFrame(rows)