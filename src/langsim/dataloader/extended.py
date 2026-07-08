from __future__ import annotations

from pathlib import Path

import pandas as pd

from langsim.utils.preprocess import add_pair_id, zscore


def load_stex_extended(
    stex_base_path: str | Path,
    stex_extended_path: str | Path,
    outcome_col: str,
) -> pd.DataFrame:
    base = pd.read_csv(stex_base_path)
    ext = pd.read_csv(stex_extended_path)

    if "observation" not in base.columns:
        raise ValueError("stex_base must contain an observation column.")

    if "observation" not in ext.columns:
        raise ValueError("stex_extended must contain an observation column.")

    if outcome_col not in base.columns:
        raise ValueError(f"stex_base is missing outcome column: {outcome_col}")

    df = base.merge(ext, on="observation", how="left", validate="one_to_one")

    if "dataset" not in df.columns:
        df["dataset"] = "stex"

    df = add_pair_id(
        df=df,
        dataset_col="dataset",
        native_col="native_code",
        target_col="target_code",
    )

    df["outcome"] = pd.to_numeric(df[outcome_col], errors="coerce")
    df["outcome_z"] = zscore(df["outcome"])

    df["is_monolingual"] = (
        df.get("auxiliary_language", pd.Series(index=df.index, dtype=object))
        .fillna("")
        .str.lower()
        .eq("monolingual")
        .astype(float)
    )

    return df