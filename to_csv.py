#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

"""
python to_csv.py \
  data/Data/Data/without_glottolog_with_lineage/family_features.npz \
  data/Data/Data/without_glottolog_with_lineage/features.npz \
  data/Data/Data/without_glottolog_with_lineage/script_features.npz \
  data/Data/Data/without_glottolog_with_lineage/geocoord_features.npz
"""


DEFAULT_FILES = [
    "data/Data/Data/without_glottolog_with_lineage/family_features.npz",
    "data/Data/Data/without_glottolog_with_lineage/features.npz",
    "data/Data/Data/without_glottolog_with_lineage/script_features.npz",
    "data/Data/Data/without_glottolog_with_lineage/geocoord_features.npz",
]


def make_unique(names: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []

    for name in names:
        name = str(name)
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
    return out


def collapse_to_2d(arr: np.ndarray, n_langs: int) -> np.ndarray:
    """
    Collapse a data array to shape (n_langs, n_features_flat).

    Examples:
      (8171, 800, 1) -> (8171, 800)
      (8171, 800)    -> (8171, 800)
      (8171,)        -> (8171, 1)
      (8171, a, b)   -> (8171, a*b)
    """
    arr = np.asarray(arr)

    if arr.shape[0] != n_langs:
        raise ValueError(
            f"First dimension of data ({arr.shape[0]}) does not match "
            f"number of langs ({n_langs})."
        )

    if arr.ndim == 1:
        return arr.reshape(n_langs, 1)

    return arr.reshape(n_langs, -1)


def infer_headers(npz: np.lib.npyio.NpzFile, n_cols: int) -> list[str]:
    """
    Prefer feature names from key 'feats' when possible.
    Otherwise fall back to generic column names.
    """
    if "feats" in npz:
        feats = np.asarray(npz["feats"], dtype=object).reshape(-1)
        if len(feats) == n_cols:
            return make_unique([str(x) for x in feats])

    return [f"feature_{i}" for i in range(n_cols)]


def write_csv(npz_path: Path, out_path: Path) -> None:
    with np.load(npz_path, allow_pickle=True) as npz:
        if "data" not in npz:
            raise KeyError(f"{npz_path} does not contain key 'data'")
        if "langs" not in npz:
            raise KeyError(f"{npz_path} does not contain key 'langs'")

        langs = np.asarray(npz["langs"], dtype=object).reshape(-1)
        data_2d = collapse_to_2d(npz["data"], len(langs))
        headers = infer_headers(npz, data_2d.shape[1])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lang", *headers])

        for lang, row in zip(langs, data_2d, strict=True):
            writer.writerow([str(lang), *row.tolist()])

    print(f"Wrote {out_path} with shape {data_2d.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NPZ feature matrices (key='data') to CSV."
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_FILES,
        help="NPZ files to convert. Defaults to the four expected feature files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the same directory as each input file.",
    )
    args = parser.parse_args()

    for file_str in args.files:
        npz_path = Path(file_str)
        if not npz_path.exists():
            print(f"Skipping missing file: {npz_path}")
            continue

        out_dir = args.outdir if args.outdir is not None else npz_path.parent
        out_path = out_dir / f"{npz_path.stem}.csv"
        write_csv(npz_path, out_path)


if __name__ == "__main__":
    main()