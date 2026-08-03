#!/usr/bin/env python3
"""Rebuild STEX HDI/WDI confounders using a target-year priority rule.

Selection rule, applied independently to every country and indicator:

1. Use the target year when available.
2. Otherwise use the latest available year before the target year.
3. Otherwise use the earliest available year after the target year.
4. Otherwise leave the value missing.

Rows without a usable ISO-3 country code, including historical Yugoslavia
observations in STEX, are retained. Their country-level confounders are set to
missing and they are not mapped to a successor country.

Outputs overwritten:
    <data-dir>/stex_extended.csv
    <data-dir>/stex_confounder_metadata.csv

Default source files:
    <data-dir>/unprocessed_indicators/human-development-index.csv
    <data-dir>/unprocessed_indicators/wdi_data.csv
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

SCRIPT_VERSION = "2026-08-03-v4-country-alias-safe"
DEFAULT_TARGET_YEAR = 2010

NA_VALUES = ["..", "...", "NA", "N/A", "null", "NULL"]
YEAR_COLUMN_RE = re.compile(r"^(\d{4})\s*\[YR\1\]$")

HDI_VALUE_COLUMN = "Human Development Index"
HDI_REGION_COLUMN = "World region according to OWID"


@dataclass(frozen=True)
class Indicator:
    output_column: str
    source: str
    series_code: str | None = None


INDICATORS: tuple[Indicator, ...] = (
    Indicator("hdi", "HDI"),
    Indicator("gdp_per_capita", "WDI", "NY.GDP.PCAP.PP.KD"),
    Indicator("adult_literacy", "WDI", "SE.ADT.LITR.ZS"),
    Indicator("internet_use", "WDI", "IT.NET.USER.ZS"),
    Indicator("education_expenditure", "WDI", "SE.XPD.TOTL.GD.ZS"),
    Indicator("tertiary_enrolment", "WDI", "SE.TER.ENRR"),
    Indicator("female_tertiary_enrolment", "WDI", "SE.TER.ENRR.FE"),
    Indicator("male_tertiary_enrolment", "WDI", "SE.TER.ENRR.MA"),
    Indicator("urban_population", "WDI", "SP.URB.TOTL.IN.ZS"),
)

INDICATOR_COLUMNS = tuple(item.output_column for item in INDICATORS)
WDI_CODE_TO_OUTPUT = {
    item.series_code: item.output_column
    for item in INDICATORS
    if item.source == "WDI" and item.series_code is not None
}

METADATA_COLUMNS = (
    "country",
    "country_name",
    "country_code",
    "confounder",
    "source",
    "source_name",
    "year",
    "timing",
    "region",
)


def normalize_country_codes(series: pd.Series) -> pd.Series:
    """Normalize ISO-3 codes while preserving missing values."""
    result = series.astype("string").str.strip().str.upper()
    return result.mask(result.eq(""), pd.NA)


def require_columns(frame: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def first_nonmissing(values: Iterable[object]) -> object:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return value
    return pd.NA


def timing_label(selected_year: int, target_year: int) -> str:
    if selected_year == target_year:
        return f"exact_{target_year}"
    if selected_year < target_year:
        return f"latest_available_before_{target_year}"
    return f"earliest_available_after_{target_year}"


def select_preferred(
    candidates: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    target_year: int,
) -> pd.DataFrame:
    """Choose one row per group using exact, latest-pre, earliest-post order."""
    required = [*group_columns, "year", "value"]
    require_columns(candidates, required, Path("candidate table"))

    work = candidates.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=required)

    if work.empty:
        work["timing"] = pd.Series(dtype="string")
        return work

    work["year"] = work["year"].astype(int)

    duplicate_keys = [*group_columns, "year"]
    conflicts = (
        work.groupby(duplicate_keys, dropna=False)["value"]
        .nunique(dropna=True)
        .loc[lambda x: x > 1]
    )
    if not conflicts.empty:
        raise ValueError(
            "Conflicting values exist for the same country, indicator, and year:\n"
            + conflicts.head(20).reset_index().to_string(index=False)
        )
    work = work.drop_duplicates(duplicate_keys, keep="first")

    # Sort key implements the deliberately asymmetric preference:
    # exact target -> any pre-target value -> any post-target value.
    work["_period"] = 2
    work.loc[work["year"] < target_year, "_period"] = 1
    work.loc[work["year"] == target_year, "_period"] = 0

    # For pre-target years, descending year is preferred. For post-target years,
    # ascending year is preferred.
    work["_year_order"] = work["year"]
    pre_mask = work["year"] < target_year
    work.loc[pre_mask, "_year_order"] = -work.loc[pre_mask, "year"]
    work.loc[work["year"] == target_year, "_year_order"] = 0

    work = work.sort_values(
        [*group_columns, "_period", "_year_order"],
        kind="stable",
    )
    chosen = work.drop_duplicates(list(group_columns), keep="first").copy()
    chosen["timing"] = chosen["year"].map(
        lambda year: timing_label(int(year), target_year)
    )
    return chosen.drop(columns=["_period", "_year_order"])


def load_hdi(
    path: Path,
    *,
    target_year: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    hdi = pd.read_csv(
        path,
        dtype={"Entity": "string", "Code": "string"},
        na_values=NA_VALUES,
        low_memory=False,
    )
    require_columns(
        hdi,
        ["Entity", "Code", "Year", HDI_VALUE_COLUMN, HDI_REGION_COLUMN],
        path,
    )

    hdi["country_code"] = normalize_country_codes(hdi["Code"])
    hdi["year"] = pd.to_numeric(hdi["Year"], errors="coerce")
    hdi["value"] = pd.to_numeric(hdi[HDI_VALUE_COLUMN], errors="coerce")

    region_map = (
        hdi.dropna(subset=["country_code", HDI_REGION_COLUMN])
        .groupby("country_code", sort=False)[HDI_REGION_COLUMN]
        .agg(first_nonmissing)
        .to_dict()
    )

    candidates = hdi[
        ["country_code", "Entity", "year", "value"]
    ].rename(columns={"Entity": "source_name"})
    selected = select_preferred(
        candidates,
        group_columns=["country_code"],
        target_year=target_year,
    )
    selected["confounder"] = "hdi"
    selected["source"] = "HDI"
    return selected, region_map


def load_wdi(path: Path, *, target_year: int) -> pd.DataFrame:
    wdi = pd.read_csv(
        path,
        dtype={
            "Country Name": "string",
            "Country Code": "string",
            "Series Name": "string",
            "Series Code": "string",
        },
        na_values=NA_VALUES,
        low_memory=False,
    )
    require_columns(
        wdi,
        ["Country Name", "Country Code", "Series Name", "Series Code"],
        path,
    )

    year_columns: dict[str, int] = {}
    for column in wdi.columns:
        match = YEAR_COLUMN_RE.fullmatch(str(column).strip())
        if match:
            year_columns[column] = int(match.group(1))
    if not year_columns:
        raise ValueError(f"No WDI year columns were found in {path}")

    wdi["Series Code"] = wdi["Series Code"].astype("string").str.strip()
    wdi = wdi[wdi["Series Code"].isin(WDI_CODE_TO_OUTPUT)].copy()

    missing_series = sorted(set(WDI_CODE_TO_OUTPUT) - set(wdi["Series Code"].dropna()))
    if missing_series:
        raise ValueError(f"{path} is missing required WDI series: {missing_series}")

    wdi["country_code"] = normalize_country_codes(wdi["Country Code"])
    long = wdi.melt(
        id_vars=[
            "country_code",
            "Country Name",
            "Series Name",
            "Series Code",
        ],
        value_vars=list(year_columns),
        var_name="year_column",
        value_name="value",
    )
    long["year"] = long["year_column"].map(year_columns)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["confounder"] = long["Series Code"].map(WDI_CODE_TO_OUTPUT)
    long = long.rename(columns={"Country Name": "source_name"})

    selected = select_preferred(
        long[
            [
                "country_code",
                "confounder",
                "source_name",
                "Series Name",
                "Series Code",
                "year",
                "value",
            ]
        ],
        group_columns=["country_code", "confounder"],
        target_year=target_year,
    )
    selected["source"] = "WDI"
    return selected


def load_existing_region_map(metadata_path: Path) -> dict[str, object]:
    if not metadata_path.exists():
        return {}

    metadata = pd.read_csv(
        metadata_path,
        dtype={"country_code": "string"},
        na_values=NA_VALUES,
        low_memory=False,
    )
    if not {"country_code", "region"}.issubset(metadata.columns):
        return {}

    metadata["_code"] = normalize_country_codes(metadata["country_code"])
    metadata = metadata.dropna(subset=["_code", "region"])
    return (
        metadata.groupby("_code", sort=False)["region"]
        .agg(first_nonmissing)
        .to_dict()
    )


def representative_label(values: pd.Series) -> object:
    """Choose a deterministic label: most frequent, then first observed."""
    labels = values.astype("string").str.strip()
    labels = labels[labels.notna() & labels.ne("")]
    if labels.empty:
        return pd.NA

    counts = labels.value_counts(sort=False)
    maximum = int(counts.max())
    tied = set(counts[counts.eq(maximum)].index.tolist())
    for label in labels:
        if label in tied:
            return label
    raise AssertionError("Could not select a representative country label")


def build_country_identity_table(extended: pd.DataFrame) -> pd.DataFrame:
    """Build one metadata identity per ISO code without altering STEX rows.

    STEX can contain multiple historical or spelling labels attached to the
    same usable ISO-3 code. All such observations still receive the same
    country-level indicators through the ISO key. Metadata needs only one
    display identity per code, so conflicting labels are resolved
    deterministically instead of aborting the rebuild.
    """
    columns = ["country", "country_name", "country_code", "_country_code_key"]
    countries = extended.loc[extended["_country_code_key"].notna(), columns].copy()

    conflict_rows: list[dict[str, object]] = []
    for code, group in countries.groupby("_country_code_key", sort=False):
        for column in ("country", "country_name"):
            labels = group[column].astype("string").str.strip()
            labels = labels[labels.notna() & labels.ne("")].drop_duplicates()
            if len(labels) > 1:
                conflict_rows.append(
                    {
                        "country_code": code,
                        "field": column,
                        "labels": " | ".join(labels.astype(str).tolist()),
                    }
                )

    if conflict_rows:
        conflicts = pd.DataFrame(conflict_rows)
        print(
            "WARNING: STEX contains multiple display labels for "
            f"{conflicts['country_code'].nunique():,} usable ISO-3 code(s). "
            "Indicators will still be joined by ISO code. Metadata will use "
            "the most frequent label, with first appearance breaking ties."
        )
        print(conflicts.head(20).to_string(index=False))

    identities = (
        countries.groupby("_country_code_key", sort=False, as_index=False)
        .agg(
            country=("country", representative_label),
            country_name=("country_name", representative_label),
        )
    )
    identities["country_code"] = identities["_country_code_key"]
    return identities.loc[
        :, ["country", "country_name", "country_code", "_country_code_key"]
    ]


def build_metadata(
    countries: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    region_map: dict[str, object],
) -> pd.DataFrame:
    if selected.duplicated(["country_code", "confounder"]).any():
        duplicates = selected.loc[
            selected.duplicated(["country_code", "confounder"], keep=False),
            ["country_code", "confounder", "year", "value"],
        ]
        raise ValueError(
            "Duplicate selected country/indicator rows:\n"
            + duplicates.head(20).to_string(index=False)
        )

    lookup = selected.set_index(["country_code", "confounder"])
    parts: list[pd.DataFrame] = []

    for confounder in INDICATOR_COLUMNS:
        part = countries.copy()
        keys = pd.MultiIndex.from_arrays(
            [
                part["_country_code_key"],
                pd.Series(confounder, index=part.index, dtype="string"),
            ],
            names=["country_code", "confounder"],
        )

        for column in ("source", "source_name", "year", "timing", "value"):
            part[column] = lookup[column].reindex(keys).to_numpy()

        part = part[part["value"].notna()].copy()
        part["confounder"] = confounder
        part["year"] = pd.array(part["year"], dtype="Int64")
        part["region"] = part["_country_code_key"].map(region_map)
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=METADATA_COLUMNS)

    metadata = pd.concat(parts, ignore_index=True)
    return metadata.loc[:, list(METADATA_COLUMNS)]


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.stem}.",
            suffix=".csv",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def backup_once(path: Path, target_year: int) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.pre_{target_year}{path.suffix}")
    if backup.exists():
        return None
    shutil.copy2(path, backup)
    return backup


def rebuild(
    *,
    data_dir: Path,
    hdi_path: Path,
    wdi_path: Path,
    target_year: int,
    make_backups: bool,
) -> None:
    extended_path = data_dir / "stex_extended.csv"
    metadata_path = data_dir / "stex_confounder_metadata.csv"

    for path in (extended_path, hdi_path, wdi_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    extended = pd.read_csv(
        extended_path,
        dtype={"country_code": "string"},
        na_values=NA_VALUES,
        low_memory=False,
    )
    require_columns(
        extended,
        ["country", "country_name", "country_code", *INDICATOR_COLUMNS],
        extended_path,
    )

    original_columns = extended.columns.tolist()
    original_row_count = len(extended)
    original_observations = (
        extended["observation"].copy() if "observation" in extended.columns else None
    )

    extended["_country_code_key"] = normalize_country_codes(extended["country_code"])
    unresolved = extended["_country_code_key"].isna()

    if unresolved.any():
        display_columns = [
            column
            for column in ("observation", "country", "country_name", "country_code")
            if column in extended.columns
        ]
        entities = extended.loc[unresolved, ["country", "country_name"]].drop_duplicates()
        print(
            f"WARNING: {int(unresolved.sum()):,} row(s) across "
            f"{len(entities):,} unresolved historical entity/entities have no "
            "usable ISO-3 code. These rows will be retained, their nine country "
            "indicators will be blank, and no successor-state mapping will be used."
        )
        print(extended.loc[unresolved, display_columns].head(20).to_string(index=False))

    countries = build_country_identity_table(extended)
    old_region_map = load_existing_region_map(metadata_path)
    hdi_selected, hdi_region_map = load_hdi(hdi_path, target_year=target_year)
    wdi_selected = load_wdi(wdi_path, target_year=target_year)
    selected = pd.concat([hdi_selected, wdi_selected], ignore_index=True, sort=False)

    region_map = dict(old_region_map)
    region_map.update(hdi_region_map)

    # This assignment intentionally blanks all nine columns first. Coded rows
    # receive selected source values; unresolved historical rows remain blank.
    for confounder in INDICATOR_COLUMNS:
        indicator_values = selected.loc[
            selected["confounder"].eq(confounder),
            ["country_code", "value"],
        ]
        if indicator_values.duplicated("country_code").any():
            raise ValueError(f"Duplicate selected values for {confounder}")
        value_map = indicator_values.set_index("country_code")["value"]
        extended[confounder] = extended["_country_code_key"].map(value_map)

    # Explicit assertion for the behavior that caused the earlier failure.
    if extended.loc[unresolved, list(INDICATOR_COLUMNS)].notna().any().any():
        raise AssertionError(
            "Rows without country codes unexpectedly received country indicators"
        )

    metadata = build_metadata(countries, selected, region_map=region_map)

    extended = extended.drop(columns="_country_code_key").loc[:, original_columns]
    if len(extended) != original_row_count:
        raise AssertionError("The rebuild changed the STEX row count")
    if extended.columns.tolist() != original_columns:
        raise AssertionError("The rebuild changed the STEX column order")
    if original_observations is not None and not extended["observation"].equals(
        original_observations
    ):
        raise AssertionError("The rebuild changed the observation order or identifiers")

    if make_backups:
        for path in (extended_path, metadata_path):
            backup = backup_once(path, target_year)
            if backup is not None:
                print(f"Backup created: {backup}")

    atomic_write_csv(extended, extended_path)
    atomic_write_csv(metadata, metadata_path)

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Wrote {extended_path} ({len(extended):,} rows)")
    print(f"Wrote {metadata_path} ({len(metadata):,} rows)")
    print(f"Target year: {target_year}")

    if not metadata.empty:
        summary = (
            metadata.groupby(["source", "timing"], dropna=False)
            .size()
            .rename("country_indicator_pairs")
            .reset_index()
        )
        print("\nSelection summary:")
        print(summary.to_string(index=False))

    coverage_rows = []
    coded_country_count = int(countries["_country_code_key"].nunique())
    for confounder in INDICATOR_COLUMNS:
        covered = int(
            metadata.loc[
                metadata["confounder"].eq(confounder), "country_code"
            ].nunique()
        )
        coverage_rows.append(
            {
                "confounder": confounder,
                "coded_countries": coded_country_count,
                "countries_with_value": covered,
                "countries_missing": coded_country_count - covered,
            }
        )
    print("\nCoverage among STEX entities with usable country codes:")
    print(pd.DataFrame(coverage_rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    default_data_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild STEX HDI/WDI confounders using exact target year, then "
            "latest pre-target, then earliest post-target values."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing stex_extended.csv and metadata.",
    )
    parser.add_argument(
        "--hdi-path",
        type=Path,
        default=None,
        help=(
            "HDI CSV path. Default: "
            "<data-dir>/unprocessed_indicators/human-development-index.csv"
        ),
    )
    parser.add_argument(
        "--wdi-path",
        type=Path,
        default=None,
        help=(
            "WDI CSV path. Default: "
            "<data-dir>/unprocessed_indicators/wdi_data.csv"
        ),
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=DEFAULT_TARGET_YEAR,
        help=f"Reference year; default is {DEFAULT_TARGET_YEAR}.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create one-time pre-target-year backup files.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=SCRIPT_VERSION,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    hdi_path = (
        args.hdi_path.expanduser().resolve()
        if args.hdi_path is not None
        else data_dir / "unprocessed_indicators" / "human-development-index.csv"
    )
    wdi_path = (
        args.wdi_path.expanduser().resolve()
        if args.wdi_path is not None
        else data_dir / "unprocessed_indicators" / "wdi_data.csv"
    )

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Data directory: {data_dir}")
    print(f"HDI input: {hdi_path}")
    print(f"WDI input: {wdi_path}")
    print(f"Target year: {args.target_year}")

    rebuild(
        data_dir=data_dir,
        hdi_path=hdi_path,
        wdi_path=wdi_path,
        target_year=args.target_year,
        make_backups=not args.no_backup,
    )


if __name__ == "__main__":
    main()