from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from langsim.utils.io import ensure_dir
from langsim.utils.preprocess import (
    available_columns,
    columns_ordered_by_family,
    family_boundaries,
)


def _first_existing_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    return None


def _safe_filename(value: str) -> str:
    out = str(value).strip()
    out = out.replace(" ", "_")
    out = out.replace("/", "_")
    out = out.replace("\\", "_")
    out = out.replace(":", "_")
    out = out.replace(";", "_")
    out = out.replace(",", "_")
    return out.lower()


def _savefig(fig: plt.Figure, outpath: str | Path, dpi: int = 200) -> None:
    outpath = Path(outpath)
    ensure_dir(outpath.parent)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def _plot_family_separators(
    ax: plt.Axes,
    ordered_cols: Sequence[str],
    families: Mapping[str, Sequence[str]] | None,
) -> None:
    for boundary in family_boundaries(ordered_cols, families):
        ax.axhline(boundary - 0.5, linewidth=0.8)


def _plot_family_separators_2d(
    ax: plt.Axes,
    ordered_cols: Sequence[str],
    families: Mapping[str, Sequence[str]] | None,
) -> None:
    for boundary in family_boundaries(ordered_cols, families):
        ax.axhline(boundary - 0.5, linewidth=0.8)
        ax.axvline(boundary - 0.5, linewidth=0.8)


def _confidence_intervals_from_se(
    estimate: pd.Series,
    se: pd.Series,
    multiplier: float = 1.96,
) -> tuple[pd.Series, pd.Series]:
    ci_low = estimate - multiplier * se
    ci_high = estimate + multiplier * se
    return ci_low, ci_high


def _errorbar_xerr(
    estimate: pd.Series,
    ci_low: pd.Series,
    ci_high: pd.Series,
) -> list[pd.Series]:
    xerr_low = np.maximum(estimate - ci_low, 0)
    xerr_high = np.maximum(ci_high - estimate, 0)
    return [xerr_low, xerr_high]


def plot_distance_correlation_heatmap(
    df: pd.DataFrame,
    distance_cols: Sequence[str],
    outpath: str | Path,
    families: Mapping[str, Sequence[str]] | None = None,
    title: str = "Spearman correlation among distance measures",
) -> None:
    distance_cols = columns_ordered_by_family(
        available_columns(df, distance_cols),
        families=families,
    )

    if len(distance_cols) == 0:
        return

    corr = df[distance_cols].apply(pd.to_numeric, errors="coerce").corr(
        method="spearman",
        min_periods=3,
    )

    fig_width = max(8, len(distance_cols) * 0.35)
    fig_height = max(6, len(distance_cols) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(corr.to_numpy(), aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(distance_cols)))
    ax.set_yticks(np.arange(len(distance_cols)))
    ax.set_xticklabels(distance_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(distance_cols, fontsize=7)
    ax.set_title(title)

    _plot_family_separators_2d(ax, distance_cols, families)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _savefig(fig, outpath)


def plot_distance_correlation_heatmaps_by_scope(
    df: pd.DataFrame,
    distance_cols: Sequence[str],
    dataset_col: str,
    outdir: str | Path,
    families: Mapping[str, Sequence[str]] | None = None,
) -> None:
    outdir = Path(outdir)
    ensure_dir(outdir)

    plot_distance_correlation_heatmap(
        df=df,
        distance_cols=distance_cols,
        outpath=outdir / "distance_spearman_heatmap_pooled.png",
        families=families,
        title="Spearman correlation among distance measures, pooled",
    )

    if dataset_col not in df.columns:
        return

    for dataset, sub in df.groupby(dataset_col, dropna=False):
        label = str(dataset)

        plot_distance_correlation_heatmap(
            df=sub,
            distance_cols=distance_cols,
            outpath=outdir / f"distance_spearman_heatmap_{_safe_filename(label)}.png",
            families=families,
            title=f"Spearman correlation among distance measures, {label}",
        )


def plot_single_effects_forest(
    effects: pd.DataFrame,
    outpath: str | Path,
    scope: str = "pooled",
    families: Mapping[str, Sequence[str]] | None = None,
    title: str | None = None,
    order_by: str = "beta",
    ascending: bool = True,
) -> None:
    required = ["scope", "measure", "beta", "ci_low", "ci_high"]

    if any(col not in effects.columns for col in required):
        return

    sub = effects[effects["scope"].astype(str) == str(scope)].copy()
    sub = sub.dropna(subset=["beta", "ci_low", "ci_high"])

    if sub.empty:
        return

    if order_by in sub.columns:
        sub = sub.sort_values(order_by, ascending=ascending)
    else:
        sub = sub.sort_values("beta", ascending=ascending)

    y_pos = np.arange(sub.shape[0])

    fig, ax = plt.subplots(figsize=(9, max(6, sub.shape[0] * 0.28)))

    ax.errorbar(
        x=sub["beta"],
        y=y_pos,
        xerr=_errorbar_xerr(sub["beta"], sub["ci_low"], sub["ci_high"]),
        fmt="o",
        capsize=2,
    )

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["measure"], fontsize=7)
    ax.set_xlabel("Standardised coefficient")

    if title is None:
        title = f"Single-distance effects, {scope}"

    ax.set_title(title)

    _savefig(fig, outpath)


def plot_single_effects_forests_by_scope(
    effects: pd.DataFrame,
    outdir: str | Path,
    families: Mapping[str, Sequence[str]] | None = None,
    scope_col: str = "scope",
) -> None:
    outdir = Path(outdir)
    ensure_dir(outdir)

    if scope_col not in effects.columns:
        return

    scopes = effects[scope_col].dropna().astype(str).unique()

    for scope in scopes:
        plot_single_effects_forest(
            effects=effects,
            outpath=outdir / f"single_effects_forest_{_safe_filename(scope)}.png",
            scope=scope,
            families=families,
            title=f"Single-distance effects, {scope}",
        )


def plot_single_effects_by_scope_forest(
    effects: pd.DataFrame,
    outpath: str | Path,
    families: Mapping[str, Sequence[str]] | None = None,
    scopes: Sequence[str] | None = None,
    title: str = "Single-distance effects by dataset",
    ascending: bool = True,
) -> None:
    required = ["scope", "measure", "beta", "ci_low", "ci_high"]

    if any(col not in effects.columns for col in required):
        return

    sub = effects.dropna(subset=["beta", "ci_low", "ci_high"]).copy()
    sub["scope"] = sub["scope"].astype(str)
    sub["measure"] = sub["measure"].astype(str)

    if scopes is not None:
        scopes = [str(scope) for scope in scopes]
        sub = sub[sub["scope"].isin(scopes)]

    if sub.empty:
        return

    available_scopes = list(sub["scope"].drop_duplicates())

    if "pooled" in available_scopes:
        scope_order = ["pooled"] + [
            scope for scope in available_scopes if scope != "pooled"
        ]
    else:
        scope_order = available_scopes

    if "pooled" in sub["scope"].unique():
        order_df = (
            sub[sub["scope"] == "pooled"]
            .loc[:, ["measure", "beta"]]
            .drop_duplicates(subset=["measure"])
            .sort_values("beta", ascending=ascending)
        )
        ordered_measures = order_df["measure"].tolist()
    else:
        order_df = (
            sub.groupby("measure", dropna=False)["beta"]
            .mean()
            .reset_index()
            .sort_values("beta", ascending=ascending)
        )
        ordered_measures = order_df["measure"].tolist()

    remaining_measures = [
        measure
        for measure in sub["measure"].drop_duplicates().tolist()
        if measure not in ordered_measures
    ]
    ordered_measures = ordered_measures + remaining_measures

    measure_to_y = {
        measure: idx for idx, measure in enumerate(ordered_measures)
    }

    offsets = np.linspace(-0.25, 0.25, max(len(scope_order), 1))

    fig, ax = plt.subplots(figsize=(10, max(6, len(ordered_measures) * 0.30)))

    for scope_idx, scope in enumerate(scope_order):
        scope_sub = sub[sub["scope"] == scope].copy()
        scope_sub = scope_sub[scope_sub["measure"].isin(measure_to_y)]

        if scope_sub.empty:
            continue

        y_pos = (
            scope_sub["measure"]
            .map(measure_to_y)
            .astype(float)
            + offsets[scope_idx]
        )

        ax.errorbar(
            x=scope_sub["beta"],
            y=y_pos,
            xerr=_errorbar_xerr(
                scope_sub["beta"],
                scope_sub["ci_low"],
                scope_sub["ci_high"],
            ),
            fmt="o",
            capsize=2,
            label=scope,
        )

    ax.axvline(0, linewidth=1)
    ax.set_yticks(np.arange(len(ordered_measures)))
    ax.set_yticklabels(ordered_measures, fontsize=7)
    ax.set_xlabel("Standardised coefficient")
    ax.set_title(title)
    ax.legend(fontsize=7)

    _savefig(fig, outpath)


def plot_family_joint_effects(
    family_effects: pd.DataFrame,
    outpath: str | Path,
    metric: str = "partial_r2",
) -> None:
    required = ["family", metric]

    if any(col not in family_effects.columns for col in required):
        return

    sub = family_effects.dropna(subset=[metric]).copy()

    if sub.empty:
        return

    sub = sub.sort_values(metric)

    fig, ax = plt.subplots(figsize=(8, max(4, sub.shape[0] * 0.45)))
    ax.barh(sub["family"], sub[metric])
    ax.set_xlabel(metric)
    ax.set_title(f"Family joint effects: {metric}")

    _savefig(fig, outpath)


def plot_family_direction_coherence(
    family_effects: pd.DataFrame,
    outpath: str | Path,
) -> None:
    metric = "direction_coherence_negative"
    required = ["family", metric]

    if any(col not in family_effects.columns for col in required):
        return

    sub = family_effects.dropna(subset=[metric]).copy()

    if sub.empty:
        return

    sub = sub.sort_values(metric)

    fig, ax = plt.subplots(figsize=(8, max(4, sub.shape[0] * 0.45)))
    ax.barh(sub["family"], sub[metric])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of coefficients with expected negative sign")
    ax.set_title("Direction coherence by family")

    _savefig(fig, outpath)


def plot_agreement_summary(
    agreement: pd.DataFrame,
    outpath: str | Path,
    scope: str = "pooled",
    metric: str = "mean_abs_spearman",
) -> None:
    required = ["scope", "family", metric]

    if any(col not in agreement.columns for col in required):
        return

    sub = agreement[agreement["scope"].astype(str) == str(scope)].copy()
    sub = sub.dropna(subset=[metric])

    if sub.empty:
        return

    sub = sub.sort_values(metric)

    fig, ax = plt.subplots(figsize=(8, max(4, sub.shape[0] * 0.45)))
    ax.barh(sub["family"], sub[metric])
    ax.set_xlabel(metric)
    ax.set_title(f"Agreement by family, {scope}")

    _savefig(fig, outpath)


def plot_agreement_scatter(
    agreement: pd.DataFrame,
    outpath: str | Path,
    scope: str = "pooled",
) -> None:
    required = ["scope", "family", "mean_abs_spearman", "pc1_variance_explained"]

    if any(col not in agreement.columns for col in required):
        return

    sub = agreement[agreement["scope"].astype(str) == str(scope)].copy()
    sub = sub.dropna(subset=["mean_abs_spearman", "pc1_variance_explained"])

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub["mean_abs_spearman"], sub["pc1_variance_explained"])

    for _, row in sub.iterrows():
        ax.annotate(
            str(row["family"]),
            (row["mean_abs_spearman"], row["pc1_variance_explained"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Mean absolute within-family Spearman")
    ax.set_ylabel("PC1 variance explained")
    ax.set_title(f"Internal agreement summaries, {scope}")

    _savefig(fig, outpath)


def plot_modality_effects_forest(
    modality_effects: pd.DataFrame,
    outpath: str | Path,
    title: str = "Independent modality effects",
) -> None:
    required = ["modality_score", "beta", "ci_low", "ci_high"]

    if any(col not in modality_effects.columns for col in required):
        return

    sub = modality_effects.dropna(subset=["beta", "ci_low", "ci_high"]).copy()

    if sub.empty:
        return

    sub = sub.sort_values("beta")
    y_pos = np.arange(sub.shape[0])

    fig, ax = plt.subplots(figsize=(8, max(4, sub.shape[0] * 0.45)))

    ax.errorbar(
        x=sub["beta"],
        y=y_pos,
        xerr=_errorbar_xerr(sub["beta"], sub["ci_low"], sub["ci_high"]),
        fmt="o",
        capsize=2,
    )

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["modality_score"], fontsize=8)
    ax.set_xlabel("Standardised coefficient")
    ax.set_title(title)

    _savefig(fig, outpath)


def plot_modality_drop_one_r2(
    modality_effects: pd.DataFrame,
    outpath: str | Path,
) -> None:
    metric = "drop_one_partial_r2"
    required = ["modality_score", metric]

    if any(col not in modality_effects.columns for col in required):
        return

    sub = modality_effects.dropna(subset=[metric]).copy()

    if sub.empty:
        return

    sub = sub.sort_values(metric)

    fig, ax = plt.subplots(figsize=(8, max(4, sub.shape[0] * 0.45)))
    ax.barh(sub["modality_score"], sub[metric])
    ax.set_xlabel("Drop-one partial $R^2$")
    ax.set_title("Independent explanatory contribution by modality")

    _savefig(fig, outpath)


def plot_prediction_summary(
    summary: pd.DataFrame,
    outpath: str | Path,
    metric: str = "r2_mean",
) -> None:
    model_col = _first_existing_column(summary, ["model", "extended_model"])

    if model_col is None or metric not in summary.columns:
        return

    sub = summary.dropna(subset=[metric]).copy()

    if sub.empty:
        return

    sub = sub.sort_values(metric)

    fig, ax = plt.subplots(figsize=(8, max(4, sub.shape[0] * 0.45)))

    std_col = metric.replace("_mean", "_std") if metric.endswith("_mean") else None

    if std_col is not None and std_col in sub.columns:
        ax.barh(sub[model_col], sub[metric], xerr=sub[std_col])
    else:
        ax.barh(sub[model_col], sub[metric])

    ax.axvline(0, linewidth=1)
    ax.set_xlabel(metric)
    ax.set_title("Predictive performance")

    _savefig(fig, outpath)


def plot_prediction_metrics(
    summary: pd.DataFrame,
    outdir: str | Path,
    metrics: Sequence[str] = ("rmse_mean", "mae_mean", "r2_mean", "spearman_mean"),
) -> None:
    outdir = Path(outdir)
    ensure_dir(outdir)

    for metric in metrics:
        if metric in summary.columns:
            plot_prediction_summary(
                summary=summary,
                outpath=outdir / f"predictive_{_safe_filename(metric)}.png",
                metric=metric,
            )


def plot_prediction_per_fold(
    per_fold: pd.DataFrame,
    outpath: str | Path,
    metric: str = "r2",
) -> None:
    model_col = _first_existing_column(per_fold, ["model", "extended_model"])
    required = ["fold", metric]

    if model_col is None or any(col not in per_fold.columns for col in required):
        return

    sub = per_fold.dropna(subset=[metric]).copy()

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for model, model_sub in sub.groupby(model_col, dropna=False):
        model_sub = model_sub.sort_values("fold")
        ax.plot(
            model_sub["fold"],
            model_sub[metric],
            marker="o",
            label=str(model),
        )

    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric)
    ax.set_title(f"Per-fold predictive performance: {metric}")
    ax.legend(fontsize=7)

    _savefig(fig, outpath)


def plot_unadjusted_vs_adjusted_effects(
    attenuation: pd.DataFrame,
    outpath: str | Path,
    label_top_n: int = 8,
) -> None:
    required = ["measure", "beta_unadjusted", "beta_adjusted"]

    if any(col not in attenuation.columns for col in required):
        return

    sub = attenuation.dropna(subset=["beta_unadjusted", "beta_adjusted"]).copy()

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(sub["beta_unadjusted"], sub["beta_adjusted"])
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    min_val = np.nanmin(
        [sub["beta_unadjusted"].min(), sub["beta_adjusted"].min()]
    )
    max_val = np.nanmax(
        [sub["beta_unadjusted"].max(), sub["beta_adjusted"].max()]
    )
    ax.plot([min_val, max_val], [min_val, max_val], linewidth=1)

    if "partial_r2_unadjusted" in sub.columns:
        label_sub = (
            sub.sort_values("partial_r2_unadjusted", ascending=False)
            .head(label_top_n)
        )
    else:
        label_sub = (
            sub.reindex(sub["beta_unadjusted"].abs().sort_values(ascending=False).index)
            .head(label_top_n)
        )

    for _, row in label_sub.iterrows():
        ax.annotate(
            str(row["measure"]),
            (row["beta_unadjusted"], row["beta_adjusted"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("Unadjusted beta")
    ax.set_ylabel("Adjusted beta")
    ax.set_title("Unadjusted versus adjusted distance effects")

    _savefig(fig, outpath)


def plot_attenuation_summary(
    attenuation: pd.DataFrame,
    outpath: str | Path,
    top_n: int | None = 20,
    sort_by: str = "partial_r2_unadjusted",
) -> None:
    required = ["measure", "attenuation"]

    if any(col not in attenuation.columns for col in required):
        return

    sub = attenuation.dropna(subset=["attenuation"]).copy()

    if sub.empty:
        return

    if sort_by in sub.columns:
        sub = sub.sort_values(sort_by, ascending=False)
    else:
        sub = sub.sort_values("attenuation", ascending=False)

    if top_n is not None:
        sub = sub.head(top_n)

    sub = sub.sort_values("attenuation")

    fig, ax = plt.subplots(figsize=(9, max(5, sub.shape[0] * 0.32)))
    ax.barh(sub["measure"], sub["attenuation"])
    ax.axvline(0, linewidth=1)
    ax.axvline(1, linewidth=1)
    ax.set_xlabel("Attenuation")
    ax.set_title("Effect attenuation after STEX adjustment")

    _savefig(fig, outpath)


def plot_adjusted_effects_forest(
    attenuation: pd.DataFrame,
    outpath: str | Path,
    top_n: int | None = None,
    sort_by: str = "partial_r2_adjusted",
    title: str = "Adjusted STEX distance effects",
) -> None:
    required = ["measure", "beta_adjusted", "se_adjusted"]

    if any(col not in attenuation.columns for col in required):
        return

    sub = attenuation.dropna(subset=["beta_adjusted", "se_adjusted"]).copy()

    if sub.empty:
        return

    sub["ci_low"], sub["ci_high"] = _confidence_intervals_from_se(
        sub["beta_adjusted"],
        sub["se_adjusted"],
    )

    if sort_by in sub.columns:
        sub = sub.sort_values(sort_by, ascending=False)
    else:
        sub = sub.reindex(
            sub["beta_adjusted"].abs().sort_values(ascending=False).index
        )

    if top_n is not None:
        sub = sub.head(top_n)

    sub = sub.sort_values("beta_adjusted")
    y_pos = np.arange(sub.shape[0])

    fig, ax = plt.subplots(figsize=(9, max(5, sub.shape[0] * 0.32)))

    ax.errorbar(
        x=sub["beta_adjusted"],
        y=y_pos,
        xerr=_errorbar_xerr(sub["beta_adjusted"], sub["ci_low"], sub["ci_high"]),
        fmt="o",
        capsize=2,
    )

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["measure"], fontsize=7)
    ax.set_xlabel("Adjusted beta")
    ax.set_title(title)

    _savefig(fig, outpath)


def make_core_plots(
    df: pd.DataFrame,
    distance_cols: Sequence[str],
    single_effects: pd.DataFrame,
    family_effects: pd.DataFrame,
    modality_effects: pd.DataFrame,
    predictive_summary: pd.DataFrame,
    predictive_per_fold: pd.DataFrame,
    outdir: str | Path,
    dataset_col: str = "dataset",
    families: Mapping[str, Sequence[str]] | None = None,
    agreement: pd.DataFrame | None = None,
) -> None:
    outdir = Path(outdir)
    ensure_dir(outdir)

    plot_distance_correlation_heatmap(
        df=df,
        distance_cols=distance_cols,
        outpath=outdir / "distance_spearman_heatmap.png",
        families=families,
    )

    plot_distance_correlation_heatmaps_by_scope(
        df=df,
        distance_cols=distance_cols,
        dataset_col=dataset_col,
        outdir=outdir / "distance_heatmaps_by_scope",
        families=families,
    )

    plot_single_effects_forest(
        effects=single_effects,
        outpath=outdir / "single_effects_forest_pooled.png",
        scope="pooled",
        families=families,
    )

    plot_single_effects_forests_by_scope(
        effects=single_effects,
        outdir=outdir / "single_effects_forests_by_scope",
        families=families,
    )

    plot_single_effects_by_scope_forest(
        effects=single_effects,
        outpath=outdir / "single_effects_forest_by_dataset.png",
        families=families,
    )

    plot_family_joint_effects(
        family_effects=family_effects,
        outpath=outdir / "family_joint_partial_r2.png",
        metric="partial_r2",
    )

    plot_family_direction_coherence(
        family_effects=family_effects,
        outpath=outdir / "family_direction_coherence.png",
    )

    plot_modality_effects_forest(
        modality_effects=modality_effects,
        outpath=outdir / "modality_effects_forest.png",
    )

    plot_modality_drop_one_r2(
        modality_effects=modality_effects,
        outpath=outdir / "modality_drop_one_partial_r2.png",
    )

    plot_prediction_metrics(
        summary=predictive_summary,
        outdir=outdir / "prediction_metrics",
    )

    plot_prediction_per_fold(
        per_fold=predictive_per_fold,
        outpath=outdir / "predictive_per_fold_r2.png",
        metric="r2",
    )

    plot_prediction_per_fold(
        per_fold=predictive_per_fold,
        outpath=outdir / "predictive_per_fold_spearman.png",
        metric="spearman",
    )

    if agreement is not None:
        plot_agreement_summary(
            agreement=agreement,
            outpath=outdir / "agreement_mean_abs_spearman.png",
            scope="pooled",
            metric="mean_abs_spearman",
        )

        plot_agreement_summary(
            agreement=agreement,
            outpath=outdir / "agreement_pc1_variance.png",
            scope="pooled",
            metric="pc1_variance_explained",
        )

        plot_agreement_scatter(
            agreement=agreement,
            outpath=outdir / "agreement_scatter.png",
            scope="pooled",
        )


def make_extended_plots(
    attenuation: pd.DataFrame,
    predictive_summary: pd.DataFrame,
    predictive_per_fold: pd.DataFrame,
    outdir: str | Path,
) -> None:
    outdir = Path(outdir)
    ensure_dir(outdir)

    plot_unadjusted_vs_adjusted_effects(
        attenuation=attenuation,
        outpath=outdir / "unadjusted_vs_adjusted_betas.png",
    )

    plot_attenuation_summary(
        attenuation=attenuation,
        outpath=outdir / "attenuation_top_unadjusted_partial_r2.png",
        top_n=20,
        sort_by="partial_r2_unadjusted",
    )

    plot_adjusted_effects_forest(
        attenuation=attenuation,
        outpath=outdir / "adjusted_effects_forest_top_partial_r2.png",
        top_n=20,
        sort_by="partial_r2_adjusted",
    )

    plot_prediction_metrics(
        summary=predictive_summary,
        outdir=outdir / "prediction_metrics",
    )

    plot_prediction_per_fold(
        per_fold=predictive_per_fold,
        outpath=outdir / "predictive_per_fold_r2.png",
        metric="r2",
    )

    plot_prediction_per_fold(
        per_fold=predictive_per_fold,
        outpath=outdir / "predictive_per_fold_spearman.png",
        metric="spearman",
    )