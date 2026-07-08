from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from langsim.utils.io import ensure_dir


def plot_distance_correlation_heatmap(
    df: pd.DataFrame,
    distance_cols: list[str],
    outpath: str | Path,
) -> None:
    outpath = Path(outpath)
    ensure_dir(outpath.parent)

    corr = df[distance_cols].apply(pd.to_numeric, errors="coerce").corr(
        method="spearman",
        min_periods=3,
    )

    fig, ax = plt.subplots(figsize=(max(8, len(distance_cols) * 0.35), max(6, len(distance_cols) * 0.35)))
    image = ax.imshow(corr.to_numpy(), aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(distance_cols)))
    ax.set_yticks(np.arange(len(distance_cols)))
    ax.set_xticklabels(distance_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(distance_cols, fontsize=7)

    ax.set_title("Spearman correlation among distance measures")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_single_effects_forest(
    effects: pd.DataFrame,
    outpath: str | Path,
    scope: str = "pooled",
) -> None:
    outpath = Path(outpath)
    ensure_dir(outpath.parent)

    sub = effects[effects["scope"] == scope].copy()
    sub = sub.dropna(subset=["beta", "ci_low", "ci_high"])

    if sub.empty:
        return

    sub = sub.sort_values("beta")
    y_pos = np.arange(sub.shape[0])

    fig, ax = plt.subplots(figsize=(9, max(6, sub.shape[0] * 0.28)))

    ax.errorbar(
        x=sub["beta"],
        y=y_pos,
        xerr=[
            sub["beta"] - sub["ci_low"],
            sub["ci_high"] - sub["beta"],
        ],
        fmt="o",
        capsize=2,
    )

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["measure"], fontsize=7)
    ax.set_xlabel("Standardised coefficient")
    ax.set_title(f"Single-distance effects, scope={scope}")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_prediction_summary(
    summary: pd.DataFrame,
    outpath: str | Path,
    metric: str = "r2_mean",
) -> None:
    outpath = Path(outpath)
    ensure_dir(outpath.parent)

    if metric not in summary.columns:
        return

    sub = summary.sort_values(metric).copy()

    fig, ax = plt.subplots(figsize=(8, max(4, sub.shape[0] * 0.45)))
    ax.barh(sub["model"], sub[metric])
    ax.set_xlabel(metric)
    ax.set_title("Predictive performance")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)