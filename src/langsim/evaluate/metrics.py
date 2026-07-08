from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)

    mask = y_true.notna() & y_pred.notna()

    if mask.sum() == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "spearman": np.nan,
        }

    yt = y_true[mask].to_numpy()
    yp = y_pred[mask].to_numpy()

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))

    if len(np.unique(yt)) > 1:
        r2 = float(r2_score(yt, yp))
    else:
        r2 = np.nan

    if len(yt) >= 3 and len(np.unique(yt)) > 1 and len(np.unique(yp)) > 1:
        rho = spearmanr(yt, yp).correlation
        spearman = float(rho) if np.isfinite(rho) else np.nan
    else:
        spearman = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman": spearman,
    }