from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def add_constant(x: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(x, has_constant="add")


def residual_sum_of_squares(y: pd.Series, x: pd.DataFrame) -> float:
    model = sm.OLS(y, add_constant(x)).fit()
    resid = model.resid
    return float(np.sum(np.square(resid)))


def partial_r2(
    y: pd.Series,
    full_x: pd.DataFrame,
    reduced_x: pd.DataFrame,
) -> float:
    full_rss = residual_sum_of_squares(y, full_x)
    reduced_rss = residual_sum_of_squares(y, reduced_x)

    if reduced_rss <= 0 or not np.isfinite(reduced_rss):
        return np.nan

    value = 1.0 - full_rss / reduced_rss
    return float(max(value, 0.0))


def fit_ols(
    y: pd.Series,
    x: pd.DataFrame,
    cov_type: str = "HC3",
    groups: pd.Series | None = None,
):
    x = add_constant(x)

    if cov_type == "cluster":
        if groups is None:
            raise ValueError("groups must be provided for cluster-robust covariance.")

        try:
            return sm.OLS(y, x).fit(
                cov_type="cluster",
                cov_kwds={"groups": groups},
            )
        except Exception:
            return sm.OLS(y, x).fit(cov_type="HC3")

    return sm.OLS(y, x).fit(cov_type=cov_type)


def wald_joint_pvalue(result, term_names: list[str]) -> float:
    param_names = list(result.params.index)
    active_terms = [term for term in term_names if term in param_names]

    if not active_terms:
        return np.nan

    r = np.zeros((len(active_terms), len(param_names)))

    for i, term in enumerate(active_terms):
        r[i, param_names.index(term)] = 1.0

    try:
        test = result.wald_test(r, scalar=True)
        return float(test.pvalue)
    except Exception:
        return np.nan


def confidence_interval(result, term: str, alpha: float = 0.05) -> tuple[float, float]:
    if term not in result.params.index:
        return np.nan, np.nan

    ci = result.conf_int(alpha=alpha).loc[term]
    return float(ci.iloc[0]), float(ci.iloc[1])