from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from langsim.models.transformers import ModalityPCATransformer


class PredictiveModel(ABC):
    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "PredictiveModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class MeanPredictor(PredictiveModel):
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "MeanPredictor":
        self.mean_ = float(np.nanmean(y))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(x.shape[0], self.mean_, dtype=float)


class RidgePredictor(PredictiveModel):
    def __init__(
        self,
        feature_cols: Sequence[str],
        alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    ):
        self.feature_cols = list(feature_cols)
        self.alphas = list(alphas)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "RidgePredictor":
        if not self.feature_cols:
            self.model_ = MeanPredictor().fit(x, y)
            self.is_mean_ = True
            return self

        self.is_mean_ = False
        self.model_ = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=self.alphas)),
            ]
        )
        self.model_.fit(x[self.feature_cols], y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.is_mean_:
            return self.model_.predict(x)
        return self.model_.predict(x[self.feature_cols])


class BestSingleDistancePredictor(PredictiveModel):
    def __init__(
        self,
        candidate_cols: Sequence[str],
        alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    ):
        self.candidate_cols = list(candidate_cols)
        self.alphas = list(alphas)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "BestSingleDistancePredictor":
        best_col = None
        best_score = -np.inf

        for col in self.candidate_cols:
            if col not in x.columns:
                continue

            values = pd.to_numeric(x[col], errors="coerce")
            mask = values.notna() & pd.notna(y)

            if mask.sum() < 3:
                continue

            corr = spearmanr(values[mask], y[mask], nan_policy="omit").correlation
            score = abs(corr) if np.isfinite(corr) else -np.inf

            if score > best_score:
                best_col = col
                best_score = score

        self.selected_col_ = best_col

        if best_col is None:
            self.model_ = MeanPredictor().fit(x, y)
        else:
            self.model_ = RidgePredictor(
                feature_cols=[best_col],
                alphas=self.alphas,
            ).fit(x, y)

        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(x)


class ModalityRidgePredictor(PredictiveModel):
    def __init__(
        self,
        families: Mapping[str, Sequence[str]],
        alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    ):
        self.families = families
        self.alphas = list(alphas)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "ModalityRidgePredictor":
        self.model_ = Pipeline(
            steps=[
                ("modality", ModalityPCATransformer(self.families)),
                ("ridge", RidgeCV(alphas=self.alphas)),
            ]
        )

        try:
            self.model_.fit(x, y)
            self.is_mean_ = False
        except Exception:
            self.model_ = MeanPredictor().fit(x, y)
            self.is_mean_ = True

        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(x)