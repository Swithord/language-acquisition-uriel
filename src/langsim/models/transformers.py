from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class ModalityPCATransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        families: Mapping[str, Sequence[str]],
        min_nonmissing: int = 3,
    ):
        self.families = families
        self.min_nonmissing = min_nonmissing

    def fit(self, x: pd.DataFrame, y=None):
        if not isinstance(x, pd.DataFrame):
            raise TypeError("ModalityPCATransformer expects a pandas DataFrame.")

        self.blocks_ = []
        self.feature_names_out_ = []

        for family, cols in self.families.items():
            available = [
                col
                for col in cols
                if col in x.columns and pd.to_numeric(x[col], errors="coerce").notna().sum() >= self.min_nonmissing
            ]

            if not available:
                continue

            raw = x[available].apply(pd.to_numeric, errors="coerce")

            if len(available) == 1:
                imputer = SimpleImputer(strategy="mean")
                scaler = StandardScaler()
                values = scaler.fit_transform(imputer.fit_transform(raw))

                self.blocks_.append(
                    {
                        "family": family,
                        "columns": available,
                        "imputer": imputer,
                        "scaler": scaler,
                        "pca": None,
                        "sign": 1.0,
                    }
                )
            else:
                imputer = SimpleImputer(strategy="mean")
                scaler = StandardScaler()
                pca = PCA(n_components=1)

                values = scaler.fit_transform(imputer.fit_transform(raw))
                score = pca.fit_transform(values).ravel()
                mean_distance = np.nanmean(values, axis=1)

                corr = np.corrcoef(score, mean_distance)[0, 1]
                sign = -1.0 if np.isfinite(corr) and corr < 0 else 1.0

                self.blocks_.append(
                    {
                        "family": family,
                        "columns": available,
                        "imputer": imputer,
                        "scaler": scaler,
                        "pca": pca,
                        "sign": sign,
                    }
                )

            self.feature_names_out_.append(f"modality_{family}_pc1")

        return self

    def transform(self, x: pd.DataFrame):
        if not isinstance(x, pd.DataFrame):
            raise TypeError("ModalityPCATransformer expects a pandas DataFrame.")

        scores = []

        for block in self.blocks_:
            raw = x[block["columns"]].apply(pd.to_numeric, errors="coerce")
            values = block["scaler"].transform(block["imputer"].transform(raw))

            if block["pca"] is None:
                score = values[:, 0]
            else:
                score = block["pca"].transform(values).ravel() * block["sign"]

            scores.append(score)

        if not scores:
            return np.empty((x.shape[0], 0))

        return np.column_stack(scores)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)