from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.features.feature_extraction import feature_matrix


@dataclass
class FeatureTransformer:
    explained_variance_threshold: float = 0.95

    def __post_init__(self) -> None:
        self.scaler = StandardScaler()
        self.pca = None
        self.n_components_ = None
        self.explained_variance_ratio_sum_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_explained_variance_ = None

    def fit(self, records: Sequence[dict]) -> "FeatureTransformer":
        matrix = feature_matrix(records)
        z = self.scaler.fit_transform(matrix)
        pca_full = PCA(svd_solver="full")
        pca_full.fit(z)
        cumulative = np.cumsum(pca_full.explained_variance_ratio_)
        self.explained_variance_ratio_ = pca_full.explained_variance_ratio_.copy()
        self.cumulative_explained_variance_ = cumulative.copy()
        self.n_components_ = int(np.searchsorted(cumulative, self.explained_variance_threshold) + 1)
        self.explained_variance_ratio_sum_ = float(cumulative[self.n_components_ - 1])
        self.pca = PCA(n_components=self.n_components_, whiten=True, svd_solver="full")
        self.pca.fit(z)
        return self

    def transform(self, records: Sequence[dict]) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("FeatureTransformer must be fitted before transform().")
        matrix = feature_matrix(records)
        z = self.scaler.transform(matrix)
        return self.pca.transform(z)

    def fit_transform(self, records: Sequence[dict]) -> np.ndarray:
        self.fit(records)
        return self.transform(records)
