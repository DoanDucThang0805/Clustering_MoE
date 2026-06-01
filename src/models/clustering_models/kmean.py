from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted


class KMeansClustering:
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        n_init: int = 20,
        random_state: int = 42,
    ):
        self.model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
    

    def fit(self, features: np.ndarray) -> "KMeansClustering":
        self.model.fit(features)
        return self


    def fit_predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return self.model.fit_predict(features)


    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(features)


    def transform(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        self._check_fitted()
        return self.model.transform(features)


    @property
    def centroids(self) -> np.ndarray:
        self._check_fitted()
        return self.model.cluster_centers_


    @property
    def labels(self) -> np.ndarray:
        self._check_fitted()
        return self.model.labels_


    def save(self, path: str | Path) -> None:
        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(self.model, path)


    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "KMeansClustering":

        model: KMeans = joblib.load(path)

        instance = cls(
            n_clusters=model.n_clusters,
            max_iter=model.max_iter,
            n_init=model.n_init,
            random_state=model.random_state,
        )

        instance.model = model
        return instance


    def _check_fitted(self) -> None:
        check_is_fitted(self.model)
