"""
Organizational clustering via K-Means.

Groups departments by satisfaction profile to identify
archetypes and target interventions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


@dataclass
class ClusterProfile:
    """Profile for a single cluster archetype."""
    cluster_id: int
    label: str
    size: int
    centroid: dict[str, float]
    avg_satisfaction: float


@dataclass
class ClusteringResult:
    """Full clustering analysis results."""
    profiles: list[ClusterProfile]
    silhouette: float
    inertia: float
    labels: np.ndarray
    optimal_k: int


class OrganizationalClusterAnalyzer:
    """K-Means clustering of departments by satisfaction metrics.

    Parameters
    ----------
    max_k : int
        Maximum number of clusters to evaluate (default: 8).
    """

    ARCHETYPE_LABELS = {
        0: "High Performers",
        1: "Steady State",
        2: "Resource Constrained",
        3: "At Risk",
        4: "Emerging",
    }

    def __init__(self, max_k: int = 8) -> None:
        self.max_k = max_k
        self._scaler = StandardScaler()
        self._model: KMeans | None = None

    def fit(self, df: pd.DataFrame, features: list[str]) -> ClusteringResult:
        """Run clustering analysis with automatic k selection.

        Parameters
        ----------
        df : pd.DataFrame
            Department-level aggregated metrics.
        features : list[str]
            Column names to use for clustering.

        Returns
        -------
        ClusteringResult
            Cluster profiles, labels, and quality metrics.
        """
        X = self._scaler.fit_transform(df[features])
        optimal_k = self._find_optimal_k(X)

        self._model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = self._model.fit_predict(X)

        sil_score = silhouette_score(X, labels) if optimal_k > 1 else 0.0

        profiles = self._build_profiles(df, features, labels, optimal_k)

        return ClusteringResult(
            profiles=profiles,
            silhouette=round(sil_score, 4),
            inertia=round(self._model.inertia_, 2),
            labels=labels,
            optimal_k=optimal_k,
        )

    def _find_optimal_k(self, X: np.ndarray) -> int:
        """Use silhouette score to find optimal cluster count."""
        scores = {}
        for k in range(2, min(self.max_k + 1, len(X))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            scores[k] = silhouette_score(X, labels)

        optimal = max(scores, key=scores.get)
        logger.info("Optimal k=%d (silhouette=%.3f)", optimal, scores[optimal])
        return optimal

    def _build_profiles(
        self, df: pd.DataFrame, features: list[str],
        labels: np.ndarray, k: int,
    ) -> list[ClusterProfile]:
        """Build descriptive profiles for each cluster."""
        profiles = []
        for i in range(k):
            mask = labels == i
            cluster_data = df[mask]
            centroid = {f: float(cluster_data[f].mean()) for f in features}

            profiles.append(ClusterProfile(
                cluster_id=i,
                label=self.ARCHETYPE_LABELS.get(i, f"Cluster {i}"),
                size=int(mask.sum()),
                centroid=centroid,
                avg_satisfaction=float(centroid.get("csfi_score", 0)),
            ))
        return profiles
