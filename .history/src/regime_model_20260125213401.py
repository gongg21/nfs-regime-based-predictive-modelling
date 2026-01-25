"""Regime identification utilities.

This module supports three regime-identification approaches used in the
statistical model notebook:
1) Euclidean distance matrix (baseline similarity)
2) KNN with Mahalanobis distance (similarity-based regimes)
3) KNN with Correlation distance (pattern-based regimes)
4) GaussianHMM (latent regime states)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

__all__ = [
    "RegimeModelConfig",
    "load_feature_matrix",
    "euclidean_distance_matrix",
    "knn_mahalanobis",
    "knn_correlation",
    "neighbors_to_frame",
    "fit_hmm",
    "label_regimes",
    "transition_matrix",
    "nearest_regimes_by_hmm_no_gap",
]


@dataclass
class RegimeModelConfig:
    standardize: bool = True
    knn_k: int = 5
    n_components: int = 3
    covariance_type: str = "full"
    n_iter: int = 1000
    random_state: int = 42
    label_order_feature: str = "volatility_transformed"


def load_feature_matrix(path: str | Path, date_col: str = "date") -> pd.DataFrame:
    """Load the feature matrix and set a monthly DateTimeIndex."""
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()


def euclidean_distance_matrix(
    features: pd.DataFrame,
    cfg: RegimeModelConfig | None = None,
) -> pd.DataFrame:
    """Compute Euclidean distance matrix between dates."""
    cfg = cfg or RegimeModelConfig()
    X, idx = _prepare_features(features, cfg.standardize)
    dist = euclidean_distances(X)
    return pd.DataFrame(dist, index=idx, columns=idx)


def _prepare_features(
    features: pd.DataFrame,
    standardize: bool,
) -> Tuple[np.ndarray, pd.Index]:
    if features.empty:
        raise ValueError("features is empty")
    X = features.to_numpy(dtype=float)
    if standardize:
        X = StandardScaler().fit_transform(X)
    return X, features.index


def knn_mahalanobis(
    features: pd.DataFrame,
    cfg: RegimeModelConfig | None = None,
) -> Tuple[NearestNeighbors, np.ndarray, np.ndarray]:
    """Fit KNN model using Mahalanobis distance."""
    cfg = cfg or RegimeModelConfig()
    X, _ = _prepare_features(features, cfg.standardize)
    VI = np.linalg.pinv(np.cov(X, rowvar=False))
    model = NearestNeighbors(metric="mahalanobis", metric_params={"VI": VI})
    model.fit(X)
    distances, indices = model.kneighbors(X, n_neighbors=cfg.knn_k + 1)
    distances, indices = _drop_self_neighbor(distances, indices)
    return model, distances, indices


def knn_correlation(
    features: pd.DataFrame,
    cfg: RegimeModelConfig | None = None,
) -> Tuple[NearestNeighbors, np.ndarray, np.ndarray]:
    """Fit KNN model using Correlation distance."""
    cfg = cfg or RegimeModelConfig()
    X, _ = _prepare_features(features, cfg.standardize)
    model = NearestNeighbors(metric="correlation")
    model.fit(X)
    distances, indices = model.kneighbors(X, n_neighbors=cfg.knn_k + 1)
    distances, indices = _drop_self_neighbor(distances, indices)
    return model, distances, indices


def _drop_self_neighbor(
    distances: np.ndarray,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove the self neighbor (distance 0 at index 0)."""
    if distances.shape[1] <= 1:
        return distances, indices
    return distances[:, 1:], indices[:, 1:]


def neighbors_to_frame(
    indices: np.ndarray,
    features_index: pd.Index,
    k: int | None = None,
) -> pd.DataFrame:
    """Convert neighbor index array to a DataFrame of neighbor dates."""
    k = k or indices.shape[1]
    k = min(k, indices.shape[1])
    neighbor_dates = features_index.values[indices[:, :k]]
    columns = [f"neighbor_{i+1}" for i in range(k)]
    return pd.DataFrame(neighbor_dates, index=features_index, columns=columns)


def fit_hmm(
    features: pd.DataFrame,
    cfg: RegimeModelConfig | None = None,
) -> Tuple[GaussianHMM, np.ndarray, np.ndarray]:
    """Fit a GaussianHMM and return model, scaled features, and regimes."""
    cfg = cfg or RegimeModelConfig()
    X, _ = _prepare_features(features, cfg.standardize)
    model = GaussianHMM(
        n_components=cfg.n_components,
        covariance_type=cfg.covariance_type,
        n_iter=cfg.n_iter,
        random_state=cfg.random_state,
    )
    model.fit(X)
    regimes = model.predict(X)
    return model, X, regimes


def _pick_order_feature(features: pd.DataFrame, preferred: str) -> str:
    if preferred in features.columns:
        return preferred
    # Try to find volatility column with new naming convention
    for col in features.columns:
        if "volatility" in col.lower():
            return col
    return features.columns[0]


def label_regimes(
    features: pd.DataFrame,
    regimes: np.ndarray,
    cfg: RegimeModelConfig | None = None,
) -> Tuple[pd.Series, Dict[int, str], str]:
    """Map raw regimes to ordered labels based on a chosen feature."""
    cfg = cfg or RegimeModelConfig()
    feature = _pick_order_feature(features, cfg.label_order_feature)
    means = pd.Series(regimes, index=features.index).groupby(regimes).apply(
        lambda idx: features.loc[idx.index, feature].mean()
    )
    order = means.sort_values().index.tolist()
    if cfg.n_components == 3:
        names = ["Low", "Mid", "High"]
    else:
        names = [f"Regime {i+1}" for i in range(cfg.n_components)]
    mapping = {state: name for state, name in zip(order, names)}
    labels = pd.Series(regimes, index=features.index).map(mapping)
    return labels.rename("regime_label"), mapping, feature


def transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """Compute transition probability matrix for regime labels."""
    if not isinstance(regimes, pd.Series):
        regimes = pd.Series(regimes)
    next_regime = regimes.shift(-1)
    transitions = pd.crosstab(regimes, next_regime, normalize="index")
    return transitions.fillna(0.0)


def nearest_regimes_by_hmm_no_gap(
    regime_df: pd.DataFrame,
    date: pd.Timestamp,
    k: int = 5,
    exclude_self: bool = True,
) -> pd.Series:
    """Find nearest regimes by L2 distance in HMM probability space."""
    prob_cols = [c for c in regime_df.columns if c.startswith("regime_")]
    if not prob_cols:
        raise ValueError("regime_df must contain columns named 'regime_*'")
    p0 = regime_df.loc[date, prob_cols].astype(float).values
    P = regime_df[prob_cols].astype(float).values
    d = np.linalg.norm(P - p0, axis=1)
    dist = pd.Series(d, index=regime_df.index)
    if exclude_self and date in dist.index:
        dist = dist.drop(date)
    return dist.sort_values().head(k)
