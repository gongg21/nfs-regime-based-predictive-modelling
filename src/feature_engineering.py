# Level/Trend/Volatility computation

"""
Feature engineering utilities for regime modeling.

Implements Level / Trend / Volatility features for a panel of monthly
macroeconomic time series, with leakage-safe scaling and sensible defaults.

Design Goals
- Level: normalized current value (expanding-window z-score)
- Trend: recent direction over past N months (rolling slope or 12m momentum)
- Volatility: historical variability over past N months (rolling std)

All functions accept a pandas Series indexed by a DateTimeIndex at monthly
frequency (end-of-month recommended).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Tuple
import numpy as np
import pandas as pd

__all__ = [
    "FeatureConfig",
    "compute_level",
    "compute_trend",
    "compute_volatility",
    "build_feature_matrix",
]

# ------------------------------
# Config dataclass
# ------------------------------

@dataclass
class FeatureConfig:
    trend_window: int = 12            # months
    vol_window: int = 12              # months
    min_periods: int = 12             # minimum months to compute features
    winsorize_lower: float = 0.01     # 1st percentile
    winsorize_upper: float = 0.99     # 99th percentile
    standardize: bool = True          # expanding-window z-score for outputs
    trend_method: Literal["slope", "mom12"] = "slope"  # OLS slope or 12m momentum


# ------------------------------
# Helpers
# ------------------------------

def _validate_monthly_index(idx: pd.Index) -> None:
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Index must be a pandas DateTimeIndex at monthly frequency.")
    # Tolerate end-of-month irregularities; warn if freq not set
    # Users can set df = df.asfreq('M') upstream if needed.


def _winsorize(s: pd.Series, lower: float, upper: float) -> pd.Series:
    """Winsorize a Series by clipping to quantiles [lower, upper]."""
    if s.empty:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def _expanding_zscore(s: pd.Series, min_periods: int = 12) -> pd.Series:
    """Leakage-safe expanding-window z-score: (x - mean_t-1)/std_t-1.

    We use shift(1) so that at time t, the normalization uses information
    available up to t-1.
    """
    mean = s.expanding(min_periods=min_periods).mean().shift(1)
    std = s.expanding(min_periods=min_periods).std(ddof=0).shift(1)
    z = (s - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope of y on time over the specified window.

    Returns the slope per month. Time regressor is 0,1,...,window-1.
    """
    if len(y) < window:
        return pd.Series(index=y.index, dtype=float)

    x = np.arange(window)
    x = (x - x.mean()) / x.std(ddof=0)  # normalize x for stability

    def _fit(arr: np.ndarray) -> float:
        # slope = cov(x,y)/var(x) with normalized x -> var(x)=1
        return float(np.cov(x, arr, ddof=0)[0, 1])

    return y.rolling(window=window, min_periods=window).apply(_fit, raw=True)


# ------------------------------
# Core feature computations
# ------------------------------

def compute_level(series: pd.Series, cfg: FeatureConfig) -> pd.Series:
    """Level feature: normalized current value (expanding z-score).

    Steps:
    1) Winsorize raw series (optional via cfg quantiles)
    2) Expanding-window z-score with shift to avoid look-ahead
    """
    _validate_monthly_index(series.index)
    s = series.copy().astype(float)
    s = _winsorize(s, cfg.winsorize_lower, cfg.winsorize_upper)
    level = _expanding_zscore(s, min_periods=cfg.min_periods) if cfg.standardize else s
    return level.rename((series.name or "var") + "__level")


def compute_trend(series: pd.Series, cfg: FeatureConfig) -> pd.Series:
    """Trend feature: recent direction over the past N months.

    Methods supported:
    - "slope": rolling OLS slope of winsorized series (leakage-safe z-score optional)
    - "mom12": 12-month momentum = (x_t / x_{t-12} - 1) for ratio-like series,
               or (x_t - x_{t-12}) for level-like series when negatives present.

    Output is optionally standardized via expanding-window z-score.
    """
    _validate_monthly_index(series.index)
    s = series.copy().astype(float)
    s = _winsorize(s, cfg.winsorize_lower, cfg.winsorize_upper)

    if cfg.trend_method == "slope":
        raw_trend = _rolling_slope(s, window=cfg.trend_window)
    elif cfg.trend_method == "mom12":
        lag = s.shift(cfg.trend_window)
        # Use returns when values are non-negative; otherwise differences
        if (s.dropna() >= 0).all() and (lag.dropna() >= 0).all():
            raw_trend = (s / lag) - 1.0
        else:
            raw_trend = s - lag
    else:
        raise ValueError("Unsupported trend_method: %s" % cfg.trend_method)

    trend = _expanding_zscore(raw_trend, min_periods=cfg.min_periods) if cfg.standardize else raw_trend
    return trend.rename((series.name or "var") + "__trend")


def compute_volatility(series: pd.Series, cfg: FeatureConfig) -> pd.Series:
    """Volatility feature: rolling std of monthly changes over past N months.

    We first compute monthly changes (diff), then rolling std. Finally, we
    optionally standardize with expanding-window z-score.
    """
    _validate_monthly_index(series.index)
    s = series.copy().astype(float)
    s = _winsorize(s, cfg.winsorize_lower, cfg.winsorize_upper)
    monthly_changes = s.diff()
    raw_vol = monthly_changes.rolling(window=cfg.vol_window, min_periods=cfg.vol_window).std(ddof=0)
    vol = _expanding_zscore(raw_vol, min_periods=cfg.min_periods) if cfg.standardize else raw_vol
    return vol.rename((series.name or "var") + "__vol")


# ------------------------------
# High-level API
# ------------------------------

def build_feature_matrix(
    data: pd.DataFrame,
    cfg: FeatureConfig | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build Level/Trend/Vol features for each input column.

    Parameters
    ----------
    data : DataFrame
        Columns are macro variables; index is monthly DateTimeIndex.
    cfg : FeatureConfig
        Hyperparameters for feature construction.
    dropna : bool
        If True, drop rows with any NA at the end.

    Returns
    -------
    DataFrame of engineered features with multi-variable columns like
    `<var>__level`, `<var>__trend`, `<var>__vol`.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        return pd.DataFrame(index=data.index)

    _validate_monthly_index(data.index)
    cfg = cfg or FeatureConfig()

    features = []
    for col in data.columns:
        s = data[col]
        lvl = compute_level(s, cfg)
        trd = compute_trend(s, cfg)
        vol = compute_volatility(s, cfg)
        features.append(pd.concat([lvl, trd, vol], axis=1))

    out = pd.concat(features, axis=1)
    return out.dropna() if dropna else out


# ------------------------------
# Example usage (for quick testing)
# ------------------------------
if __name__ == "__main__":
    idx = pd.date_range("1990-01-31", periods=240, freq="M")
    gdp = pd.Series(np.cumsum(np.random.randn(len(idx))) + 100, index=idx, name="gdp")
    cpi = pd.Series(np.cumsum(np.random.randn(len(idx))) + 50, index=idx, name="cpi")

    df = pd.concat([gdp, cpi], axis=1)
    cfg = FeatureConfig(trend_method="slope")

    X = build_feature_matrix(df, cfg)
    print(X.tail())
