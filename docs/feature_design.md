# Feature Engineering Design for Regime Identification

This document outlines the pseudocode and feature plan for constructing **Level**, **Trend**, and **Volatility** features for regime modeling. These engineered features form the input to PCA, clustering (K-Means / HMM) and similarity-based regime identification.

---

## 1. Feature Definitions

### Level

* **Definition**: Normalized current value of a macro variable.
* **Formula**:
  $$\text{Level}_{t} = \frac{x_t - \mu_{1:t-1}}{\sigma_{1:t-1}}$$
  where mean and std are expanding-window statistics up to time (t-1).
* **Interpretation**: Shows whether the current value is unusually high or low relative to history.

### Trend

* **Definition**: Recent direction of a macro variable over the past N months.
* **Options**:

  * **Slope**: OLS slope of variable over past 12 months.
  * **Momentum**: $( (x_t/x_{t-12}) - 1 )$ for ratio-like variables, or $( x_t - x_{t-12} )$ for level variables.
* **Interpretation**: Captures whether the variable is trending upward or downward.

### Volatility

* **Definition**: Historical variability of monthly changes over past N months.
* **Formula**:
  $ \text{Vol}_t = \text{std}(\ x_{t-N+1}, ..., \ x_t) $
* **Interpretation**: Identifies whether a variable is stable or turbulent.

---

## 2. Pseudocode

```python
# input: pandas Series with DateTimeIndex at monthly frequency

# level
def compute_level(series):
    winsorize series at [1%, 99%]
    mean = expanding_mean(series).shift(1)
    std  = expanding_std(series).shift(1)
    level = (series - mean) / std
    return level

# trend
def compute_trend(series, method="slope"):
    winsorize series
    if method == "slope":
        trend = rolling_OLS_slope(series, window=12)
    elif method == "momentum":
        trend = (series / series.shift(12)) - 1
    return expanding_zscore(trend)

# volatility
def compute_volatility(series):
    winsorize series
    vol = rolling_std(series, window=12)
    return expanding_zscore(vol)

# high-level feature matrix
def build_feature_matrix(df):
    for each variable in df:
        level = compute_level(variable)
        trend = compute_trend(variable)
        vol   = compute_volatility(variable)
        concat into feature set
    return feature_matrix
```

---

## 3. Feature Plan

* **Input Variables**: Macro & financial indicators (inflation, interest rates, spreads, GDP growth, volatility, etc.).
* **Transformations**: Winsorization, expanding-window normalization to avoid lookahead bias.
* **Final Feature Matrix**: Wide dataframe with all Level/Trend/Vol columns, ready for:

  1. PCA (dimension reduction)
  2. K-Means clustering (regime taxonomy)
  3. Euclidean similarity (local analogues)
  4. HMM (regime persistence)

---
