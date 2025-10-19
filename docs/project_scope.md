Project Scope & Success Metrics
1. Project Title
Regime-Based Predictive Modeling of Equity Factor Returns

2. Project Scope
Objective
To investigate whether market regimes, defined by combinations of macroeconomic, financial, and sentiment indicators, can be used to predict equity factor returns and construct a dynamic long–short factor portfolio that outperforms a static benchmark.
Core Research Questions
Do distinct macro-financial regimes exist that systematically affect factor performance (e.g., Value, Momentum, Size, Market)?


Can regime similarity to historical periods improve forecasts of next-period factor returns?


Does a regime-aware factor allocation approach produce higher risk-adjusted returns compared to static equal-weight or momentum-weighted factor portfolios?


Scope of Work
Component
Description
Macroeconomic Data Selection/ Market Sentiment 
Identify and source key macro variables (inflation, interest rate, credit spread, GDP growth, volatility index, sentiment index) that influence factor performance.
Feature Engineering
Construct interpretable features (Level, Trend, Volatility) to capture dynamic macroeconomic behavior.
Regime Identification Model
Develop methods to define and detect market regimes based on macro features using similarity-based or unsupervised learning approaches (Euclidean distance, PCA, K-Means, or HMM).
Factor Return Modeling
Use regime information to forecast next-period factor returns and determine long/short positions per factor.
Portfolio Construction
Construct a market-neutral portfolio based on predicted factor performance, optionally weighting by confidence or volatility.
Performance Evaluation
Evaluate predictive power and portfolio efficiency across regimes using multiple financial and statistical metrics.

Inclusions
Historical monthly data on macroeconomic variables and equity factors .


Development of an interpretable regime identification algorithm.


A GitHub codebase with reproducible pipeline (data → regime → prediction → evaluation).


Quantitative backtesting framework for factor portfolio performance.


Exclusions
No deep-learning or black-box models (focus on interpretability).


No high-frequency or intraday data.


No trading cost modeling in the first phase (to be added later if time allows).
