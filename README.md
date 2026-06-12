# Portfolio Risk Analytics

![Tests](https://github.com/eugen-goebel/portfolio-risk-analytics/actions/workflows/tests.yml/badge.svg)
![Data pipeline](https://github.com/eugen-goebel/portfolio-risk-analytics/actions/workflows/pipeline-health.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Market data platform that ingests real daily prices, stores them in SQL and computes portfolio risk metrics: volatility, Sharpe ratio, maximum drawdown and asset correlations.

Prices come from the public Yahoo Finance chart endpoint with no API key. Metrics are computed on adjusted closes, so dividends and splits are accounted for. The same code runs on SQLite for local work and PostgreSQL in production, and the CI suite runs against both.

## Quick Start

```bash
git clone https://github.com/eugen-goebel/portfolio-risk-analytics.git
cd portfolio-risk-analytics

# Install dependencies (https://docs.astral.sh/uv/)
uv sync

# Load five years of daily prices
uv run main.py ingest SPY AAPL MSFT

# Risk metrics for one symbol
uv run main.py metrics SPY --risk-free-rate 0.03
```

```
Symbol:                SPY
Observations:          1256
Total return:          +83.71%
Annualized volatility: 17.10%
Sharpe ratio:          0.624
Max drawdown:          -24.50%
```

Hourly intraday prices are loaded with `uv run main.py ingest-intraday SPY --interval 1h` and feed the daily realized volatility served at `GET /assets/{symbol}/realized-volatility`.

Without network access, `uv run main.py ingest --demo demo-a demo-b` generates deterministic demo data.

### Exchange rates

```bash
uv run main.py ingest-fx USD GBP
```

Daily reference rates come from the official ECB data API, also without a key, and are stored as EURUSD and EURGBP so the metrics and forecast commands work on them directly.

## Dashboard

```bash
uv run streamlit run app.py
```

The dashboard has four views. Single Asset shows the headline metrics, the price history and the running drawdown for one symbol. Portfolio takes a set of assets with weights, normalizes them, and shows portfolio volatility, Sharpe ratio, max drawdown, the cumulative value curve and a correlation heatmap. Optimization draws a cloud of random long-only portfolios in risk-return space, colored by Sharpe ratio, and marks the closed-form minimum variance and maximum Sharpe portfolios with their weight tables. Model Monitor compares the three volatility forecasters in a walk-forward test, with next-day volatility per model and an error chart, and checks the recent return distribution for drift against the stored history. With an empty database it offers to load demo data.

## Factsheets

The report command turns a stored symbol into a one-page PDF factsheet: the headline risk metrics as a table, plus the price history and the running drawdown as charts. The default output path is `{symbol}-factsheet.pdf`.

```bash
uv run main.py report SPY --output spy-factsheet.pdf --risk-free-rate 0.03
```

## REST API

```bash
uv run uvicorn api.main:app --reload
```

| Endpoint | Description |
|----------|-------------|
| `GET /assets` | Stored assets with their price counts |
| `GET /assets/{symbol}/prices` | Daily closing prices |
| `GET /assets/{symbol}/metrics` | Risk metrics for one asset |
| `GET /assets/{symbol}/realized-volatility` | Daily realized volatility from stored intraday prices |
| `POST /portfolio/metrics` | Metrics for a weighted portfolio, including the correlation matrix |

Portfolio request body:

```json
{
  "weights": {"SPY": 0.6, "AAPL": 0.4},
  "risk_free_rate": 0.03
}
```

Interactive documentation is served at `/docs`.

## Docker

```bash
docker compose up
```

This starts PostgreSQL, the API on port 8000 and the dashboard on port 8501, all wired together. Load data into the running stack with:

```bash
docker compose exec api python main.py ingest SPY AAPL MSFT
```

CI builds the stack on every change, seeds demo data through the API container and checks both services before merging is allowed.

## Metrics

| Metric | Definition |
|--------|------------|
| Total return | Price change from the first to the last observation |
| Annualized volatility | Sample standard deviation of daily returns, scaled by the square root of 252 |
| Sharpe ratio | Annualized excess return over the risk free rate, divided by annualized volatility |
| Max drawdown | Largest peak-to-trough decline of the price series |
| VaR 95% | Empirical 5% quantile of daily returns, sign-flipped: the loss that only the worst 5% of days exceeded |
| Expected shortfall 95% | Mean daily loss on the days at or beyond the VaR threshold |
| Correlations | Pairwise correlation of daily returns between portfolio assets |
| Beta | Covariance of asset and benchmark daily returns, divided by the benchmark variance |
| Alpha | Annualized CAPM alpha: the return left after subtracting what beta exposure to the benchmark explains |
| Tracking error | Annualized standard deviation of the daily active returns, asset minus benchmark |
| Information ratio | Annualized mean active return divided by the tracking error |

The benchmark-relative metrics are served at `GET /assets/{symbol}/benchmark` and through `uv run main.py benchmark AAPL --benchmark SPY`. The metric functions are tested against hand-computed values, not against their own output.

## VaR validation

The Kupiec proportion-of-failures test checks whether the VaR model breaches as often as its confidence level promises, at 95% roughly one day in twenty should lose more than the VaR. The backtest walks forward over the stored history, estimates the VaR at each day from the trailing window only, and compares the observed breach count against the promised rate with a likelihood ratio test. A p-value below 0.05 rejects the VaR model, whether it breached too often or suspiciously rarely, also served at `GET /assets/{symbol}/var-validation`.

```bash
uv run main.py var-test SPY --window 250 --confidence 0.95
```

## Monte Carlo simulation

The simulate command resamples an asset's own daily return history to generate thousands of possible value paths, each starting at 100, and summarizes where they end: the percentile range of final values, the probability of ending below the start, and the expected final value. The honest caveat is part of the method, resampling history assumes the future resembles the past, so the ranges describe a market like the stored sample and say nothing about events it never contained. A normal method that draws from a fitted normal distribution is available for comparison, and the report is also served at `GET /assets/{symbol}/montecarlo`.

```bash
uv run main.py simulate SPY --horizon 252 --paths 2000 --method bootstrap --seed 7
```

## Backtesting

The backtest simulates the same target weights twice over the stored history: once bought and held, where the weights drift with performance, and once rebalanced back to target on the first trading day of each month or quarter. Both value paths are summarized with the usual risk metrics, so the comparison shows directly what rebalancing discipline costs or earns against buy and hold.

```
POST /portfolio/backtest
```

```json
{
  "weights": {"SPY": 0.6, "AAPL": 0.4},
  "rebalance": "monthly",
  "risk_free_rate": 0.03
}
```

## Portfolio optimization

Closed-form Markowitz mean-variance optimization on the stored history: the global minimum variance portfolio and the maximum Sharpe (tangency) portfolio, computed directly from the annualized sample moments with plain linear algebra and no numerical optimizer. The solutions are the textbook unconstrained ones, so negative weights, meaning short positions, are allowed.

```
POST /portfolio/optimize
```

```json
{
  "symbols": ["SPY", "AAPL", "MSFT"],
  "risk_free_rate": 0.03
}
```

## Volatility forecasting

Daily returns are close to unpredictable, their volatility is not: turbulent days cluster. The forecast module compares three one-day-ahead volatility forecasters in a walk-forward test where no model ever sees the future.

| Model | Idea |
|-------|------|
| rolling | Trailing 22-day mean squared return, the naive baseline |
| ewma | RiskMetrics exponentially weighted recursion (decay 0.94) |
| har | Heterogeneous autoregressive regression on daily, weekly and monthly squared returns, fit by least squares |

```bash
uv run main.py forecast SPY
```

```
Walk-forward test over the last 250 trading days

model         MAE %   RMSE %  next-day vol (ann.) %
rolling      0.4265   0.5464                  13.07
ewma         0.4527   0.5641                  14.09
har          0.5282   0.6193                  17.99

Lowest RMSE: rolling
```

The comparison is also served at `GET /assets/{symbol}/forecast`. Whichever model wins depends on the market regime, in calm stretches the simple baseline is hard to beat, which is exactly what the honest test shows.

## Drift monitoring

A forecast fitted on a long reference window assumes that recent returns still look like that history, so the drift module compares the last 60 returns against the 500 before them. The population stability index measures how the share of observations per reference quantile bin has shifted, the Kolmogorov-Smirnov statistic is the largest distance between the two empirical distribution functions. A PSI above 0.2 or a KS statistic above 0.15 flags drift at the conventional industry thresholds, also served at `GET /assets/{symbol}/drift`.

```bash
uv run main.py drift SPY
```

## Architecture

```
portfolio-risk-analytics/
├── ingestion/     # Yahoo Finance and ECB clients, demo data generator, idempotent storage
├── db/            # SQLAlchemy models (assets, daily and intraday prices)
├── analytics/     # Metric functions and price loaders on pandas
├── api/           # FastAPI endpoints
├── reporting/     # One-page PDF factsheets (matplotlib + fpdf2)
├── tests/         # 158 tests, run on SQLite and PostgreSQL in CI
└── main.py        # CLI for ingestion and quick metric checks
```

The database connection is configured through `DATABASE_URL`. It defaults to a local SQLite file, PostgreSQL works without code changes:

```bash
export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/market
```

## Testing

```bash
uv run pytest -v
```

CI runs Ruff, mypy, the test suite with a coverage floor on Python 3.12 and 3.13, the same suite against a real PostgreSQL service container, and CodeQL scanning.

## License

MIT
