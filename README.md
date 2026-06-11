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

Without network access, `uv run main.py ingest --demo demo-a demo-b` generates deterministic demo data.

## Dashboard

```bash
uv run streamlit run app.py
```

The dashboard has two views. Single Asset shows the headline metrics, the price history and the running drawdown for one symbol. Portfolio takes a set of assets with weights, normalizes them, and shows portfolio volatility, Sharpe ratio, max drawdown, the cumulative value curve and a correlation heatmap. With an empty database it offers to load demo data.

## REST API

```bash
uv run uvicorn api.main:app --reload
```

| Endpoint | Description |
|----------|-------------|
| `GET /assets` | Stored assets with their price counts |
| `GET /assets/{symbol}/prices` | Daily closing prices |
| `GET /assets/{symbol}/metrics` | Risk metrics for one asset |
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
| Correlations | Pairwise correlation of daily returns between portfolio assets |

The metric functions are tested against hand-computed values, not against their own output.

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

## Architecture

```
portfolio-risk-analytics/
├── ingestion/     # Yahoo Finance client, demo data generator, idempotent storage
├── db/            # SQLAlchemy models (assets, daily prices)
├── analytics/     # Metric functions and price loaders on pandas
├── api/           # FastAPI endpoints
├── tests/         # 33 tests, run on SQLite and PostgreSQL in CI
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
