"""REST API for stored market data and portfolio risk metrics.

Run locally with:
    uv run uvicorn api.main:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from analytics.backtest import BacktestReport, run_backtest
from analytics.drift import DriftReport, evaluate_drift
from analytics.forecast import ForecastReport, evaluate_models
from analytics.loader import load_close_frame, load_close_series, load_intraday_closes
from analytics.metrics import (
    BenchmarkReport,
    MetricsSummary,
    compare_to_benchmark,
    correlation_matrix,
    daily_returns,
    expected_shortfall,
    historical_var,
    portfolio_returns,
    summarize,
)
from analytics.metrics import annualized_volatility as ann_vol
from analytics.metrics import max_drawdown as mdd
from analytics.metrics import sharpe_ratio as sharpe
from analytics.montecarlo import MonteCarloReport, run_monte_carlo
from analytics.optimize import OptimizationReport, optimize_portfolio
from analytics.realized import annualized_realized_volatility_pct
from analytics.var_validation import VarValidationReport, validate_var
from db.database import get_db, init_db
from db.models import Asset, DailyPrice


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(
    title="Portfolio Risk Analytics",
    description="Market data and portfolio risk metrics on daily prices",
    lifespan=lifespan,
)


class AssetOut(BaseModel):
    symbol: str
    name: str
    price_count: int


class PricePoint(BaseModel):
    day: str
    close: float


class RealizedVolPoint(BaseModel):
    day: str
    realized_vol_pct: float


class PortfolioRequest(BaseModel):
    weights: dict[str, float] = Field(description="Symbol to weight, weights sum to 1")
    risk_free_rate: float = 0.0


class BacktestRequest(BaseModel):
    weights: dict[str, float] = Field(description="Symbol to weight, weights sum to 1")
    rebalance: str = "monthly"
    risk_free_rate: float = 0.0


class OptimizeRequest(BaseModel):
    symbols: list[str] = Field(description="Symbols to include in the optimization")
    risk_free_rate: float = 0.0


class PortfolioMetrics(BaseModel):
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    var_95_pct: float
    expected_shortfall_95_pct: float
    correlations: dict[str, dict[str, float]]


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    return {"status": "running", "docs": "/docs"}


@app.get("/assets", response_model=list[AssetOut], tags=["Assets"])
def list_assets(db: Session = Depends(get_db)) -> list[AssetOut]:
    rows = db.execute(
        select(Asset.symbol, Asset.name, func.count(DailyPrice.id))
        .outerjoin(DailyPrice)
        .group_by(Asset.id)
        .order_by(Asset.symbol)
    ).all()
    return [AssetOut(symbol=r[0], name=r[1], price_count=r[2]) for r in rows]


@app.get("/assets/{symbol}/prices", response_model=list[PricePoint], tags=["Assets"])
def get_prices(symbol: str, limit: int = 250, db: Session = Depends(get_db)) -> list[PricePoint]:
    try:
        series = load_close_series(db, symbol)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    tail = series.tail(limit)
    return [PricePoint(day=str(d.date()), close=float(v)) for d, v in tail.items()]


@app.get(
    "/assets/{symbol}/realized-volatility",
    response_model=list[RealizedVolPoint],
    tags=["Metrics"],
)
def get_realized_volatility(symbol: str, db: Session = Depends(get_db)) -> list[RealizedVolPoint]:
    try:
        closes = load_intraday_closes(db, symbol)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    series = annualized_realized_volatility_pct(closes)
    return [
        RealizedVolPoint(day=str(d.date()), realized_vol_pct=round(float(v), 2))
        for d, v in series.items()
    ]


@app.get("/assets/{symbol}/metrics", response_model=MetricsSummary, tags=["Metrics"])
def get_metrics(
    symbol: str, risk_free_rate: float = 0.0, db: Session = Depends(get_db)
) -> MetricsSummary:
    try:
        series = load_close_series(db, symbol)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return summarize(symbol, series, risk_free_rate)


@app.get("/assets/{symbol}/forecast", response_model=ForecastReport, tags=["Metrics"])
def get_forecast(
    symbol: str, test_size: int = 250, db: Session = Depends(get_db)
) -> ForecastReport:
    try:
        series = load_close_series(db, symbol)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return evaluate_models(symbol, daily_returns(series), test_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/assets/{symbol}/montecarlo", response_model=MonteCarloReport, tags=["Metrics"])
def get_monte_carlo(
    symbol: str,
    horizon: int = 252,
    n_paths: int = 2000,
    method: str = "bootstrap",
    seed: int | None = None,
    db: Session = Depends(get_db),
) -> MonteCarloReport:
    try:
        series = load_close_series(db, symbol)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return run_monte_carlo(symbol, daily_returns(series), horizon, n_paths, method, seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/assets/{symbol}/var-validation", response_model=VarValidationReport, tags=["Metrics"])
def get_var_validation(
    symbol: str, window: int = 250, confidence: float = 0.95, db: Session = Depends(get_db)
) -> VarValidationReport:
    try:
        series = load_close_series(db, symbol)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return validate_var(symbol, daily_returns(series), window, confidence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/assets/{symbol}/drift", response_model=DriftReport, tags=["Metrics"])
def get_drift(
    symbol: str, reference_size: int = 500, recent_size: int = 60, db: Session = Depends(get_db)
) -> DriftReport:
    try:
        series = load_close_series(db, symbol)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return evaluate_drift(symbol, daily_returns(series), reference_size, recent_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/assets/{symbol}/benchmark", response_model=BenchmarkReport, tags=["Metrics"])
def get_benchmark(
    symbol: str,
    benchmark: str = "SPY",
    risk_free_rate: float = 0.0,
    db: Session = Depends(get_db),
) -> BenchmarkReport:
    try:
        series = load_close_series(db, symbol)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        benchmark_series = load_close_series(db, benchmark)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return compare_to_benchmark(symbol, series, benchmark, benchmark_series, risk_free_rate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/portfolio/metrics", response_model=PortfolioMetrics, tags=["Metrics"])
def get_portfolio_metrics(
    request: PortfolioRequest, db: Session = Depends(get_db)
) -> PortfolioMetrics:
    symbols = list(request.weights)
    try:
        frame = load_close_frame(db, symbols)
        returns = portfolio_returns(frame, request.weights)
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    portfolio_prices = (1 + returns).cumprod()
    corr = correlation_matrix(frame)
    return PortfolioMetrics(
        annualized_volatility_pct=round(ann_vol(returns) * 100, 2),
        sharpe_ratio=round(sharpe(returns, request.risk_free_rate), 3),
        max_drawdown_pct=round(mdd(portfolio_prices) * 100, 2),
        var_95_pct=round(historical_var(returns) * 100, 2),
        expected_shortfall_95_pct=round(expected_shortfall(returns) * 100, 2),
        correlations={c: {i: round(float(v), 3) for i, v in corr[c].items()} for c in corr.columns},
    )


@app.post("/portfolio/backtest", response_model=BacktestReport, tags=["Metrics"])
def run_portfolio_backtest(
    request: BacktestRequest, db: Session = Depends(get_db)
) -> BacktestReport:
    symbols = list(request.weights)
    try:
        frame = load_close_frame(db, symbols)
        return run_backtest(frame, request.weights, request.rebalance, request.risk_free_rate)
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/portfolio/optimize", response_model=OptimizationReport, tags=["Metrics"])
def run_portfolio_optimization(
    request: OptimizeRequest, db: Session = Depends(get_db)
) -> OptimizationReport:
    try:
        frame = load_close_frame(db, request.symbols)
        return optimize_portfolio(frame, request.risk_free_rate)
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
