"""Streamlit dashboard for stored market data and portfolio risk metrics.

Run locally with:
    uv run streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from analytics.drift import evaluate_drift
from analytics.forecast import evaluate_models
from analytics.loader import load_close_frame, load_close_series
from analytics.metrics import (
    TRADING_DAYS,
    annualized_volatility,
    correlation_matrix,
    daily_returns,
    drawdown_series,
    expected_shortfall,
    historical_var,
    max_drawdown,
    portfolio_returns,
    sharpe_ratio,
    summarize,
)
from analytics.optimize import OptimizationReport, optimize_portfolio
from db.database import SessionLocal, init_db
from db.models import Asset

st.set_page_config(page_title="Portfolio Risk Analytics", page_icon="📊", layout="wide")

init_db()
db = SessionLocal()


def stored_symbols() -> list[str]:
    return list(db.scalars(select(Asset.symbol).order_by(Asset.symbol)))


def seed_demo_data() -> None:
    from ingestion.demo import generate_demo_bars
    from ingestion.store import store_bars

    for symbol in ("demo-equity", "demo-bonds", "demo-gold"):
        store_bars(db, symbol, generate_demo_bars(symbol, days=750))


st.title("Portfolio Risk Analytics")

symbols = stored_symbols()
if not symbols:
    st.info(
        "No price data stored yet. Load demo data below, or ingest real prices "
        "with `uv run main.py ingest SPY AAPL MSFT`."
    )
    if st.button("Load demo data"):
        seed_demo_data()
        st.rerun()
    st.stop()

tab_asset, tab_portfolio, tab_optimize, tab_monitor = st.tabs(
    ["Single Asset", "Portfolio", "Optimization", "Model Monitor"]
)


with tab_asset:
    col_select, col_rate = st.columns([2, 1])
    symbol = col_select.selectbox("Asset", symbols)
    risk_free = col_rate.number_input(
        "Risk free rate (yearly)", min_value=0.0, max_value=0.20, value=0.03, step=0.005
    )

    prices = load_close_series(db, symbol)
    metrics = summarize(symbol, prices, risk_free)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total return", f"{metrics.total_return_pct:+.1f}%")
    m2.metric("Volatility (ann.)", f"{metrics.annualized_volatility_pct:.1f}%")
    m3.metric("Sharpe ratio", f"{metrics.sharpe_ratio:.2f}")
    m4.metric("Max drawdown", f"{metrics.max_drawdown_pct:.1f}%")

    m5, m6, _, _ = st.columns(4)
    m5.metric("VaR 95% (daily)", f"{metrics.var_95_pct:.1f}%")
    m6.metric("Expected shortfall 95%", f"{metrics.expected_shortfall_95_pct:.1f}%")

    fig_price = px.line(prices, title=f"{symbol} closing prices")
    fig_price.update_layout(showlegend=False, yaxis_title="Price", xaxis_title="")
    st.plotly_chart(fig_price, use_container_width=True)

    dd = drawdown_series(prices) * 100
    fig_dd = go.Figure(
        go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", line={"color": "#d62728"})
    )
    fig_dd.update_layout(title=f"{symbol} drawdown", yaxis_title="Drawdown %", xaxis_title="")
    st.plotly_chart(fig_dd, use_container_width=True)


with tab_portfolio:
    chosen = st.multiselect("Assets", symbols, default=symbols[: min(3, len(symbols))])
    if len(chosen) < 2:
        st.info("Pick at least two assets to analyze a portfolio.")
    else:
        st.caption("Raw weights are normalized so they sum to one.")
        cols = st.columns(len(chosen))
        raw_weights = {
            s: cols[i].number_input(f"{s}", min_value=0.0, value=1.0, step=0.1, key=f"w_{s}")
            for i, s in enumerate(chosen)
        }
        total_weight = sum(raw_weights.values())
        if total_weight == 0:
            st.warning("At least one weight has to be above zero.")
        else:
            weights = {s: w / total_weight for s, w in raw_weights.items()}

            frame = load_close_frame(db, chosen)
            returns = portfolio_returns(frame, weights)
            portfolio_value = (1 + returns).cumprod() * 100

            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("Volatility (ann.)", f"{annualized_volatility(returns) * 100:.1f}%")
            p2.metric("Sharpe ratio", f"{sharpe_ratio(returns, 0.03):.2f}")
            p3.metric("Max drawdown", f"{max_drawdown(portfolio_value) * 100:.1f}%")
            p4.metric("VaR 95% (daily)", f"{historical_var(returns) * 100:.1f}%")
            p5.metric("Expected shortfall 95%", f"{expected_shortfall(returns) * 100:.1f}%")

            fig_value = px.line(portfolio_value, title="Portfolio value (start = 100)")
            fig_value.update_layout(showlegend=False, yaxis_title="Value", xaxis_title="")
            st.plotly_chart(fig_value, use_container_width=True)

            corr = correlation_matrix(frame)
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation of daily returns",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            st.dataframe(
                pd.DataFrame({"weight": pd.Series(weights).round(3)}),
                use_container_width=False,
            )


with tab_optimize:
    col_assets, col_rf = st.columns([2, 1])
    opt_chosen = col_assets.multiselect(
        "Assets", symbols, default=symbols[: min(3, len(symbols))], key="optimize_assets"
    )
    opt_risk_free = col_rf.number_input(
        "Risk free rate (yearly)",
        min_value=0.0,
        max_value=0.20,
        value=0.03,
        step=0.005,
        key="optimize_risk_free",
    )

    if len(opt_chosen) < 2:
        st.info("Pick at least two assets to optimize a portfolio.")
    else:
        opt_frame = load_close_frame(db, opt_chosen)
        opt_returns = opt_frame.pct_change().dropna()
        mu = opt_returns.mean().to_numpy() * TRADING_DAYS
        sigma = opt_returns.cov(ddof=1).to_numpy() * TRADING_DAYS

        rng = np.random.default_rng(42)
        cloud_weights = rng.dirichlet(np.ones(len(opt_chosen)), size=2000)
        cloud_return = cloud_weights @ mu
        cloud_vol = np.sqrt(np.einsum("ij,jk,ik->i", cloud_weights, sigma, cloud_weights))
        cloud = pd.DataFrame(
            {
                "Volatility %": cloud_vol * 100,
                "Return %": cloud_return * 100,
                "Sharpe": (cloud_return - opt_risk_free) / cloud_vol,
            }
        )

        fig_cloud = px.scatter(
            cloud,
            x="Volatility %",
            y="Return %",
            color="Sharpe",
            color_continuous_scale="Viridis",
            title="Random long-only portfolios and the closed-form optima",
        )

        report: OptimizationReport | None
        try:
            report = optimize_portfolio(opt_frame, opt_risk_free)
        except ValueError as exc:
            report = None
            st.info(str(exc))
        else:
            fig_cloud.add_trace(
                go.Scatter(
                    x=[report.minimum_variance.volatility_pct],
                    y=[report.minimum_variance.expected_return_pct],
                    mode="markers",
                    name="Minimum variance",
                    marker={"symbol": "star", "size": 18, "color": "#d62728"},
                )
            )
            fig_cloud.add_trace(
                go.Scatter(
                    x=[report.maximum_sharpe.volatility_pct],
                    y=[report.maximum_sharpe.expected_return_pct],
                    mode="markers",
                    name="Maximum Sharpe",
                    marker={"symbol": "diamond", "size": 14, "color": "#1f77b4"},
                )
            )
            fig_cloud.update_layout(legend={"orientation": "h", "y": -0.2})

        st.plotly_chart(fig_cloud, use_container_width=True)

        if report is not None:
            col_mv, col_ms = st.columns(2)
            col_mv.subheader("Minimum variance")
            col_mv.dataframe(
                pd.DataFrame({"weight": pd.Series(report.minimum_variance.weights).round(3)}),
                use_container_width=False,
            )
            col_ms.subheader("Maximum Sharpe")
            col_ms.dataframe(
                pd.DataFrame({"weight": pd.Series(report.maximum_sharpe.weights).round(3)}),
                use_container_width=False,
            )

        st.caption(
            "The cloud samples long-only portfolios, while the marked closed-form "
            "solutions allow short positions, which is why they can lie outside the cloud."
        )


with tab_monitor:
    col_monitor, col_test = st.columns([2, 1])
    monitor_symbol = col_monitor.selectbox("Asset", symbols, key="monitor_asset")
    test_size = col_test.number_input("Test size (days)", min_value=30, value=250, step=10)

    monitor_prices = load_close_series(db, monitor_symbol)
    monitor_returns = daily_returns(monitor_prices)

    st.subheader("Volatility forecast")
    try:
        forecast = evaluate_models(monitor_symbol, monitor_returns, test_size)
    except ValueError as exc:
        st.info(str(exc))
    else:
        vol_cols = st.columns(3)
        for i, (model, vol) in enumerate(forecast.next_day_volatility_pct.items()):
            label = f"{model} (best)" if model == forecast.best_model else model
            vol_cols[i].metric(label, f"{vol:.1f}%")
        st.caption(
            f"Next-day annualized volatility per model, best model by RMSE "
            f"over the last {forecast.test_observations} test days."
        )

        models = [score.model for score in forecast.scores]
        fig_scores = go.Figure(
            [
                go.Bar(name="MAE", x=models, y=[score.mae_pct for score in forecast.scores]),
                go.Bar(name="RMSE", x=models, y=[score.rmse_pct for score in forecast.scores]),
            ]
        )
        fig_scores.update_layout(
            barmode="group",
            title=f"{monitor_symbol} walk-forward forecast errors",
            yaxis_title="Error (daily %)",
            xaxis_title="",
        )
        st.plotly_chart(fig_scores, use_container_width=True)
        st.caption("Errors are measured against the absolute next-day return in daily percent.")

    st.subheader("Drift monitor")
    try:
        drift = evaluate_drift(monitor_symbol, monitor_returns)
    except ValueError as exc:
        st.info(str(exc))
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("PSI", f"{drift.psi:.3f}")
        d2.metric("KS statistic", f"{drift.ks:.3f}")
        d3.metric("Mean shift", f"{drift.mean_shift:+.3f}%")
        d4.metric("Volatility ratio", f"{drift.volatility_ratio:.2f}")
        if drift.drift_detected:
            st.warning("Distribution drift detected, the forecasts deserve scrutiny.")
        else:
            st.success("No distribution drift detected")
        st.caption(
            f"Last {drift.recent_size} returns against the {drift.reference_size} before them. "
            "Drift is flagged at a PSI above 0.2 or a KS statistic above 0.15."
        )
