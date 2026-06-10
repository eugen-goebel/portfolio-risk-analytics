"""Streamlit dashboard for stored market data and portfolio risk metrics.

Run locally with:
    uv run streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from analytics.loader import load_close_frame, load_close_series
from analytics.metrics import (
    annualized_volatility,
    correlation_matrix,
    drawdown_series,
    max_drawdown,
    portfolio_returns,
    sharpe_ratio,
    summarize,
)
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

tab_asset, tab_portfolio = st.tabs(["Single Asset", "Portfolio"])


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
        st.stop()

    st.caption("Raw weights are normalized so they sum to one.")
    cols = st.columns(len(chosen))
    raw_weights = {
        s: cols[i].number_input(f"{s}", min_value=0.0, value=1.0, step=0.1, key=f"w_{s}")
        for i, s in enumerate(chosen)
    }
    total_weight = sum(raw_weights.values())
    if total_weight == 0:
        st.warning("At least one weight has to be above zero.")
        st.stop()
    weights = {s: w / total_weight for s, w in raw_weights.items()}

    frame = load_close_frame(db, chosen)
    returns = portfolio_returns(frame, weights)
    portfolio_value = (1 + returns).cumprod() * 100

    p1, p2, p3 = st.columns(3)
    p1.metric("Volatility (ann.)", f"{annualized_volatility(returns) * 100:.1f}%")
    p2.metric("Sharpe ratio", f"{sharpe_ratio(returns, 0.03):.2f}")
    p3.metric("Max drawdown", f"{max_drawdown(portfolio_value) * 100:.1f}%")

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
