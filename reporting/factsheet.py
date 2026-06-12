"""One-page PDF factsheets for stored assets.

A factsheet condenses what the dashboard shows into a single tangible
page, the kind of artifact a fund manager hands out: the headline risk
metrics from analytics.metrics as a table, the price history and the
running drawdown as charts. Charts are rendered with matplotlib on the
Agg backend into temporary PNG files and embedded with fpdf2.

The built-in Helvetica font only supports latin-1, so every string
drawn on the page passes through _sanitize, which replaces characters
outside that range instead of crashing on them.
"""

import os
import tempfile
from datetime import date

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from fpdf import FPDF  # noqa: E402

from analytics.metrics import drawdown_series, summarize  # noqa: E402

PRICE_COLOR = "#1f6f8b"
DRAWDOWN_COLOR = "#c0392b"
CHART_SIZE = (8.0, 3.0)
CHART_DPI = 150


def _sanitize(text: str) -> str:
    """Replace characters outside latin-1, the built-in Helvetica range."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _render_price_chart(prices: pd.Series, path: str) -> None:
    """Plot the closing prices into a PNG file."""
    fig, ax = plt.subplots(figsize=CHART_SIZE)
    ax.plot(prices.index, prices.to_numpy(), color=PRICE_COLOR, linewidth=1.3)
    ax.set_title("Price history", fontsize=11, loc="left")
    ax.grid(alpha=0.3)
    ax.margins(x=0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_DPI)
    plt.close(fig)


def _render_drawdown_chart(prices: pd.Series, path: str) -> None:
    """Plot the running drawdown in percent into a PNG file."""
    drawdowns = drawdown_series(prices) * 100.0
    fig, ax = plt.subplots(figsize=CHART_SIZE)
    ax.fill_between(drawdowns.index, drawdowns.to_numpy(), 0.0, color=DRAWDOWN_COLOR, alpha=0.35)
    ax.plot(drawdowns.index, drawdowns.to_numpy(), color=DRAWDOWN_COLOR, linewidth=0.8)
    ax.set_title("Running drawdown (%)", fontsize=11, loc="left")
    ax.grid(alpha=0.3)
    ax.margins(x=0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_DPI)
    plt.close(fig)


def generate_factsheet(
    symbol: str, prices: pd.Series, output_path: str, risk_free_rate: float = 0.0
) -> str:
    """Write a one-page PDF factsheet for one price series.

    The page holds a title, the date range covered, the metrics from
    summarize and the two charts. Returns the written path.
    """
    summary = summarize(symbol, prices, risk_free_rate)
    first_day = prices.index.min().date().isoformat()
    last_day = prices.index.max().date().isoformat()

    pdf = FPDF(orientation="portrait", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    pdf.set_font("Helvetica", style="B", size=18)
    pdf.cell(0, 10, _sanitize(f"{symbol} Risk Factsheet"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(90, 90, 90)
    subtitle = f"{first_day} to {last_day}, generated {date.today().isoformat()}"
    pdf.cell(0, 6, _sanitize(subtitle), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    rows: list[tuple[str, str]] = [
        ("Total return", f"{summary.total_return_pct:+.2f}%"),
        ("Annualized volatility", f"{summary.annualized_volatility_pct:.2f}%"),
        ("Sharpe ratio", f"{summary.sharpe_ratio:.3f}"),
        ("Max drawdown", f"{summary.max_drawdown_pct:.2f}%"),
        ("VaR 95%", f"{summary.var_95_pct:.2f}%"),
        ("Expected shortfall 95%", f"{summary.expected_shortfall_95_pct:.2f}%"),
        ("Observations", str(summary.observations)),
    ]
    label_width = pdf.epw * 0.6
    value_width = pdf.epw * 0.4
    for label, value in rows:
        pdf.set_font("Helvetica", size=10)
        pdf.cell(label_width, 7, _sanitize(label), border="B")
        pdf.set_font("Helvetica", style="B", size=10)
        pdf.cell(
            value_width, 7, _sanitize(value), border="B", align="R", new_x="LMARGIN", new_y="NEXT"
        )
    pdf.ln(6)

    with tempfile.TemporaryDirectory() as tmp_dir:
        price_png = os.path.join(tmp_dir, "price.png")
        drawdown_png = os.path.join(tmp_dir, "drawdown.png")
        _render_price_chart(prices, price_png)
        _render_drawdown_chart(prices, drawdown_png)
        pdf.image(price_png, w=pdf.epw)
        pdf.ln(3)
        pdf.image(drawdown_png, w=pdf.epw)
        pdf.output(output_path)
    return output_path
