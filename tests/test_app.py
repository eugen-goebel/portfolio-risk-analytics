"""The Streamlit dashboard renders and seeds its demo series on its own."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def app() -> AppTest:
    # The four tabs run an optimizer, a walk forward forecast and a drift
    # check on every script run, so the default timeout is far too tight.
    return AppTest.from_file(APP, default_timeout=120).run()


def test_dashboard_renders(app: AppTest) -> None:
    assert not app.exception
    assert app.title[0].value == "Portfolio Risk Analytics"


def test_demo_series_load_without_a_click(app: AppTest) -> None:
    # Regression: the dashboard used to open empty and hide its data behind a
    # "Load demo data" button that a first visitor never finds.
    assert [button.label for button in app.button] == []
    assert app.selectbox[0].options == ["demo-bonds", "demo-equity", "demo-gold"]


def test_bonds_stay_calmer_than_equity_in_the_metrics(app: AppTest) -> None:
    # The metric strip is where an unrealistic demo shows first: bonds used to
    # report equity volatility because every symbol shared one profile.
    volatility = {}
    for symbol in ("demo-bonds", "demo-equity"):
        app.selectbox[0].set_value(symbol).run()
        # The portfolio tab repeats this label, and the single asset tab
        # renders first, so read the first match rather than the last.
        shown = next(m.value for m in app.metric if m.label == "Volatility (ann.)")
        volatility[symbol] = float(shown.rstrip("%"))
    assert volatility["demo-bonds"] < volatility["demo-equity"] / 2


def test_portfolio_sharpe_follows_its_risk_free_input(app: AppTest) -> None:
    # Regression: the portfolio tab had no rate input and passed a hardcoded
    # 0.03 to sharpe_ratio, so the three tabs silently disagreed and the
    # number never moved no matter what the user set elsewhere.
    rate_inputs = [n for n in app.number_input if n.label == "Risk free rate (yearly)"]
    assert len(rate_inputs) == 3, "each tab needs its own risk free rate input"

    def portfolio_sharpe() -> float:
        # The single-asset tab renders its Sharpe first, the portfolio tab second.
        return float([m.value for m in app.metric if m.label == "Sharpe ratio"][1])

    before = portfolio_sharpe()
    app.number_input(key="portfolio_risk_free").set_value(0.15).run()
    after = portfolio_sharpe()

    # A higher risk free rate has to lower the Sharpe ratio.
    assert after < before


def test_risk_metrics_carry_an_explanation(app: AppTest) -> None:
    # The metric labels are finance jargon. A visitor without that background
    # cannot tell whether a Sharpe ratio of 0.8 is good or bad.
    explained = {m.label for m in app.metric if m.help}
    for label in (
        "Sharpe ratio",
        "Max drawdown",
        "VaR 95% (daily)",
        "Expected shortfall 95%",
        "Volatility (ann.)",
    ):
        assert label in explained, f"metric without help text: {label}"


def test_risk_free_inputs_explain_their_unit(app: AppTest) -> None:
    rate_inputs = [n for n in app.number_input if n.label == "Risk free rate (yearly)"]
    for rate_input in rate_inputs:
        assert rate_input.help, "the bare value 0.03 needs to say it means 3 percent"


def test_footer_points_at_the_portfolio(app: AppTest) -> None:
    assert any("github.com/eugen-goebel" in block.value for block in app.markdown)
