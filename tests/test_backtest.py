"""Backtest simulation checked against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from analytics.backtest import buy_and_hold, run_backtest, simulate

WEIGHTS = {"A": 0.5, "B": 0.5}


def hand_frame() -> pd.DataFrame:
    """Four business days across a month boundary, fully hand-computable."""
    idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
    return pd.DataFrame(
        {"A": [100.0, 110.0, 110.0, 121.0], "B": [100.0, 100.0, 100.0, 100.0]}, index=idx
    )


def extended_frame() -> pd.DataFrame:
    """The hand frame padded with flat prices to pass the 30-row guard.

    Flat prices keep both value paths constant after 2024-02-02, so
    the final values stay hand-computable.
    """
    idx = pd.date_range("2024-01-30", periods=40, freq="B")
    return pd.DataFrame({"A": [100.0, 110.0, 110.0] + [121.0] * 37, "B": [100.0] * 40}, index=idx)


def random_frame(days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-02", periods=days, freq="B")
    return pd.DataFrame(
        {
            "A": 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.010, size=days)),
            "B": 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.008, size=days)),
        },
        index=idx,
    )


class TestBuyAndHold:
    def test_hand_computed_path(self):
        # shares: A = 100 * 0.5 / 100 = 0.5, B = 0.5, never touched
        path = buy_and_hold(hand_frame(), WEIGHTS)
        assert list(path) == pytest.approx([100.0, 105.0, 105.0, 110.5])

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            buy_and_hold(hand_frame(), {"A": 0.5})

    def test_unknown_symbol_rejected(self):
        with pytest.raises(ValueError, match="No price data"):
            buy_and_hold(hand_frame(), {"A": 0.5, "zzz": 0.5})


class TestSimulate:
    def test_hand_computed_monthly_path(self):
        # identical to buy and hold until the rebalance on 2024-02-01
        # (V = 105). The reset puts 52.5 into each asset: shares
        # A = 52.5 / 110 and B = 0.525, so the last day is
        # 52.5 / 110 * 121 + 52.5 = 57.75 + 52.5 = 110.25
        path = simulate(hand_frame(), WEIGHTS, rebalance="monthly")
        assert list(path) == pytest.approx([100.0, 105.0, 105.0, 110.25])

    def test_quarterly_has_no_reset_inside_one_quarter(self):
        # both months lie in Q1 and the quarter starts at the first
        # row, so the quarterly schedule never fires
        path = simulate(hand_frame(), WEIGHTS, rebalance="quarterly")
        assert list(path) == pytest.approx([100.0, 105.0, 105.0, 110.5])

    def test_no_rebalance_date_equals_buy_and_hold(self):
        # all days in one month, so no rebalance date after the first row
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        frame = pd.DataFrame(
            {"A": [100.0, 110.0, 105.0, 121.0], "B": [100.0, 90.0, 95.0, 100.0]}, index=idx
        )
        assert list(simulate(frame, WEIGHTS)) == pytest.approx(list(buy_and_hold(frame, WEIGHTS)))

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            simulate(hand_frame(), {"A": 0.7, "B": 0.7})

    def test_unknown_rebalance_rejected(self):
        with pytest.raises(ValueError, match="rebalance"):
            simulate(hand_frame(), WEIGHTS, rebalance="weekly")


class TestRunBacktest:
    def test_hand_computed_final_values(self):
        report = run_backtest(extended_frame(), WEIGHTS)
        assert report.buy_and_hold.final_value == pytest.approx(110.5)
        assert report.rebalanced.final_value == pytest.approx(110.25)

    def test_report_structure(self):
        frame = random_frame()
        report = run_backtest(frame, WEIGHTS, rebalance="quarterly", risk_free_rate=0.02)
        assert report.symbols == ["A", "B"]
        assert report.rebalance == "quarterly"
        assert report.observations == len(frame)
        for result in (report.rebalanced, report.buy_and_hold):
            assert result.final_value > 0
            assert result.annualized_volatility_pct > 0
            assert result.max_drawdown_pct <= 0

    def test_short_frame_rejected(self):
        with pytest.raises(ValueError, match="at least 30"):
            run_backtest(hand_frame(), WEIGHTS)
