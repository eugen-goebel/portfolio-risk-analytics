"""Yahoo chart parsing, demo data, and idempotent storage."""

import itertools
import math
from datetime import UTC, datetime

import pandas as pd
import pytest

from ingestion.base import ProviderError
from ingestion.demo import PROFILES, TRADING_DAYS, generate_demo_bars
from ingestion.store import store_bars, store_intraday_bars
from ingestion.yahoo import fetch_intraday, parse_chart_json, parse_intraday_json


def _close_series(symbol: str, days: int = 750) -> pd.Series:
    """Closing prices the way the dashboard loads them, indexed by day."""
    return pd.Series({bar.day: bar.close for bar in generate_demo_bars(symbol, days=days)})

# 2024-01-02 and 2024-01-03 at 21:00 UTC, like Yahoo timestamps them
SAMPLE_CHART = {
    "chart": {
        "result": [
            {
                "timestamp": [1704229200, 1704315600],
                "indicators": {
                    "quote": [
                        {
                            "open": [470.49, 468.71],
                            "high": [472.10, 470.04],
                            "low": [468.18, 466.43],
                            "close": [468.94, 467.27],
                            "volume": [82488700, 81919800],
                        }
                    ],
                    "adjclose": [{"adjclose": [465.10, 463.45]}],
                },
            }
        ],
        "error": None,
    }
}


class TestParseChartJson:
    def test_valid_payload_uses_adjusted_close(self):
        bars = parse_chart_json(SAMPLE_CHART)
        assert len(bars) == 2
        assert bars[0].day.isoformat() == "2024-01-02"
        assert bars[0].close == pytest.approx(465.10)
        assert bars[1].volume == pytest.approx(81919800)

    def test_null_rows_are_skipped(self):
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1704229200, 1704315600],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [470.49, None],
                                    "high": [472.10, None],
                                    "low": [468.18, None],
                                    "close": [468.94, None],
                                    "volume": [82488700, None],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }
        assert len(parse_chart_json(payload)) == 1

    def test_error_payload(self):
        payload = {"chart": {"result": None, "error": {"code": "Not Found"}}}
        with pytest.raises(ProviderError, match="error"):
            parse_chart_json(payload)

    def test_empty_result(self):
        with pytest.raises(ProviderError, match="no result"):
            parse_chart_json({"chart": {"result": [], "error": None}})

    def test_no_usable_bars(self):
        payload = {
            "chart": {
                "result": [{"timestamp": [], "indicators": {"quote": [{}]}}],
                "error": None,
            }
        }
        with pytest.raises(ProviderError, match="no usable bars"):
            parse_chart_json(payload)


# 2024-01-02 at 14:30 and 15:30 UTC, hourly bars without adjclose
SAMPLE_INTRADAY = {
    "chart": {
        "result": [
            {
                "timestamp": [1704205800, 1704209400],
                "indicators": {
                    "quote": [
                        {
                            "open": [470.49, 469.20],
                            "high": [471.00, 470.10],
                            "low": [469.90, 468.80],
                            "close": [470.20, 469.55],
                            "volume": [5200100, 4100300],
                        }
                    ]
                },
            }
        ],
        "error": None,
    }
}


class TestParseIntradayJson:
    def test_valid_payload_keeps_utc_timestamps(self):
        bars = parse_intraday_json(SAMPLE_INTRADAY)
        assert len(bars) == 2
        assert bars[0].ts == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
        assert bars[1].ts == datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
        assert bars[0].close == pytest.approx(470.20)
        assert bars[1].volume == pytest.approx(4100300)

    def test_null_rows_are_skipped(self):
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1704205800, 1704209400],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [470.49, None],
                                    "high": [471.00, None],
                                    "low": [469.90, None],
                                    "close": [470.20, None],
                                    "volume": [5200100, None],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }
        assert len(parse_intraday_json(payload)) == 1

    def test_invalid_interval_raises(self):
        with pytest.raises(ProviderError, match="interval"):
            fetch_intraday("SPY", interval="1d")


class TestDemoData:
    def test_deterministic_per_symbol(self):
        a = generate_demo_bars("demo-a", days=100)
        b = generate_demo_bars("demo-a", days=100)
        assert [bar.close for bar in a] == [bar.close for bar in b]

    def test_different_symbols_differ(self):
        a = generate_demo_bars("demo-a", days=100)
        b = generate_demo_bars("demo-b", days=100)
        assert [bar.close for bar in a] != [bar.close for bar in b]

    def test_no_weekend_bars(self):
        assert all(bar.day.weekday() < 5 for bar in generate_demo_bars("demo-a", days=50))

    @pytest.mark.parametrize("symbol", sorted(PROFILES))
    def test_named_profiles_hit_their_volatility(self, symbol):
        # Guards the demo's credibility: bonds used to come out at equity
        # volatility because every symbol shared one set of parameters.
        closes = _close_series(symbol)
        realized = closes.pct_change().dropna().std() * math.sqrt(TRADING_DAYS)
        assert realized == pytest.approx(PROFILES[symbol].volatility, rel=0.15)

    def test_bonds_are_calmer_than_equity(self):
        bonds = _close_series("demo-bonds").pct_change().dropna().std()
        equity = _close_series("demo-equity").pct_change().dropna().std()
        assert bonds < equity / 2

    def test_correlation_follows_the_product_of_betas(self):
        # The tolerance carries the sampling error of a correlation over the
        # ~540 bars the dashboard shows, which is around 0.04 per pair.
        frame = pd.DataFrame({s: _close_series(s) for s in PROFILES})
        realized = frame.pct_change().dropna().corr()
        for a, b in itertools.combinations(PROFILES, 2):
            expected = PROFILES[a].beta * PROFILES[b].beta
            assert realized.loc[a, b] == pytest.approx(expected, abs=0.09)

    def test_equity_sharpe_stays_believable(self):
        # A market seed drawing a two year bull run reads as a broken demo.
        closes = _close_series("demo-equity")
        returns = closes.pct_change().dropna()
        sharpe = (returns.mean() * TRADING_DAYS - 0.03) / (returns.std() * math.sqrt(TRADING_DAYS))
        assert 0.2 < sharpe < 1.2


class TestStore:
    def test_insert_and_idempotent_rerun(self, db):
        bars = generate_demo_bars("demo-a", days=50)
        first = store_bars(db, "demo-a", bars)
        assert first == len(bars)
        second = store_bars(db, "demo-a", bars)
        assert second == 0

    def test_incremental_insert(self, db):
        bars = generate_demo_bars("demo-a", days=50)
        store_bars(db, "demo-a", bars[:10])
        added = store_bars(db, "demo-a", bars)
        assert added == len(bars) - 10

    def test_intraday_insert_and_idempotent_rerun(self, db):
        bars = parse_intraday_json(SAMPLE_INTRADAY)
        first = store_intraday_bars(db, "demo-a", bars)
        assert first == len(bars)
        second = store_intraday_bars(db, "demo-a", bars)
        assert second == 0
