"""Yahoo chart parsing, demo data, and idempotent storage."""

import pytest

from ingestion.base import ProviderError
from ingestion.demo import generate_demo_bars
from ingestion.store import store_bars
from ingestion.yahoo import parse_chart_json

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
