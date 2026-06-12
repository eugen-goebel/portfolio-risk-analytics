"""Factsheet generation checked on deterministic demo data."""

from pathlib import Path

import pandas as pd

from ingestion.demo import generate_demo_bars
from reporting.factsheet import _sanitize, generate_factsheet


def demo_series(symbol: str) -> pd.Series:
    bars = generate_demo_bars(symbol)
    return pd.Series(
        [bar.close for bar in bars],
        index=pd.DatetimeIndex([bar.day for bar in bars]),
        name=symbol,
    )


def assert_valid_pdf(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 5 * 1024
    assert path.read_bytes()[:4] == b"%PDF"


class TestGenerateFactsheet:
    def test_writes_a_valid_pdf(self, tmp_path):
        target = tmp_path / "demo-factsheet.pdf"
        generate_factsheet("DEMO", demo_series("DEMO"), str(target), risk_free_rate=0.03)
        assert_valid_pdf(target)

    def test_returns_the_requested_path(self, tmp_path):
        target = tmp_path / "demo-factsheet.pdf"
        result = generate_factsheet("DEMO", demo_series("DEMO"), str(target))
        assert result == str(target)

    def test_symbol_outside_latin1_does_not_crash(self, tmp_path):
        target = tmp_path / "unicode-factsheet.pdf"
        generate_factsheet("TEST→X", demo_series("TEST"), str(target))
        assert_valid_pdf(target)


class TestSanitize:
    def test_latin1_text_is_unchanged(self):
        assert _sanitize("Volatility 17.10%") == "Volatility 17.10%"

    def test_characters_outside_latin1_are_replaced(self):
        assert _sanitize("TEST→X") == "TEST?X"
