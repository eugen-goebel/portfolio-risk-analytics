"""ECB exchange rate CSV parsing."""

import pytest

from ingestion.base import ProviderError
from ingestion.ecb import parse_exr_csv

# Trimmed but structurally faithful csvdata response for D.USD.EUR.SP00.A
SAMPLE_CSV = (
    "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n"
    "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2026-06-08,1.1411\n"
    "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2026-06-09,1.1487\n"
    "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2026-06-10,1.1539\n"
)


class TestParseExrCsv:
    def test_valid_csv(self):
        bars = parse_exr_csv(SAMPLE_CSV)
        assert len(bars) == 3
        assert bars[0].day.isoformat() == "2026-06-08"
        assert bars[0].close == pytest.approx(1.1411)
        assert bars[2].day.isoformat() == "2026-06-10"
        assert bars[2].close == pytest.approx(1.1539)

    def test_single_daily_fix_fills_all_fields(self):
        bar = parse_exr_csv(SAMPLE_CSV)[1]
        assert bar.open == bar.high == bar.low == bar.close == pytest.approx(1.1487)
        assert bar.volume == 0.0

    def test_empty_body(self):
        with pytest.raises(ProviderError, match="empty"):
            parse_exr_csv("")

    def test_missing_columns(self):
        text = "KEY,TIME_PERIOD\nEXR.D.USD.EUR.SP00.A,2026-06-10\n"
        with pytest.raises(ProviderError, match="missing columns"):
            parse_exr_csv(text)

    def test_empty_obs_value_is_skipped(self):
        text = (
            "KEY,TIME_PERIOD,OBS_VALUE\n"
            "EXR.D.USD.EUR.SP00.A,2026-06-09,1.1487\n"
            "EXR.D.USD.EUR.SP00.A,2026-06-10,\n"
        )
        bars = parse_exr_csv(text)
        assert len(bars) == 1
        assert bars[0].day.isoformat() == "2026-06-09"

    def test_header_only(self):
        with pytest.raises(ProviderError, match="no usable rates"):
            parse_exr_csv("KEY,TIME_PERIOD,OBS_VALUE\n")
