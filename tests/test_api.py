"""API endpoints against an in-memory database with demo data."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from api.main import app
from db.database import get_db
from ingestion.base import IntradayBar
from ingestion.demo import generate_demo_bars
from ingestion.store import store_bars, store_intraday_bars


def intraday_bar(ts: datetime, close: float) -> IntradayBar:
    return IntradayBar(ts=ts, open=close, high=close, low=close, close=close, volume=0.0)


@pytest.fixture()
def client(db):
    store_bars(db, "demo-a", generate_demo_bars("demo-a", days=300))
    store_bars(db, "demo-b", generate_demo_bars("demo-b", days=300))

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestHealth:
    def test_root(self, client):
        body = client.get("/").json()
        assert body["status"] == "running"


class TestAssets:
    def test_list_assets(self, client):
        body = client.get("/assets").json()
        assert [a["symbol"] for a in body] == ["demo-a", "demo-b"]
        assert all(a["price_count"] > 0 for a in body)

    def test_prices(self, client):
        body = client.get("/assets/demo-a/prices?limit=10").json()
        assert len(body) == 10
        assert set(body[0]) == {"day", "close"}

    def test_unknown_symbol_is_404(self, client):
        assert client.get("/assets/nope/prices").status_code == 404

    def test_realized_volatility(self, client, db):
        # closes 100 -> 101 -> 99.99 give RV sqrt(0.0002) = 0.0141421,
        # annualized: 0.0141421 * sqrt(252) * 100 = 22.4497, rounded 22.45
        store_intraday_bars(
            db,
            "demo-a",
            [
                intraday_bar(datetime(2024, 1, 2, 10, tzinfo=UTC), 100.0),
                intraday_bar(datetime(2024, 1, 2, 11, tzinfo=UTC), 101.0),
                intraday_bar(datetime(2024, 1, 2, 12, tzinfo=UTC), 99.99),
            ],
        )
        body = client.get("/assets/demo-a/realized-volatility").json()
        assert body == [{"day": "2024-01-02", "realized_vol_pct": 22.45}]

    def test_realized_volatility_without_intraday_data_is_404(self, client):
        assert client.get("/assets/demo-a/realized-volatility").status_code == 404


class TestMetrics:
    def test_single_asset_metrics(self, client):
        body = client.get("/assets/demo-a/metrics").json()
        assert body["symbol"] == "demo-a"
        assert body["observations"] > 0
        assert body["max_drawdown_pct"] <= 0
        assert body["var_95_pct"] >= 0
        assert body["expected_shortfall_95_pct"] >= body["var_95_pct"]

    def test_portfolio_metrics(self, client):
        body = client.post(
            "/portfolio/metrics",
            json={"weights": {"demo-a": 0.6, "demo-b": 0.4}},
        ).json()
        assert body["annualized_volatility_pct"] > 0
        assert body["var_95_pct"] >= 0
        assert body["expected_shortfall_95_pct"] >= body["var_95_pct"]
        assert body["correlations"]["demo-a"]["demo-a"] == pytest.approx(1.0)

    def test_forecast(self, client):
        body = client.get("/assets/demo-a/forecast?test_size=60").json()
        assert body["symbol"] == "demo-a"
        assert body["test_observations"] == 60
        assert body["best_model"] in {"rolling", "ewma", "har"}
        assert len(body["scores"]) == 3

    def test_forecast_unknown_symbol_is_404(self, client):
        assert client.get("/assets/nope/forecast").status_code == 404

    def test_drift(self, client):
        body = client.get("/assets/demo-a/drift").json()
        assert body["symbol"] == "demo-a"
        assert body["recent_size"] == 60
        assert body["psi"] >= 0
        assert 0 <= body["ks"] <= 1
        assert isinstance(body["drift_detected"], bool)

    def test_drift_unknown_symbol_is_404(self, client):
        assert client.get("/assets/nope/drift").status_code == 404

    def test_montecarlo(self, client):
        body = client.get("/assets/demo-a/montecarlo?horizon=60&n_paths=300&seed=7").json()
        assert body["symbol"] == "demo-a"
        assert body["method"] == "bootstrap"
        assert body["horizon_days"] == 60
        assert body["n_paths"] == 300
        assert body["start_value"] == 100.0
        p = body["percentiles"]
        assert p["p5"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p95"]
        assert 0.0 <= body["prob_loss"] <= 1.0
        assert body["expected_final"] > 0

    def test_montecarlo_unknown_symbol_is_404(self, client):
        assert client.get("/assets/nope/montecarlo").status_code == 404

    def test_var_validation(self, client):
        body = client.get("/assets/demo-a/var-validation?window=100").json()
        assert body["symbol"] == "demo-a"
        assert body["window"] == 100
        assert body["confidence"] == 0.95
        assert body["observations"] > 0
        assert body["expected_breaches"] == pytest.approx(0.05 * body["observations"], abs=0.1)
        assert 0 <= body["p_value"] <= 1
        assert isinstance(body["model_rejected"], bool)

    def test_var_validation_unknown_symbol_is_404(self, client):
        assert client.get("/assets/nope/var-validation").status_code == 404

    def test_benchmark(self, client):
        body = client.get("/assets/demo-a/benchmark?benchmark=demo-b").json()
        assert body["symbol"] == "demo-a"
        assert body["benchmark"] == "demo-b"
        assert body["observations"] > 0
        assert body["tracking_error_pct"] >= 0

    def test_benchmark_unknown_benchmark_is_404(self, client):
        resp = client.get("/assets/demo-a/benchmark?benchmark=nope")
        assert resp.status_code == 404
        assert "nope" in resp.json()["detail"]

    def test_drift_short_history_is_400(self, client):
        assert client.get("/assets/demo-a/drift?recent_size=200").status_code == 400

    def test_portfolio_backtest(self, client):
        body = client.post(
            "/portfolio/backtest",
            json={"weights": {"demo-a": 0.6, "demo-b": 0.4}},
        ).json()
        assert body["symbols"] == ["demo-a", "demo-b"]
        assert body["rebalance"] == "monthly"
        assert body["observations"] > 0
        assert body["rebalanced"]["final_value"] > 0
        assert body["buy_and_hold"]["final_value"] > 0

    def test_backtest_bad_weights_are_400(self, client):
        resp = client.post("/portfolio/backtest", json={"weights": {"demo-a": 0.5}})
        assert resp.status_code == 400

    def test_portfolio_optimize(self, client):
        body = client.post(
            "/portfolio/optimize",
            json={"symbols": ["demo-a", "demo-b"]},
        ).json()
        assert body["symbols"] == ["demo-a", "demo-b"]
        assert body["observations"] > 0
        for key in ("minimum_variance", "maximum_sharpe"):
            assert sum(body[key]["weights"].values()) == pytest.approx(1.0, abs=1e-3)
            assert body[key]["volatility_pct"] > 0

    def test_optimize_single_symbol_is_400(self, client):
        resp = client.post("/portfolio/optimize", json={"symbols": ["demo-a"]})
        assert resp.status_code == 400

    def test_bad_weights_are_400(self, client):
        resp = client.post("/portfolio/metrics", json={"weights": {"demo-a": 0.5}})
        assert resp.status_code == 400

    def test_unknown_portfolio_symbol_is_400(self, client):
        resp = client.post(
            "/portfolio/metrics",
            json={"weights": {"demo-a": 0.5, "ghost": 0.5}},
        )
        assert resp.status_code == 400
