"""API endpoints against an in-memory database with demo data."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from db.database import get_db
from ingestion.demo import generate_demo_bars
from ingestion.store import store_bars


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


class TestMetrics:
    def test_single_asset_metrics(self, client):
        body = client.get("/assets/demo-a/metrics").json()
        assert body["symbol"] == "demo-a"
        assert body["observations"] > 0
        assert body["max_drawdown_pct"] <= 0

    def test_portfolio_metrics(self, client):
        body = client.post(
            "/portfolio/metrics",
            json={"weights": {"demo-a": 0.6, "demo-b": 0.4}},
        ).json()
        assert body["annualized_volatility_pct"] > 0
        assert body["correlations"]["demo-a"]["demo-a"] == pytest.approx(1.0)

    def test_forecast(self, client):
        body = client.get("/assets/demo-a/forecast?test_size=60").json()
        assert body["symbol"] == "demo-a"
        assert body["test_observations"] == 60
        assert body["best_model"] in {"rolling", "ewma", "har"}
        assert len(body["scores"]) == 3

    def test_forecast_unknown_symbol_is_404(self, client):
        assert client.get("/assets/nope/forecast").status_code == 404

    def test_bad_weights_are_400(self, client):
        resp = client.post("/portfolio/metrics", json={"weights": {"demo-a": 0.5}})
        assert resp.status_code == 400

    def test_unknown_portfolio_symbol_is_400(self, client):
        resp = client.post(
            "/portfolio/metrics",
            json={"weights": {"demo-a": 0.5, "ghost": 0.5}},
        )
        assert resp.status_code == 400
