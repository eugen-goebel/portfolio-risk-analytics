"""The demo seeder rebuilds only when the stored data version is stale."""

from sqlalchemy import select

from db.models import Asset, DailyPrice, DemoDataVersion
from ingestion.demo import DEMO_DATA_VERSION, DEMO_SYMBOLS, generate_demo_bars
from ingestion.demo_seed import ensure_demo_data
from ingestion.store import store_bars


def test_seeds_an_empty_database(db):
    assert ensure_demo_data(db, days=120) is True
    assert sorted(db.scalars(select(Asset.symbol))) == sorted(DEMO_SYMBOLS)
    assert db.scalar(select(DemoDataVersion.version)) == DEMO_DATA_VERSION


def test_is_idempotent_when_already_current(db):
    ensure_demo_data(db, days=120)
    assert ensure_demo_data(db, days=120) is False


def test_rebuilds_when_the_version_is_stale(db):
    # Look like an old deploy: demo rows plus an out of date stamp.
    for symbol in DEMO_SYMBOLS:
        store_bars(db, symbol, generate_demo_bars(symbol, days=120))
    db.add(DemoDataVersion(version="0"))
    db.commit()
    # Corrupt one close so we can prove the rows were actually replaced.
    stale = db.scalars(select(DailyPrice)).first()
    stale.close = -999.0
    db.commit()

    assert ensure_demo_data(db, days=120) is True
    assert db.scalar(select(DemoDataVersion.version)) == DEMO_DATA_VERSION
    assert db.scalar(select(DailyPrice).where(DailyPrice.close == -999.0)) is None


def test_leaves_real_ingested_prices_alone(db):
    store_bars(db, "SPY", generate_demo_bars("demo-equity", days=120), name="SPY")
    assert ensure_demo_data(db) is False
    assert list(db.scalars(select(Asset.symbol))) == ["SPY"]
