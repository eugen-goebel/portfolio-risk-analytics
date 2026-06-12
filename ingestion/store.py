"""Persist price bars into the database.

Inserts are idempotent: bars whose (asset, day) or (asset, ts) pair
already exists are skipped, so re-running an ingest never duplicates
rows. The check works the same on SQLite and PostgreSQL.
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Asset, DailyPrice, IntradayPrice
from ingestion.base import IntradayBar, PriceBar


def get_or_create_asset(db: Session, symbol: str, name: str = "") -> Asset:
    asset = db.scalar(select(Asset).where(Asset.symbol == symbol))
    if asset is None:
        asset = Asset(symbol=symbol, name=name or symbol)
        db.add(asset)
        db.flush()
    return asset


def store_bars(db: Session, symbol: str, bars: list[PriceBar], name: str = "") -> int:
    """Store bars for a symbol and return how many new rows were inserted."""
    asset = get_or_create_asset(db, symbol, name)
    existing = set(db.scalars(select(DailyPrice.day).where(DailyPrice.asset_id == asset.id)).all())

    inserted = 0
    for bar in bars:
        if bar.day in existing:
            continue
        db.add(
            DailyPrice(
                asset_id=asset.id,
                day=bar.day,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
        )
        inserted += 1

    db.commit()
    return inserted


def _naive_utc(ts: datetime) -> datetime:
    """Wall-clock UTC for the timezone naive DateTime column."""
    return ts.astimezone(UTC).replace(tzinfo=None)


def store_intraday_bars(db: Session, symbol: str, bars: list[IntradayBar], name: str = "") -> int:
    """Store intraday bars for a symbol and return how many new rows were inserted."""
    asset = get_or_create_asset(db, symbol, name)
    existing = set(
        db.scalars(select(IntradayPrice.ts).where(IntradayPrice.asset_id == asset.id)).all()
    )

    inserted = 0
    for bar in bars:
        ts = _naive_utc(bar.ts)
        if ts in existing:
            continue
        db.add(
            IntradayPrice(
                asset_id=asset.id,
                ts=ts,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
        )
        inserted += 1

    db.commit()
    return inserted
