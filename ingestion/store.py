"""Persist price bars into the database.

Inserts are idempotent: bars whose (asset, day) pair already exists are
skipped, so re-running an ingest never duplicates rows. The check works
the same on SQLite and PostgreSQL.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Asset, DailyPrice
from ingestion.base import PriceBar


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
