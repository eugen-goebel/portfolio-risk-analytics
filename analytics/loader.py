"""Load price series from the database as pandas objects."""

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Asset, DailyPrice, IntradayPrice


def load_close_series(db: Session, symbol: str) -> pd.Series:
    """Closing prices for one symbol, indexed by date, oldest first."""
    rows = db.execute(
        select(DailyPrice.day, DailyPrice.close)
        .join(Asset)
        .where(Asset.symbol == symbol)
        .order_by(DailyPrice.day)
    ).all()
    if not rows:
        raise LookupError(f"No price data stored for {symbol}")
    series = pd.Series(
        [r.close for r in rows],
        index=pd.DatetimeIndex([r.day for r in rows]),
        name=symbol,
    )
    return series


def load_intraday_closes(db: Session, symbol: str) -> pd.Series:
    """Intraday closing prices for one symbol, indexed by timestamp, oldest first."""
    rows = db.execute(
        select(IntradayPrice.ts, IntradayPrice.close)
        .join(Asset)
        .where(Asset.symbol == symbol)
        .order_by(IntradayPrice.ts)
    ).all()
    if not rows:
        raise LookupError(f"No intraday data stored for {symbol}")
    series = pd.Series(
        [r.close for r in rows],
        index=pd.DatetimeIndex([r.ts for r in rows]),
        name=symbol,
    )
    return series


def load_close_frame(db: Session, symbols: list[str]) -> pd.DataFrame:
    """Closing prices for several symbols aligned on shared dates."""
    frame = pd.DataFrame({s: load_close_series(db, s) for s in symbols})
    return frame.dropna()
