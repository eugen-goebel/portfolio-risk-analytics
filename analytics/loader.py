"""Load price series from the database as pandas objects."""

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Asset, DailyPrice


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


def load_close_frame(db: Session, symbols: list[str]) -> pd.DataFrame:
    """Closing prices for several symbols aligned on shared dates."""
    frame = pd.DataFrame({s: load_close_series(db, s) for s in symbols})
    return frame.dropna()
