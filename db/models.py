"""ORM models for assets and their daily and intraday prices."""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.database import Base


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), default="")

    prices: Mapped[list["DailyPrice"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )
    intraday_prices: Mapped[list["IntradayPrice"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )


class DemoDataVersion(Base):
    """Stamps the database with the demo-data version that populated it.

    A hosted deploy (e.g. Streamlit Cloud) keeps its SQLite file across
    redeploys, so without this stamp a changed generator would never reach the
    live demo. The seeder rebuilds when the stamp is missing or out of date.
    """

    __tablename__ = "demo_data_version"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(32))


class DailyPrice(Base):
    __tablename__ = "daily_prices"
    __table_args__ = (UniqueConstraint("asset_id", "day", name="uq_price_per_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), index=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0.0)

    asset: Mapped[Asset] = relationship(back_populates="prices")


class IntradayPrice(Base):
    __tablename__ = "intraday_prices"
    __table_args__ = (UniqueConstraint("asset_id", "ts", name="uq_intraday_per_ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0.0)

    asset: Mapped[Asset] = relationship(back_populates="intraday_prices")
