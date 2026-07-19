"""Keep a database seeded with the current demo series.

A hosted deploy (Streamlit Cloud) keeps its database across redeploys, so a
changed generator would otherwise never reach the live demo: the old rows are
still there and store_bars, being idempotent per (asset, day), never overwrites
them. Stamping the database with a version and rebuilding on a mismatch fixes
that, and works the same on SQLite and PostgreSQL.
"""

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from db.models import Asset, DemoDataVersion
from ingestion.demo import DEMO_DATA_VERSION, DEMO_SYMBOLS, generate_demo_bars
from ingestion.store import store_bars


def ensure_demo_data(db: Session, *, days: int = 750) -> bool:
    """Seed the demo series unless the database is already current.

    Returns True when it (re)built the demo data. A database that only holds
    real ingested prices is left untouched; a database whose demo stamp is
    missing or older than DEMO_DATA_VERSION has its demo assets rebuilt.
    """
    demo_present = (
        db.scalar(select(Asset.id).where(Asset.symbol.in_(DEMO_SYMBOLS)).limit(1)) is not None
    )
    stored_version = db.scalar(select(DemoDataVersion.version).limit(1))
    if demo_present and stored_version == DEMO_DATA_VERSION:
        return False
    if not demo_present and db.scalar(select(Asset.id).limit(1)) is not None:
        # Only real ingested prices live here, so there is no demo to manage.
        return False

    for asset in db.scalars(select(Asset).where(Asset.symbol.in_(DEMO_SYMBOLS))).all():
        db.delete(asset)  # cascades to the asset's price rows
    db.execute(delete(DemoDataVersion))
    db.commit()

    for symbol in DEMO_SYMBOLS:
        store_bars(db, symbol, generate_demo_bars(symbol, days=days))
    db.add(DemoDataVersion(version=DEMO_DATA_VERSION))
    db.commit()
    return True
