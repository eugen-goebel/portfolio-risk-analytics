"""Shared fixtures.

By default tests run on an in-memory SQLite database. When
TEST_DATABASE_URL is set (the CI PostgreSQL job does this) the same
suite runs against that server instead, with a clean schema per test.
"""

import os
import tempfile
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# db.database reads DATABASE_URL when it is imported, and the dashboard test
# drives app.py against that engine. Point it at a throwaway file before the
# import below, so a test run never creates or touches the repo's market.db.
os.environ.setdefault("DATABASE_URL", f"sqlite:///{tempfile.mkdtemp()}/market.db")

from db import models  # noqa: E402,F401  (registers the tables on Base)
from db.database import Base  # noqa: E402

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")


@pytest.fixture()
def db() -> Iterator[Session]:
    if TEST_DATABASE_URL:
        engine = create_engine(TEST_DATABASE_URL)
    else:
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()
    engine.dispose()
