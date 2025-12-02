"""Database engine and session management for TimescaleDB and PostgreSQL."""

import urllib.parse
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, scoped_session

from core.config import settings


def _build_postgres_url(user, password, host, port, db):
    """Build PostgreSQL connection URL with URL-encoded password."""
    pw = urllib.parse.quote_plus(password)
    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"


# Singleton engines with connection pooling for concurrent access
_timescale_engine = None
_metadata_engine = None


def get_timescale_engine() -> Engine:
    """Create or return singleton SQLAlchemy engine for TimescaleDB."""
    global _timescale_engine
    if _timescale_engine is None:
        _timescale_engine = create_engine(
            _build_postgres_url(
                settings.TIMESCALE_USER,
                settings.TIMESCALE_PASSWORD,
                settings.TIMESCALE_HOST,
                settings.TIMESCALE_PORT,
                settings.TIMESCALE_DB,
            ),
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            future=True,
        )
    return _timescale_engine


def get_metadata_engine() -> Engine:
    """Create or return singleton SQLAlchemy engine for PostgreSQL metadata."""
    global _metadata_engine
    if _metadata_engine is None:
        _metadata_engine = create_engine(
            _build_postgres_url(
                settings.POSTGRES_USER,
                settings.POSTGRES_PASSWORD,
                settings.POSTGRES_HOST,
                settings.POSTGRES_PORT,
                settings.POSTGRES_DB,
            ),
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            future=True,
        )
    return _metadata_engine


# Session factories - create new sessions, don't share across threads
TimescaleSessionLocal = sessionmaker(
    bind=get_timescale_engine(), autocommit=False, autoflush=False, future=True
)
MetadataSessionLocal = sessionmaker(
    bind=get_metadata_engine(), autocommit=False, autoflush=False, future=True
)


@contextmanager
def get_timescale_db():
    """Context manager for TimescaleDB sessions with auto commit/rollback."""
    session = TimescaleSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_metadata_db():
    """Context manager for metadata DB sessions with auto commit/rollback."""
    session = MetadataSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()