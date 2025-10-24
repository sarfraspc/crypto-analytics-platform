from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from core.config import settings
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import urllib.parse

def _build_postgres_url(user, password, host, port, db):
    pw = urllib.parse.quote_plus(password)
    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"

def get_timescale_engine() -> Engine:
    return create_engine(
        _build_postgres_url(
            settings.TIMESCALE_USER,
            settings.TIMESCALE_PASSWORD,
            settings.TIMESCALE_HOST,
            settings.TIMESCALE_PORT,
            settings.TIMESCALE_DB,
        ),
        pool_pre_ping=True,
        future=True,
    )

def get_metadata_engine() -> Engine:
    return create_engine(
        _build_postgres_url(
            settings.POSTGRES_USER,
            settings.POSTGRES_PASSWORD,
            settings.POSTGRES_HOST,
            settings.POSTGRES_PORT,
            settings.POSTGRES_DB,
        ),
        pool_pre_ping=True,
        future=True,
    )

TimescaleSessionLocal = sessionmaker(bind=get_timescale_engine(), autocommit=False, autoflush=False, future=True)
MetadataSessionLocal = sessionmaker(bind=get_metadata_engine(), autocommit=False, autoflush=False, future=True)

@contextmanager
def get_timescale_db():
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
    session = MetadataSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()