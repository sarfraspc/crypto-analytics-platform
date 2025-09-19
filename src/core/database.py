from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from core.config import settings
import urllib.parse

def _build_postgres_url(user, password, host, port, db):
    # password may contain special chars
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
    # metadata db (if separate)
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