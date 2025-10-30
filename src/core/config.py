try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from typing import Dict, List
import json
from pydantic import field_validator
from ast import literal_eval

class Settings(BaseSettings):
    # Postgres (metadata DB)
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    # Timescale
    TIMESCALE_USER: str
    TIMESCALE_PASSWORD: str
    TIMESCALE_DB: str
    TIMESCALE_HOST: str
    TIMESCALE_PORT: int

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int

    # Vector DB (Qdrant)
    QDRANT_URL: str 
    QDRANT_COLLECTION: str

    # MLflow
    MLFLOW_TRACKING_URI: str
    MLFLOW_ARTIFACT_ROOT: str

    # API Keys 
    COINGECKO_API_KEY: str | None = None
    BINANCE_API_KEY: str | None = None
    BINANCE_API_SECRET: str | None = None
    INFURA_PROJECT_ID: str | None = None
    CRYPTOPANIC_API_KEY: str | None = None
    REDDIT_CLIENT_ID: str | None = None
    REDDIT_CLIENT_SECRET: str | None = None
    REDDIT_USER_AGENT: str | None = None

    # App
    APP_ENV: str = "development"
    APP_PORT: int = 8000

    # MCP
    MCP_DISCOVERY: bool = True
    MCP_TIMEOUT: int = 30

    # Infura 
    INFURA_HTTPS: str
    INFURA_WSS: str

    # Other
    ALTERNATIVE_ME_URL: str
    CRYPTOPANIC_URL: str
    EXCHANGE_ADDRESSES: Dict[str, List[str]]

    @field_validator('EXCHANGE_ADDRESSES', mode='before')
    @classmethod
    def parse_exchange_addrs(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return literal_eval(v)
        return v

    class Config:
        env_file = ".env"

    @property
    def exchange_addrs(self) -> set[str]:
        return set(addr.lower() for addrs in self.EXCHANGE_ADDRESSES.values() for addr in addrs)

settings = Settings()