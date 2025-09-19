try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

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

    # MLflow
    MLFLOW_TRACKING_URI: str
    MLFLOW_ARTIFACT_ROOT: str

    # API Keys (optional)
    COINGECKO_API_KEY: str | None = None
    BINANCE_API_KEY: str | None = None
    BINANCE_API_SECRET: str | None = None
    ALCHEMY_API_KEY: str | None = None
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

    class Config:
        env_file = ".env"

settings = Settings()