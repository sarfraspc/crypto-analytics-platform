try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from typing import Dict, List

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
    GROQ_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None

    # App
    APP_ENV: str = "development"
    APP_PORT: int = 8000

    # MCP
    MCP_DISCOVERY: bool = True
    MCP_TIMEOUT: int = 30

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Infura 
    INFURA_HTTPS: str
    INFURA_WSS: str

    # Other
    ALTERNATIVE_ME_URL: str
    CRYPTOPANIC_URL: str
    EXCHANGE_ADDRESSES: Dict[str, List[str]] = {
        "binance": [
            "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",
            "0xf977814e90da44bfa03b6295a0616a897441acec",
            "0x28c6c06298d514db089934071355e5743bf21d60"
        ],
        "coinbase": [
            "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43",
            "0x71660c4005ba85c37ccec55d0c4493e66fe775d3",
            "0x77696bb39917c91a0c3908d577d5e322095425ca",
            "0xcd531ae9efcce479654c4926dec5f6209531ca7b"
        ],
        "kraken": [
            "0xe9f7ecae3a53d2a67105292894676b00d1fab785",
            "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0"
        ],
        "gemini": ["0xd24400ae8bfebb18ca49be86258a3c749cf46853"],
        "bitfinex": [
            "0x742d35cc6634c0532925a3b844bc454e4438f44e",
            "0x77134cbc06cb00b66f4c7e623d5fdbf6777635ec"
        ]
    }

    STATIC_TOKEN_METADATA: Dict[str, Dict[str, str]] = {
        # address (lowercase) -> {coingecko_id, symbol, decimals}
        "0xdac17f958d2ee523a2206206994597c13d831ec7": {
            "coingecko_id": "tether",
            "symbol": "USDT",
            # Stored as string for compatibility with Dict[str, str]
            "decimals": "6",
        },
    }

    class Config:
        env_file = ".env"

settings = Settings()
