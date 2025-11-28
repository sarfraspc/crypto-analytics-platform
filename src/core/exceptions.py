"""Custom exceptions for the Crypto Analytics platform."""


class CryptoAnalyticsError(Exception):
    """Base exception for all platform-specific errors."""

    pass


class APIError(CryptoAnalyticsError):
    """Raised when external API calls fail (CoinGecko, Binance, etc.)."""

    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class DatabaseError(CryptoAnalyticsError):
    """Raised when database operations fail (Postgres, TimescaleDB, Redis)."""

    pass


class ModelError(CryptoAnalyticsError):
    """Raised when ML model operations fail (loading, inference, training)."""

    pass
