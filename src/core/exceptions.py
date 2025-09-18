class CryptoAnalyticsError(Exception):
    """Base exception for the platform"""
    pass

class APIError(CryptoAnalyticsError):
    """Raised when external API calls fail"""
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class DatabaseError(CryptoAnalyticsError):
    """Raised when database operations fail"""
    pass

class ModelError(CryptoAnalyticsError):
    """Raised when ML model operations fail"""
    pass
