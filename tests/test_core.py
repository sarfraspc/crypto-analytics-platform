"""
Tests for core configuration, database, and exception modules.

Covers settings validation, database connection management,
and custom exception handling.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.core.config import Settings
from src.core.database import (
    _build_postgres_url,
    get_metadata_db,
    get_timescale_db,
)
from src.core.exceptions import APIError


# Configuration Tests

def test_settings_load_defaults():
    """
    Test that settings fail to load when required vars are missing.
    We pass _env_file=None to ensure Pydantic doesn't read your local .env file.
    """
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

def test_settings_valid_env_vars():
    """Test that settings correctly parse environment variables."""
    env_vars = {
        "POSTGRES_USER": "test_user",
        "POSTGRES_PASSWORD": "test_password",
        "POSTGRES_DB": "test_db",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "TIMESCALE_USER": "ts_user",
        "TIMESCALE_PASSWORD": "ts_password",
        "TIMESCALE_DB": "ts_db",
        "TIMESCALE_HOST": "localhost",
        "TIMESCALE_PORT": "5432",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_DB": "0",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_COLLECTION": "crypto",
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "MLFLOW_ARTIFACT_ROOT": "./mlruns",
        "INFURA_HTTPS": "https://mainnet.infura.io/v3/xyz",
        "INFURA_WSS": "wss://mainnet.infura.io/v3/xyz",
        "ALTERNATIVE_ME_URL": "https://api.alternative.me/",
        "CRYPTOPANIC_URL": "https://cryptopanic.com/api/",
        "ALLOWED_ORIGINS": "http://localhost:3000,http://example.com"
    }

    with patch.dict(os.environ, env_vars):
        # _env_file=None ensures use the mocked env_vars, not the real .env file
        settings = Settings(_env_file=None)
        
        assert settings.POSTGRES_USER == "test_user"
        assert settings.APP_ENV == "development"
        
        # CORRECTED: config defines this as a str, so test for str
        assert isinstance(settings.ALLOWED_ORIGINS, str)
        assert "http://localhost:3000" in settings.ALLOWED_ORIGINS

def test_settings_missing_required_field():
    """Test that Pydantic raises an error if a required field is missing."""
    incomplete_vars = {"POSTGRES_USER": "test_user"}
    
    with patch.dict(os.environ, incomplete_vars, clear=True):
        with pytest.raises(ValidationError) as excinfo:
            Settings(_env_file=None)
        
        errors = excinfo.value.errors()
        assert any(e['loc'][0] == 'POSTGRES_PASSWORD' for e in errors)

# Database Tests

def test_build_postgres_url():
    """Test the helper function for constructing DB URLs."""
    url = _build_postgres_url("user", "pass", "localhost", 5432, "db")
    assert url == "postgresql+psycopg2://user:pass@localhost:5432/db"

    # Test password escaping
    url_special = _build_postgres_url("user", "pass@word#", "localhost", 5432, "db")
    assert "pass%40word%23" in url_special

# Timescale DB Tests

@patch("src.core.database.TimescaleSessionLocal")
def test_get_timescale_db_success(mock_session_cls):
    """Test that the context manager commits on success."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    with get_timescale_db() as session:
        assert session == mock_session
    
    mock_session.commit.assert_called_once()
    mock_session.close.assert_called_once()
    mock_session.rollback.assert_not_called()

@patch("src.core.database.TimescaleSessionLocal")
def test_get_timescale_db_exception(mock_session_cls):
    """Test that the context manager rolls back on exception."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    with pytest.raises(ValueError):
        with get_timescale_db() as session:
            raise ValueError("Simulated DB Error")

    mock_session.rollback.assert_called_once()
    mock_session.close.assert_called_once()

# Metadata DB Tests

@patch("src.core.database.MetadataSessionLocal")
def test_get_metadata_db_success(mock_session_cls):
    """Test that the metadata DB context manager commits on success."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    with get_metadata_db() as session:
        assert session == mock_session
    
    mock_session.commit.assert_called_once()
    mock_session.close.assert_called_once()
    mock_session.rollback.assert_not_called()

@patch("src.core.database.MetadataSessionLocal")
def test_get_metadata_db_exception(mock_session_cls):
    """Test that the metadata DB context manager rolls back on exception."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    with pytest.raises(ValueError):
        with get_metadata_db() as session:
            raise ValueError("Simulated Metadata DB Error")

    mock_session.rollback.assert_called_once()
    mock_session.close.assert_called_once()

# Exception Tests

def test_api_error_initialization():
    """Test the custom APIError class."""
    error = APIError(message="Rate limit exceeded", status_code=429)
    
    assert str(error) == "Rate limit exceeded"
    assert error.message == "Rate limit exceeded"
    assert error.status_code == 429
    assert isinstance(error, Exception)