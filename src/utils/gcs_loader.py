"""
Google Cloud Storage utilities for model loading.

Provides upload and download functions for ML models with
RAM disk caching for improved performance.
"""

import os
from pathlib import Path

from google.cloud import storage


def _bucket():
    """Get GCS bucket name from environment."""
    bucket = os.getenv("GCS_MODEL_BUCKET")
    if not bucket:
        raise RuntimeError("GCS_MODEL_BUCKET env var is missing.")
    return bucket


def _client():
    """Create GCS client using application credentials."""
    # Uses GOOGLE_APPLICATION_CREDENTIALS automatically from your .env
    return storage.Client()

def upload_to_gcs(local_path: str | Path, remote_path: str) -> None:
    """
    Upload a local model file to GCS under the configured GCS_MODEL_BUCKET.

    Args:
        local_path: local filesystem path to the file to upload
        remote_path: path inside bucket (ex: forecasting/prophet/binanceus/1h/prophet_BTC.pkl)
    """
    bucket_name = _bucket()
    client = _client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(str(local_path))

def load_from_gcs(remote_path: str, local_name: str = None) -> Path:
    """
    Downloads a model file from GCS into a fast local path.
    
    Args:
        remote_path: path inside bucket (ex: forecasting/prophet/prophet_BTC.pkl)
        local_name: filename after download (optional)
    
    Returns:
        Path: The local path where the file is now stored
    """
    bucket_name = _bucket()
    client = _client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(remote_path)

    if local_name is None:
        local_name = os.path.basename(remote_path)

    # PERFORMANCE TRICK: 
    # /dev/shm is a RAM Disk (Memory). 
    # Writing here is 10x faster than writing to the hard drive.
    # We fall back to /tmp if we are on Windows/Mac (which don't have /dev/shm by default).
    base = Path("/dev/shm") if Path("/dev/shm").exists() else Path("/tmp")
    
    cache_dir = base / "models"
    cache_dir.mkdir(exist_ok=True, parents=True)

    local_path = cache_dir / local_name
    
    # Only download if haven't already (Simple Caching)
    if not local_path.exists():
        print(f"Downloading {remote_path} to {local_path}...")
        blob.download_to_filename(str(local_path))
    else:
        print(f"{local_name} found in cache. Skipping download.")

    return local_path
