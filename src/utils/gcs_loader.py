import os
from pathlib import Path
from google.cloud import storage

def _bucket():
    bucket = os.getenv("GCS_MODEL_BUCKET")
    if not bucket:
        raise RuntimeError("GCS_MODEL_BUCKET env var is missing.")
    return bucket

def _client():
    # Uses GOOGLE_APPLICATION_CREDENTIALS automatically from your .env
    return storage.Client()

def load_from_gcs(remote_path: str, local_name: str = None) -> Path:
    """
    Downloads a model file from GCS into a fast local path.
    
    Args:
        remote_path: path inside bucket (ex: forecasting/prophet/BTC.pkl)
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