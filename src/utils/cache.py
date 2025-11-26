from io import StringIO
import json
import logging
from typing import Any, Optional

import os
import pandas as pd
import redis

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, host="localhost", port=6379, db=0, expire_seconds: int = 3600):
        # Allow Docker / env-configured Redis (e.g. REDIS_HOST=redis inside containers).
        resolved_host = os.getenv("REDIS_HOST", host)
        resolved_port = int(os.getenv("REDIS_PORT", port))
        resolved_db = int(os.getenv("REDIS_DB", db))
        self.client = redis.Redis(
            host=resolved_host,
            port=resolved_port,
            db=resolved_db,
            decode_responses=True,
        )
        self.expire_seconds = expire_seconds

    def set_json(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> None:
        try:
            ttl = expire_seconds if expire_seconds is not None else self.expire_seconds
            self.client.set(key, json.dumps(value), ex=ttl)
            logger.debug(f"[Redis] Cached key={key} (expire={ttl}s)")
        except Exception as e:
            logger.warning(f"[Redis] Failed to cache key={key}: {e}")

    def get_json(self, key: str):
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"[Redis] Failed to fetch key={key}: {e}")
            return None

    def set_dataframe(self, key: str, df: pd.DataFrame, expire_seconds: Optional[int] = None):
        try:
            ttl = expire_seconds if expire_seconds is not None else self.expire_seconds
            payload = df.to_json(orient="split", date_format="iso")
            self.client.set(key, payload, ex=ttl)
            logger.debug(f"[Redis] Cached DataFrame key={key} (expire={ttl}s)")
        except Exception as e:
            logger.warning(f"[Redis] Failed to cache DataFrame key={key}: {e}")

    def get_dataframe(self, key: str):
        try:
            data = self.client.get(key)
            if not data:
                return None
            if isinstance(data, str):
                data = StringIO(data)
            return pd.read_json(data, orient="split")
        except Exception as e:
            logger.warning(f"[Redis] Failed to fetch DataFrame key={key}: {e}")
            return None

    def delete_by_pattern(self, pattern: str):
        try:
            keys_to_delete = [key for key in self.client.scan_iter(match=pattern)]
            if keys_to_delete:
                self.client.delete(*keys_to_delete)
                logger.debug(f"[Redis] Deleted {len(keys_to_delete)} keys matching pattern={pattern}")
        except Exception as e:
            logger.warning(f"[Redis] Failed to delete keys with pattern={pattern}: {e}")

    def delete(self, key: str):
        """Delete a cache key safely."""
        try:
            return self.client.delete(key)
        except Exception as e:
            logger.warning(f"Failed to delete cache key {key}: {e}")
            return 0

    def get_stats(self, pattern: str = "*"):
        try:
            count = 0
            for _ in self.client.scan_iter(match=pattern):
                count += 1
            return {"keys": count}
        except Exception as e:
            logger.warning(f"[Redis] Failed to get stats for pattern={pattern}: {e}")
            return {}
