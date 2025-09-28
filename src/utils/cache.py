import json
import logging
from typing import Any, Optional

import pandas as pd
import redis

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, host="localhost", port=6379, db=0, expire_seconds: int = 3600):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.expire_seconds = expire_seconds

    def set_json(self, key: str, value: Any) -> None:
        try:
            self.client.set(key, json.dumps(value), ex=self.expire_seconds)
            logger.debug(f"[Redis] Cached key={key} (expire={self.expire_seconds}s)")
        except Exception as e:
            logger.warning(f"[Redis] Failed to cache key={key}: {e}")

    def get_json(self, key: str):
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"[Redis] Failed to fetch key={key}: {e}")
            return None

    def set_dataframe(self, key: str, df: pd.DataFrame):
        try:
            payload = df.to_json(orient="split", date_format="iso")
            self.client.set(key, payload, ex=self.expire_seconds)
            logger.debug(f"[Redis] Cached DataFrame key={key}")
        except Exception as e:
            logger.warning(f"[Redis] Failed to cache DataFrame key={key}: {e}")

    def get_dataframe(self, key: str):
        try:
            data = self.client.get(key)
            if not data:
                return None
            return pd.read_json(data, orient="split")
        except Exception as e:
            logger.warning(f"[Redis] Failed to fetch DataFrame key={key}: {e}")
            return None