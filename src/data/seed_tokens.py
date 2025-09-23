from pycoingecko import CoinGeckoAPI
from core.database import get_metadata_engine
from sqlalchemy import text
from core.config import settings
import logging

logger = logging.getLogger(__name__)
CG = CoinGeckoAPI()
ENG = get_metadata_engine()

def seed_top_n(n=200):
    coins = CG.get_coins_markets(vs_currency='usd', per_page=250, page=1)
    with ENG.begin() as conn:
        for c in coins[:n]:
            symbol = c['symbol'].upper()
            conn.execute(text("""
                INSERT INTO tokens (symbol, coingecko_id, name, decimals, metadata)
                VALUES (:symbol, :cid, :name, :dec, :meta)
                ON CONFLICT (symbol) DO UPDATE SET coingecko_id = EXCLUDED.coingecko_id, name = EXCLUDED.name, metadata = EXCLUDED.metadata
            """), {
                'symbol': symbol,
                'cid': c['id'],
                'name': c['name'],
                'dec': 'NULL',
                'meta': c
            })
    logger.info("Seeded top %d tokens", n)
