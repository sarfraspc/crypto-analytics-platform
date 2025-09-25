from pycoingecko import CoinGeckoAPI
from core.database import get_metadata_engine
from sqlalchemy import text
from core.config import settings
import logging
import json

logger = logging.getLogger(__name__)
CG = CoinGeckoAPI()
ENG = get_metadata_engine()

def seed_top_n(n=200):
    try:
        coins = CG.get_coins_markets(vs_currency='usd', per_page=250, page=1)
    except Exception as e:
        logger.error(f"CoinGecko API error: {e}")
        return

    params = [
        {
            'symbol': c['symbol'].upper(),
            'cid': c['id'],
            'name': c['name'],
            'dec': None,  
            'meta': json.dumps(c) 
        }
        for c in coins[:n] if 'symbol' in c and 'id' in c and 'name' in c
    ]

    insert_sql = text("""
        INSERT INTO tokens (symbol, coingecko_id, name, decimals, metadata)
        VALUES (:symbol, :cid, :name, :dec, :meta)
        ON CONFLICT (symbol) DO UPDATE SET
            coingecko_id = EXCLUDED.coingecko_id,
            name = EXCLUDED.name,
            metadata = EXCLUDED.metadata
    """)

    try:
        with ENG.begin() as conn:
            conn.execute(insert_sql, params)
        logger.info(f"Seeded/updated top {len(params)} tokens")
    except Exception as e:
        logger.error(f"DB error during seeding: {e}")

if __name__ == "__main__":
    seed_top_n()    