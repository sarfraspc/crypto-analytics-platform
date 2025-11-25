from pycoingecko import CoinGeckoAPI
from core.database import get_metadata_db
from data.storage.models import Token as TokenModel
import logging
from sqlalchemy import select
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
CG = CoinGeckoAPI()

def seed_top_n(n=200):
    try:
        coins = CG.get_coins_markets(vs_currency='usd', per_page=250, page=1)
    except Exception as e:
        logger.error("CoinGecko API error: %s", e)
        return

    with get_metadata_db() as db:
        try:
            seen_symbols = set()
            updated_count = 0
            inserted_count = 0
            for c in coins[:n]:
                if 'symbol' in c and 'id' in c and 'name' in c:
                    try:
                        symbol = c['symbol'].upper()
                        if symbol in seen_symbols:
                            continue
                        seen_symbols.add(symbol)
                        existing_token = db.execute(
                            select(TokenModel).where(TokenModel.symbol == symbol)
                        ).scalar_one_or_none()
                        
                        if existing_token:
                            existing_token.coingecko_id = c['id']
                            existing_token.name = c['name']
                            existing_token.decimals = None
                            existing_token.token_metadata = c
                            logger.debug("Updating token: %s", symbol)
                            updated_count += 1
                        else:
                            new_token = TokenModel(
                                symbol=symbol,
                                coingecko_id=c['id'],
                                name=c['name'],
                                decimals=None,
                                token_metadata=c
                            )
                            db.add(new_token)
                            logger.debug("Inserting new token: %s", symbol)
                            inserted_count += 1
                    except Exception as token_e:
                        logger.warning("Skipping invalid token %s: %s", c.get('symbol'), token_e)
            
            db.commit()
            logger.info("Seeded/updated %d tokens. Inserted: %d, Updated: %d", (inserted_count + updated_count), inserted_count, updated_count)

        except Exception as e:
            db.rollback()
            logger.error("DB error during seeding: %s", e)

if __name__ == "__main__":
    seed_top_n()
