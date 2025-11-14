from web3 import Web3
from sqlalchemy import select, func
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy.orm import Session
import logging
import time
import random
from pycoingecko import CoinGeckoAPI  
from tenacity import retry, wait_exponential, stop_after_attempt

from core.config import settings
from data.validation import WhaleAlert, ChainState
from data.storage.crud import upsert_whale_alerts, update_chain_state
from core.logging_config import setup_logging
from data.storage.models import Token as TokenModel 

setup_logging()
logger = logging.getLogger(__name__)
CG = CoinGeckoAPI()

TRANSFER_TOPIC = "0x" + Web3.keccak(text="Transfer(address,address,uint256)").hex()

EXCHANGE_ADDRESSES = settings.EXCHANGE_ADDRESSES
ALL_EXCHANGE_ADDRS = set(addr.lower() for addrs in EXCHANGE_ADDRESSES.values() for addr in addrs)

def get_top_eth_tokens(db: Session, limit: int = 20) -> List[Dict[str, str]]:
    try:
        result = db.execute(
            select(TokenModel)
            .where(
                func.jsonb_extract_path_text(TokenModel.token_metadata, 'market_cap_rank').isnot(None),
                func.jsonb_extract_path_text(TokenModel.token_metadata, 'detail_platforms', 'ethereum').isnot(None)
            )
            .order_by(
                func.cast(func.jsonb_extract_path_text(TokenModel.token_metadata, 'market_cap_rank'), func.Integer)
            )
            .limit(limit)
        ).scalars().all()

        top_tokens = []
        for token in result:
            metadata = token.token_metadata or {}
            platforms = metadata.get('detail_platforms', {})
            eth_addr = platforms.get('ethereum')
            if eth_addr:
                top_tokens.append({
                    'contract_addr': eth_addr.lower(),
                    'coingecko_id': token.coingecko_id,
                    'symbol': token.symbol
                })
        
        logger.info(f"Loaded {len(top_tokens)} top ETH tokens from DB (limit={limit})")
        return top_tokens
    except Exception as e:
        logger.error(f"Error fetching top ETH tokens from DB: {e}", exc_info=True)
        return [
            {'contract_addr': addr.lower(), 'coingecko_id': 'weth', 'symbol': 'WETH'}  
            for addr in settings.STATIC_KNOWN_TOKENS
        ][:limit]

def fetch_prices_bulk(coingecko_ids: List[str]) -> Dict[str, float]:
    if not coingecko_ids:
        return {}
    
    try:
        ids_str = ','.join(coingecko_ids)
        prices_data = CG.get_price(ids=ids_str, vs_currencies='usd')
        return {cg_id: data['usd'] for cg_id, data in prices_data.items()}
    except Exception as e:
        logger.warning(f"Failed to bulk fetch prices for {len(coingecko_ids)} IDs: {e}", exc_info=True)
        return {}

def get_last_chain_block(db: Session, chain: str = 'ethereum'):
    from data.storage.models import ChainState as ChainStateModel
    state = db.execute(select(ChainStateModel).where(ChainStateModel.chain == chain)).scalar_one_or_none()
    return state.last_block if state else None

@retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(10))
def _get_logs_with_retry(w3, from_block, to_block, known_tokens: List[str], step=50):
    all_logs = []
    for start in range(from_block, to_block + 1, step):
        end = min(start + step - 1, to_block)
        try:
            logs = w3.eth.get_logs({
                "fromBlock": start,
                "toBlock": end,
                "topics": [TRANSFER_TOPIC],
                "address": [Web3.to_checksum_address(addr) for addr in known_tokens]  
            })
            all_logs.extend(logs)
            time.sleep(1.0 + random.uniform(0, 0.5))  
        except Exception as e:
            logger.warning(f"Partial fetch failed for blocks {start}-{end}: {e}", exc_info=True)
            continue
    return all_logs

def is_erc20(w3, address: str) -> bool:
    SYMBOL_ABI = [{"constant": True, "name": "symbol", "outputs": [{"type": "string"}], "stateMutability": "view", "type": "function"}]
    try:
        token = w3.eth.contract(address=Web3.to_checksum_address(address), abi=SYMBOL_ABI)
        token.functions.symbol().call()
        return True
    except Exception:
        return False

def get_token_decimals(w3, token_address: str) -> Optional[int]:
    if not hasattr(get_token_decimals, "_cache"):
        get_token_decimals._cache = {}
    
    if token_address in get_token_decimals._cache:
        return get_token_decimals._cache[token_address]

    if not is_erc20(w3, token_address):
        logger.warning(f"Skipping non-ERC20 token: {token_address}")
        get_token_decimals._cache[token_address] = None
        return None

    DECIMALS_ABI = [{"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "payable": False, "stateMutability": "view", "type": "function"}]
    
    try:
        token_contract = w3.eth.contract(address=Web3.to_checksum_address(token_address), abi=DECIMALS_ABI)
        decimals = token_contract.functions.decimals().call()
        get_token_decimals._cache[token_address] = decimals
        return decimals
    except Exception as e:
        logger.warning(f"Could not fetch decimals for {token_address}: {e}", exc_info=True)
        get_token_decimals._cache[token_address] = None
        return None

def clean_hexbytes(obj: Any):
    if isinstance(obj, dict):
        return {k: clean_hexbytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_hexbytes(item) for item in obj]
    elif hasattr(obj, 'hex'):
        return obj.hex()
    return obj

def process_transfer_log(w3, prices: Dict[str, float], log: Dict, threshold_usd: float) -> tuple[Optional[WhaleAlert], Set[str]]:
    try:
        if len(log['topics']) < 3:
            logger.warning(f"Skipping malformed log (short topics): {log.get('transactionHash', 'unknown').hex() if hasattr(log.get('transactionHash'), 'hex') else 'unknown'}")
            return None, set()

        token_address = log['address'].lower()
        from_addr = '0x' + log['topics'][1].hex()[-40:].lower()
        to_addr = '0x' + log['topics'][2].hex()[-40:].lower()
        amount_raw = int.from_bytes(log['data'], 'big')
        
        addresses = {from_addr, to_addr}
        
        if from_addr in ALL_EXCHANGE_ADDRS and to_addr in ALL_EXCHANGE_ADDRS:
            logger.debug(f"Skipping internal exchange transfer: {from_addr} -> {to_addr}")
            return None, addresses

        if amount_raw < 10**15:  
            return None, addresses

        decimals = get_token_decimals(w3, token_address)
        if decimals is None:
            decimals = 18 
            logger.debug(f"Using fallback decimals=18 for {token_address}")
        amount = Decimal(amount_raw) / Decimal(10**decimals)

        price_usd = prices.get(token_address, 0.0)  
        value_usd = float(amount) * price_usd
        if value_usd < threshold_usd:
            return None, addresses

        block_number = log['blockNumber']
        block_data = w3.eth.get_block(block_number)
        block_timestamp = block_data['timestamp']

        raw_log = clean_hexbytes(dict(log))
        
        usd_value = Decimal(str(value_usd))

        alert = WhaleAlert(
            time=datetime.fromtimestamp(block_timestamp, tz=timezone.utc),
            tx_hash=log['transactionHash'].hex(),
            chain='ethereum',
            from_address=from_addr,
            to_address=to_addr,
            amount=amount,  
            usd_value=usd_value,
            asset=token_address,
            raw=raw_log
        )
        logger.info(f"Whale alert: {amount} {token_address} ({value_usd:.2f} USD) from {from_addr} to {to_addr}")
        return alert, addresses
        
    except Exception as e:
        logger.error(f"Error processing transfer log: {e}", exc_info=True)
        return None, set()

def scan_eth_transfers(db: Session, batch_size: int = 100, threshold_usd: float = 500000.0): 
    logger.info(f"Starting Ethereum whale scan with USD threshold: {threshold_usd:,}")
    try:
        w3 = Web3(Web3.HTTPProvider(settings.INFURA_HTTPS))
        if not w3.is_connected():
            raise Exception("Failed to connect to Ethereum")

        top_eth_tokens = get_top_eth_tokens(db, limit=20)
        known_tokens = [t['contract_addr'] for t in top_eth_tokens]
        coingecko_ids = [t['coingecko_id'] for t in top_eth_tokens]

        prices_dict = fetch_prices_bulk(coingecko_ids)
        prices = {t['contract_addr']: prices_dict.get(t['coingecko_id'], 0.0) for t in top_eth_tokens}

        last_block = get_last_chain_block(db, 'ethereum')  
        current_block = w3.eth.block_number
        lag = current_block - (last_block or 0)
        if lag > 10000:  # >2 hours lag
            batch_size = min(5000, lag // 10)  # Scale to 5k max, or 10% lag
            logger.info(f"Lag {lag} blocks; scaling batch to {batch_size}")
        if last_block is None:
            last_block = current_block - 100
        if last_block >= current_block:
            return {'whale_alerts': 0}

        start_block = last_block + 1
        end_block = min(current_block, last_block + batch_size)

        logger.info(f"Processing blocks {start_block} to {end_block} for {len(known_tokens)} ETH tokens")
        
        logs = _get_logs_with_retry(w3, start_block, end_block, known_tokens)
        logger.info(f"Fetched {len(logs)} transfer logs")

        if not logs:
            update_chain_state(db, ChainState(chain='ethereum', last_block=end_block, last_updated=datetime.now(timezone.utc)))
            return {'whale_alerts': 0}

        alerts: List[WhaleAlert] = []
        unique_addrs: Set[str] = set()

        for log in logs:
            alert, addrs = process_transfer_log(w3, prices, log, threshold_usd)
            if alert:
                alerts.append(alert)
            unique_addrs.update(addrs)

        if alerts:
            upsert_whale_alerts(db, alerts)
            logger.info(f"Upserted {len(alerts)} whale alerts")

        update_chain_state(db, ChainState(chain='ethereum', last_block=end_block, last_updated=datetime.now(timezone.utc)))

        logger.info(f"Scan complete: {len(alerts)} alerts from {len(unique_addrs)} unique addresses")
        return {'whale_alerts': len(alerts)}

    except Exception as e:
        logger.error(f"Ethereum scan failed: {e}", exc_info=True)
        if db.in_transaction():
            db.rollback()
        return {'whale_alerts': 0}