import time
import logging
from datetime import datetime, timezone
from typing import Optional

from web3 import Web3

from core.config import settings
from core.database import get_timescale_engine
from sqlalchemy import text
from data.validation import WhaleAlert
from data.storage.crud import upsert_whale_alerts

logger = logging.getLogger(__name__)

ALCHEMY_KEY = settings.ALCHEMY_API_KEY
W3 = None
if ALCHEMY_KEY:
    W3 = Web3(Web3.HTTPProvider(f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}"))

TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)").hex() if W3 else None

TS_ENG = get_timescale_engine()

def get_last_chain_block(chain: str = 'ethereum') -> Optional[int]:
    with TS_ENG.begin() as conn:
        r = conn.execute(text("SELECT last_block FROM chain_state WHERE chain=:chain"), {'chain': chain}).fetchone()
        return int(r[0]) if r else None

def set_last_chain_block(chain: str, blk: int):
    with TS_ENG.begin() as conn:
        conn.execute(text("""
            INSERT INTO chain_state (chain, last_block, last_updated)
            VALUES (:chain, :blk, now())
            ON CONFLICT (chain) DO UPDATE SET last_block = EXCLUDED.last_block, last_updated = now()
        """), {'chain': chain, 'blk': int(blk)})

def scan_eth_transfers(batch_blocks: int = 2000, threshold_wei: int = 10**18, max_blocks_per_call: int = 10, chain: str = 'ethereum'):
    """Scan new blocks for transfers (deltas from last_block)."""
    if not W3:
        logger.warning("Alchemy key not configured; skipping on-chain scan")
        return

    last = get_last_chain_block(chain)  # Use param
    head = W3.eth.block_number
    if last is None:
        last = max(head - 1000, 0)
    to_block = min(head, last + batch_blocks)
    if to_block < last + 1:
        logger.info("No new blocks to scan (%s -> %s)", last+1, to_block)
        return

    logger.info("Scanning blocks %d -> %d (total %d) with chunk_size=%d", last+1, to_block, to_block - last, max_blocks_per_call)

    alerts = []
    start = last + 1
    while start <= to_block:
        end = min(start + max_blocks_per_call - 1, to_block)
        logger.debug("Requesting logs %d -> %d", start, end)
        try:
            logs = W3.eth.get_logs({'fromBlock': start, 'toBlock': end, 'topics': [TRANSFER_TOPIC]})
        except Exception as e:
            logger.exception("get_logs failed for range %d-%d: %s", start, end, e)
            start = end + 1
            time.sleep(0.5)
            continue

        for l in logs:
            topics = l.get('topics', []) if isinstance(l, dict) else []
            if len(topics) < 3:
                continue
            try:
                from_addr = '0x' + topics[1].hex()[-40:]
                to_addr = '0x' + topics[2].hex()[-40:]
            except Exception:
                continue

            data_field = l.get('data', b'0x0')
            amount = 0
            try:
                if isinstance(data_field, (bytes, bytearray)):
                    amount = int.from_bytes(data_field, byteorder='big')
                elif isinstance(data_field, str):
                    hexstr = data_field
                    if hexstr.startswith(('0x', '0X')):
                        hexstr = hexstr[2:]
                    amount = int(hexstr, 16)
                else:
                    if hasattr(data_field, 'hex'):
                        amount = int(data_field.hex(), 16)
            except Exception:
                continue

            if amount >= threshold_wei:
                try:
                    blk = W3.eth.get_block(l['blockNumber'])
                    t = datetime.fromtimestamp(blk['timestamp'], tz=timezone.utc)
                    tx_hash_obj = l.get('transactionHash')
                    tx_hash = tx_hash_obj.hex() if hasattr(tx_hash_obj, 'hex') else str(tx_hash_obj)
                    wa = WhaleAlert(
                        time=t, tx_hash=tx_hash, chain=chain,  # Use param
                        from_address=from_addr, to_address=to_addr,
                        amount=amount, asset=None, raw=dict(l) if isinstance(l, dict) else {}
                    )
                    alerts.append(wa)
                except Exception:
                    logger.exception("Failed to create WhaleAlert from log: %s", l)
                    continue

        time.sleep(0.05)  
        start = end + 1

    if alerts:
        upsert_whale_alerts(alerts)
        logger.info("Inserted %d whale alerts for %s", len(alerts), chain)
    set_last_chain_block(chain, to_block)