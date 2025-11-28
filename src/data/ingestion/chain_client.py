"""Ethereum blockchain client for whale transfer detection and ingestion."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.orm import Session
from web3 import Web3

from core.config import settings
from core.logging_config import setup_logging
from data.storage.crud import upsert_whale_alerts
from data.validation import WhaleAlert

setup_logging()
logger = logging.getLogger(__name__)

TRANSFER_TOPIC = "0x" + Web3.keccak(text="Transfer(address,address,uint256)").hex()

EXCHANGE_ADDRESSES = settings.EXCHANGE_ADDRESSES
ALL_EXCHANGE_ADDRS = set(addr.lower() for addrs in EXCHANGE_ADDRESSES.values() for addr in addrs)

def get_logs_in_chunks(
    w3,
    from_block: int,
    to_block: int,
    known_tokens: List[str],
    chunk_size: int = 120,
) -> List[Dict]:
    """
    Fetch Transfer logs in smaller block chunks to avoid provider
    limits (e.g., Infura's 10k log cap per query).
    """
    all_logs: List[Dict] = []
    if from_block > to_block:
        return all_logs

    # Start with a safe-ish chunk size and shrink dynamically
    # if the provider reports >10k logs (Infura code -32005).
    current_chunk = max(20, int(chunk_size))
    start = from_block

    while start <= to_block:
        end = min(start + current_chunk - 1, to_block)
        try:
            logs = w3.eth.get_logs(
                {
                    "fromBlock": start,
                    "toBlock": end,
                    "topics": [TRANSFER_TOPIC],
                    "address": [Web3.to_checksum_address(addr) for addr in known_tokens],
                }
            )
            all_logs.extend(logs)
            logger.info(f"Chunk {start}-{end} (size={current_chunk}) fetched {len(logs)} logs")
            start = end + 1
        except Exception as e:
            msg = str(e)
            too_many_logs = "more than 10000 results" in msg or "10000 results" in msg or "-32005" in msg
            if too_many_logs and current_chunk > 20:
                # Shrink chunk and retry same start range
                new_chunk = max(20, current_chunk // 2)
                logger.warning(
                    f"Chunk {start}-{end} hit provider log cap; "
                    f"shrinking chunk_size {current_chunk} -> {new_chunk} and retrying"
                )
                current_chunk = new_chunk
                continue

            logger.warning(f"Chunk get_logs failed for blocks {start}-{end}: {e}", exc_info=True)
            # Skip this problematic range and move on
            start = end + 1

    logger.info(f"Total logs fetched across chunks: {len(all_logs)}")
    return all_logs

def get_token_decimals(w3, token_address: str) -> Optional[int]:
    """Get token decimals from cache, static metadata, or on-chain call."""
    if not hasattr(get_token_decimals, "_cache"):
        get_token_decimals._cache = {}
    
    if token_address in get_token_decimals._cache:
        return get_token_decimals._cache[token_address]

    # First, try static metadata (no network call)
    meta = settings.STATIC_TOKEN_METADATA.get(token_address.lower())
    if meta and "decimals" in meta:
        try:
            decimals = int(meta["decimals"])
            get_token_decimals._cache[token_address] = decimals
            return decimals
        except (TypeError, ValueError):
            logger.warning(f"Invalid decimals value in STATIC_TOKEN_METADATA for {token_address}: {meta.get('decimals')}")

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
    """Recursively convert HexBytes objects to hex strings for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_hexbytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_hexbytes(item) for item in obj]
    elif hasattr(obj, 'hex'):
        return obj.hex()
    return obj

def process_transfer_log(
    w3,
    prices: Dict[str, float],
    log: Dict,
    block_timestamps: Dict[int, int],
    threshold_usd: float,
    min_timestamp: Optional[int] = None
) -> tuple[Optional[WhaleAlert], Set[str]]:
    """Process a single ERC-20 Transfer log and return WhaleAlert if above threshold."""
    try:
        if len(log['topics']) < 3:
            logger.warning(f"Skipping malformed log (short topics): {log.get('transactionHash', 'unknown').hex() if hasattr(log.get('transactionHash'), 'hex') else 'unknown'}")
            return None, set()

        token_address = log['address'].lower()
        from_addr = '0x' + log['topics'][1].hex()[-40:].lower()
        to_addr = '0x' + log['topics'][2].hex()[-40:].lower()
        amount_raw = int.from_bytes(log['data'], 'big')
        
        addresses = {from_addr, to_addr}
        
        block_number = log['blockNumber']
        if block_number in block_timestamps:
            block_timestamp = int(block_timestamps[block_number])
        else:
            block_data = w3.eth.get_block(block_number)
            block_timestamp = int(block_data['timestamp'])

        if min_timestamp is not None and block_timestamp < min_timestamp:
            return None, addresses
        
        if from_addr in ALL_EXCHANGE_ADDRS and to_addr in ALL_EXCHANGE_ADDRS:
            logger.debug(f"Skipping internal exchange transfer: {from_addr} -> {to_addr}")
            return None, addresses

        decimals = get_token_decimals(w3, token_address)
        if decimals is None:
            logger.warning(f"Skipping token with unknown decimals: {token_address}")
            return None, addresses
        amount = Decimal(amount_raw) / Decimal(10**decimals)

        price_usd = prices.get(token_address)
        if not price_usd or price_usd <= 0:
            logger.warning(f"Skipping token with missing/zero price: {token_address}")
            return None, addresses

        value_usd = float(amount) * price_usd
        if value_usd < threshold_usd:
            return None, addresses

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

def scan_eth_transfers(db: Session, threshold_usd: float = 500000.0):
    """
    Timestamp-based 24h rolling ingestion for Ethereum whale transfers.

    This no longer relies on ChainState or incremental block scanning.
    Each run:
      - Estimates a block range covering the last 24h
      - Fetches logs once for that range
      - Filters by timestamp and USD value
      - Upserts whale alerts idempotently into Timescale
    """
    logger.info(f"Starting Ethereum 24h whale scan (USD threshold={threshold_usd:,})")
    try:
        w3 = Web3(Web3.HTTPProvider(settings.INFURA_HTTPS))
        if not w3.is_connected():
            raise Exception("Failed to connect to Ethereum")

        now_ts = int(datetime.now(timezone.utc).timestamp())
        min_ts = now_ts - 24 * 60 * 60

        latest_block = w3.eth.get_block("latest")
        latest_number = latest_block["number"]

        # Estimate average seconds per block from recent history
        lookback_blocks = min(10_000, max(100, int(24 * 60 * 60 / 12)))
        sample_start = max(1, latest_number - lookback_blocks)
        sample_block = w3.eth.get_block(sample_start)

        seconds_diff = max(1, int(latest_block["timestamp"]) - int(sample_block["timestamp"]))
        blocks_diff = latest_number - sample_start or 1
        avg_seconds_per_block = seconds_diff / blocks_diff

        estimated_window_blocks = int((24 * 60 * 60) / avg_seconds_per_block) + 200
        from_block = max(0, latest_number - estimated_window_blocks)
        to_block = latest_number

        logger.info(
            f"Fetching logs for ~24h window: blocks {from_block}–{to_block} "
            f"(avg_seconds_per_block≈{avg_seconds_per_block:.2f})"
        )

        # Single-token (USDT) setup: use static metadata and assume 1 token ≈ 1 USD.
        known_tokens = [addr.lower() for addr in settings.STATIC_TOKEN_METADATA.keys()]
        prices = {addr: 1.0 for addr in known_tokens}

        logs = get_logs_in_chunks(
            w3=w3,
            from_block=from_block,
            to_block=to_block,
            known_tokens=known_tokens,
            chunk_size=120,
        )
        total_logs = len(logs)
        logger.info(f"Fetched {total_logs} transfer logs in estimated 24h window (chunked)")

        if not logs:
            return {"whale_alerts": 0}

        block_numbers = sorted({log["blockNumber"] for log in logs})
        block_timestamps: Dict[int, int] = {}
        for bn in block_numbers:
            try:
                block_data = w3.eth.get_block(bn)
                block_timestamps[bn] = int(block_data["timestamp"])
            except Exception as e:
                logger.warning(f"Failed to fetch block {bn} timestamp: {e}", exc_info=True)
        logger.info(f"Pre-fetched timestamps for {len(block_timestamps)} unique blocks")

        alerts: List[WhaleAlert] = []
        unique_addrs: Set[str] = set()

        for idx, log in enumerate(logs, start=1):
            alert, addrs = process_transfer_log(
                w3,
                prices,
                log,
                block_timestamps,
                threshold_usd,
                min_timestamp=min_ts,
            )
            if alert:
                alerts.append(alert)
            unique_addrs.update(addrs)

            if idx % 10000 == 0 or idx == total_logs:
                logger.info(f"Processed {idx:,} / {total_logs:,} logs for whale detection...")

        if alerts:
            upsert_whale_alerts(db, alerts)
            logger.info(f"Upserted {len(alerts)} whale alerts from {len(unique_addrs)} unique addresses")
        else:
            logger.info("No whale alerts found in the last 24h window above threshold")

        return {"whale_alerts": len(alerts)}

    except Exception as e:
        logger.error(f"Ethereum 24h scan failed: {e}", exc_info=True)
        if db.in_transaction():
            db.rollback()
        return {"whale_alerts": 0}
