from web3 import Web3
from sqlalchemy import select
from typing import Optional
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy.orm import Session
import logging

from core.config import settings
from data.validation import WhaleAlert, ChainState
from data.storage.crud import upsert_whale_alerts, update_chain_state
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)").hex()

def get_last_chain_block(db: Session, chain: str = 'ethereum'):
    from data.storage.models import ChainState
    state = db.execute(select(ChainState).where(ChainState.chain == chain)).scalar_one_or_none()
    return state.last_block if state else None

def set_last_chain_block(db: Session, chain: str, blk: int):
    from data.validation import ChainState
    from data.storage.crud import update_chain_state
    update_chain_state(db, ChainState(chain=chain, last_block=blk, last_updated=datetime.now(timezone.utc)))

def scan_eth_transfers(db: Session, batch_blocks: int = 100, threshold_wei: int = 10**18): 
    try:
        try:
            w3 = Web3(Web3.WebsocketProvider(f"wss://eth-mainnet.g.alchemy.com/v2/{settings.ALCHEMY_API_KEY}"))
            if not w3.is_connected():
                raise Exception("WebSocket connection failed")
            logger.info("Connected to Ethereum via WebSocket.")
        except Exception as e:
            logger.warning(f"WebSocket connection failed: {e}. Falling back to HTTP.")
            w3 = Web3(Web3.HTTPProvider(f"https://eth-mainnet.g.alchemy.com/v2/{settings.ALCHEMY_API_KEY}"))
            if not w3.is_connected():
                logger.error("Failed to connect to Ethereum via HTTP.")
                return 0
            logger.info("Connected to Ethereum via HTTP.")
            
        latest_block = w3.eth.block_number
        logger.info(f"Latest block: {latest_block}")
        
        from_block = get_last_chain_block(db, 'ethereum')
        if from_block is None:
            from_block = latest_block - batch_blocks
            logger.info(f"No previous block found, starting from block {from_block}")
        else:
            from_block += 1

        to_block = min(from_block + batch_blocks, latest_block)

        if from_block > to_block:
            logger.info("All blocks are scanned up to %s, nothing to do.", latest_block)
            return 0

        logger.info(f"Scanning ethereum from block {from_block} to {to_block}")

        try:
            logs = w3.eth.get_logs({
                "fromBlock": from_block,
                "toBlock": to_block,
                "topics": [TRANSFER_TOPIC]
            })
            logger.info(f"Successfully fetched {len(logs)} logs")
            
        except Exception as e:
            logger.error(f"get_logs failed: {e}")
            if batch_blocks > 10:
                logger.info("Trying with smaller batch size...")
                return scan_eth_transfers(db, batch_blocks=10, threshold_wei=threshold_wei)
            return 0

        alerts = []
        for log in logs:
            try:
                token_address = log['address']
                from_addr = '0x' + log['topics'][1].hex()[-40:]
                to_addr = '0x' + log['topics'][2].hex()[-40:]
                amount = int.from_bytes(log['data'], 'big')
                
                if amount >= threshold_wei:
                    block_timestamp = w3.eth.get_block(log['blockNumber'])['timestamp']
                    wa = WhaleAlert(
                        time=datetime.fromtimestamp(block_timestamp, tz=timezone.utc),
                        tx_hash=log['transactionHash'].hex(),
                        chain='ethereum',
                        from_address=from_addr,
                        to_address=to_addr,
                        amount=Decimal(amount),
                        asset=token_address,
                        raw=dict(log)
                    )
                    alerts.append(wa)
            except Exception as e:
                logger.error(f"Error processing log: {e}")
                continue
        
        if alerts:
            upsert_whale_alerts(db, alerts)
            logger.info(f"Inserted {len(alerts)} whale alerts")
        
        set_last_chain_block(db, 'ethereum', to_block)
        db.commit()
        logger.info(f"Finished scanning ethereum up to block {to_block}")
        return len(alerts)  

    except Exception as e:
        logger.error(f"Whale alert scanning failed: {e}")
        return 0  

