from web3 import Web3
from sqlalchemy import select
from typing import List, Optional, Any
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy.orm import Session
import logging
import time
from tenacity import retry, wait_exponential, stop_after_attempt

from core.config import settings
from data.validation import WhaleAlert, ChainState, OnchainMetric
from data.storage.crud import upsert_whale_alerts, update_chain_state, upsert_onchain_metrics
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

TRANSFER_TOPIC = "0x" + Web3.keccak(text="Transfer(address,address,uint256)").hex()

def get_last_chain_block(db: Session, chain: str = 'ethereum'):
    from data.storage.models import ChainState as ChainStateModel
    state = db.execute(select(ChainStateModel).where(ChainStateModel.chain == chain)).scalar_one_or_none()
    return state.last_block if state else None

@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def _get_logs_with_retry(w3, from_block, to_block):
    return w3.eth.get_logs({
        "fromBlock": from_block,
        "toBlock": to_block,
        "topics": [TRANSFER_TOPIC]
    })

def get_token_decimals(w3, token_address: str):
    if not hasattr(get_token_decimals, "_cache"):
        get_token_decimals._cache = {}
    
    if token_address in get_token_decimals._cache:
        return get_token_decimals._cache[token_address]

    DECIMALS_ABI = [{"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "payable": False, "stateMutability": "view", "type": "function"}]
    
    try:
        token_contract = w3.eth.contract(address=Web3.to_checksum_address(token_address), abi=DECIMALS_ABI)
        decimals = token_contract.functions.decimals().call()
        get_token_decimals._cache[token_address] = decimals
        return decimals
    except Exception as e:
        logger.warning(f"Could not fetch decimals for {token_address}: {e}")
        return 18  

def clean_hexbytes(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: clean_hexbytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_hexbytes(item) for item in obj]
    elif hasattr(obj, 'hex'):
        return obj.hex()
    return obj

def calculate_metrics_from_logs(w3, logs: List, alerts: List[WhaleAlert], unique_addrs: set, total_logs: int, block_timestamp: int):
    if not logs:
        return []
    
    try:
        total_tx = len(set(log['transactionHash'].hex() for log in logs))  
        total_logs = len(logs)
        
        latest_block = w3.eth.get_block('latest')
        avg_gas_price = Decimal(latest_block.get('baseFeePerGas', 0) or 20000000000)  
        active_addrs_total = len(unique_addrs) or 1000  
        
        time = datetime.fromtimestamp(block_timestamp, tz=timezone.utc)
        
        eth_net_flow = Decimal(0)
        token_net_flows = {}
        for alert in alerts:
            if alert.asset and (alert.from_address and alert.from_address.lower() in settings.exchange_addrs or alert.to_address and alert.to_address.lower() in settings.exchange_addrs):
                decimals = get_token_decimals(w3, alert.asset)
                normalized_amount = alert.amount / Decimal(10**decimals)
                
                if alert.from_address and alert.from_address.lower() in settings.exchange_addrs:
                    token_net_flows[alert.asset] = token_net_flows.get(alert.asset, Decimal(0)) - normalized_amount
                if alert.to_address and alert.to_address.lower() in settings.exchange_addrs:
                    token_net_flows[alert.asset] = token_net_flows.get(alert.asset, Decimal(0)) + normalized_amount

        metrics_list = [
            OnchainMetric(time=time, chain='ethereum', metric='total_transactions', value=Decimal(total_tx)),
            OnchainMetric(time=time, chain='ethereum', metric='avg_gas_price_wei', value=avg_gas_price),
            OnchainMetric(time=time, chain='ethereum', metric='erc20_transfer_count', value=Decimal(total_logs)),
            OnchainMetric(time=time, chain='ethereum', metric='active_addresses_total', value=Decimal(active_addrs_total)),
            OnchainMetric(time=time, chain='ethereum', metric='exchange_net_flow_eth_wei', value=eth_net_flow),
        ]

        for asset, flow in token_net_flows.items():
            metrics_list.append(OnchainMetric(time=time, chain='ethereum', metric=f'exchange_net_flow_{asset.lower()}_total', value=flow))

        return metrics_list
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return []

def process_transfer_log(log, threshold_wei: int, block_timestamp: int):
    try:
        token_address = log['address']
        from_addr = '0x' + log['topics'][1].hex()[-40:]
        to_addr = '0x' + log['topics'][2].hex()[-40:]
        amount = int.from_bytes(log['data'], 'big')
        
        addresses = {from_addr.lower(), to_addr.lower()}
        
        if amount >= threshold_wei:
            raw = clean_hexbytes(dict(log)) 
            
            alert = WhaleAlert(
                time=datetime.fromtimestamp(block_timestamp, tz=timezone.utc),
                tx_hash=log['transactionHash'].hex(),
                chain='ethereum',
                from_address=from_addr,
                to_address=to_addr,
                amount=Decimal(amount),
                asset=token_address,
                raw=raw
            )
            return alert, addresses
        
        return None, addresses
    except Exception as e:
        logger.error(f"Error processing transfer log: {e}")
        return None, set()

def scan_eth_transfers(db: Session, batch_size: int = 500, threshold_eth: float = 500.0):
    logger.info(f"Starting Ethereum scan with threshold: {threshold_eth} ETH")
    try:
        w3 = Web3(Web3.HTTPProvider(settings.INFURA_HTTPS))
        if not w3.is_connected():
            raise Exception("Failed to connect to Ethereum")

        last_block = get_last_chain_block(db, 'ethereum')  
        current_block = w3.eth.block_number
        if last_block is None:
            last_block = current_block - 100
        if last_block >= current_block:
            return {'whale_alerts': 0, 'onchain_metrics': 0}

        start_block = last_block + 1
        end_block = min(current_block, last_block + batch_size)

        logger.info(f"Processing blocks {start_block} to {end_block}")
        
        logs = _get_logs_with_retry(w3, start_block, end_block)
        logger.info(f"Fetched {len(logs)} transfer logs")

        end_block_data = w3.eth.get_block(end_block)
        block_timestamp = end_block_data['timestamp']

        alerts = []
        unique_addrs = set()
        total_transfer_logs = len(logs)

        for log in logs:
            alert, addrs = process_transfer_log(log, int(threshold_eth * 10**18), block_timestamp)
            if alert:
                decimals = get_token_decimals(w3, alert.asset)
                alert.amount = alert.amount / Decimal(10**decimals)
                alerts.append(alert)
            unique_addrs.update(addrs)

        metrics = calculate_metrics_from_logs(w3, logs, alerts, unique_addrs, total_transfer_logs, block_timestamp)

        if alerts:
            upsert_whale_alerts(db, alerts)
        if metrics:
            upsert_onchain_metrics(db, metrics)
        update_chain_state(db, ChainState(chain='ethereum', last_block=end_block, last_updated=datetime.now(timezone.utc)))

        logger.info(f"Scan complete: {len(alerts)} alerts, {len(metrics)} metrics")
        return {'whale_alerts': len(alerts), 'onchain_metrics': len(metrics)}

    except Exception as e:
        logger.error(f"Ethereum scan failed: {e}")
        if db.in_transaction():
            db.rollback()
        return {'whale_alerts': 0, 'onchain_metrics': 0}