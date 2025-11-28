"""Data ingestion clients for market, news, and on-chain data sources."""

from data.ingestion import chain_client, market_client, news_client

__all__ = ["chain_client", "market_client", "news_client"]
