#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p logs

echo "[$(date -Iseconds)] Starting scheduled ingestion cycle" >> logs/cron_ingestion.log 2>&1

# 1) Market data (incremental OHLCV + TA)
docker compose exec -T backend \
  python -m data.market_client --pipeline full_cycle \
  >> logs/cron_market.log 2>&1

# 2) News data (CryptoPanic + Reddit + FNG)
docker compose exec -T backend \
  python -m data.news_client --mode cycle \
  >> logs/cron_news.log 2>&1

# 3) On-chain pipeline (ingestion + metrics)
docker compose exec -T backend \
  python -m data.onchain_client --run all \
  >> logs/cron_onchain.log 2>&1

# 4) RAG index ingestion for chatbot
docker compose exec -T backend \
  python -m modules.sentiment.rag.rag_pipeline --ingest \
  >> logs/cron_rag_ingest.log 2>&1

echo "[$(date -Iseconds)] Ingestion cycle complete" >> logs/cron_ingestion.log 2>&1

