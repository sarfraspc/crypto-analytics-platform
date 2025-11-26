#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p logs

echo "[$(date -Iseconds)] Starting scheduled Prophet retraining" >> logs/cron_retrain_prophet.log 2>&1

docker compose exec -T backend \
  python -m modules.forecasting.retrain_all \
    --models prophet \
    --retrain-if-exists \
  >> logs/cron_retrain_prophet.log 2>&1

echo "[$(date -Iseconds)] Prophet retraining finished" >> logs/cron_retrain_prophet.log 2>&1

