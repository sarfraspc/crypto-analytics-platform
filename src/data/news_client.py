import asyncio
import logging
from datetime import datetime
from typing import Dict, List

from sqlalchemy.orm import Session

from core.config import settings
from core.database import get_timescale_db
from core.logging_config import setup_logging
from data.ingestion.news_client import ingest_cryptopanic, ingest_reddit_praw, ingest_fng
from data.market_client import log_ingestion_to_mlflow, setup_mlflow

setup_logging()
logger = logging.getLogger(__name__)


async def run_news_cycle(
    db_timescale: Session,
    pipeline: str = "news_cycle",
    subreddit: str = "cryptocurrency",
    reddit_limit: int = 50,
) -> Dict:
    """
    Run a single news/Reddit/FNG ingestion cycle and log to the
    dedicated MLflow experiment for news (`news_client`).
    """
    logger.info(
        "Starting news cycle: CryptoPanic, Reddit(%s, limit=%s), FNG",
        subreddit,
        reddit_limit,
    )

    loop = asyncio.get_running_loop()
    alt_data_tasks = [
        loop.run_in_executor(None, ingest_cryptopanic, db_timescale),
        loop.run_in_executor(None, ingest_reddit_praw, db_timescale, subreddit, reddit_limit),
        loop.run_in_executor(None, ingest_fng, db_timescale),
    ]

    alt_start = datetime.now()
    alt_data_results = await asyncio.gather(*alt_data_tasks, return_exceptions=True)
    alt_duration = (datetime.now() - alt_start).total_seconds()

    cp_count, cp_skipped = (0, 0)
    reddit_count, reddit_skipped = (0, 0)
    fng_count, fng_skipped = (0, 0)

    if len(alt_data_results) >= 1:
        res0 = alt_data_results[0]
        if not isinstance(res0, Exception):
            if isinstance(res0, tuple) and len(res0) >= 2:
                cp_count, cp_skipped = res0
            else:
                cp_count = res0
    if len(alt_data_results) >= 2:
        res1 = alt_data_results[1]
        if not isinstance(res1, Exception):
            if isinstance(res1, tuple) and len(res1) >= 2:
                reddit_count, reddit_skipped = res1
            else:
                reddit_count = res1
    if len(alt_data_results) >= 3:
        res2 = alt_data_results[2]
        if not isinstance(res2, Exception):
            if isinstance(res2, tuple) and len(res2) >= 2:
                fng_count, fng_skipped = res2
            else:
                fng_count = res2

    news_details = {
        "cryptopanic_inserted": cp_count,
        "cryptopanic_skipped": cp_skipped,
        "reddit_inserted": reddit_count,
        "reddit_skipped": reddit_skipped,
        "fng_inserted": fng_count,
        "fng_skipped": fng_skipped,
        "total_inserted": cp_count + reddit_count + fng_count,
        "total_skipped": cp_skipped + reddit_skipped + fng_skipped,
    }
    log_ingestion_to_mlflow("news_client", pipeline, alt_duration, [], news_details)
    logger.info(
        "News Cycle summary: CryptoPanic Inserted: %s, CryptoPanic Skipped: %s, "
        "Reddit Inserted: %s, Reddit Skipped: %s, FNG Inserted: %s, FNG Skipped: %s. "
        "Total Inserted: %s, Total Skipped: %s",
        cp_count,
        cp_skipped,
        reddit_count,
        reddit_skipped,
        fng_count,
        fng_skipped,
        news_details["total_inserted"],
        news_details["total_skipped"],
    )

    return news_details


async def run_news_backfill(
    db_timescale: Session,
    pipeline: str = "news_backfill",
    subreddit: str = "cryptocurrency",
    reddit_limit: int = 50,
) -> Dict:
    """
    Backfill-only version of the news cycle; structurally the same,
    but with a distinct pipeline label for MLflow.
    """
    return await run_news_cycle(
        db_timescale=db_timescale,
        pipeline=pipeline,
        subreddit=subreddit,
        reddit_limit=reddit_limit,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crypto News / Reddit Ingestion Pipeline")
    parser.add_argument(
        "--mode",
        default="cycle",
        choices=["cycle", "backfill"],
        help="Run mode: 'cycle' (default) or 'backfill'",
    )
    parser.add_argument(
        "--subreddit",
        default="cryptocurrency",
        help="Subreddit to use for Reddit ingestion (default: cryptocurrency)",
    )
    parser.add_argument(
        "--reddit-limit",
        type=int,
        default=50,
        help="Number of Reddit posts to fetch (default: 50)",
    )
    args = parser.parse_args()

    setup_mlflow()
    with get_timescale_db() as db_timescale:
        if args.mode == "backfill":
            asyncio.run(
                run_news_backfill(
                    db_timescale=db_timescale,
                    subreddit=args.subreddit,
                    reddit_limit=args.reddit_limit,
                )
            )
        else:
            asyncio.run(
                run_news_cycle(
                    db_timescale=db_timescale,
                    subreddit=args.subreddit,
                    reddit_limit=args.reddit_limit,
                )
            )
