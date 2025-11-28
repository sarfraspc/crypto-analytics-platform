"""News and social media ingestion from CryptoPanic, Reddit, and Alternative.me."""

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

import praw
import requests
from prawcore import exceptions as prawcore_exceptions
from sqlalchemy.orm import Session
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from core.config import settings
from core.logging_config import setup_logging
from data.storage.crud import get_last_success, update_ingestion_job, upsert_news, upsert_reddit
from data.storage.models import IngestionJob as IngestionJobModel
from data.validation import IngestionJob, NewsArticle, RedditPost

setup_logging()
logger = logging.getLogger(__name__)
vader = SentimentIntensityAnalyzer()

def ingest_fng(db: Session):
    """Ingest Fear & Greed Index from Alternative.me API."""
    logger.info("Starting FNG ingestion")
    try:
        resp = requests.get(settings.ALTERNATIVE_ME_URL, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        data = j.get('data', [])
        if not data:
            return 0, 0  
        item = data[0]
        item = dict(item) if isinstance(item, (list, str)) else item or {}
        if isinstance(item, dict):
            classification = item.get('value_classification', '')
            classification_map = {
                'Extreme Fear': -1.0, 'Fear': -0.5, 'Neutral': 0.0, 'Greed': 0.5, 'Extreme Greed': 1.0
            }
            score = classification_map.get(classification, vader.polarity_scores(classification)['compound'])
            item['score'] = score
        else:
            item = {}
        job_model = IngestionJobModel(
            pipeline='fear_greed', last_run=datetime.now(timezone.utc), last_success=datetime.now(timezone.utc),
            details=item
        )
        db.merge(job_model)
        db.commit()
        logger.info("Ingested FNG: %s %s", item.get('value'), item.get('value_classification'))
        return 1, 0 
    except Exception as e:
        logger.exception("FNG ingestion failed: %s", e)
        return 0, 0  

def ingest_cryptopanic(db: Session, api_key: Optional[str] = None, limit: int = 50, max_retries: int = 3, ingestion_pipeline: str = 'cryptopanic_ingest'):
    """Ingest news articles from CryptoPanic API with sentiment scoring."""
    logger.info("Starting CryptoPanic ingestion")
    key = api_key or settings.CRYPTOPANIC_API_KEY
    if not key:
        logger.warning("CryptoPanic API key not configured")
        return 0, 0

    last_success_time = get_last_success(db, ingestion_pipeline)
    url = f"{settings.CRYPTOPANIC_URL}?auth_token={key}&kind=news&public=true&currencies=BTC&filter=global&limit=50&page=1&hourly=true"
    backoff = 1
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            j = resp.json()
            logger.info(f"API returned {len(j.get('results', []))} articles")
            posts = []
            skipped_count = 0
            for p in j.get('results', [])[:limit]:
                try:
                    published_time = datetime.fromisoformat(p.get('published_at').replace('Z', '+00:00')) if p.get('published_at') else None
                except ValueError:
                    published_time = None
                    logger.warning("Bad timestamp: %s", p.get('published_at'))
                if published_time is None:
                    published_time = datetime.now(timezone.utc)
                    logger.warning("Using now() for missing timestamp")

                if published_time and published_time < last_success_time:
                    skipped_count += 1
                    continue

                if p.get('id'):
                    aid = str(p.get('id'))
                else:
                    published_at = p.get('published_at') or ""
                    title = p.get('title') or ''
                    aid = hashlib.sha256((published_at + title).encode('utf-8')).hexdigest()

                title_text = p.get('title') or ''
                body_text = p.get('body') or title_text
                cleaned_text = re.sub(r'<[^>]+>', '', body_text)[:5000]

                article = NewsArticle(
                    id=aid,
                    title=title_text,
                    source=p.get('source', {}).get('title'),
                    url=p.get('url'),
                    published=published_time,
                    text=cleaned_text,
                    score=vader.polarity_scores(title_text + ' ' + cleaned_text)['compound'],
                    raw=p
                )
                posts.append(article)
            if posts:
                upsert_news(db, posts)
                logger.info("Inserted %d CryptoPanic articles, skipped %d old.", len(posts), skipped_count)
                update_ingestion_job(db, IngestionJob(
                    pipeline=ingestion_pipeline,
                    last_run=datetime.now(timezone.utc),
                    last_success=datetime.now(timezone.utc),
                    details={'fetched_count': len(posts), 'skipped_count': skipped_count}
                ))
                return len(posts), skipped_count  
            else:
                logger.info("No new CryptoPanic articles to insert. Skipped %d old ones.", skipped_count)
                return 0, skipped_count  
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "") or ""
            status_code = getattr(e.response, "status_code", "N/A")
            logger.warning("CryptoPanic HTTPError %s - body: %s", status_code, body[:1000])
            if status_code == 401:
                logger.error("Invalid API key for CryptoPanic. Aborting.")
                return 0, 0
            if status_code == 429 and ("quota" in body.lower() or "monthly quota" in body.lower()):
                logger.error("CryptoPanic monthly quota exceeded. Skipping until reset.")
                return 0, 0
            if status_code == 429:
                logger.warning("Rate limited, backing off %s sec (attempt %d)", backoff, attempt + 1)
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                logger.exception("CryptoPanic ingestion failed: %s", e)
                break
        except Exception as e:
            logger.exception("CryptoPanic error: %s", e)
            time.sleep(backoff)
            backoff *= 2
    logger.info("CryptoPanic ingestion finished.")
    return 0, 0  

def ingest_reddit_praw(db: Session, subreddit: str = "cryptocurrency", limit: int = 200, ingestion_pipeline: str = 'reddit_praw_ingest'):
    """Ingest Reddit posts from subreddit using PRAW with sentiment scoring."""
    logger.info("Starting Reddit PRAW ingestion for /r/%s", subreddit)
    cid = settings.REDDIT_CLIENT_ID
    secret = settings.REDDIT_CLIENT_SECRET
    ua = settings.REDDIT_USER_AGENT
    if not (cid and secret and ua):
        logger.warning("PRAW credentials not configured; skipping Reddit.")
        return 0, 0

    last_success_time = get_last_success(db, ingestion_pipeline)

    try:
        reddit = praw.Reddit(client_id=cid, client_secret=secret, user_agent=ua, request_timeout=20)
        posts = []
        skipped_count = 0

        for submission in reddit.subreddit(subreddit).new(limit=limit):
            created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
            if created < last_success_time:
                skipped_count += 1
                continue

            title_text = submission.title or ''
            body_text = getattr(submission, 'selftext', '') or ''
            cleaned_body = re.sub(r'<[^>]+>', '', body_text)[:5000]

            p = RedditPost(
                id=str(submission.id),
                subreddit=subreddit,
                author=str(submission.author) if submission.author else None,
                title=title_text,
                body=cleaned_body,
                upvote_score=getattr(submission, 'score', None),
                score=vader.polarity_scores(title_text + ' ' + cleaned_body)['compound'],
                created=created,
                raw={
                    'url': submission.url,
                    'num_comments': submission.num_comments,
                    'permalink': submission.permalink
                }
            )
            posts.append(p)
        if posts:
            upsert_reddit(db, posts)
            logger.info("Inserted %d Reddit posts (PRAW) for %s, skipped %d old.", len(posts), subreddit, skipped_count)
            update_ingestion_job(db, IngestionJob(
                pipeline=ingestion_pipeline,
                last_run=datetime.now(timezone.utc),
                last_success=datetime.now(timezone.utc),
                details={'fetched_count': len(posts), 'skipped_count': skipped_count}
            ))
            return len(posts), skipped_count  
        else:
            logger.info("No new PRAW posts for %s to insert. Skipped %d old ones.", subreddit, skipped_count)
            return 0, skipped_count  
    except prawcore_exceptions.ResponseException as e:
        if e.response.status_code == 401:
            logger.error("Invalid PRAW credentials. Aborting Reddit ingestion.")
            return 0, 0
        logger.exception("PRAW API error: %s", e)
    except Exception as e:
        logger.exception("PRAW ingestion failed: %s", e)
    return 0, 0  