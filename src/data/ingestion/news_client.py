import time
import logging
import json
from datetime import datetime, timezone
from typing import Optional

import requests
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from core.config import settings
from core.database import get_timescale_engine
from data.validation import NewsArticle, RedditPost
from data.storage.crud import upsert_news, upsert_reddit
from sqlalchemy import text

logger = logging.getLogger(__name__)
vader = SentimentIntensityAnalyzer()

TS_ENG = get_timescale_engine()

def ingest_cryptopanic(api_key: Optional[str] = None, limit: int = 50, max_retries: int = 3):
    """Ingest recent news from CryptoPanic (free tier)."""
    key = api_key or settings.CRYPTOPANIC_API_KEY
    if not key:
        logger.warning("CryptoPanic API key not configured")
        return
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={key}&kind=news&public=true"
    backoff = 1
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            j = resp.json()
            posts = []
            for p in j.get('results', [])[:limit]:
                aid = str(p.get('id') or (p.get('published_at') or "") + (p.get('title') or ''))
                article = NewsArticle(
                    id=aid,
                    title=p.get('title'),
                    source=p.get('source', {}).get('title'),
                    url=p.get('url'),
                    published=datetime.fromisoformat(p.get('published_at').replace('Z','+00:00')) if p.get('published_at') else None,
                    text=p.get('body') or p.get('title'),
                    sentiment_score=vader.polarity_scores((p.get('title') or '') + ' ' + (p.get('body') or ''))['compound'],
                    raw=p
                )
                posts.append(article)
            if posts:
                upsert_news(posts)
                logger.info("Inserted %d CryptoPanic articles", len(posts))
            return
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "") or ""
            logger.warning("CryptoPanic HTTPError %s - body: %s", getattr(e.response, "status_code", "N/A"), body[:1000])
            if getattr(e.response, "status_code", None) == 429 and ("quota" in body.lower() or "monthly quota" in body.lower()):
                logger.error("CryptoPanic monthly quota exceeded. Skipping until reset.")
                return
            if getattr(e.response, "status_code", None) == 429:
                logger.warning("Rate limited, backing off %s sec (attempt %d)", backoff, attempt+1)
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

def ingest_fng():
    """Ingest Fear & Greed Index (free)."""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        resp.raise_for_status()
        j = resp.json()
        data = j.get('data', [])
        if not data:
            return None
        item = data[0]
        # Cache in ingestion_jobs
        with TS_ENG.begin() as conn:
            conn.execute(text("""
                INSERT INTO ingestion_jobs (pipeline, last_run, last_success, details)
                VALUES ('fear_greed', now(), now(), :details)
                ON CONFLICT (pipeline) DO UPDATE SET last_run = now(), last_success = now(), details = EXCLUDED.details
            """), {'details': json.dumps(item)})
        logger.info("Ingested FNG: %s %s", item.get('value'), item.get('value_classification'))
        return item
    except Exception as e:
        logger.exception("FNG ingestion failed: %s", e)
        return None

def ingest_reddit_praw(subreddit: str = "cryptocurrency", limit: int = 100):
    """Ingest via PRAW (free with app creds)."""
    cid = settings.REDDIT_CLIENT_ID
    secret = settings.REDDIT_CLIENT_SECRET
    ua = settings.REDDIT_USER_AGENT
    if not (cid and secret and ua):
        logger.warning("PRAW credentials not configured; skipping Reddit.")
        return
    try:
        reddit = praw.Reddit(client_id=cid, client_secret=secret, user_agent=ua, request_timeout=20)
        posts = []
        for submission in reddit.subreddit(subreddit).new(limit=limit):
            created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
            p = RedditPost(
                id=str(submission.id),
                subreddit=subreddit,
                author=str(submission.author) if submission.author else None,
                title=submission.title,
                body=getattr(submission, 'selftext', None),
                score=getattr(submission, 'score', None),
                sentiment_score=vader.polarity_scores(submission.title + ' ' + getattr(submission, 'selftext', ''))['compound'],
                created=created,
                raw={
                    'url': submission.url,
                    'num_comments': submission.num_comments,
                    'permalink': submission.permalink
                }
            )
            posts.append(p)
        if posts:
            upsert_reddit(posts)
            logger.info("Inserted %d Reddit posts (PRAW) for %s", len(posts), subreddit)
    except Exception as e:
        logger.exception("PRAW ingestion failed: %s", e)

def ingest_reddit_pushshift(subreddit: str = "cryptocurrency", limit: int = 100, max_retries: int = 3):
    """Fallback Pushshift (no auth)."""
    url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&size={limit}&sort=desc&sort_type=created_utc"
    backoff = 1
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            j = r.json()
            posts = []
            for item in j.get('data', []):
                created = datetime.fromtimestamp(item.get('created_utc', 0), tz=timezone.utc)
                p = RedditPost(
                    id=str(item.get('id')),
                    subreddit=subreddit,
                    author=item.get('author'),
                    title=item.get('title'),
                    body=item.get('selftext'),
                    score=item.get('score'),
                    sentiment_score=vader.polarity_scores(item.get('title', '') + ' ' + item.get('selftext', ''))['compound'],
                    created=created,
                    raw=item
                )
                posts.append(p)
            if posts:
                upsert_reddit(posts)
                logger.info("Inserted %d Reddit posts (Pushshift) for %s", len(posts), subreddit)
            return
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "")
            logger.warning("Pushshift HTTPError %s - body: %s", e.response.status_code, body[:1000])
            time.sleep(backoff)
            backoff *= 2
            continue
        except Exception as e:
            logger.exception("Pushshift ingestion failed: %s", e)
            time.sleep(backoff)
            backoff *= 2
    logger.info("Pushshift ingestion finished.")