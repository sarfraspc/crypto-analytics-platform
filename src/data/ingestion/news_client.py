import time
import logging
import json
import hashlib
import re
from datetime import datetime, timezone
from typing import Optional

import requests
import praw
from prawcore import exceptions as prawcore_exceptions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from core.config import settings
from core.database import get_timescale_engine
from data.validation import NewsArticle, RedditPost, IngestionJob
from data.storage.crud import upsert_news, upsert_reddit, get_last_success, update_ingestion_job
from sqlalchemy import text

logger = logging.getLogger(__name__)
vader = SentimentIntensityAnalyzer()

TS_ENG = get_timescale_engine()

def ingest_cryptopanic(api_key: Optional[str] = None, limit: int = 50, max_retries: int = 3, ingestion_pipeline: str = 'cryptopanic_ingest'):
    key = api_key or settings.CRYPTOPANIC_API_KEY
    if not key:
        logger.warning("CryptoPanic API key not configured")
        return
    
    last_success_time = get_last_success(ingestion_pipeline)
    
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={key}&kind=news&public=true&currencies=BTC&filter=important"
    
    backoff = 1
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            j = resp.json()
            posts = []
            skipped_count = 0
            for p in j.get('results', [])[:limit]:
                try:
                    published_time = datetime.fromisoformat(p.get('published_at').replace('Z','+00:00')) if p.get('published_at') else None
                except ValueError:
                    published_time = None
                    logger.warning("Bad timestamp: %s", p.get('published_at'))
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
                upsert_news(posts)
                logger.info("Inserted %d CryptoPanic articles, skipped %d old.", len(posts), skipped_count)
                update_ingestion_job(IngestionJob(
                    pipeline=ingestion_pipeline,
                    last_run=datetime.now(timezone.utc),
                    last_success=datetime.now(timezone.utc),
                    details={'fetched_count': len(posts), 'skipped_count': skipped_count}
                ))
            else:
                logger.info("No new CryptoPanic articles to insert. Skipped %d old ones.", skipped_count)
            return
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "") or ""
            status_code = getattr(e.response, "status_code", "N/A")
            logger.warning("CryptoPanic HTTPError %s - body: %s", status_code, body[:1000])
            if status_code == 401:
                logger.error("Invalid API key for CryptoPanic. Aborting.")
                return
            if status_code == 429 and ("quota" in body.lower() or "monthly quota" in body.lower()):
                logger.error("CryptoPanic monthly quota exceeded. Skipping until reset.")
                return
            if status_code == 429:
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
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        resp.raise_for_status()
        j = resp.json()
        data = j.get('data', [])
        if not data:
            return None
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

def ingest_reddit_praw(subreddit: str = "cryptocurrency", limit: int = 200, ingestion_pipeline: str = 'reddit_praw_ingest'):
    cid = settings.REDDIT_CLIENT_ID
    secret = settings.REDDIT_CLIENT_SECRET
    ua = settings.REDDIT_USER_AGENT
    if not (cid and secret and ua):
        logger.warning("PRAW credentials not configured; skipping Reddit.")
        return
    
    last_success_time = get_last_success(ingestion_pipeline)
    
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
            upsert_reddit(posts)
            logger.info("Inserted %d Reddit posts (PRAW) for %s, skipped %d old.", len(posts), subreddit, skipped_count)
            update_ingestion_job(IngestionJob(
                pipeline=ingestion_pipeline,
                last_run=datetime.now(timezone.utc),
                last_success=datetime.now(timezone.utc),
                details={'fetched_count': len(posts), 'skipped_count': skipped_count}
            ))
        else:
            logger.info("No new PRAW posts for %s to insert. Skipped %d old ones.", subreddit, skipped_count)
    except prawcore_exceptions.ResponseException as e:
        if e.response.status_code == 401:
            logger.error("Invalid PRAW credentials. Aborting Reddit ingestion.")
            return
        logger.exception("PRAW API error: %s", e)
    except Exception as e:
        logger.exception("PRAW ingestion failed: %s", e)

def ingest_reddit_pushshift(subreddit: str = "cryptocurrency", limit: int = 100, max_retries: int = 3, ingestion_pipeline: str = 'reddit_pushshift_ingest'):
    last_success_time = get_last_success(ingestion_pipeline)
    after_ts = int(last_success_time.timestamp()) if last_success_time else 0
    url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&size={limit}&sort=desc&sort_type=created_utc&after={after_ts}"
    backoff = 1
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            j = r.json()
            posts = []
            skipped_count = 0
            for item in j.get('data', []):
                created = datetime.fromtimestamp(item.get('created_utc', 0), tz=timezone.utc)
                if created < last_success_time:
                    skipped_count += 1
                    continue

                title_text = item.get('title', '') or ''
                body_text = item.get('selftext', '') or ''
                cleaned_body = re.sub(r'<[^>]+>', '', body_text)[:5000]

                p = RedditPost(
                    id=str(item.get('id')),
                    subreddit=subreddit,
                    author=item.get('author'),
                    title=title_text,
                    body=cleaned_body,
                    upvote_score=item.get('score'),
                    score=vader.polarity_scores(title_text + ' ' + cleaned_body)['compound'],
                    created=created,
                    raw=item
                )
                posts.append(p)
            if posts:
                upsert_reddit(posts)
                logger.info("Inserted %d Reddit posts (Pushshift) for %s, skipped %d old.", len(posts), subreddit, skipped_count)
                update_ingestion_job(IngestionJob(
                    pipeline=ingestion_pipeline,
                    last_run=datetime.now(timezone.utc),
                    last_success=datetime.now(timezone.utc),
                    details={'fetched_count': len(posts), 'skipped_count': skipped_count}
                ))
            else:
                logger.info("No new Pushshift posts for %s to insert. Skipped %d old ones.", subreddit, skipped_count)
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