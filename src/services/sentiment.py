from fastapi import APIRouter, HTTPException, Body
from typing import List
import logging
from datetime import datetime
import re

from modules.agent.agent_client import call_mcp_tool
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sentiment", tags=["Sentiment"])

@router.post("/analyze")
async def analyze_sentiment(text: str = Body(..., description="Text to analyze")):
    start_time = datetime.now()
    request_id = f"sentiment_analyze_{hash(str(start_time)) % 1000000}"
    logger.info(f"[{request_id}] Request: text={text[:100]}...")

    try:
        raw_result = await call_mcp_tool("crypto-sentiment-server", "analyze_sentiment", {"text": text})

        # Improved parsing with regex for key-value extraction
        result_text = raw_result.get("raw_text", raw_result) if isinstance(raw_result, dict) and "raw_text" in raw_result else str(raw_result)
        if "Sentiment Analysis Result:" in result_text:
            # Regex for extraction
            sentiment_match = re.search(r'Sentiment:\s*([^\n,]+)', result_text)
            confidence_match = re.search(r'Confidence:\s*([\d.]+)', result_text)
            bearish_match = re.search(r'Bearish:\s*([\d.]+)', result_text)
            bullish_match = re.search(r'Bullish:\s*([\d.]+)', result_text)
            neutral_match = re.search(r'Neutral:\s*([\d.]+)', result_text)

            sentiment = sentiment_match.group(1).strip().upper() if sentiment_match else "UNKNOWN"
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0
            scores = {
                "bearish": round(float(bearish_match.group(1)) if bearish_match else 0, 3),
                "bullish": round(float(bullish_match.group(1)) if bullish_match else 0, 3),
                "neutral": round(float(neutral_match.group(1)) if neutral_match else 0, 3)
            }

            result = {
                "sentiment": sentiment,
                "confidence": round(confidence, 3),
                "scores": scores
            }
        else:
            result = {"error": "No sentiment data in response"}

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"[{request_id}] Completed in {duration_ms}ms: sentiment={result.get('sentiment')}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error calling Sentiment MCP: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upstream Sentiment server error; please retry.")

@router.post("/batch")
async def analyze_sentiment_batch(texts: List[str] = Body(..., description="List of texts to analyze")):
    start_time = datetime.now()
    request_id = f"sentiment_batch_{hash(str(start_time)) % 1000000}"
    logger.info(f"[{request_id}] Request: {len(texts)} texts")

    try:
        raw_result = await call_mcp_tool("crypto-sentiment-server", "analyze_sentiment_batch", {"texts": texts})

        # Improved regex-based parsing for batch
        result_text = raw_result.get("raw_text", raw_result) if isinstance(raw_result, dict) and "raw_text" in raw_result else str(raw_result)
        results = []
        if "Batch Sentiment Analysis Results:" in result_text:
            # Split into blocks and parse each
            blocks = re.split(r'\n\n', result_text)
            bearish_total, bullish_total, neutral_total, conf_total = 0, 0, 0, 0
            count = 0
            for block in blocks:
                if "Sentiment:" in block:
                    sentiment_match = re.search(r'Sentiment:\s*([^\n,]+)', block)
                    conf_match = re.search(r'Confidence:\s*([\d.]+)', block)
                    bear_match = re.search(r'Bearish:\s*([\d.]+)', block)
                    bull_match = re.search(r'Bullish:\s*([\d.]+)', block)
                    neut_match = re.search(r'Neutral:\s*([\d.]+)', block)

                    sentiment = sentiment_match.group(1).strip().upper() if sentiment_match else "UNKNOWN"
                    conf = float(conf_match.group(1)) if conf_match else 0.0
                    bear = float(bear_match.group(1)) if bear_match else 0.0
                    bull = float(bull_match.group(1)) if bull_match else 0.0
                    neut = float(neut_match.group(1)) if neut_match else 0.0

                    indiv_result = {
                        "sentiment": sentiment,
                        "confidence": round(conf, 3),
                        "scores": {
                            "bearish": round(bear, 3),
                            "bullish": round(bull, 3),
                            "neutral": round(neut, 3)
                        }
                    }
                    results.append(indiv_result)

                    bearish_total += bear
                    bullish_total += bull
                    neutral_total += neut
                    conf_total += conf
                    count += 1

            if count > 0:
                avg_bull = bullish_total / count
                overall_sentiment = "BULLISH" if avg_bull > 0.6 else "BEARISH" if bearish_total / count > 0.6 else "NEUTRAL"
                result = {
                    "sentiment": overall_sentiment,
                    "confidence": round(conf_total / count, 3),
                    "scores": {
                        "bearish": round(bearish_total / count, 3),
                        "bullish": round(avg_bull, 3),
                        "neutral": round(neutral_total / count, 3)
                    },
                    "individual_results": results
                }
            else:
                result = {"error": "No individual results parsed"}
        else:
            result = {"error": "No batch sentiment data in response"}

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"[{request_id}] Completed in {duration_ms}ms: overall_sentiment={result.get('sentiment')}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error calling Sentiment MCP: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upstream Sentiment server error; please retry.")