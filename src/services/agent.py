from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging
import json
import re
from datetime import datetime

from modules.agent.agent_client import call_mcp_tool 
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["Agent"])

@router.get("/insight/{symbol}")
async def get_agent_insight(
    symbol: str,
    question: Optional[str] = Query(None, description="Natural language query"),
    options: Optional[str] = Query(None, description="JSON string of options (e.g., {'k_docs': 5, 'horizon': 7})")
):
    start_time = datetime.now()
    request_id = f"agent_insight_{hash(str(start_time)) % 1000000}"  # Improved simple ID
    logger.info(f"[{request_id}] Request: symbol={symbol}, question={question[:100] if question else 'N/A'}..., options={options}")

    try:
        # Parse options if provided
        parsed_options = {}
        if options:
            try:
                parsed_options = json.loads(options)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in options parameter")

        # Call MCP tool
        mcp_args = {
            "symbol": symbol,
            "question": question or "",
            "options": parsed_options
        }
        raw_result = await call_mcp_tool("crypto-agent-server", "get_agent_insight", mcp_args)

        # Parse the result (improved: find JSON block)
        result_text = raw_result.get("raw_text", raw_result) if isinstance(raw_result, dict) and "raw_text" in raw_result else str(raw_result)
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
        else:
            # Fallback to previous split
            if "Agent Insight Result:\n\n" in result_text:
                json_str = result_text.split("\n\n", 1)[1] if "\n\n" in result_text else result_text
                result = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in MCP response")

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"[{request_id}] Completed in {duration_ms}ms: query_type={result.get('query_type', 'N/A')}, cache_hit={result.get('cache_hit', False)}")

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"[{request_id}] Malformed JSON in MCP response: {e}")
        raise HTTPException(status_code=500, detail="Invalid response format from upstream service")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error calling Agent MCP: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upstream Agent server error; please retry.")