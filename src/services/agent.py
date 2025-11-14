import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from core.logging_config import setup_logging
from modules.agent.agent_client import orchestrate_query

setup_logging()
logger = logging.getLogger(__name__)
router = APIRouter(tags=["Agent"])


def _validate_symbol(symbol: str) -> str:
    if not symbol or not symbol.isalnum():
        raise HTTPException(status_code=400, detail="Symbol must be alphanumeric.")
    return symbol.upper()


class AgentInsightRequest(BaseModel):
    question: Optional[str] = Field(
        None,
        description="Natural language question for the agent (defaults to '{symbol} market overview').",
        max_length=1000,
    )
    options: Dict[str, Any] = Field(default_factory=dict, description="Advanced agent options.")
    no_cache: bool = Field(
        default=False,
        description="Bypass both agent-level and MCP-level caches when true.",
    )

    @validator("options", pre=True)
    def ensure_dict(cls, value):
        if value in (None, ""):
            return {}
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"options payload must be valid JSON: {exc}") from exc
        if isinstance(value, dict):
            return value
        raise ValueError("options payload must be an object or JSON string.")


@router.post("/insight/{symbol}")
async def get_agent_insight(symbol: str, payload: AgentInsightRequest):
    sanitized_symbol = _validate_symbol(symbol)
    question = (payload.question or f"{sanitized_symbol} market overview").strip()
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()

    logger.info(
        "[%s] Agent insight: symbol=%s, question=%s, no_cache=%s",
        request_id,
        sanitized_symbol,
        question[:120],
        payload.no_cache,
    )

    try:
        agent_result = await orchestrate_query(
            symbol=sanitized_symbol,
            question=question,
            options=payload.options,
            no_cache=payload.no_cache,
            force_query_type=payload.options.get("force_llm"),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("[%s] Agent validation failed: %s", request_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("[%s] Agent orchestration failed: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Agent service error.") from exc

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    response = {
        "request_id": request_id,
        "duration_ms": duration_ms,
        **agent_result,
    }
    logger.info(
        "[%s] Agent insight complete: query_type=%s duration_ms=%s",
        request_id,
        agent_result.get("query_type"),
        duration_ms,
    )
    return response
