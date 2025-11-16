import asyncio
import json
import logging
import hashlib
from typing import Dict, Any, List
from datetime import datetime, timedelta, timezone

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    Tool,
    TextContent,
)

from modules.sentiment.rag.embedder import Embedder
from modules.sentiment.rag.vector_store import QdrantVectorStore
from modules.sentiment.rag.retriever import Retriever
from modules.sentiment.rag.generator import Generator
from modules.sentiment.models.sentiment_infer import (
    get_sentiment_classifier,
    analyze_sentiment_batch,
    analyze_sentiment  # For single if needed
)
from utils.cache import RedisCache
from core.database import get_timescale_db
from data.storage.models import IngestionJob as IngestionJobModel
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class PipelineMCP:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = QdrantVectorStore()
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.generator = Generator()
        self.cache = RedisCache(expire_seconds=3600)
        self.classifier = None
        self.is_initialized = False

    @staticmethod
    def _error_result(message: str) -> CallToolResult:
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=message)]
        )

    @staticmethod
    def _json_result(payload: Dict[str, Any]) -> CallToolResult:
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, default=str, indent=2))]
        )

    async def initialize(self):
        try:
            self.classifier = await asyncio.to_thread(get_sentiment_classifier)
            logger.info("Pipeline MCP initialized successfully")
            self.is_initialized = True
        except Exception as e:
            logger.exception("Pipeline initialization failed")
            raise e

    async def ingest_documents(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            raise Exception("Server not initialized")
        params = request.params if hasattr(request, "params") else None
        input_data = (params.arguments if params and params.arguments is not None else {})
        days_back = input_data.get('days_back', 30)
        try:
            await asyncio.to_thread(self.vector_store.delete_all)
            docs = await asyncio.to_thread(self.embedder.fetch_docs, days_back)
            chunks, embeddings, metadatas = await asyncio.to_thread(
                lambda: self.embedder.process_docs(docs, chunk_method='sentence')
            )
            await asyncio.to_thread(self.vector_store.add, chunks, embeddings, metadatas)
            await asyncio.to_thread(self.retriever.index_for_hybrid)
            await self._clear_cache("rag_query:*", "combined:*")
            logger.info(f"Ingested {len(docs)} docs with {len(chunks)} chunks")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Successfully ingested {len(docs)} documents ({len(chunks)} chunks) from the last {days_back} days."
                )]
            )
        except Exception as e:
            err = f"Document ingestion failed: {type(e).__name__} - {e}"
            logger.error(err, exc_info=True)
            return self._error_result(err)

    async def query_rag(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            raise Exception("Server not initialized")
        params = request.params if hasattr(request, "params") else None
        input_data = (params.arguments if params and params.arguments is not None else {})
        query = input_data.get('query', '')
        k = input_data.get('k', 5)
        if not query:
            return self._error_result("Query parameter is required")
        if await asyncio.to_thread(self._get_vector_count) == 0:
            return self._error_result("No documents ingested; run 'ingest_documents' first")
        cache_key = f"rag_query:{hashlib.sha256(query.encode()).hexdigest()}:{k}"
        cached = await asyncio.to_thread(self.cache.get_json, cache_key)
        if cached:
            logger.info(f"Returning cached RAG response for: {query[:50]}...")
            return self._format_rag_response(cached, query)
        try:
            contexts = await asyncio.to_thread(lambda: self.retriever.retrieve(query, k=k))
            response = await asyncio.to_thread(self.generator.generate, query, contexts)
            await asyncio.to_thread(
                self.cache.set_json,
                cache_key,
                {'response': response, 'contexts': contexts}
            )
            return self._format_rag_response({'response': response, 'contexts': contexts}, query)
        except Exception as e:
            err = f"RAG query failed: {type(e).__name__} - {e}"
            logger.error(err, exc_info=True)
            return self._error_result(err)

    def _format_rag_response(self, data: Dict, query: str) -> CallToolResult:
        payload = {
            "query": query,
            "response": data.get("response"),
            "contexts": data.get("contexts", []),
        }
        return self._json_result(payload)

    async def analyze_sentiment(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            raise Exception("Server not initialized")
        params = request.params if hasattr(request, "params") else None
        input_data = (params.arguments if params and params.arguments is not None else {})
        text = input_data.get('text', '')
        if not text:
            return self._error_result("Text parameter is required")
        try:
            result = await asyncio.to_thread(analyze_sentiment, self._truncate_text(text))
            payload = {
                "text_preview": f"{text[:200]}{'...' if len(text) > 200 else ''}",
                "sentiment": result.get("sentiment"),
                "confidence": result.get("confidence"),
                "scores": {
                    "bearish": result.get("bearish_score"),
                    "bullish": result.get("bullish_score"),
                    "neutral": result.get("neutral_score"),
                },
            }
            return self._json_result(payload)
        except Exception as e:
            err = f"Sentiment analysis failed: {type(e).__name__} - {e}"
            logger.error(err, exc_info=True)
            return self._error_result(err)

    async def analyze_sentiment_batch(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            raise Exception("Server not initialized")
        params = request.params if hasattr(request, "params") else None
        input_data = (params.arguments if params and params.arguments is not None else {})
        texts = input_data.get('texts', [])
        max_batch_size = 32
        if not texts or not isinstance(texts, list):
            return self._error_result("Texts must be a non-empty list")
        try:
            all_results = []
            for i in range(0, len(texts), max_batch_size):
                batch = texts[i:i + max_batch_size]
                processed = [self._truncate_text(t) for t in batch]
                results = await asyncio.to_thread(analyze_sentiment_batch, processed)
                all_results.extend(results)
            structured = []
            for text, result in zip(texts, all_results):
                structured.append({
                    "text_preview": f"{text[:100]}{'...' if len(text) > 100 else ''}",
                    "sentiment": result.get('top_sentiment', result.get('sentiment')),
                    "confidence": result.get('top_confidence', result.get('confidence')),
                    "scores": {
                        "bearish": result.get('bearish_score', result.get('BEARISH')),
                        "bullish": result.get('bullish_score', result.get('BULLISH')),
                        "neutral": result.get('neutral_score', result.get('NEUTRAL')),
                    }
                })
            payload = {
                "count": len(structured),
                "results": structured,
                "aggregated": self._aggregate_sentiment(all_results),
            }
            return self._json_result(payload)
        except Exception as e:
            err = f"Batch sentiment failed: {type(e).__name__} - {e}"
            logger.error(err, exc_info=True)
            return self._error_result(err)

    async def analyze_with_sources(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            raise Exception("Server not initialized")
        params = request.params if hasattr(request, "params") else None
        input_data = (params.arguments if params and params.arguments is not None else {})
        query = input_data.get('query', '')
        k = input_data.get('k', 5)
        include_sources = input_data.get('include_sources', True)
        request_sources = "sources" in query.lower() or "source" in query.lower()
        if not query:
            return self._error_result("Query parameter is required")
        if await asyncio.to_thread(self._get_vector_count) == 0:
            return self._error_result("No documents ingested; run ingestion first")
        cache_key = f"combined:{hashlib.sha256(query.encode()).hexdigest()}:{k}"
        cached = await asyncio.to_thread(self.cache.get_json, cache_key)
        if cached:
            logger.info(f"Returning cached combined response for: {query[:50]}...")
            return self._format_combined_response(cached, query, include_sources or request_sources)
        try:
            contexts = await asyncio.to_thread(lambda: self.retriever.retrieve(query, k=k))
            if not contexts:
                raise Exception("No relevant sources found")
            response = await asyncio.to_thread(self.generator.generate, query, contexts)
            source_texts = [c['content'] for c in contexts]
            sentiment_results = await asyncio.to_thread(analyze_sentiment_batch, source_texts)
            aggregated = self._aggregate_sentiment(sentiment_results)
            cache_data = {
                'query': query, 'response': response, 'sources': contexts,
                'sentiments': sentiment_results, 'aggregated': aggregated
            }
            await asyncio.to_thread(self.cache.set_json, cache_key, cache_data)
            return self._format_combined_response(cache_data, query, include_sources or request_sources)
        except Exception as e:
            err = f"Combined pipeline failed: {type(e).__name__} - {e}"
            logger.error(err, exc_info=True)
            return self._error_result(err)

    def _format_combined_response(self, data: Dict, query: str, show_sources: bool) -> CallToolResult:
        payload = {
            "query": query,
            "response": data.get("response"),
            "aggregated": data.get("aggregated"),
        }
        if show_sources:
            payload["sources"] = data.get("sources")
            payload["sentiments"] = data.get("sentiments")
        return self._json_result(payload)

    def _aggregate_sentiment(self, sentiments: List[Dict]) -> Dict:
        if not sentiments:
            return {'top_sentiment': 'NEUTRAL', 'top_confidence': 0.0, 'bearish_score': 0, 'bullish_score': 0, 'neutral_score': 0}
        avg_bearish = sum(s.get('bearish_score', s.get('BEARISH', 0)) for s in sentiments) / len(sentiments)
        avg_bullish = sum(s.get('bullish_score', s.get('BULLISH', 0)) for s in sentiments) / len(sentiments)
        avg_neutral = sum(s.get('neutral_score', s.get('NEUTRAL', 0)) for s in sentiments) / len(sentiments)
        scores = {'BEARISH': avg_bearish, 'BULLISH': avg_bullish, 'NEUTRAL': avg_neutral}
        top_label = max(scores, key=scores.get)
        return {
            'top_sentiment': top_label, 'top_confidence': scores[top_label],
            'bearish_score': avg_bearish, 'bullish_score': avg_bullish, 'neutral_score': avg_neutral
        }

    async def get_stats(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            raise Exception("Server not initialized")
        try:
            count = await asyncio.to_thread(self._get_vector_count)
            all_data = await asyncio.to_thread(self.vector_store.get_all)
            sources = {}
            for metadata in all_data.get("metadatas", []):
                source = metadata.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            stats_text = f"Vector Store Stats:\nTotal chunks: {count}\nSources:\n"
            for source, cnt in sources.items():
                stats_text += f"  {source}: {cnt}\n"
            cache_info = await asyncio.to_thread(self.cache.get_stats, "rag_query:*")
            if cache_info:
                stats_text += f"\nCache: {cache_info}"
            return CallToolResult(content=[TextContent(type="text", text=stats_text)])
        except Exception as e:
            err = f"Failed to get stats: {type(e).__name__} - {e}"
            logger.error(err, exc_info=True)
            return self._error_result(err)

    async def get_fng(self, request: CallToolRequest) -> CallToolResult:
        if not self.is_initialized:
            raise Exception("Server not initialized")
        
        params = request.params if hasattr(request, "params") else None
        input_data = (params.arguments if params and params.arguments is not None else {})
        limit = input_data.get('limit', 30)  # Last 30 records
        days_back = input_data.get('days_back', 30)

        try:
            def fetch_fng():
                with get_timescale_db() as db:
                    from sqlalchemy import desc
                    
                    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
                    records = db.query(IngestionJobModel).filter(
                        IngestionJobModel.pipeline == 'fear_greed',
                        IngestionJobModel.last_success >= cutoff
                    ).order_by(desc(IngestionJobModel.last_success)).limit(limit).all()
                    
                    return [
                        {
                            "time": record.last_success.isoformat(),
                            "score": details.get("score"),
                            "value": details.get("value"),
                            "timestamp": details.get("timestamp"),
                            "time_until_update": details.get("time_until_update"),
                            "value_classification": details.get("value_classification"),
                            "last_update": record.last_run.isoformat() if record.last_run else None
                        }
                        for record in records
                        if (details := record.details) and isinstance(details, dict)
                    ]
            
            fng_data = await asyncio.to_thread(fetch_fng)
            
            # Calculate current sentiment from latest FNG
            current_sentiment = "NEUTRAL"
            if fng_data:
                latest_value = float(fng_data[0].get("value", 50))
                if latest_value >= 75:
                    current_sentiment = "EXTREME GREED"
                elif latest_value >= 55:
                    current_sentiment = "GREED" 
                elif latest_value >= 45:
                    current_sentiment = "NEUTRAL"
                elif latest_value >= 25:
                    current_sentiment = "FEAR"
                else:
                    current_sentiment = "EXTREME FEAR"

            payload = {
                "current_sentiment": current_sentiment,
                "current_value": fng_data[0].get("value") if fng_data else None,
                "historical_data": fng_data,
                "count": len(fng_data)
            }
            
            return self._json_result(payload)
            
        except Exception as e:
            err = f"FNG data fetch failed: {type(e).__name__} - {e}"
            logger.error(err, exc_info=True)
            return self._error_result(err)

    async def get_fng_current(self, request: CallToolRequest) -> CallToolResult:
        """Get only the current FNG value and classification"""
        if not self.is_initialized:
            raise Exception("Server not initialized")
        
        try:
            def fetch_current_fng():
                with get_timescale_db() as db:
                    from sqlalchemy import desc
                    
                    latest = db.query(IngestionJobModel).filter(
                        IngestionJobModel.pipeline == 'fear_greed'
                    ).order_by(desc(IngestionJobModel.last_success)).first()
                    if not latest or not (details := latest.details) or not isinstance(details, dict):
                        return None
                    
                    return {
                        "time": latest.last_success.isoformat(),
                        "score": details.get("score"),
                        "value": details.get("value"),
                        "timestamp": details.get("timestamp"),
                        "time_until_update": details.get("time_until_update"),
                        "value_classification": details.get("value_classification"),
                        "last_update": latest.last_run.isoformat() if latest.last_run else None
                    }
            
            current_fng = await asyncio.to_thread(fetch_current_fng)
            
            if not current_fng:
                return self._error_result("No FNG data available")
                
            # Classify sentiment
            value = float(current_fng.get("value", 50))
            if value >= 75:
                sentiment = "EXTREME GREED"
                market_bias = "BEARISH"  # Contrarian indicator
            elif value >= 55:
                sentiment = "GREED"
                market_bias = "CAUTION"
            elif value >= 45:
                sentiment = "NEUTRAL" 
                market_bias = "NEUTRAL"
            elif value >= 25:
                sentiment = "FEAR"
                market_bias = "OPPORTUNITY"
            else:
                sentiment = "EXTREME FEAR"
                market_bias = "BULLISH"  # Contrarian indicator

            payload = {
                "current_value": current_fng.get("value"),
                "sentiment": sentiment,
                "classification": current_fng.get("value_classification"),
                "market_bias": market_bias,
                "timestamp": current_fng.get("time"),
                "last_updated": current_fng.get("last_update")
            }
            
            return self._json_result(payload)
            
        except Exception as e:
            err = f"Current FNG fetch failed: {type(e).__name__} - {e}"
            logger.error(err, exc_info=True)
            return self._error_result(err)

    def _get_vector_count(self):
        count_result = self.vector_store.count()
        return count_result if isinstance(count_result, int) else count_result.get("count", 0)

    def _truncate_text(self, text: str, max_tokens: int = 512) -> str:
        tokens = text.split()
        if len(tokens) > max_tokens:
            return ' '.join(tokens[:max_tokens]) + "... [truncated]"
        return text

    async def _clear_cache(self, *patterns):
        for pattern in patterns:
            try:
                await asyncio.to_thread(self.cache.delete_by_pattern, pattern)
            except Exception as e:
                logger.warning(f"Cache clear failed for {pattern}: {e}")

server = Server("crypto-sentiment-server")
mcp = PipelineMCP()

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="ingest_documents",
            description="Ingest recent crypto news/Reddit (prereq for queries)",
            inputSchema={
                "type": "object",
                "properties": {"days_back": {"type": "integer", "default": 30}},
                "required": []
            }
        ),
        Tool(
            name="query_rag",
            description="Query RAG for insights (without sentiment)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query text"},
                    "k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="analyze_sentiment",
            description="Analyze single text sentiment",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }
        ),
        Tool(
            name="analyze_sentiment_batch",
            description="Analyze batch of texts",
            inputSchema={
                "type": "object",
                "properties": {"texts": {"type": "array", "items": {"type": "string"}}},
                "required": ["texts"]
            }
        ),
        Tool(
            name="analyze_with_sources",
            description="Full pipeline: RAG sources + sentiment + insights",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5},
                    "include_sources": {"type": "boolean", "default": True}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_stats",
            description="Get vector store and cache stats",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_fng",
            description="Get Fear and Greed Index historical data",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 30, "description": "Number of historical records"},
                    "days_back": {"type": "integer", "default": 30, "description": "Days of history to fetch"}
                },
                "required": []
            }
        ),
        Tool(
            name="get_fng_current", 
            description="Get current Fear and Greed Index value with market bias interpretation",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    if name == "ingest_documents":
        return await mcp.ingest_documents(
            CallToolRequest(
                params=CallToolRequestParams(name=name, arguments=arguments)
            )
        )
    elif name == "query_rag":
        return await mcp.query_rag(
            CallToolRequest(
                params=CallToolRequestParams(name=name, arguments=arguments)
            )
        )
    elif name == "analyze_sentiment":
        return await mcp.analyze_sentiment(
            CallToolRequest(
                params=CallToolRequestParams(name=name, arguments=arguments)
            )
        )
    elif name == "analyze_sentiment_batch":
        return await mcp.analyze_sentiment_batch(
            CallToolRequest(
                params=CallToolRequestParams(name=name, arguments=arguments)
            )
        )
    elif name == "analyze_with_sources":
        return await mcp.analyze_with_sources(
            CallToolRequest(
                params=CallToolRequestParams(name=name, arguments=arguments)
            )
        )
    elif name == "get_stats":
        return await mcp.get_stats(
            CallToolRequest(
                params=CallToolRequestParams(name=name, arguments=arguments)
            )
        )
    elif name == "get_fng":
        return await mcp.get_fng(
            CallToolRequest(
                params=CallToolRequestParams(name=name, arguments=arguments)
            )
        )
    elif name == "get_fng_current":
        return await mcp.get_fng_current(
            CallToolRequest(
                params=CallToolRequestParams(name=name, arguments=arguments)
            )
        )
    else:
        raise Exception(f"Unknown tool: {name}")

@server.list_resources()
async def list_resources():
    return []

@server.read_resource()
async def read_resource(name: str):
    raise Exception(f"Unknown resource: {name}")


async def main():
    await mcp.initialize()
    logger.info(f"Starting {server.name}...")
    init_options = InitializationOptions(
        server_name=server.name,
        server_version="1.0.0",
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )
    logger.info(f"Init options: {init_options}")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline server stopped by user")