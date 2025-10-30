import asyncio
import logging
import sys
from typing import Dict, Any

from mcp.server import Server
from mcp.types import CallToolRequest, CallToolResult, Tool, TextContent

from modules.sentiment.models.sentiment_infer import (
    get_sentiment_classifier, 
    analyze_sentiment, 
    analyze_sentiment_batch
)
from core.logging_config import setup_logging
from utils.mcp_utils import AsyncStdioWrapper

setup_logging()
logger = logging.getLogger(__name__)

class SentimentMCP:
    def __init__(self):
        self.classifier = None
        self.is_initialized = False

    async def initialize(self):
        try:
            self.classifier = await asyncio.to_thread(get_sentiment_classifier)
            logger.info("Sentiment classifier initialized successfully")
            self.is_initialized = True
        except Exception as e:
            logger.exception("Sentiment classifier initialization failed")
            raise e

    def _truncate_text(self, text: str, max_tokens: int = 512) -> str:
        tokens = text.split() 
        if len(tokens) > max_tokens:
            truncated = ' '.join(tokens[:max_tokens]) + "... [truncated]"
            logger.warning(f"Text truncated from {len(tokens)} to {max_tokens} tokens")
            return truncated
        return text

    async def analyze_single(self, request: CallToolRequest):
        if not self.is_initialized:
            raise Exception("Server not initialized")

        input_data = request.arguments or {}
        text = input_data.get('text', '')
        
        if not text:
            raise ValueError("Text parameter is required")
        
        processed_text = self._truncate_text(text)
        
        result = await asyncio.to_thread(analyze_sentiment, processed_text)
        
        return CallToolResult(
            content=[TextContent(
                text=f"Sentiment Analysis Result:\n"
                     f"Text: {text[:200]}{'...' if len(text) > 200 else ''}\n"
                     f"Sentiment: {result['sentiment']}\n"
                     f"Confidence: {result['confidence']:.3f}\n"
                     f"Scores - Bearish: {result['bearish_score']:.3f}, "
                     f"Bullish: {result['bullish_score']:.3f}, "
                     f"Neutral: {result['neutral_score']:.3f}"
            )]
        )

    async def analyze_batch(self, request: CallToolRequest):
        if not self.is_initialized:
            raise Exception("Server not initialized")

        input_data = request.arguments or {}
        texts = input_data.get('texts', [])
        max_batch_size = 32
        
        if not texts or not isinstance(texts, list):
            raise Exception("Texts parameter must be a non-empty list")
        
        all_results = []
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i + max_batch_size]
            processed_texts = [self._truncate_text(text) for text in batch_texts]
            
            results = await asyncio.to_thread(analyze_sentiment_batch, processed_texts)
            all_results.extend(results)
        
        if not all_results:
            return CallToolResult(
                content=[TextContent(text="No text was analyzed.")]
            )

        response_text = "Batch Sentiment Analysis Results:\n\n"
        for i, (text, result) in enumerate(zip(texts, all_results)):
            bearish_score = result.get('BEARISH', result.get('bearish_score', 0))
            bullish_score = result.get('BULLISH', result.get('bullish_score', 0))
            neutral_score = result.get('NEUTRAL', result.get('neutral_score', 0))
            top_sentiment = result.get('top_sentiment', 'UNKNOWN')
            top_confidence = result.get('top_confidence', 0)
            
            response_text += (
                f"Text {i+1}: {text[:100]}...\n"
                f"  Sentiment: {top_sentiment}\n"
                f"  Confidence: {top_confidence:.3f}\n"
                f"  Bearish: {bearish_score:.3f}, "
                f"Bullish: {bullish_score:.3f}, "
                f"Neutral: {neutral_score:.3f}\n\n"
            )
        
        return CallToolResult(
            content=[TextContent(text=response_text)]
        )

async def main():
    server = Server("crypto-sentiment-server")
    mcp = SentimentMCP()
    await mcp.initialize()

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="analyze_sentiment",
                description="Analyze sentiment of a single text for crypto market sentiment",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze for sentiment"}
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="analyze_sentiment_batch",
                description="Analyze sentiment of multiple texts in batch",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texts": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "List of texts to analyze"
                        }
                    },
                    "required": ["texts"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]):
        if name == "analyze_sentiment":
            return await mcp.analyze_single(CallToolRequest(name=name, arguments=arguments))
        elif name == "analyze_sentiment_batch":
            return await mcp.analyze_batch(CallToolRequest(name=name, arguments=arguments))
        else:
            raise Exception(f"Unknown tool: {name}")

    @server.list_resources()
    async def list_resources():
        return []

    @server.read_resource()
    async def read_resource(name: str):
        raise Exception(f"Unknown resource: {name}")

    read_stream = AsyncStdioWrapper(sys.stdin.buffer, mode='r')
    write_stream = AsyncStdioWrapper(sys.stdout.buffer, mode='w')
    init_options = {"name": "crypto-sentiment-server"}

    try:
        await server.run(read_stream, write_stream, init_options)
    except Exception:
        logger.exception("Server.run failed")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Sentiment server stopped by user")