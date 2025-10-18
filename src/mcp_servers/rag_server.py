import asyncio
import logging
import sys
from typing import Dict, Any
import hashlib

from mcp.server import Server
from mcp.types import CallToolRequest, CallToolResult, Tool, TextContent

from modules.sentiment.rag.embedder import Embedder
from modules.sentiment.rag.vector_store import QdrantVectorStore
from modules.sentiment.rag.retriever import Retriever
from modules.sentiment.rag.generator import Generator
from utils.cache import RedisCache
from core.logging_config import setup_logging
from utils.mcp_utils import AsyncStdioWrapper

setup_logging()
logger = logging.getLogger(__name__)

class RAGMCP:
    def __init__(self):
        self.embedder = None
        self.vector_store = None
        self.retriever = None
        self.generator = None
        self.cache = RedisCache(expire_seconds=3600)  
        self.is_initialized = False

    async def initialize(self):
        try:
            self.embedder = Embedder()
            self.vector_store = QdrantVectorStore()
            self.retriever = Retriever(self.embedder, self.vector_store)
            self.generator = Generator()
            
            self.retriever.index_for_hybrid()
            
            logger.info("RAG system initialized successfully")
            self.is_initialized = True
        except Exception as e:
            logger.exception("RAG system initialization failed")
            raise e

    async def ingest_documents(self, request: CallToolRequest):
        if not self.is_initialized:
            raise Exception("Server not initialized")

        input_data = request.arguments or {}
        days_back = input_data.get('days_back', 30)
        
        try:
            self.vector_store.delete_all()
            
            docs = self.embedder.fetch_docs(days_back=days_back)
            chunks, embeddings, metadatas = self.embedder.process_docs(docs, chunk_method='sentence')
            
            self.vector_store.add(chunks, embeddings, metadatas)
            
            self.retriever.index_for_hybrid()
            
            await self._clear_rag_cache()
            
            return CallToolResult(
                content=[TextContent(
                    text=f"Successfully ingested {len(docs)} documents with {len(chunks)} chunks from the last {days_back} days."
                )]
            )
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise Exception(f"Document ingestion failed: {str(e)}")

    async def query_rag(self, request: CallToolRequest):
        if not self.is_initialized:
            raise Exception("Server not initialized")

        input_data = request.arguments or {}
        query = input_data.get('query', '')
        k = input_data.get('k', 5)
        
        if not query:
            raise Exception("Query parameter is required")
        
        if self.vector_store.count() == 0:
            raise Exception("No documents ingested; run 'ingest_documents' first")
        
        try:
            cache_key = f"rag_query:{hashlib.sha256(query.encode()).hexdigest()}:{k}"
            if cached := self.cache.get_json(cache_key):
                logger.info(f"Returning cached response for query: {query[:50]}...")
                response_text = f"Query: {query}\n\n"
                response_text += f"[CACHED] Generated Answer: {cached.get('response')}\n\n"
                response_text += "Retrieved Contexts:\n"
                for i, context in enumerate(cached.get('contexts', [])):
                    response_text += f"\n  Context {i+1} (Score: {context.get('score', 0):.3f}):\n"
                    response_text += f"    Source: {context.get('metadata', {}).get('source', 'unknown')}\n"
                    response_text += f"    Content: {context.get('content', '')[:200]}...\n"
                return CallToolResult(
                    content=[TextContent(text=response_text)]
                )
            
            contexts = self.retriever.retrieve(query, k=k)
            
            response = self.generator.generate(query, contexts)
            
            self.cache.set_json(cache_key, {'response': response, 'contexts': contexts, 'k': k})
            
            response_text = f"Query: {query}\n\n"
            response_text += f"Generated Answer: {response}\n\n"
            response_text += "Retrieved Contexts:\n"
            
            for i, context in enumerate(contexts):
                response_text += f"\n  Context {i+1} (Score: {context['score']:.3f}):\n"
                response_text += f"    Source: {context['metadata'].get('source', 'unknown')}\n"
                response_text += f"    Content: {context['content'][:200]}...\n"
            
            return CallToolResult(
                content=[TextContent(text=response_text)]
            )
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise Exception(f"RAG query failed: {str(e)}")

    async def get_collection_stats(self, request: CallToolRequest):
        if not self.is_initialized:
            raise Exception("Server not initialized")

        try:
            count = self.vector_store.count()
            all_data = self.vector_store.get_all()
            
            sources = {}
            for metadata in all_data["metadatas"]:
                source = metadata.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            stats_text = f"Vector Store Statistics:\n"
            stats_text += f"Total chunks: {count}\n"
            stats_text += "Sources distribution:\n"
            for source, cnt in sources.items():
                stats_text += f"  {source}: {cnt} chunks\n"
            
            cache_info = self.cache.get_stats("rag_query:*")
            if cache_info:
                stats_text += f"\nCache Info: {cache_info}"
            
            return CallToolResult(
                content=[TextContent(text=stats_text)]
            )
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise Exception(f"Failed to get collection stats: {str(e)}")

    async def _clear_rag_cache(self):
        try:
            self.cache.delete_by_pattern("rag_query:*")
            logger.info("RAG cache cleared because of new ingestion")
        except Exception as e:
            logger.warning(f"Could not clear RAG cache: {e}")

async def main():
    server = Server("crypto-rag-server")
    mcp = RAGMCP()
    await mcp.initialize()

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="ingest_documents",
                description="Ingest recent news and Reddit posts into the RAG system",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days_back": {
                            "type": "integer", 
                            "description": "Number of days back to fetch documents (default: 30)",
                            "default": 30
                        }
                    },
                    "required": []  
                }
            ),
            Tool(
                name="query_rag",
                description="Query the RAG system for crypto market insights and sentiment analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "Question about crypto market sentiment, news, or trends"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of context chunks to retrieve (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_rag_stats",
                description="Get statistics about the RAG system's document collection",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]):
        if name == "ingest_documents":
            return await mcp.ingest_documents(CallToolRequest(name=name, arguments=arguments))
        elif name == "query_rag":
            return await mcp.query_rag(CallToolRequest(name=name, arguments=arguments))
        elif name == "get_rag_stats":
            return await mcp.get_collection_stats(CallToolRequest(name=name, arguments=arguments))
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
    init_options = {"name": "crypto-rag-server"}

    try:
        await server.run(read_stream, write_stream, init_options)
    except Exception:
        logger.exception("Server.run failed")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("RAG server stopped by user")