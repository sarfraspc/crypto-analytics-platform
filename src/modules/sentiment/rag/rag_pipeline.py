"""
RAG pipeline orchestration module.

Coordinates document ingestion, retrieval, and generation
for question-answering over crypto news and social media content.
"""

import argparse
import json
import logging
import uuid

import mlflow
from modules.sentiment.evaluation.mlflow_logger import (
    end_rag_run,
    log_rag_metrics,
    setup_mlflow,
    start_rag_run,
)
from modules.sentiment.evaluation.rag_metrics import faithfulness
from modules.sentiment.rag.embedder import Embedder
from modules.sentiment.rag.generator import Generator
from modules.sentiment.rag.retriever import Retriever
from modules.sentiment.rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def ingest(embedder: Embedder, vector_store: QdrantVectorStore, retriever: Retriever):
    """Ingest documents into vector store for RAG retrieval."""
    vector_store.delete_all()
    docs = embedder.fetch_docs()
    chunks, embeddings, metadatas = embedder.process_docs(docs, chunk_method='sentence')
    vector_store.add(chunks, embeddings, metadatas)
    logger.info("Ingestion complete.")


def query_rag(query: str, retriever: Retriever, generator: Generator, k: int = 3, log_mlflow: bool = False):
    """Execute RAG query with optional MLflow logging. Caching handled by sentiment_server.py."""
    contexts = retriever.retrieve(query, k=k)
    response = generator.generate(query, contexts)

    if log_mlflow:
        run = start_rag_run(run_name="single_query_eval", params={"query": query[:100], "k": k})
        try:
            metrics = {"faithfulness": faithfulness(response, [c['content'] for c in contexts])}
            log_rag_metrics(metrics)
            artifact = {
                "query": query,
                "contexts": [c['content'] for c in contexts],
                "retrieved_ids": [c['metadata'].get('doc_id') for c in contexts if c['metadata'].get('doc_id')],
                "response": response
            }
            artifact_filename = f"query_artifacts_{uuid.uuid4().hex[:8]}.json"
            mlflow.log_text(json.dumps(artifact), artifact_filename)
        finally:
            end_rag_run()

    return response

if __name__ == "__main__":
    setup_mlflow()
    parser = argparse.ArgumentParser()
    parser.add_argument('--ingest', action='store_true')
    parser.add_argument('--query', type=str)
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    embedder = Embedder()
    vector_store = QdrantVectorStore()
    retriever = Retriever(embedder, vector_store)
    generator = Generator()

    if args.ingest:
        ingest(embedder, vector_store, retriever)
    elif args.query:
        print(query_rag(args.query, retriever, generator, log_mlflow=args.log))
    else:
        logger.error("Use --ingest or --query <text>")