"""
Tests for sentiment analysis and RAG pipeline modules.

Covers sentiment classification, text embedding, document
retrieval, and RAG query processing with caching.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np


# ROBUST MOCKING

# 1. Mock 'datasets'
mock_datasets = types.ModuleType("datasets")
mock_datasets.load_dataset = MagicMock()
mock_datasets.__spec__ = MagicMock()
mock_datasets.__spec__.name = "datasets"
mock_datasets.__path__ = [] 
sys.modules["datasets"] = mock_datasets

# 2. Mock 'qdrant_client'
mock_qdrant = types.ModuleType("qdrant_client")
mock_qdrant.QdrantClient = MagicMock()
mock_qdrant.__spec__ = MagicMock()
mock_qdrant.__spec__.name = "qdrant_client"
mock_qdrant.__path__ = [] 
sys.modules["qdrant_client"] = mock_qdrant

# 3. Mock 'qdrant_client.http'
mock_qdrant_http = types.ModuleType("qdrant_client.http")
mock_qdrant_http.__spec__ = MagicMock()
mock_qdrant_http.__spec__.name = "qdrant_client.http"
mock_qdrant_http.__path__ = []
sys.modules["qdrant_client.http"] = mock_qdrant_http

# 4. Mock 'qdrant_client.http.models'
mock_qdrant_http_models = types.ModuleType("qdrant_client.http.models")
mock_qdrant_http_models.Distance = MagicMock()
mock_qdrant_http_models.VectorParams = MagicMock()
mock_qdrant_http_models.PointStruct = MagicMock()
mock_qdrant_http_models.__spec__ = MagicMock()
mock_qdrant_http_models.__spec__.name = "qdrant_client.http.models"
sys.modules["qdrant_client.http.models"] = mock_qdrant_http_models

# 5. Mock 'qdrant_client.models'
mock_qdrant_models = types.ModuleType("qdrant_client.models")
mock_qdrant_models.Filter = MagicMock()
mock_qdrant_models.FieldCondition = MagicMock()
mock_qdrant_models.MatchText = MagicMock()
mock_qdrant_models.__spec__ = MagicMock()
mock_qdrant_models.__spec__.name = "qdrant_client.models"
sys.modules["qdrant_client.models"] = mock_qdrant_models

# IMPORTS

try:
    from src.modules.sentiment.models.sentiment_infer import SentimentClassifier
except ImportError:
    from src.modules.sentiment.models.distilroberta_sentiment import SentimentClassifier

from src.modules.sentiment.rag.embedder import Embedder
from src.modules.sentiment.rag.retriever import Retriever
from src.modules.sentiment.rag.rag_pipeline import query_rag

# Sentiment Classifier Tests

def test_sentiment_initialization():
    with patch("src.modules.sentiment.models.sentiment_infer.pipeline") as mock_pipeline:
        classifier = SentimentClassifier(model_path="fake/path")
        mock_pipeline.assert_called_once_with(
            "text-classification",
            model="fake/path",
            tokenizer="fake/path",
            return_all_scores=True,
            device=-1
        )

def test_sentiment_prediction_formatting():
    with patch("src.modules.sentiment.models.sentiment_infer.pipeline") as mock_pipeline:
        mock_output = [[
            {'label': 'LABEL_0', 'score': 0.05}, 
            {'label': 'LABEL_1', 'score': 0.90}, 
            {'label': 'LABEL_2', 'score': 0.05}  
        ]]
        mock_pipeline.return_value = lambda x: mock_output
        
        classifier = SentimentClassifier(model_path="fake")
        result = classifier.quick_predict("Bitcoin is great")
        
        assert result['sentiment'] == 'BULLISH'
        assert result['confidence'] == 0.90

# Embedder Tests

def test_embedder_chunking():
    with patch("src.modules.sentiment.rag.embedder.SentenceTransformer"):
        embedder = Embedder(chunk_size=10, overlap=0)
        text = "One Two Three Four Five"
        chunks = embedder.chunk_text(text, method='fixed')
        assert isinstance(chunks, list)
        assert len(chunks) > 0

def test_embedder_process_docs():
    with patch("src.modules.sentiment.rag.embedder.SentenceTransformer") as MockST:
        mock_model = MockST.return_value
        
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        embedder = Embedder()
        docs = [{'id': 1, 'content': 'Hello world', 'source': 'news'}]
        
        chunks, embeddings, metadatas = embedder.process_docs(docs)
        
        assert len(chunks) > 0
        assert len(embeddings) == len(chunks)
        assert metadatas[0]['doc_id'] == 1

# Retriever Tests

def test_retriever_rrf_logic():
    # 1. Setup Mocks
    mock_embedder = MagicMock()
    
    # FIX: Return numpy array because the real code calls .tolist()
    mock_embedder.model.encode.return_value = np.array([0.1, 0.2])
    
    mock_vector_store = MagicMock()
    mock_client = mock_vector_store.client
    
    Hit = MagicMock
    sem_results = [Hit(id=1, score=0.9), Hit(id=2, score=0.8)]
    key_results = [Hit(id=2, score=0.9), Hit(id=1, score=0.5)]
    
    mock_client.search.side_effect = [sem_results, key_results]
    
    mock_doc1 = MagicMock()
    mock_doc1.id = 1
    mock_doc1.payload = {"document": "Doc 1 content", "metadata": {}}
    
    mock_doc2 = MagicMock()
    mock_doc2.id = 2
    mock_doc2.payload = {"document": "Doc 2 content", "metadata": {}}
    
    mock_client.retrieve.return_value = [mock_doc1, mock_doc2]

    # 2. Run
    retriever = Retriever(mock_embedder, mock_vector_store)
    retriever.boost_keywords = {} 
    
    results = retriever.retrieve("crypto query", k=2, min_score=0.0)
    
    # 3. Assertions
    assert len(results) == 2
    contents = [r['content'] for r in results]
    assert "Doc 1 content" in contents

# RAG Pipeline Tests

def test_query_rag_basic():
    """Test basic RAG query flow (caching handled by sentiment_server, not rag_pipeline)."""
    mock_retriever = MagicMock()
    mock_generator = MagicMock()

    mock_retriever.retrieve.return_value = [{'content': 'Context 1', 'metadata': {'doc_id': 1}}]
    mock_generator.generate.return_value = "Generated Answer"

    response = query_rag("test query", mock_retriever, mock_generator)

    assert response == "Generated Answer"
    mock_retriever.retrieve.assert_called_once()
    mock_generator.generate.assert_called_once()


def test_query_rag_with_multiple_contexts():
    """Test RAG query with multiple retrieved contexts."""
    mock_retriever = MagicMock()
    mock_generator = MagicMock()

    mock_retriever.retrieve.return_value = [
        {'content': 'Context 1', 'metadata': {'doc_id': 1}},
        {'content': 'Context 2', 'metadata': {'doc_id': 2}},
        {'content': 'Context 3', 'metadata': {'doc_id': 3}},
    ]
    mock_generator.generate.return_value = "Answer based on 3 contexts"

    response = query_rag("test query", mock_retriever, mock_generator, k=3)

    assert response == "Answer based on 3 contexts"
    mock_retriever.retrieve.assert_called_once_with("test query", k=3)