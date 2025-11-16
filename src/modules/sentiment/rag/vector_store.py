import logging
from typing import List, Dict, Any, Optional
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from core.config import settings

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    def __init__(self, url: Optional[str] = None, collection_name: Optional[str] = None):
        resolved_url = url or settings.QDRANT_URL
        resolved_collection = collection_name or settings.QDRANT_COLLECTION
        self.client = QdrantClient(url=resolved_url)
        self.collection_name = resolved_collection
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.warning(f"Collection check failed (creating if needed): {e}")

    def add(self, documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        if not documents:
            return

        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            vector_size = len(embeddings[0])
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}' with vector_size={vector_size}")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        points = []
        for i, (doc, emb, meta) in enumerate(zip(documents, embeddings, metadatas)):
            point = PointStruct(
                id=ids[i],
                vector=emb,
                payload={
                    "document": doc,
                    "metadata": meta
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Added {len(documents)} chunks to Qdrant.")

    def query_semantic(self, query_embedding: List[float], n_results: int = 5):
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=n_results,
            with_payload=True
        )
        return {
            "documents": [hit.payload["document"] for hit in search_result],
            "distances": [hit.score for hit in search_result],
            "metadatas": [hit.payload["metadata"] for hit in search_result],
            "ids": [str(hit.id) for hit in search_result]
        }

    def get_all(self):
        count = self.count()
        if count == 0:
            return {"documents": [], "metadatas": [], "ids": []}
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            limit=count,
            with_payload=True,
            with_vectors=False
        )[0]
        return {
            "documents": [p.payload["document"] for p in scroll_result],
            "metadatas": [p.payload["metadata"] for p in scroll_result],
            "ids": [str(p.id) for p in scroll_result]
        }

    def delete_all(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        logger.info("Cleared Qdrant collection.")

    def count(self):
        try:
            return self.client.get_collection(self.collection_name).points_count
        except Exception:
            return 0