import logging
from typing import Dict
import numpy as np
from qdrant_client import models

from modules.sentiment.rag.vector_store import QdrantVectorStore
from modules.sentiment.rag.embedder import Embedder

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, embedder: Embedder, vector_store: QdrantVectorStore, alpha: float = 0.5, boost_keywords: Dict[str, float] = None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.client = vector_store.client
        self.collection_name = vector_store.collection_name
        self.model = self.embedder.model
        self.alpha = alpha
        self.boost_keywords = boost_keywords or {"sentiment": 1.2, "btc": 1.2}

    def retrieve(self, query: str, k: int = 5, min_score: float = 0.05, raw_min_score: float = 0.005):
        query_emb = self.model.encode(query)
        sem_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_emb.tolist(),  
            limit=k * 4, 
            with_payload=True
        )
        
        query_words = [word for word in query.lower().split() if len(word) > 2] 
        if not query_words:
            logger.warning(f"No meaningful query words for keyword filter: '{query}'")
            return []
        
        keyword_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="document", 
                    match=models.MatchText(text=word)
                ) for word in query_words
            ]
        )
        
        key_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_emb.tolist(),
            query_filter=keyword_filter,
            limit=k * 4
        )
        
        rrf_scores = {} 

        for rank, hit in enumerate(sem_results):
            if hit.id not in rrf_scores:
                rrf_scores[hit.id] = 0.0
            rrf_scores[hit.id] += self.alpha / (60 + rank + 1)

        for rank, hit in enumerate(key_results):
            if hit.id not in rrf_scores:
                rrf_scores[hit.id] = 0.0
            rrf_scores[hit.id] += (1 - self.alpha) / (60 + rank + 1)
            
        if not rrf_scores:
            logger.warning(f"No results found for query: '{query}'")
            return []

        raw_scores = list(rrf_scores.values())
        max_raw = max(raw_scores)
        if max_raw < raw_min_score:
            logger.warning(f"Low raw RRF scores for '{query}'; max raw score: {max_raw:.4f}")
            return []

        for doc_id in rrf_scores:
            rrf_scores[doc_id] /= max_raw

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        top_k_ids = sorted_ids[:k]
        
        final_docs = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[id for id in top_k_ids],  
            with_payload=True
        )
        
        doc_map = {doc.id: doc for doc in final_docs}
        
        retrieved = []
        for doc_id in top_k_ids:
            if doc_id not in doc_map:
                continue
            doc = doc_map[doc_id]
            score = rrf_scores.get(doc_id, 0.0)
            content = doc.payload.get("document", "")
            content_trunc = content[:300] + "..." if len(content) > 300 else content
            
            content_lower = content.lower()
            for keyword, multiplier in self.boost_keywords.items():
                if keyword in content_lower:
                    score *= multiplier
            
            retrieved.append({
                'content': content_trunc,
                'metadata': doc.payload.get("metadata", {}),
                'score': score
            })
        
        good_retrieved = [r for r in retrieved if r['score'] >= min_score]
        if not good_retrieved:
            logger.warning(f"No results met min_score threshold for '{query}' after boosting")
            return []
        
        good_retrieved.sort(key=lambda x: x['score'], reverse=True)
        retrieved = good_retrieved[:k]
        
        avg_score = np.mean([r['score'] for r in retrieved])
        logger.debug(f"Retrieved {len(retrieved)} chunks (avg score: {avg_score:.3f}) for '{query}'")
        return retrieved