import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np
from modules.sentiment.rag.vector_store import QdrantVectorStore
from modules.sentiment.rag.embedder import Embedder

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, embedder: Embedder, vector_store: QdrantVectorStore, alpha: float = 0.7, expansion_terms: List[str] = None, boost_keywords: Dict[str, float] = None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.alpha = alpha  
        self.model = self.embedder.model
        self.bm25 = None
        self.indexed_docs = []  
        self.indexed_metadatas = []  
        self.expansion_terms = expansion_terms or ["bitcoin", "crypto", "market", "price", "discussion", "fear", "greed", "sentiment"]
        self.boost_keywords = boost_keywords or {"sentiment": 1.2, "btc": 1.2}

    def index_for_hybrid(self):
        results = self.vector_store.get_all()
        self.indexed_docs = results["documents"]
        self.indexed_metadatas = results["metadatas"]
        tokenized_docs = [doc.lower().split() for doc in self.indexed_docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"Built BM25 index over {len(self.indexed_docs)} docs.")

    def retrieve(self, query: str, k: int = 5, min_score: float = 0.05):  
        expanded_query = f"{query} {' '.join(self.expansion_terms)}"
        query_lower = expanded_query.lower()
        query_tokens = query_lower.split()
        query_emb = self.model.encode([expanded_query])

        sem_results = self.vector_store.query_semantic(query_emb[0], n_results=k*4)
        num_sem_results = len(sem_results["documents"])
        logger.debug(f"Semantic search returned {num_sem_results} results for expanded query: '{expanded_query}'")
        sem_scores = np.array(sem_results["distances"]) if num_sem_results > 0 else np.array([])
        sem_ranks = np.argsort(-sem_scores) if num_sem_results > 0 else np.array([])

        if self.bm25:
            bm25_scores = self.bm25.get_scores(query_tokens)
            bm25_ranks = np.argsort(-bm25_scores)[:k*4]
        else:
            bm25_scores = np.zeros(len(self.indexed_docs))
            bm25_ranks = np.arange(min(k*4, len(self.indexed_docs)))
        num_bm25 = len(bm25_ranks)

        rrf_scores = np.zeros(len(self.indexed_docs))
        max_iter = max(len(sem_ranks), num_bm25, k*4)
        for i in range(max_iter):
            if num_sem_results > 0 and i < len(sem_ranks):
                doc_idx_sem = sem_ranks[i]
                if 0 <= doc_idx_sem < len(rrf_scores):
                    rrf_scores[doc_idx_sem] += self.alpha / (60 + i + 1)
            if i < num_bm25:
                doc_idx_bm25 = bm25_ranks[i]
                if 0 <= doc_idx_bm25 < len(rrf_scores):
                    rrf_scores[doc_idx_bm25] += (1 - self.alpha) / (60 + i + 1)

        if np.max(rrf_scores) > 0:
            rrf_scores = rrf_scores / np.max(rrf_scores) 

        top_indices = np.argsort(-rrf_scores)[:k]        
        good_indices = [idx for idx in top_indices if rrf_scores[idx] >= min_score]
        if len(good_indices) >= k or len(good_indices) > 0:
            top_indices = good_indices[:k]
        else:
            logger.warning(f"Low scores for '{query}'; using top k")
        retrieved = []
        for idx in top_indices:
            score = float(rrf_scores[idx])
            content = self.indexed_docs[idx]
            for keyword, multiplier in self.boost_keywords.items():
                if keyword in content.lower():
                    score *= multiplier
            retrieved.append({
                'content': content[:300],
                'metadata': self.indexed_metadatas[idx],
                'score': score
            })
        logger.debug(f"Retrieved {len(retrieved)} chunks (avg score: {np.mean([r['score'] for r in retrieved]):.3f}) for '{query}'")
        return retrieved