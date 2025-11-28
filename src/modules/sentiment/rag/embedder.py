"""
Text embedding module for RAG pipeline.

Provides document fetching, text chunking, and embedding generation
using sentence transformers for semantic search capabilities.
"""

import logging
from datetime import datetime, timedelta
from typing import List

import nltk
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import sessionmaker

from core.database import get_timescale_engine
from data.storage.models import NewsArticle, RedditPost

logger = logging.getLogger(__name__)
nltk.download('punkt', quiet=True)

class Embedder:
    """
    Text embedder for document processing and vector generation.

    Handles fetching documents from database, chunking text into
    manageable pieces, and generating embeddings using sentence transformers.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 500, overlap: int = 50):
        """Initialize embedder with model and chunking parameters."""
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._engine = None

    @property
    def engine(self):
        """Lazy-load database engine."""
        if self._engine is None:
            self._engine = get_timescale_engine()
        return self._engine

    def fetch_docs(self, days_back: int = 30):
        """Fetch news articles and Reddit posts from the last N days."""
        cutoff = datetime.now() - timedelta(days=days_back)
        
        Session = sessionmaker(bind=self.engine)
        with Session() as session:
            news_articles = session.query(NewsArticle).filter(NewsArticle.published > cutoff).all()
            reddit_posts = session.query(RedditPost).filter(RedditPost.created > cutoff).all()
        
        docs = []
        for article in news_articles:
            docs.append({
                'id': article.id,
                'content': f"{article.title} {article.text or ''}",
                'source': 'news'
            })
            
        for post in reddit_posts:
            docs.append({
                'id': post.id,
                'content': f"{post.title} {post.body or ''}",
                'source': 'reddit'
            })
        
        logger.info(f"Fetched {len(docs)} docs from last {days_back} days.")
        return docs

    def chunk_text(self, text: str, method: str = 'fixed'):
        """Split text into chunks using fixed-size or sentence-based method."""
        if method == 'sentence':
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk = []
            current_len = 0
            for sent in sentences:
                if current_len + len(sent) > self.chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sent]
                    current_len = len(sent)
                else:
                    current_chunk.append(sent)
                    current_len += len(sent) + 1
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            return chunks
        else: 
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.overlap):
                chunks.append(text[i:i + self.chunk_size])
            return chunks

    def embed_chunks(self, chunks: List[str]):
        """Generate embeddings for a list of text chunks."""
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings.tolist()  

    def process_docs(self, docs: List[dict], chunk_method: str = 'sentence'):
        """Process documents into chunks with embeddings and metadata.""" 
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        for doc in docs:
            chunks = self.chunk_text(doc['content'], chunk_method)
            chunk_embs = self.embed_chunks(chunks)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk.strip()) 
                all_embeddings.append(chunk_embs[i])
                all_metadata.append({
                    'doc_id': doc['id'],
                    'source': doc['source'],
                    'chunk_idx': i
                })
        logger.info(f"Processed {len(all_chunks)} chunks into embeddings.")
        return all_chunks, all_embeddings, all_metadata