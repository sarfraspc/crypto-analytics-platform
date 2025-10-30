import logging
from typing import List
from datetime import datetime, timedelta
import nltk
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from core.database import get_timescale_engine

logger = logging.getLogger(__name__)
nltk.download('punkt', quiet=True)

class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 500, overlap: int = 50):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = get_timescale_engine()
        return self._engine

    def fetch_docs(self, days_back: int = 30):
        cutoff = datetime.now() - timedelta(days=days_back)
        query = text("""
            SELECT id, title || ' ' || COALESCE(text, '') AS content, 'news' AS source
            FROM news_articles WHERE published > :cutoff
            UNION ALL
            SELECT id, title || ' ' || COALESCE(body, '') AS content, 'reddit' AS source
            FROM reddit_posts WHERE created > :cutoff
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {'cutoff': cutoff}).fetchall()
        docs = [{'id': row[0], 'content': row[1], 'source': row[2]} for row in result]
        logger.info(f"Fetched {len(docs)} docs from last {days_back} days.")
        return docs

    def chunk_text(self, text: str, method: str = 'fixed'):
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
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings.tolist()  

    def process_docs(self, docs: List[dict], chunk_method: str = 'sentence'): 
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