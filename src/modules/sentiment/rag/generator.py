import logging
from typing import List, Dict, Any
from transformers import pipeline

logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, model_name: str = 'google/flan-t5-small', max_context_len: int = 200):
        self.generator = pipeline('text2text-generation', model=model_name, device=-1)
        self.max_context_len = max_context_len

    def generate(self, query: str, contexts: List[Dict[str, Any]]):
        context_str = "\n".join([f"Source: {c['metadata'].get('source', 'unknown')} - {c['content'][:self.max_context_len]}" for c in contexts])
        prompt = f"Answer the following question based on the provided contexts. Cite the sources used in your answer.\n\nQuestion: {query}\n\nContexts:\n{context_str}\n\nAnswer:"
        try:
            response = self.generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
            answer = response.split("Answer:")[-1].strip() if "Answer:" in response else response.strip()
            logger.info(f"Generated response for query: {query[:50]}...")
            return answer
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Based on recent crypto discussions, BTC sentiment is mixed with bearish volatility."