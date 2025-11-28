"""
Text generation module for RAG pipeline.

Provides answer generation from retrieved contexts using
transformer-based text-to-text generation models.
"""

import logging
from typing import Any, Dict, List

from transformers import pipeline

logger = logging.getLogger(__name__)


class Generator:
    """
    Text generator for RAG response synthesis.

    Uses transformer models to generate coherent answers
    based on retrieved context documents and user queries.
    """

    def __init__(self, model_name: str = 'google/flan-t5-base', max_context_len: int = 200):
        """Initialize generator with model and context length settings."""
        self.generator = pipeline('text2text-generation', model=model_name, device=-1)
        self.max_context_len = max_context_len

    def generate(self, query: str, contexts: List[Dict[str, Any]]):
        """Generate a response based on query and retrieved contexts."""
        context_str = "\n".join([f"Source: {c['metadata'].get('source', 'unknown')} - {c['content'][:self.max_context_len]}" for c in contexts])
        prompt = f"Summarize what the following contexts mean in relation to the question. Only use information from the contexts.\n\nQuestion: {query}\n\nContexts:\n{context_str}\n\nSummary:"

        try:
            response = self.generator(
                prompt,
                max_new_tokens=300,
                num_return_sequences=1,
                do_sample=False,
                truncation=True,
                max_length=512
            )[0]['generated_text']

            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response.replace(prompt, "").strip()

            if len(answer) < 10:
                logger.warning(f"Short response detected (len: {len(answer)}); possible truncation")
                answer = "Based on the contexts, a detailed answer couldn't be fully generated. Key insights: " + answer

            logger.info(f"Generated response for query: {query[:50]}... (len: {len(answer)})")
            return answer

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I'm sorry, I couldn't generate a confident answer based on the retrieved contexts."