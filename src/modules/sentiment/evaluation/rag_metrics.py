"""
RAG evaluation metrics module.

Provides metrics for evaluating RAG pipeline quality including
faithfulness scoring between generated responses and source contexts.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def faithfulness(generated: str, contexts: List[str]):
    """Calculate word overlap between generated text and source contexts."""
    gen_words = set(generated.lower().split())
    context_words = set(' '.join(contexts).lower().split())
    overlap = len(gen_words & context_words) / len(gen_words) if gen_words else 0.0
    return overlap
