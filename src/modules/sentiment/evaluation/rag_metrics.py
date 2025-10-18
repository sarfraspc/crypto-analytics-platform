import logging
from typing import List

logger = logging.getLogger(__name__)

def faithfulness(generated: str, contexts: List[str]):
    gen_words = set(generated.lower().split())
    context_words = set(' '.join(contexts).lower().split())
    overlap = len(gen_words & context_words) / len(gen_words) if gen_words else 0.0
    return overlap
