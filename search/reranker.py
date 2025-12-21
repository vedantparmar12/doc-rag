"""
Reranker for improving search result quality.

Reranking refines initial search results by:
1. Fast retrieval gets top 20-50 candidates (keyword/semantic)
2. Reranker scores each candidate against the query
3. Return top 5-10 most relevant results

This gives better quality than embeddings alone!
"""

import os
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """Result after reranking."""
    original_score: float
    rerank_score: float
    final_score: float
    content: str
    metadata: dict


class CrossEncoderReranker:
    """
    Cross-encoder reranker for better relevance.

    Unlike bi-encoders (embeddings), cross-encoders:
    - Score query + document pairs directly
    - More accurate (but slower)
    - Perfect for reranking top candidates

    FREE! Runs locally with HuggingFace models.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker.

        Popular models:
        - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good)
        - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, better)
        - cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingual)
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers required for reranking. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker loaded")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float, dict]],
        top_k: int = 10
    ) -> List[RerankedResult]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: User query
            candidates: List of (content, score, metadata)
            top_k: Number of results to return

        Returns:
            Top-k reranked results
        """
        if not candidates:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, content) for content, _, _ in candidates]

        # Get reranker scores
        rerank_scores = self.model.predict(pairs, show_progress_bar=False)

        # Combine with original scores
        results = []
        for i, (content, orig_score, metadata) in enumerate(candidates):
            rerank_score = float(rerank_scores[i])

            # Weighted combination (70% reranker, 30% original)
            final_score = 0.7 * rerank_score + 0.3 * orig_score

            results.append(RerankedResult(
                original_score=orig_score,
                rerank_score=rerank_score,
                final_score=final_score,
                content=content,
                metadata=metadata
            ))

        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)

        return results[:top_k]


class NoOpReranker:
    """
    No-op reranker (passthrough).
    Use when reranking is disabled.
    """

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float, dict]],
        top_k: int = 10
    ) -> List[RerankedResult]:
        """Just pass through original results."""
        results = []
        for content, score, metadata in candidates[:top_k]:
            results.append(RerankedResult(
                original_score=score,
                rerank_score=score,
                final_score=score,
                content=content,
                metadata=metadata
            ))
        return results


def create_reranker(
    enabled: bool = True,
    model: Optional[str] = None
) -> CrossEncoderReranker | NoOpReranker:
    """
    Factory function to create reranker.

    Args:
        enabled: Whether to enable reranking
        model: Model name (if None, use default)

    Returns:
        Reranker instance

    Example:
        # With reranking (better quality)
        reranker = create_reranker(enabled=True)

        # Without reranking (faster)
        reranker = create_reranker(enabled=False)
    """
    if not enabled:
        logger.info("Reranking disabled, using pass-through")
        return NoOpReranker()

    model = model or os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return CrossEncoderReranker(model_name=model)


# Recommended configurations
RERANKER_CONFIGS = {
    "fast": {
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "notes": "Fast, good quality (default)"
    },
    "quality": {
        "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "notes": "Better quality, slower"
    },
    "multilingual": {
        "model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "notes": "Multilingual support"
    }
}


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RERANKER CONFIGURATIONS")
    print("="*60)

    for name, config in RERANKER_CONFIGS.items():
        print(f"\n{name.upper()}:")
        print(f"  Model: {config['model']}")
        print(f"  Notes: {config['notes']}")

    print("\n" + "="*60)
    print("\nUsage:")
    print("  export ENABLE_RERANKING=true")
    print("  export RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("="*60 + "\n")
