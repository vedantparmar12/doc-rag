"""
Query Decomposition System - Break complex queries into sub-queries.

Features:
- Detects multi-part questions
- Extracts individual sub-queries
- Maintains query intent
- Combines results intelligently
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SubQuery:
    """A sub-query extracted from complex query."""
    text: str
    query_type: str  # 'what', 'how', 'where', 'when', 'why', 'which'
    priority: int  # 1 (high) to 3 (low)
    keywords: List[str]
    original_index: int


class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries.

    Examples:
    - "How do I deploy to production and what are the prerequisites?"
      → ["How do I deploy to production?", "What are the prerequisites?"]

    - "Explain kubernetes and docker differences"
      → ["Explain kubernetes", "Explain docker", "kubernetes vs docker"]

    - "What is the API endpoint and how do I authenticate?"
      → ["What is the API endpoint?", "How do I authenticate?"]
    """

    def __init__(self):
        # Patterns for detecting multiple questions
        self.separator_patterns = [
            r'\s+and\s+',
            r'\s+or\s+',
            r'\s*,\s*and\s+',
            r'\s*;\s*',
            r'\s*\.\s+',
            r'\s+also\s+',
            r'\s+plus\s+'
        ]

        # Question word patterns
        self.question_words = {
            'what': r'\bwhat\b',
            'how': r'\bhow\b',
            'where': r'\bwhere\b',
            'when': r'\bwhen\b',
            'why': r'\bwhy\b',
            'which': r'\bwhich\b',
            'who': r'\bwho\b'
        }

    def decompose(self, query: str) -> List[SubQuery]:
        """
        Decompose a complex query into sub-queries.

        Args:
            query: Original user query

        Returns:
            List of sub-queries (returns original if simple)
        """
        query = query.strip()

        if not query:
            return []

        # Check if query is complex
        if not self._is_complex_query(query):
            # Simple query - return as-is
            return [SubQuery(
                text=query,
                query_type=self._detect_query_type(query),
                priority=1,
                keywords=self._extract_keywords(query),
                original_index=0
            )]

        # Decompose complex query
        sub_queries = self._split_query(query)

        logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
        for i, sq in enumerate(sub_queries):
            logger.debug(f"  {i+1}. [{sq.query_type}] {sq.text}")

        return sub_queries

    def _is_complex_query(self, query: str) -> bool:
        """Check if query is complex (multiple questions)."""
        # Check for multiple question words
        question_count = sum(
            1 for pattern in self.question_words.values()
            if re.search(pattern, query, re.IGNORECASE)
        )

        if question_count > 1:
            return True

        # Check for separators with question indicators
        for sep_pattern in self.separator_patterns:
            if re.search(sep_pattern, query, re.IGNORECASE):
                # Check if both parts have query-like structure
                parts = re.split(sep_pattern, query, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    return True

        return False

    def _split_query(self, query: str) -> List[SubQuery]:
        """Split complex query into sub-queries."""
        sub_queries = []

        # Strategy 1: Split by explicit question separators
        parts = self._split_by_questions(query)

        if len(parts) > 1:
            # Multiple explicit questions found
            for i, part in enumerate(parts):
                if part.strip():
                    sub_queries.append(SubQuery(
                        text=part.strip(),
                        query_type=self._detect_query_type(part),
                        priority=1 if i == 0 else 2,
                        keywords=self._extract_keywords(part),
                        original_index=i
                    ))
            return sub_queries

        # Strategy 2: Split by conjunctions with question words
        parts = self._split_by_conjunctions(query)

        if len(parts) > 1:
            for i, part in enumerate(parts):
                if part.strip():
                    sub_queries.append(SubQuery(
                        text=part.strip(),
                        query_type=self._detect_query_type(part),
                        priority=1 if i == 0 else 2,
                        keywords=self._extract_keywords(part),
                        original_index=i
                    ))
            return sub_queries

        # Strategy 3: Split by topics (for comparison queries)
        parts = self._split_by_topics(query)

        if len(parts) > 1:
            for i, part in enumerate(parts):
                if part.strip():
                    sub_queries.append(SubQuery(
                        text=part.strip(),
                        query_type=self._detect_query_type(part),
                        priority=1,
                        keywords=self._extract_keywords(part),
                        original_index=i
                    ))
            return sub_queries

        # Fallback: return original
        return [SubQuery(
            text=query,
            query_type=self._detect_query_type(query),
            priority=1,
            keywords=self._extract_keywords(query),
            original_index=0
        )]

    def _split_by_questions(self, query: str) -> List[str]:
        """Split by explicit question marks."""
        # Split on question marks, keep the ?
        parts = re.split(r'(\?)', query)

        questions = []
        current = ""

        for part in parts:
            current += part
            if part == '?':
                questions.append(current.strip())
                current = ""

        if current.strip():
            questions.append(current.strip())

        return [q for q in questions if q]

    def _split_by_conjunctions(self, query: str) -> List[str]:
        """Split by conjunctions (and, or, also)."""
        # Find conjunction positions with question words
        parts = []

        # Pattern: question word ... and ... question word
        pattern = r'\b(and|or|also)\s+(?=what|how|where|when|why|which|who)'

        segments = re.split(pattern, query, flags=re.IGNORECASE)

        # Reconstruct parts (regex split includes captured groups)
        current = ""
        for i, segment in enumerate(segments):
            if segment.lower() in ['and', 'or', 'also']:
                if current.strip():
                    parts.append(current.strip())
                current = ""
            else:
                current += segment

        if current.strip():
            parts.append(current.strip())

        return parts if len(parts) > 1 else [query]

    def _split_by_topics(self, query: str) -> List[str]:
        """Split by topics (for comparison queries)."""
        # Pattern: "explain X and Y"
        # Pattern: "difference between X and Y"
        # Pattern: "compare X and Y"

        comparison_patterns = [
            r'(explain|describe|what is)\s+(\w+)\s+and\s+(\w+)',
            r'(difference|compare|contrast)\s+(?:between\s+)?(\w+)\s+and\s+(\w+)',
            r'(\w+)\s+vs\.?\s+(\w+)',
            r'(\w+)\s+versus\s+(\w+)'
        ]

        for pattern in comparison_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()

                if len(groups) >= 3:
                    # Pattern with action word
                    action = groups[0]
                    topic1 = groups[1]
                    topic2 = groups[2]

                    return [
                        f"{action} {topic1}",
                        f"{action} {topic2}",
                        f"{action} {topic1} vs {topic2}"
                    ]
                elif len(groups) == 2:
                    # Simple comparison (vs/versus)
                    topic1 = groups[0]
                    topic2 = groups[1]

                    return [
                        f"explain {topic1}",
                        f"explain {topic2}",
                        f"compare {topic1} and {topic2}"
                    ]

        return [query]

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of question."""
        query_lower = query.lower()

        for qtype, pattern in self.question_words.items():
            if re.search(pattern, query_lower):
                return qtype

        # Check for imperative forms
        if re.search(r'\b(explain|describe|show|list|tell)\b', query_lower):
            return 'explain'

        if re.search(r'\b(find|search|get|locate)\b', query_lower):
            return 'find'

        return 'general'

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove question words and common words
        stopwords = {
            'what', 'how', 'where', 'when', 'why', 'which', 'who',
            'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but',
            'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'do', 'does', 'did', 'can', 'could', 'should', 'would'
        }

        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        return keywords

    def combine_results(
        self,
        sub_query_results: List[Tuple[SubQuery, List[Any]]]
    ) -> List[Any]:
        """
        Combine results from multiple sub-queries.

        Strategy:
        1. Prioritize results from higher priority queries
        2. Remove duplicates
        3. Boost results that match multiple sub-queries
        4. Return top results
        """
        # Collect all results with scores
        all_results = {}  # path -> (result, score)

        for sub_query, results in sub_query_results:
            priority_boost = 1.0 / sub_query.priority  # Higher priority = higher boost

            for result in results:
                result_id = getattr(result, 'path', str(result))

                if result_id in all_results:
                    # Boost score for appearing in multiple sub-queries
                    existing_score = all_results[result_id][1]
                    new_score = existing_score + (result.score * priority_boost * 0.5)
                    all_results[result_id] = (result, new_score)
                else:
                    # New result
                    all_results[result_id] = (result, result.score * priority_boost)

        # Sort by combined score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x[1],
            reverse=True
        )

        # Update scores and return
        combined = []
        for result, score in sorted_results:
            # Update result score
            if hasattr(result, 'score'):
                result.score = min(100.0, score)  # Cap at 100
            combined.append(result)

        return combined


# Convenience function
async def decompose_and_search(
    query: str,
    search_engine: Any,
    limit: int = 10
) -> List[Any]:
    """
    Decompose query and search for each sub-query, then combine results.

    Args:
        query: User query
        search_engine: Search engine with fast_search method
        limit: Max results to return

    Returns:
        Combined search results
    """
    decomposer = QueryDecomposer()

    # Decompose query
    sub_queries = decomposer.decompose(query)

    if len(sub_queries) == 1:
        # Simple query - direct search
        return await search_engine.fast_search(query, limit=limit)

    # Search for each sub-query
    sub_query_results = []

    for sub_query in sub_queries:
        results = await search_engine.fast_search(
            sub_query.text,
            limit=limit * 2  # Get more candidates
        )
        sub_query_results.append((sub_query, results))

    # Combine results
    combined = decomposer.combine_results(sub_query_results)

    return combined[:limit]
