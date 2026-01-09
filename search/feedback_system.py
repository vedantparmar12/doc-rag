"""
Feedback System - Learn from user corrections and improve results.

Features:
- Tracks user feedback (helpful/not helpful)
- Learns from corrections
- Adjusts search relevance
- Improves over time
- Triggers reindexing if needed
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    query: str
    result_path: str
    feedback_type: str  # 'helpful', 'not_helpful', 'correction'
    correction_text: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class QueryPattern:
    """Learned pattern from user queries."""
    keywords: List[str]
    preferred_docs: List[str]  # Docs users found helpful
    avoided_docs: List[str]  # Docs users said weren't helpful
    correction_count: int
    last_updated: str


class FeedbackSystem:
    """
    Manages user feedback and learns from it.

    Workflow:
    1. User searches and gets results
    2. User provides feedback (helpful/not helpful/correction)
    3. System records feedback
    4. System adjusts future search results
    5. If many corrections, triggers reindexing
    """

    def __init__(
        self,
        feedback_file: Path,
        patterns_file: Path,
        reindex_threshold: int = 10  # Reindex after N corrections
    ):
        self.feedback_file = Path(feedback_file)
        self.patterns_file = Path(patterns_file)
        self.reindex_threshold = reindex_threshold

        self.feedback_history: List[FeedbackEntry] = []
        self.query_patterns: Dict[str, QueryPattern] = {}

        self.correction_count = 0

        # Load existing data
        self._load_feedback()
        self._load_patterns()

    def record_feedback(
        self,
        query: str,
        result_path: str,
        feedback_type: str,
        correction_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record user feedback.

        Args:
            query: User's search query
            result_path: Path of the result document
            feedback_type: 'helpful', 'not_helpful', or 'correction'
            correction_text: If correction, the corrected answer

        Returns:
            Response with action taken
        """
        # Create feedback entry
        entry = FeedbackEntry(
            query=query,
            result_path=result_path,
            feedback_type=feedback_type,
            correction_text=correction_text
        )

        self.feedback_history.append(entry)

        # Update query patterns
        self._update_patterns(entry)

        # Save feedback
        self._save_feedback()
        self._save_patterns()

        # Check if reindexing needed
        needs_reindex = False
        if feedback_type == 'correction':
            self.correction_count += 1
            if self.correction_count >= self.reindex_threshold:
                needs_reindex = True
                logger.warning(f"Reindex threshold reached ({self.correction_count} corrections)")

        logger.info(f"Recorded {feedback_type} feedback for query: '{query}'")

        return {
            'status': 'recorded',
            'feedback_type': feedback_type,
            'needs_reindex': needs_reindex,
            'total_feedback': len(self.feedback_history),
            'correction_count': self.correction_count
        }

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about feedback."""
        if not self.feedback_history:
            return {
                'total': 0,
                'helpful': 0,
                'not_helpful': 0,
                'corrections': 0,
                'patterns_learned': 0
            }

        feedback_counts = defaultdict(int)
        for entry in self.feedback_history:
            feedback_counts[entry.feedback_type] += 1

        return {
            'total': len(self.feedback_history),
            'helpful': feedback_counts['helpful'],
            'not_helpful': feedback_counts['not_helpful'],
            'corrections': feedback_counts['correction'],
            'patterns_learned': len(self.query_patterns),
            'correction_count': self.correction_count,
            'needs_reindex': self.correction_count >= self.reindex_threshold
        }

    def adjust_search_results(
        self,
        query: str,
        results: List[Any]
    ) -> List[Any]:
        """
        Adjust search results based on learned patterns.

        Args:
            query: User query
            results: Search results from engine

        Returns:
            Adjusted results
        """
        # Extract keywords from query
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Find matching patterns
        relevant_patterns = []
        for pattern_key, pattern in self.query_patterns.items():
            # Check if query matches pattern keywords
            pattern_keywords = set(pattern.keywords)
            if pattern_keywords & query_words:  # Intersection
                relevant_patterns.append(pattern)

        if not relevant_patterns:
            # No patterns match - return results as-is
            return results

        logger.info(f"Found {len(relevant_patterns)} matching patterns for query")

        # Adjust scores based on patterns
        adjusted_results = []

        for result in results:
            result_path = getattr(result, 'path', str(result))
            original_score = getattr(result, 'score', 100.0)

            # Apply pattern adjustments
            score_adjustment = 0.0

            for pattern in relevant_patterns:
                # Boost if document was helpful in the past
                if result_path in pattern.preferred_docs:
                    boost = 10.0  # +10 points
                    score_adjustment += boost

                # Penalize if document was not helpful in the past
                if result_path in pattern.avoided_docs:
                    penalty = -15.0  # -15 points
                    score_adjustment += penalty

            # Apply adjustment
            if score_adjustment != 0:
                new_score = max(0, min(100, original_score + score_adjustment))
                if hasattr(result, 'score'):
                    result.score = new_score
                logger.debug(
                    f"Adjusted {result_path}: {original_score:.1f} -> {new_score:.1f} "
                    f"({score_adjustment:+.1f})"
                )

            adjusted_results.append(result)

        # Re-sort by adjusted scores
        adjusted_results.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)

        return adjusted_results

    def get_corrections_for_reindex(self) -> List[Dict[str, Any]]:
        """
        Get corrections that should be applied during reindexing.

        Returns:
            List of corrections with paths and suggested changes
        """
        corrections = []

        for entry in self.feedback_history:
            if entry.feedback_type == 'correction' and entry.correction_text:
                corrections.append({
                    'query': entry.query,
                    'path': entry.result_path,
                    'correction': entry.correction_text,
                    'timestamp': entry.timestamp
                })

        return corrections

    def reset_correction_count(self):
        """Reset correction count after reindexing."""
        self.correction_count = 0
        self._save_patterns()
        logger.info("Reset correction count after reindexing")

    def _update_patterns(self, entry: FeedbackEntry):
        """Update query patterns based on feedback."""
        # Extract keywords from query
        keywords = [w.lower() for w in entry.query.split() if len(w) > 2]

        if not keywords:
            return

        # Create pattern key (sorted keywords)
        pattern_key = '_'.join(sorted(keywords[:3]))  # Use top 3 keywords

        # Get or create pattern
        if pattern_key not in self.query_patterns:
            self.query_patterns[pattern_key] = QueryPattern(
                keywords=keywords,
                preferred_docs=[],
                avoided_docs=[],
                correction_count=0,
                last_updated=datetime.now().isoformat()
            )

        pattern = self.query_patterns[pattern_key]

        # Update based on feedback type
        if entry.feedback_type == 'helpful':
            if entry.result_path not in pattern.preferred_docs:
                pattern.preferred_docs.append(entry.result_path)
            # Remove from avoided if it was there
            if entry.result_path in pattern.avoided_docs:
                pattern.avoided_docs.remove(entry.result_path)

        elif entry.feedback_type == 'not_helpful':
            if entry.result_path not in pattern.avoided_docs:
                pattern.avoided_docs.append(entry.result_path)
            # Remove from preferred if it was there
            if entry.result_path in pattern.preferred_docs:
                pattern.preferred_docs.remove(entry.result_path)

        elif entry.feedback_type == 'correction':
            pattern.correction_count += 1

        pattern.last_updated = datetime.now().isoformat()

    def _load_feedback(self):
        """Load feedback history from disk."""
        if not self.feedback_file.exists():
            self.feedback_history = []
            return

        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.feedback_history = [
                FeedbackEntry(**entry) for entry in data.get('feedback', [])
            ]

            logger.info(f"Loaded {len(self.feedback_history)} feedback entries")

        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")
            self.feedback_history = []

    def _save_feedback(self):
        """Save feedback history to disk."""
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'feedback': [asdict(entry) for entry in self.feedback_history],
            'last_updated': datetime.now().isoformat()
        }

        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _load_patterns(self):
        """Load learned patterns from disk."""
        if not self.patterns_file.exists():
            self.query_patterns = {}
            self.correction_count = 0
            return

        try:
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.query_patterns = {
                key: QueryPattern(**pattern_data)
                for key, pattern_data in data.get('patterns', {}).items()
            }

            self.correction_count = data.get('correction_count', 0)

            logger.info(f"Loaded {len(self.query_patterns)} learned patterns")

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            self.query_patterns = {}
            self.correction_count = 0

    def _save_patterns(self):
        """Save learned patterns to disk."""
        self.patterns_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'patterns': {
                key: asdict(pattern)
                for key, pattern in self.query_patterns.items()
            },
            'correction_count': self.correction_count,
            'last_updated': datetime.now().isoformat()
        }

        with open(self.patterns_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
