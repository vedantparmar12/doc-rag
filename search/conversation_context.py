"""
Conversation Context Manager - Remember previous queries for better answers.

Features:
- Tracks conversation history
- Maintains query context
- Resolves follow-up questions
- Expands references (e.g., "it", "that", "the previous one")
- Session management
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    query: str
    results: List[Dict[str, Any]]  # Simplified result data
    timestamp: str
    session_id: str
    turn_id: int


@dataclass
class ConversationSession:
    """A conversation session."""
    session_id: str
    turns: List[ConversationTurn]
    created_at: str
    last_active: str
    total_queries: int = 0


class ConversationContextManager:
    """
    Manages conversation context across multiple queries.

    Features:
    - Resolves follow-up questions using previous context
    - Expands references like "it", "that", "the one"
    - Maintains session history
    - Auto-expires old sessions
    """

    def __init__(
        self,
        context_file: Path,
        max_history: int = 10,
        session_timeout_hours: int = 24
    ):
        self.context_file = Path(context_file)
        self.max_history = max_history
        self.session_timeout = timedelta(hours=session_timeout_hours)

        self.sessions: Dict[str, ConversationSession] = {}
        self.active_session_id: Optional[str] = None

        # Reference words that indicate follow-up queries
        self.reference_words = {
            'it', 'this', 'that', 'these', 'those',
            'the one', 'the same', 'same thing',
            'previous', 'earlier', 'before', 'above'
        }

        # Follow-up indicators
        self.followup_patterns = [
            'also', 'additionally', 'furthermore', 'more about',
            'what about', 'how about', 'and', 'or',
            'tell me more', 'explain', 'details', 'elaborate'
        ]

        # Load existing context
        self._load_context()

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session or resume existing.

        Args:
            session_id: Optional session ID to resume

        Returns:
            Session ID
        """
        if session_id and session_id in self.sessions:
            # Resume existing session
            session = self.sessions[session_id]
            session.last_active = datetime.now().isoformat()
            self.active_session_id = session_id
            logger.info(f"Resumed session: {session_id}")
        else:
            # Create new session
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                turns=[],
                created_at=datetime.now().isoformat(),
                last_active=datetime.now().isoformat()
            )
            self.active_session_id = session_id
            logger.info(f"Started new session: {session_id}")

        # Clean up expired sessions
        self._cleanup_expired_sessions()

        return session_id

    def add_turn(
        self,
        query: str,
        results: List[Any],
        session_id: Optional[str] = None
    ) -> int:
        """
        Add a conversation turn to the active session.

        Args:
            query: User query
            results: Search results
            session_id: Optional session ID (uses active if None)

        Returns:
            Turn ID
        """
        # Use active session if not specified
        if session_id is None:
            session_id = self.active_session_id

        if not session_id or session_id not in self.sessions:
            # Start new session if none exists
            session_id = self.start_session()

        session = self.sessions[session_id]

        # Create turn
        turn_id = len(session.turns)

        # Simplify results for storage (keep only essential data)
        simplified_results = []
        for result in results[:5]:  # Keep top 5
            simplified_results.append({
                'path': getattr(result, 'path', str(result)),
                'title': getattr(result, 'title', ''),
                'score': getattr(result, 'score', 0)
            })

        turn = ConversationTurn(
            query=query,
            results=simplified_results,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            turn_id=turn_id
        )

        session.turns.append(turn)
        session.last_active = datetime.now().isoformat()
        session.total_queries += 1

        # Limit history size
        if len(session.turns) > self.max_history:
            session.turns = session.turns[-self.max_history:]

        # Save context
        self._save_context()

        logger.debug(f"Added turn {turn_id} to session {session_id}")

        return turn_id

    def expand_query(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Expand a query using conversation context.

        Args:
            query: Original query (may have references)
            session_id: Optional session ID (uses active if None)

        Returns:
            (expanded_query, context_info)
        """
        # Use active session if not specified
        if session_id is None:
            session_id = self.active_session_id

        if not session_id or session_id not in self.sessions:
            # No context available
            return query, {'has_context': False}

        session = self.sessions[session_id]

        if not session.turns:
            # No previous turns
            return query, {'has_context': False}

        # Check if query is a follow-up
        is_followup = self._is_followup_query(query)

        if not is_followup:
            # Not a follow-up - return as-is
            return query, {
                'has_context': True,
                'is_followup': False,
                'previous_queries': len(session.turns)
            }

        # Expand query using context
        expanded = self._expand_with_context(query, session)

        logger.info(f"Expanded query: '{query}' -> '{expanded}'")

        return expanded, {
            'has_context': True,
            'is_followup': True,
            'original_query': query,
            'expanded_query': expanded,
            'previous_queries': len(session.turns)
        }

    def get_session_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 5
    ) -> List[ConversationTurn]:
        """
        Get conversation history for a session.

        Args:
            session_id: Optional session ID (uses active if None)
            limit: Max turns to return

        Returns:
            List of conversation turns
        """
        # Use active session if not specified
        if session_id is None:
            session_id = self.active_session_id

        if not session_id or session_id not in self.sessions:
            return []

        session = self.sessions[session_id]
        return session.turns[-limit:]

    def _is_followup_query(self, query: str) -> bool:
        """Check if query is a follow-up question."""
        query_lower = query.lower()

        # Check for reference words
        for ref_word in self.reference_words:
            if ref_word in query_lower:
                return True

        # Check for follow-up patterns
        for pattern in self.followup_patterns:
            if pattern in query_lower:
                return True

        # Very short queries are often follow-ups
        if len(query.split()) <= 3:
            return True

        return False

    def _expand_with_context(
        self,
        query: str,
        session: ConversationSession
    ) -> str:
        """Expand query using conversation context."""
        if not session.turns:
            return query

        # Get previous turn
        prev_turn = session.turns[-1]
        prev_query = prev_turn.query
        prev_results = prev_turn.results

        expanded_parts = []

        # Strategy 1: Replace pronouns with previous topic
        expanded_query = query

        # Extract topic from previous query (main nouns)
        prev_topic = self._extract_topic(prev_query)

        if prev_topic:
            # Replace pronouns
            for ref_word in self.reference_words:
                pattern = r'\b' + ref_word + r'\b'
                import re
                expanded_query = re.sub(
                    pattern,
                    prev_topic,
                    expanded_query,
                    flags=re.IGNORECASE
                )

        # Strategy 2: Combine with previous query for short follow-ups
        if len(query.split()) <= 3:
            # Very short query - likely needs previous context
            # Example: "and kubernetes?" -> "docker and kubernetes?"
            if not any(word in query.lower() for word in ['what', 'how', 'where', 'when', 'why']):
                # Add previous topic as context
                if prev_topic:
                    expanded_query = f"{prev_topic} {query}"

        # Strategy 3: Add context from previous results
        if prev_results:
            # Get top result from previous query
            top_result = prev_results[0]
            context_doc = top_result.get('title', '')

            # If query is very ambiguous, add context
            if len(query.split()) <= 2 and context_doc:
                expanded_query = f"{expanded_query} (context: {context_doc})"

        return expanded_query

    def _extract_topic(self, query: str) -> Optional[str]:
        """Extract main topic from query."""
        # Remove question words
        question_words = {
            'what', 'how', 'where', 'when', 'why', 'which', 'who',
            'is', 'are', 'the', 'a', 'an', 'to', 'for', 'of', 'in', 'on'
        }

        words = query.lower().split()
        topic_words = [w for w in words if w not in question_words and len(w) > 2]

        if topic_words:
            # Return first significant word (likely the main topic)
            return topic_words[0]

        return None

    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired = []

        for session_id, session in self.sessions.items():
            last_active = datetime.fromisoformat(session.last_active)
            if now - last_active > self.session_timeout:
                expired.append(session_id)

        for session_id in expired:
            del self.sessions[session_id]
            logger.info(f"Removed expired session: {session_id}")

        if expired:
            self._save_context()

    def _load_context(self):
        """Load conversation context from disk."""
        if not self.context_file.exists():
            self.sessions = {}
            return

        try:
            with open(self.context_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct sessions
            for session_data in data.get('sessions', []):
                session = ConversationSession(
                    session_id=session_data['session_id'],
                    turns=[
                        ConversationTurn(**turn_data)
                        for turn_data in session_data['turns']
                    ],
                    created_at=session_data['created_at'],
                    last_active=session_data['last_active'],
                    total_queries=session_data.get('total_queries', len(session_data['turns']))
                )
                self.sessions[session.session_id] = session

            logger.info(f"Loaded {len(self.sessions)} conversation sessions")

        except Exception as e:
            logger.error(f"Failed to load conversation context: {e}")
            self.sessions = {}

    def _save_context(self):
        """Save conversation context to disk."""
        self.context_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'sessions': [
                {
                    'session_id': session.session_id,
                    'turns': [asdict(turn) for turn in session.turns],
                    'created_at': session.created_at,
                    'last_active': session.last_active,
                    'total_queries': session.total_queries
                }
                for session in self.sessions.values()
            ],
            'last_updated': datetime.now().isoformat()
        }

        with open(self.context_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


# Convenience function for MCP server
async def search_with_context(
    query: str,
    search_engine: Any,
    context_manager: ConversationContextManager,
    session_id: Optional[str] = None,
    limit: int = 10
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Search with conversation context awareness.

    Args:
        query: User query
        search_engine: Search engine instance
        context_manager: Conversation context manager
        session_id: Optional session ID
        limit: Max results

    Returns:
        (results, context_info)
    """
    # Expand query using context
    expanded_query, context_info = context_manager.expand_query(query, session_id)

    # Search with expanded query
    if expanded_query != query:
        logger.info(f"Using expanded query for search: {expanded_query}")
        results = await search_engine.fast_search(expanded_query, limit=limit)
    else:
        results = await search_engine.fast_search(query, limit=limit)

    # Add turn to conversation history
    context_manager.add_turn(query, results, session_id)

    return results, context_info
