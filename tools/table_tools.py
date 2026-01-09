"""
Table search tools for finding and querying tables in documentation.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


@dataclass
class TableSearchResult:
    """Table search result."""
    file_path: str
    file_title: str
    table_index: int
    headers: List[str]
    rows: List[List[str]]
    matching_rows: List[List[str]]
    score: float
    caption: Optional[str] = None
    row_count: int = 0
    col_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'file_path': self.file_path,
            'file_title': self.file_title,
            'table_index': self.table_index,
            'headers': self.headers,
            'rows': self.rows,
            'matching_rows': self.matching_rows,
            'score': self.score,
            'caption': self.caption,
            'row_count': self.row_count,
            'col_count': self.col_count
        }


class TableTools:
    """Tools for searching and querying tables in documentation."""

    def __init__(self, search_engine, docs_folder: Path):
        """
        Initialize table tools.

        Args:
            search_engine: HybridSearchEngine instance
            docs_folder: Root documentation folder
        """
        self.search_engine = search_engine
        self.docs_folder = docs_folder

    async def search_tables(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.3
    ) -> List[TableSearchResult]:
        """
        Search for tables matching query.

        Args:
            query: Search query
            limit: Maximum results to return
            min_score: Minimum relevance score

        Returns:
            List of matching tables

        Examples:
            - "configuration settings" → finds config tables
            - "API endpoints" → finds API reference tables
            - "deployment options" → finds deployment comparison tables
        """
        results = []

        # Search through all indexed files
        for path, metadata in self.search_engine.file_index.items():
            if not metadata.get('tables'):
                continue

            # Check each table
            for idx, table in enumerate(metadata['tables']):
                score = self._score_table(query, table)

                if score >= min_score:
                    # Find matching rows
                    matching_rows = self._find_matching_rows(query, table)

                    results.append(TableSearchResult(
                        file_path=path,
                        file_title=metadata.get('title', path),
                        table_index=idx,
                        headers=table['headers'],
                        rows=table['rows'],
                        matching_rows=matching_rows or table['rows'][:5],  # Top 5 if no specific match
                        score=score,
                        caption=table.get('caption'),
                        row_count=table['row_count'],
                        col_count=table['col_count']
                    ))

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def _score_table(self, query: str, table: Dict[str, Any]) -> float:
        """
        Score table relevance to query.

        Args:
            query: Search query
            table: Table dictionary

        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        query_lower = query.lower()

        # Score caption (30% weight)
        if table.get('caption'):
            caption_score = fuzz.partial_ratio(query_lower, table['caption'].lower()) / 100
            score += caption_score * 0.3

        # Score headers (40% weight)
        headers_text = ' '.join(table['headers']).lower()
        header_score = fuzz.partial_ratio(query_lower, headers_text) / 100
        score += header_score * 0.4

        # Score rows (30% weight)
        rows_text = ' '.join([' '.join(row) for row in table['rows']]).lower()
        rows_score = fuzz.partial_ratio(query_lower, rows_text) / 100
        score += rows_score * 0.3

        return min(score, 1.0)

    def _find_matching_rows(
        self,
        query: str,
        table: Dict[str, Any]
    ) -> Optional[List[List[str]]]:
        """
        Find rows that match query.

        Args:
            query: Search query
            table: Table dictionary

        Returns:
            List of matching rows or None
        """
        query_lower = query.lower()
        matching_rows = []
        threshold = 60  # Minimum fuzzy match score

        for row in table['rows']:
            row_text = ' '.join(row).lower()
            score = fuzz.partial_ratio(query_lower, row_text)

            if score >= threshold:
                matching_rows.append(row)

        return matching_rows if matching_rows else None

    async def get_table(
        self,
        file_path: str,
        table_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get specific table from file.

        Args:
            file_path: Path to file
            table_index: Index of table in file

        Returns:
            Table dictionary or None if not found
        """
        metadata = self.search_engine.file_index.get(file_path)

        if not metadata or not metadata.get('tables'):
            return None

        if table_index >= len(metadata['tables']):
            return None

        return metadata['tables'][table_index]

    async def list_all_tables(self) -> List[Dict[str, Any]]:
        """
        List all tables in the documentation.

        Returns:
            List of table summaries
        """
        all_tables = []

        for path, metadata in self.search_engine.file_index.items():
            if not metadata.get('tables'):
                continue

            for idx, table in enumerate(metadata['tables']):
                all_tables.append({
                    'file_path': path,
                    'file_title': metadata.get('title', path),
                    'table_index': idx,
                    'caption': table.get('caption'),
                    'headers': table['headers'],
                    'row_count': table['row_count'],
                    'col_count': table['col_count']
                })

        return all_tables

    def format_table_for_display(
        self,
        table: Dict[str, Any],
        max_rows: int = 10
    ) -> str:
        """
        Format table as markdown for display.

        Args:
            table: Table dictionary
            max_rows: Maximum rows to display

        Returns:
            Markdown-formatted table
        """
        lines = []

        if table.get('caption'):
            lines.append(f"**{table['caption']}**\n")

        # Header row
        headers = table['headers']
        lines.append(f"| {' | '.join(headers)} |")

        # Separator
        lines.append(f"|{'---|' * len(headers)}")

        # Data rows
        rows = table['rows'][:max_rows]
        for row in rows:
            # Pad row if needed
            padded_row = row + [''] * (len(headers) - len(row))
            lines.append(f"| {' | '.join(padded_row)} |")

        if table['row_count'] > max_rows:
            lines.append(f"\n*... and {table['row_count'] - max_rows} more rows*")

        return '\n'.join(lines)


def format_table_search_results(results: List[TableSearchResult]) -> str:
    """
    Format table search results for display.

    Args:
        results: List of table search results

    Returns:
        Formatted string
    """
    if not results:
        return "*No tables found matching your query.*"

    lines = [f"**Found {len(results)} table(s):**\n"]

    for i, result in enumerate(results, 1):
        lines.append(f"## {i}. {result.file_title}")

        if result.caption:
            lines.append(f"**{result.caption}**")

        lines.append(f"📁 Location: `{result.file_path}`")
        lines.append(f"🎯 Relevance: {result.score:.2f}")
        lines.append(f"📊 Size: {result.row_count} rows × {result.col_count} columns")
        lines.append("")

        # Show headers
        lines.append(f"**Columns:** {', '.join(result.headers)}")
        lines.append("")

        # Show matching rows (or first few rows)
        if result.matching_rows:
            lines.append("**Matching rows:**")
            lines.append(f"| {' | '.join(result.headers)} |")
            lines.append(f"|{'---|' * len(result.headers)}")

            for row in result.matching_rows[:3]:  # Show max 3 matching rows
                padded_row = row + [''] * (len(result.headers) - len(row))
                lines.append(f"| {' | '.join(padded_row)} |")

            if len(result.matching_rows) > 3:
                lines.append(f"\n*... and {len(result.matching_rows) - 3} more matching rows*")

        lines.append("\n")

    return '\n'.join(lines)
