"""
Rich context builder for detailed LLM responses.
Combines search results with tables, images, and related documents.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build comprehensive context for LLM responses."""

    def __init__(self, search_engine, docs_folder: Path):
        """
        Initialize context builder.

        Args:
            search_engine: HybridSearchEngine instance
            docs_folder: Root documentation folder
        """
        self.search_engine = search_engine
        self.docs_folder = docs_folder

    async def build_rich_context(
        self,
        query: str,
        search_results: List[Any],
        include_tables: bool = True,
        include_images: bool = True,
        include_links: bool = True,
        include_related: bool = True,
        max_context_length: int = 8000
    ) -> str:
        """
        Build comprehensive context from search results.

        Args:
            query: User's search query
            search_results: List of search results
            include_tables: Include table data
            include_images: Include image information
            include_links: Include related links
            include_related: Include related documents
            max_context_length: Maximum context length in characters

        Returns:
            Markdown-formatted rich context
        """
        context_parts = []
        current_length = 0

        # Add query header
        header = f"# Search Results for: {query}\n\n"
        context_parts.append(header)
        current_length += len(header)

        # Add summary
        summary = self._build_summary(search_results)
        context_parts.append(summary)
        current_length += len(summary)

        # Process each search result
        for i, result in enumerate(search_results, 1):
            if current_length >= max_context_length:
                context_parts.append("\n\n*[Additional results truncated...]*")
                break

            # Get full metadata
            metadata = self.search_engine.file_index.get(result.path)
            if not metadata:
                continue

            # Build result section
            section = self._build_result_section(
                result_num=i,
                result=result,
                metadata=metadata,
                include_tables=include_tables,
                include_images=include_images,
                include_links=include_links,
                include_related=include_related
            )

            # Check if section fits
            if current_length + len(section) > max_context_length:
                # Add truncated version
                remaining = max_context_length - current_length
                if remaining > 200:  # Only add if meaningful space left
                    truncated = section[:remaining] + "\n\n*[Content truncated...]*"
                    context_parts.append(truncated)
                break

            context_parts.append(section)
            current_length += len(section)

        return '\n\n'.join(context_parts)

    def _build_summary(self, results: List[Any]) -> str:
        """Build summary section."""
        if not results:
            return "*No results found.*\n"

        summary_parts = [
            f"**Found {len(results)} relevant document(s)**\n"
        ]

        # Count files with special content
        with_tables = sum(1 for r in results if self._has_tables(r.path))
        with_images = sum(1 for r in results if self._has_images(r.path))

        if with_tables > 0:
            summary_parts.append(f"- {with_tables} with tables")
        if with_images > 0:
            summary_parts.append(f"- {with_images} with images")

        return '\n'.join(summary_parts) + '\n'

    def _build_result_section(
        self,
        result_num: int,
        result: Any,
        metadata: Dict[str, Any],
        include_tables: bool,
        include_images: bool,
        include_links: bool,
        include_related: bool
    ) -> str:
        """Build detailed section for one search result."""
        parts = []

        # Header
        parts.append(f"## Result {result_num}: {result.title}")
        parts.append(f"**📁 Location:** `{result.path}`")
        parts.append(f"**🏷️ Category:** {result.team}/{result.category}")
        parts.append(f"**🎯 Match:** {result.match_type} (score: {result.score:.2f})")
        parts.append("")

        # Main content excerpt
        parts.append("### 📄 Content")
        parts.append(result.excerpt)
        parts.append("")

        # Document structure (headings)
        if metadata.get('headings'):
            parts.append("### 📋 Document Structure")
            headings = metadata['headings'][:8]  # First 8 headings
            for heading in headings:
                level = len(heading) - len(heading.lstrip('#'))
                indent = "  " * (level - 1) if level > 1 else ""
                parts.append(f"{indent}- {heading.lstrip('#').strip()}")
            if len(metadata['headings']) > 8:
                parts.append(f"  *(... and {len(metadata['headings']) - 8} more sections)*")
            parts.append("")

        # Tables
        if include_tables and metadata.get('tables'):
            parts.append("### 📊 Tables in this Document")
            for idx, table in enumerate(metadata['tables']):
                parts.append(f"\n**Table {idx + 1}**")
                if table.get('caption'):
                    parts.append(f"*{table['caption']}*")

                # Show headers
                parts.append(f"\n**Columns:** {', '.join(table['headers'])}")
                parts.append(f"**Rows:** {table['row_count']}")

                # Include sample data for small tables
                if table['row_count'] <= 5:
                    parts.append("\n```")
                    parts.append(table['markdown'])
                    parts.append("```")
                else:
                    # Show first 3 rows for large tables
                    parts.append("\n*Sample (first 3 rows):*")
                    parts.append(f"| {' | '.join(table['headers'])} |")
                    parts.append(f"|{'---|' * len(table['headers'])}")
                    for row in table['rows'][:3]:
                        parts.append(f"| {' | '.join(row)} |")
                    parts.append(f"*... and {table['row_count'] - 3} more rows*")

            parts.append("")

        # Images
        if include_images and metadata.get('images'):
            parts.append("### 🖼️ Images in this Document")
            for img in metadata['images']:
                parts.append(f"\n**{img.get('alt', 'Untitled Image')}**")
                parts.append(f"- Path: `{img['path']}`")

                if img.get('type') == 'local':
                    parts.append(f"- Size: {img.get('width')}x{img.get('height')} ({img.get('format')})")

                if img.get('ocr_text'):
                    ocr_preview = img['ocr_text'][:200]
                    if len(img['ocr_text']) > 200:
                        ocr_preview += "..."
                    parts.append(f"- Extracted text: *{ocr_preview}*")

            parts.append("")

        # Related documents via links
        if include_related and metadata.get('related_files'):
            parts.append("### 🔗 Related Documents")
            related_files = metadata['related_files'][:5]  # Top 5
            for related_path in related_files:
                # Try to get title
                related_meta = self.search_engine.file_index.get(related_path)
                if related_meta:
                    title = related_meta.get('title', related_path)
                    parts.append(f"- [{title}]({related_path})")
                else:
                    parts.append(f"- {related_path}")

            if len(metadata['related_files']) > 5:
                parts.append(f"  *(... and {len(metadata['related_files']) - 5} more)*")

            parts.append("")

        # Link stats
        if include_links and metadata.get('link_stats'):
            stats = metadata['link_stats']
            if stats['total'] > 0:
                parts.append(f"**Links:** {stats['internal']} internal, {stats['external']} external")
                if stats['broken'] > 0:
                    parts.append(f"⚠️ *{stats['broken']} broken link(s)*")
                parts.append("")

        return '\n'.join(parts)

    def _has_tables(self, path: str) -> bool:
        """Check if document has tables."""
        metadata = self.search_engine.file_index.get(path)
        return bool(metadata and metadata.get('tables'))

    def _has_images(self, path: str) -> bool:
        """Check if document has images."""
        metadata = self.search_engine.file_index.get(path)
        return bool(metadata and metadata.get('images'))


def format_table_for_llm(table: Dict[str, Any]) -> str:
    """
    Format a table in a way that's easy for LLMs to understand.

    Args:
        table: Table dictionary

    Returns:
        Formatted table string
    """
    parts = []

    if table.get('caption'):
        parts.append(f"Table: {table['caption']}\n")

    # Add markdown table
    parts.append(table['markdown'])

    # Add text summary
    parts.append(f"\n(Table has {table['row_count']} rows and {table['col_count']} columns)")

    return '\n'.join(parts)


def format_image_for_llm(image: Dict[str, Any]) -> str:
    """
    Format image information for LLMs.

    Args:
        image: Image dictionary

    Returns:
        Formatted image description
    """
    parts = []

    parts.append(f"Image: {image.get('alt', 'Untitled')}")

    if image.get('ocr_text'):
        parts.append(f"Text content: {image['ocr_text']}")

    if image.get('type') == 'local':
        parts.append(f"({image.get('width')}x{image.get('height')} {image.get('format')})")

    return '\n'.join(parts)
