"""Search tools for MCP server."""

import re
import logging
from pathlib import Path
from typing import List, Optional

from mcp.types import TextContent

from search.hybrid_search import HybridSearchEngine, SearchResult

logger = logging.getLogger(__name__)


class SearchTools:
    """Search tool implementations."""

    def __init__(self, search_engine: HybridSearchEngine, docs_folder: Path):
        self.search_engine = search_engine
        self.docs_folder = docs_folder

    async def search_docs(
        self,
        query: str,
        folder: Optional[str] = None,
        limit: int = 5
    ) -> List[TextContent]:
        """Fast keyword search."""
        try:
            results = await self.search_engine.fast_search(
                query=query,
                folder=folder,
                limit=limit
            )

            if not results:
                return [TextContent(
                    type="text",
                    text=f"No documentation found for '{query}'"
                )]

            # Format results
            response_parts = [
                f"Found {len(results)} documentation matches for '{query}':\n"
            ]

            for i, result in enumerate(results, 1):
                response_parts.append(
                    f"\n{i}. **{result.title}**\n"
                    f"   Path: `{result.path}`\n"
                    f"   Team: {result.team} | Category: {result.category}\n"
                    f"   Score: {result.score:.1f} ({result.match_type} match)\n"
                    f"   {result.excerpt}\n"
                )

                if result.has_images:
                    response_parts.append(
                        f"   📷 Contains {result.image_count} image(s)\n"
                    )

            return [TextContent(
                type="text",
                text="".join(response_parts)
            )]

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Search error: {str(e)}"
            )]

    async def semantic_search(
        self,
        query: str,
        limit: int = 5
    ) -> List[TextContent]:
        """Semantic search for conceptual queries."""
        try:
            results = await self.search_engine.semantic_search(
                query=query,
                limit=limit
            )

            if not results:
                # Fallback to fast search
                logger.info("Semantic search returned no results, falling back to keyword search")
                return await self.search_docs(query=query, limit=limit)

            # Format results
            response_parts = [
                f"Found {len(results)} conceptually related docs for '{query}':\n"
            ]

            for i, result in enumerate(results, 1):
                response_parts.append(
                    f"\n{i}. **{result.title}**\n"
                    f"   Path: `{result.path}`\n"
                    f"   Relevance: {result.score:.1f}%\n"
                    f"   {result.excerpt}\n"
                )

            return [TextContent(
                type="text",
                text="".join(response_parts)
            )]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Semantic search error: {str(e)}"
            )]

    async def find_configs(
        self,
        query: str,
        language: Optional[str] = None
    ) -> List[TextContent]:
        """Find configuration files and code blocks."""
        try:
            # Search for docs containing config/code
            results = await self.search_engine.fast_search(
                query=f"{query} config",
                limit=10
            )

            if not results:
                return [TextContent(
                    type="text",
                    text=f"No configuration examples found for '{query}'"
                )]

            # Extract code blocks from results
            config_examples = []

            for result in results:
                # Read file content
                file_path = self.docs_folder / result.path
                if not file_path.exists():
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Extract code blocks
                    code_blocks = self._extract_code_blocks(content, language)

                    if code_blocks:
                        config_examples.append({
                            'file': result.path,
                            'title': result.title,
                            'blocks': code_blocks
                        })

                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")

            if not config_examples:
                return [TextContent(
                    type="text",
                    text=f"Found {len(results)} docs, but no code blocks extracted"
                )]

            # Format response
            response_parts = [
                f"Found {len(config_examples)} files with configuration examples:\n"
            ]

            for example in config_examples:
                response_parts.append(f"\n## {example['title']}\n")
                response_parts.append(f"File: `{example['file']}`\n\n")

                for block in example['blocks'][:3]:  # Max 3 blocks per file
                    lang = block['language'] or 'text'
                    response_parts.append(f"```{lang}\n{block['code']}\n```\n\n")

            return [TextContent(
                type="text",
                text="".join(response_parts)
            )]

        except Exception as e:
            logger.error(f"Config search failed: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Config search error: {str(e)}"
            )]

    def _extract_code_blocks(
        self,
        content: str,
        language: Optional[str] = None
    ) -> List[dict]:
        """Extract code blocks from markdown."""
        # Pattern: ```language\ncode\n```
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)

        blocks = []
        for lang, code in matches:
            # Filter by language if specified
            if language and lang != language:
                continue

            blocks.append({
                'language': lang or None,
                'code': code.strip()
            })

        return blocks
