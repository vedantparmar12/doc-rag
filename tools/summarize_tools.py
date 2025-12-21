"""Summarization tools for MCP server."""

import os
import logging
from pathlib import Path
from typing import List, Optional

from mcp.types import TextContent
import openai

from search.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)


class SummarizeTools:
    """Summarization tool implementations."""

    def __init__(self, search_engine: HybridSearchEngine, docs_folder: Path):
        self.search_engine = search_engine
        self.docs_folder = docs_folder
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def summarize_topic(
        self,
        topic: str,
        folder: Optional[str] = None
    ) -> List[TextContent]:
        """Summarize all documentation about a topic."""
        try:
            # Search for related docs
            results = await self.search_engine.fast_search(
                query=topic,
                folder=folder,
                limit=10
            )

            if not results:
                return [TextContent(
                    type="text",
                    text=f"No documentation found for topic: {topic}"
                )]

            # Read content from top results
            doc_contents = []
            for result in results[:5]:  # Top 5 docs
                file_path = self.docs_folder / result.path
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Limit content to avoid token limits
                            if len(content) > 4000:
                                content = content[:4000] + "\n...(truncated)"
                            doc_contents.append({
                                'title': result.title,
                                'path': result.path,
                                'content': content
                            })
                    except Exception as e:
                        logger.error(f"Failed to read {file_path}: {e}")

            if not doc_contents:
                return [TextContent(
                    type="text",
                    text=f"Found {len(results)} docs but couldn't read content"
                )]

            # Generate summary using LLM
            summary = await self._generate_summary(topic, doc_contents)

            # Format response
            response_parts = [
                f"# Summary: {topic}\n\n",
                f"Analyzed {len(doc_contents)} documents:\n"
            ]

            for doc in doc_contents:
                response_parts.append(f"- {doc['title']} (`{doc['path']}`)\n")

            response_parts.append(f"\n## Overview\n\n{summary}\n")

            return [TextContent(
                type="text",
                text="".join(response_parts)
            )]

        except Exception as e:
            logger.error(f"Summarize topic failed: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Summarization error: {str(e)}"
            )]

    async def _generate_summary(
        self,
        topic: str,
        doc_contents: List[dict]
    ) -> str:
        """Generate summary using OpenAI."""
        # Build prompt
        docs_text = "\n\n---\n\n".join([
            f"Document: {doc['title']}\n{doc['content']}"
            for doc in doc_contents
        ])

        prompt = f"""You are analyzing documentation about "{topic}".

Please provide a comprehensive summary that includes:
1. Overview of what {topic} is
2. Key features and capabilities
3. How to get started
4. Common use cases
5. Important notes or gotchas

Documentation:
{docs_text}

Provide a clear, concise summary suitable for someone new to this topic."""

        try:
            response = await self.client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a helpful documentation assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM summary failed: {e}")
            return f"[Summary generation failed: {str(e)}]"
