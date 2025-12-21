#!/usr/bin/env python3
"""
MCP Documentation RAG Server
Smart documentation search with hybrid approach
"""

import asyncio
import os
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from search.hybrid_search import HybridSearchEngine
from tools.search_tools import SearchTools
from tools.file_tools import FileTools
from tools.summarize_tools import SummarizeTools
from indexer.doc_indexer import DocIndexer

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
app = Server("docs-rag")

# Global state
search_engine: HybridSearchEngine = None
search_tools: SearchTools = None
file_tools: FileTools = None
summarize_tools: SummarizeTools = None
docs_folder: Path = None


async def initialize_server():
    """Initialize the search engine and tools."""
    global search_engine, search_tools, file_tools, summarize_tools, docs_folder

    # Get docs folder from environment
    docs_folder_str = os.getenv("DOCS_FOLDER")
    if not docs_folder_str:
        raise ValueError("DOCS_FOLDER environment variable is required")

    docs_folder = Path(docs_folder_str)
    if not docs_folder.exists():
        raise ValueError(f"Docs folder not found: {docs_folder}")

    logger.info(f"Initializing MCP server for docs folder: {docs_folder}")

    # Initialize search engine
    index_path = Path(os.getenv("INDEX_PATH", ".index"))
    search_engine = HybridSearchEngine(
        docs_folder=docs_folder,
        index_path=index_path,
        enable_semantic=os.getenv("ENABLE_SEMANTIC_SEARCH", "true").lower() == "true",
        enable_vlm=os.getenv("ENABLE_IMAGE_UNDERSTANDING", "false").lower() == "true",
        enable_reranking=os.getenv("ENABLE_RERANKING", "false").lower() == "true"
    )

    # Load or build index
    await search_engine.initialize()

    # Initialize tool handlers
    search_tools = SearchTools(search_engine, docs_folder)
    file_tools = FileTools(docs_folder)
    summarize_tools = SummarizeTools(search_engine, docs_folder)

    logger.info("MCP server initialized successfully")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="search_docs",
            description=(
                "Fast keyword search across documentation. "
                "Searches file names, headings, and content. "
                "Supports folder filtering (e.g., team='platform-team'). "
                "Use for: 'find windows vm docs', 'search pipeline configs'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords or phrase)"
                    },
                    "folder": {
                        "type": "string",
                        "description": "Optional folder filter (e.g., 'platform-team/ci-cd')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="semantic_search",
            description=(
                "Semantic search for conceptual queries. "
                "Understands intent and meaning, not just keywords. "
                "Use for: 'how to deploy', 'best practices', 'what is the process for'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question or description"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_file",
            description=(
                "Get full content of a documentation file. "
                "Returns markdown with image descriptions if available. "
                "Use for: 'show me the windows VM doc', 'get pipeline config file'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to file (e.g., 'platform-team/iaas/windows-vm.md')"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="list_structure",
            description=(
                "List folder structure and available docs. "
                "Shows teams, categories, and files. "
                "Use for: 'what teams exist', 'show me the docs structure'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional folder path to list (default: root)",
                        "default": ""
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Maximum depth to show (default: 3)",
                        "default": 3
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="summarize_topic",
            description=(
                "Summarize all documentation about a specific topic. "
                "Finds related docs and creates comprehensive summary. "
                "Use for: 'summarize kubernetes docs', 'overview of CI/CD pipelines'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to summarize"
                    },
                    "folder": {
                        "type": "string",
                        "description": "Optional folder filter"
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="find_configs",
            description=(
                "Extract configuration examples and code blocks. "
                "Finds YAML, JSON, code snippets related to query. "
                "Use for: 'show pipeline YAML', 'find terraform configs'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What type of config to find"
                    },
                    "language": {
                        "type": "string",
                        "description": "Code language filter (yaml, json, bash, etc.)"
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""
    try:
        if name == "search_docs":
            return await search_tools.search_docs(**arguments)

        elif name == "semantic_search":
            return await search_tools.semantic_search(**arguments)

        elif name == "get_file":
            return await file_tools.get_file(**arguments)

        elif name == "list_structure":
            return await file_tools.list_structure(**arguments)

        elif name == "summarize_topic":
            return await summarize_tools.summarize_topic(**arguments)

        elif name == "find_configs":
            return await search_tools.find_configs(**arguments)

        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]


async def main():
    """Run the MCP server."""
    # Initialize server state
    await initialize_server()

    # Run stdio server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server running on stdio")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise
