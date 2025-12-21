"""File operations tools for MCP server."""

import os
import logging
from pathlib import Path
from typing import List, Optional

from mcp.types import TextContent

logger = logging.getLogger(__name__)


class FileTools:
    """File operation tool implementations."""

    def __init__(self, docs_folder: Path):
        self.docs_folder = docs_folder

    async def get_file(self, path: str) -> List[TextContent]:
        """Get full content of a documentation file."""
        try:
            file_path = self.docs_folder / path

            if not file_path.exists():
                return [TextContent(
                    type="text",
                    text=f"File not found: {path}"
                )]

            if not file_path.is_file():
                return [TextContent(
                    type="text",
                    text=f"Path is not a file: {path}"
                )]

            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Get file stats
            stats = file_path.stat()
            size_kb = stats.st_size / 1024

            # Format response
            response = (
                f"# {file_path.name}\n\n"
                f"**Path:** `{path}`  \n"
                f"**Size:** {size_kb:.1f} KB  \n"
                f"**Team:** {self._extract_team(path)}  \n\n"
                f"---\n\n"
                f"{content}"
            )

            return [TextContent(
                type="text",
                text=response
            )]

        except Exception as e:
            logger.error(f"Get file failed: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error reading file: {str(e)}"
            )]

    async def list_structure(
        self,
        path: str = "",
        depth: int = 3
    ) -> List[TextContent]:
        """List folder structure."""
        try:
            start_path = self.docs_folder / path if path else self.docs_folder

            if not start_path.exists():
                return [TextContent(
                    type="text",
                    text=f"Path not found: {path}"
                )]

            # Build tree
            tree_lines = [f"Documentation Structure: {start_path.name}\n"]
            tree_lines.extend(self._build_tree(start_path, depth=depth))

            return [TextContent(
                type="text",
                text="".join(tree_lines)
            )]

        except Exception as e:
            logger.error(f"List structure failed: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error listing structure: {str(e)}"
            )]

    def _build_tree(
        self,
        path: Path,
        prefix: str = "",
        depth: int = 3,
        current_depth: int = 0
    ) -> List[str]:
        """Build tree structure recursively."""
        if current_depth >= depth:
            return []

        lines = []

        try:
            # Get all items, sorted (folders first, then files)
            items = sorted(
                path.iterdir(),
                key=lambda x: (not x.is_dir(), x.name.lower())
            )

            # Filter out hidden files and common excludes
            items = [
                item for item in items
                if not item.name.startswith('.')
                and item.name not in ['node_modules', '__pycache__', '.git']
            ]

            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = prefix + ("    " if is_last else "│   ")

                # Format item
                if item.is_dir():
                    # Count markdown files in folder
                    md_count = len(list(item.rglob("*.md")))
                    lines.append(f"{prefix}{current_prefix}📁 {item.name}/ ({md_count} docs)\n")

                    # Recurse into folder
                    lines.extend(self._build_tree(
                        item,
                        prefix=next_prefix,
                        depth=depth,
                        current_depth=current_depth + 1
                    ))
                else:
                    # File
                    if item.suffix == '.md':
                        # Get file size
                        size_kb = item.stat().st_size / 1024
                        lines.append(
                            f"{prefix}{current_prefix}📄 {item.name} ({size_kb:.1f} KB)\n"
                        )

        except PermissionError:
            lines.append(f"{prefix}[Permission Denied]\n")

        return lines

    def _extract_team(self, path: str) -> str:
        """Extract team name from path."""
        parts = Path(path).parts
        return parts[0] if len(parts) > 0 else "unknown"
