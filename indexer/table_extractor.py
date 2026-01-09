"""
Markdown table extraction and parsing.
Extracts tables from markdown files for structured search.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MarkdownTable:
    """Extracted markdown table with metadata."""
    headers: List[str]
    rows: List[List[str]]
    raw_markdown: str
    line_start: int
    line_end: int
    caption: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'headers': self.headers,
            'rows': self.rows,
            'markdown': self.raw_markdown,
            'line_range': [self.line_start, self.line_end],
            'caption': self.caption,
            'row_count': len(self.rows),
            'col_count': len(self.headers)
        }

    def to_text(self) -> str:
        """Convert table to searchable plain text."""
        text_parts = []

        if self.caption:
            text_parts.append(f"Table: {self.caption}")

        # Add headers
        text_parts.append(" | ".join(self.headers))

        # Add all rows
        for row in self.rows:
            text_parts.append(" | ".join(row))

        return "\n".join(text_parts)


class MarkdownTableExtractor:
    """Extract and parse markdown tables from content."""

    # Markdown table patterns
    TABLE_ROW_PATTERN = re.compile(r'^\s*\|(.+)\|\s*$')
    SEPARATOR_PATTERN = re.compile(r'^\s*\|[\s\-:|]+\|\s*$')

    def extract_tables(
        self,
        content: str,
        file_path: str
    ) -> List[MarkdownTable]:
        """
        Extract all markdown tables from content.

        Args:
            content: Markdown file content
            file_path: Path to file (for logging)

        Returns:
            List of extracted tables
        """
        lines = content.split('\n')
        tables = []

        i = 0
        while i < len(lines):
            # Look for potential table start (header row)
            if self.TABLE_ROW_PATTERN.match(lines[i]):
                # Check if next line is separator
                if i + 1 < len(lines) and self.SEPARATOR_PATTERN.match(lines[i + 1]):
                    table = self._parse_table(lines, i, file_path)
                    if table:
                        tables.append(table)
                        i = table.line_end + 1
                        continue
            i += 1

        logger.debug(f"Extracted {len(tables)} tables from {file_path}")
        return tables

    def _parse_table(
        self,
        lines: List[str],
        start_idx: int,
        file_path: str
    ) -> Optional[MarkdownTable]:
        """
        Parse a single markdown table starting at given index.

        Args:
            lines: All lines in the file
            start_idx: Index of header row
            file_path: File path for error reporting

        Returns:
            Parsed table or None if parsing fails
        """
        try:
            # Parse header row
            header_line = lines[start_idx]
            headers = self._parse_row(header_line)

            if not headers:
                return None

            # Skip separator line (already validated)
            current_idx = start_idx + 2

            # Parse data rows
            rows = []
            raw_lines = [lines[start_idx], lines[start_idx + 1]]

            while current_idx < len(lines):
                line = lines[current_idx]

                # Check if still a table row
                if not self.TABLE_ROW_PATTERN.match(line):
                    break

                # Skip additional separator rows (for multi-header tables)
                if self.SEPARATOR_PATTERN.match(line):
                    raw_lines.append(line)
                    current_idx += 1
                    continue

                # Parse data row
                row_data = self._parse_row(line)
                if row_data:
                    rows.append(row_data)
                    raw_lines.append(line)

                current_idx += 1

            # Look for caption (line before table or after)
            caption = None
            if start_idx > 0:
                prev_line = lines[start_idx - 1].strip()
                # Caption is non-empty line that's not a heading
                if prev_line and not prev_line.startswith('#') and not prev_line.startswith('|'):
                    caption = prev_line

            return MarkdownTable(
                headers=headers,
                rows=rows,
                raw_markdown='\n'.join(raw_lines),
                line_start=start_idx,
                line_end=current_idx - 1,
                caption=caption
            )

        except Exception as e:
            logger.error(f"Failed to parse table in {file_path} at line {start_idx}: {e}")
            return None

    def _parse_row(self, line: str) -> List[str]:
        """
        Parse a single table row into cells.

        Args:
            line: Table row line

        Returns:
            List of cell contents
        """
        # Remove leading/trailing whitespace
        line = line.strip()

        # Remove leading/trailing pipes
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]

        # Split by pipe and clean cells
        cells = []
        for cell in line.split('|'):
            # Strip whitespace and markdown formatting
            cell_text = cell.strip()
            # Remove common markdown (bold, italic, code)
            cell_text = re.sub(r'\*\*(.*?)\*\*', r'\1', cell_text)  # Bold
            cell_text = re.sub(r'\*(.*?)\*', r'\1', cell_text)      # Italic
            cell_text = re.sub(r'`(.*?)`', r'\1', cell_text)        # Inline code
            cells.append(cell_text)

        return cells


def extract_tables_from_file(file_path: Path) -> List[MarkdownTable]:
    """
    Convenience function to extract tables from a markdown file.

    Args:
        file_path: Path to markdown file

    Returns:
        List of extracted tables
    """
    extractor = MarkdownTableExtractor()

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return extractor.extract_tables(content, str(file_path))


if __name__ == "__main__":
    # Test extraction
    import sys

    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        if test_file.exists():
            tables = extract_tables_from_file(test_file)
            print(f"Found {len(tables)} tables:\n")

            for i, table in enumerate(tables, 1):
                print(f"Table {i}:")
                if table.caption:
                    print(f"  Caption: {table.caption}")
                print(f"  Headers: {table.headers}")
                print(f"  Rows: {len(table.rows)}")
                print(f"  Lines: {table.line_start}-{table.line_end}\n")
        else:
            print(f"File not found: {test_file}")
    else:
        print("Usage: python table_extractor.py <markdown_file>")
