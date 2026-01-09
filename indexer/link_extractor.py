"""
Link extraction and validation for markdown files.
Tracks internal links, external links, and validates targets.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class Link:
    """Extracted link with metadata."""
    url: str
    text: str
    type: str  # 'internal', 'external', 'anchor'
    target_file: str = None
    is_valid: bool = True
    line_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'url': self.url,
            'text': self.text,
            'type': self.type,
            'target': self.target_file,
            'valid': self.is_valid,
            'line': self.line_number
        }


class LinkExtractor:
    """Extract and validate links from markdown content."""

    # Markdown link patterns
    MARKDOWN_LINK = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
    HTML_LINK = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>', re.IGNORECASE)
    AUTOLINK = re.compile(r'<(https?://[^>]+)>')

    def __init__(self, docs_folder: Path):
        """
        Initialize link extractor.

        Args:
            docs_folder: Root folder of documentation
        """
        self.docs_folder = docs_folder

    def extract_links(
        self,
        content: str,
        file_path: Path
    ) -> List[Link]:
        """
        Extract all links from markdown content.

        Args:
            content: Markdown file content
            file_path: Path to the markdown file

        Returns:
            List of extracted links
        """
        links = []

        # Extract markdown-style links [text](url)
        for match in self.MARKDOWN_LINK.finditer(content):
            text = match.group(1)
            url = match.group(2)
            line_number = content[:match.start()].count('\n') + 1

            link = self._classify_link(url, text, file_path, line_number)
            links.append(link)

        # Extract HTML-style links <a href="url">text</a>
        for match in self.HTML_LINK.finditer(content):
            url = match.group(1)
            text = match.group(2) or url
            line_number = content[:match.start()].count('\n') + 1

            link = self._classify_link(url, text, file_path, line_number)
            links.append(link)

        # Extract autolinks <https://example.com>
        for match in self.AUTOLINK.finditer(content):
            url = match.group(1)
            text = url
            line_number = content[:match.start()].count('\n') + 1

            link = self._classify_link(url, text, file_path, line_number)
            links.append(link)

        logger.debug(f"Extracted {len(links)} links from {file_path}")
        return links

    def _classify_link(
        self,
        url: str,
        text: str,
        file_path: Path,
        line_number: int
    ) -> Link:
        """
        Classify and validate a link.

        Args:
            url: Link URL
            text: Link text
            file_path: Current markdown file path
            line_number: Line number where link appears

        Returns:
            Classified link
        """
        # Strip whitespace
        url = url.strip()
        text = text.strip()

        # External link (http/https)
        if url.startswith('http://') or url.startswith('https://'):
            return Link(
                url=url,
                text=text,
                type='external',
                line_number=line_number
            )

        # Anchor link (#section)
        if url.startswith('#'):
            return Link(
                url=url,
                text=text,
                type='anchor',
                line_number=line_number
            )

        # Internal link to file
        return self._validate_internal_link(url, text, file_path, line_number)

    def _validate_internal_link(
        self,
        url: str,
        text: str,
        file_path: Path,
        line_number: int
    ) -> Link:
        """
        Validate an internal file link.

        Args:
            url: Relative path to target file
            text: Link text
            file_path: Current file path
            line_number: Line number

        Returns:
            Validated link
        """
        try:
            # Remove anchor if present
            clean_url = url.split('#')[0] if '#' in url else url

            if not clean_url:
                # Pure anchor link (already handled, but just in case)
                return Link(
                    url=url,
                    text=text,
                    type='anchor',
                    line_number=line_number
                )

            # Resolve relative to current file
            target_path = (file_path.parent / clean_url).resolve()

            # Check if target exists
            is_valid = target_path.exists()

            # Get relative path from docs root
            try:
                rel_target = str(target_path.relative_to(self.docs_folder))
            except ValueError:
                # Target is outside docs folder
                rel_target = str(target_path)
                is_valid = False

            return Link(
                url=url,
                text=text,
                type='internal',
                target_file=rel_target if is_valid else None,
                is_valid=is_valid,
                line_number=line_number
            )

        except Exception as e:
            logger.debug(f"Failed to validate link '{url}' in {file_path}: {e}")
            return Link(
                url=url,
                text=text,
                type='internal',
                is_valid=False,
                line_number=line_number
            )

    def get_related_files(self, links: List[Link]) -> List[str]:
        """
        Get list of related files from valid internal links.

        Args:
            links: List of extracted links

        Returns:
            List of unique related file paths
        """
        related = []
        for link in links:
            if link.type == 'internal' and link.is_valid and link.target_file:
                # Only add markdown files
                if link.target_file.endswith('.md'):
                    related.append(link.target_file)

        return list(set(related))  # Unique

    def get_link_stats(self, links: List[Link]) -> Dict[str, Any]:
        """
        Get statistics about links.

        Args:
            links: List of extracted links

        Returns:
            Dictionary with link statistics
        """
        stats = {
            'total': len(links),
            'internal': 0,
            'external': 0,
            'anchor': 0,
            'valid': 0,
            'broken': 0
        }

        for link in links:
            stats[link.type] += 1
            if link.is_valid:
                stats['valid'] += 1
            else:
                stats['broken'] += 1

        return stats


if __name__ == "__main__":
    # Test link extraction
    import sys

    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        if test_file.exists():
            docs_folder = test_file.parent
            extractor = LinkExtractor(docs_folder)

            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            links = extractor.extract_links(content, test_file)
            stats = extractor.get_link_stats(links)

            print(f"Link extraction results for {test_file.name}:\n")
            print(f"Total links: {stats['total']}")
            print(f"  Internal: {stats['internal']} ({stats['valid'] - stats['anchor']} valid)")
            print(f"  External: {stats['external']}")
            print(f"  Anchors: {stats['anchor']}")
            print(f"  Broken: {stats['broken']}\n")

            if stats['broken'] > 0:
                print("Broken links:")
                for link in links:
                    if not link.is_valid:
                        print(f"  Line {link.line_number}: {link.url} -> {link.text}")
        else:
            print(f"File not found: {test_file}")
    else:
        print("Usage: python link_extractor.py <markdown_file>")
