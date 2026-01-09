"""
Simplified Document Indexer for Markdown-only RAG.

Optimized for:
- 400 fixed markdown files
- Tables extraction
- JSON code blocks extraction
- No images/OCR
- Fast one-time indexing
"""

import os
import re
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MarkdownDocument:
    """A processed markdown document."""
    path: str
    title: str
    content: str
    headings: List[str]
    tables: List[Dict[str, Any]]
    json_blocks: List[Dict[str, Any]]
    links: List[str]
    metadata: Dict[str, Any]


class SimpleDocIndexer:
    """
    Simplified indexer for markdown-only documents.

    Features:
    - Fast parallel processing
    - Table extraction
    - JSON code block extraction
    - Link extraction
    - No image processing
    """

    def __init__(
        self,
        docs_folder: Path,
        index_path: Path,
        max_workers: Optional[int] = None
    ):
        self.docs_folder = Path(docs_folder)
        self.index_path = Path(index_path)
        self.max_workers = max_workers or os.cpu_count() - 1 or 1

        # Compiled patterns for speed
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.table_pattern = re.compile(r'^\|.+\|$', re.MULTILINE)
        self.json_block_pattern = re.compile(r'```json\s*([\s\S]*?)```', re.MULTILINE)
        self.link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')

    async def build_index(self):
        """Build index for all markdown files."""
        logger.info(f"🚀 Starting indexing of {self.docs_folder}")

        # Find all markdown files
        md_files = list(self.docs_folder.rglob("*.md"))
        total_files = len(md_files)

        if total_files == 0:
            logger.error("No markdown files found!")
            return

        logger.info(f"📄 Found {total_files} markdown files")

        # Process files in parallel batches
        batch_size = 20
        batches = [md_files[i:i + batch_size] for i in range(0, total_files, batch_size)]

        all_docs = []
        tables_count = 0
        json_blocks_count = 0

        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} files)")

            tasks = [self._process_file(file_path) for file_path in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, MarkdownDocument):
                    all_docs.append(result)
                    tables_count += len(result.tables)
                    json_blocks_count += len(result.json_blocks)

        logger.info(f"✅ Processed {len(all_docs)} documents")
        logger.info(f"📊 Extracted {tables_count} tables")
        logger.info(f"📝 Extracted {json_blocks_count} JSON blocks")

        # Save index
        await self._save_index(all_docs)

        logger.info(f"🎉 Indexing complete!")

    async def _process_file(self, file_path: Path) -> Optional[MarkdownDocument]:
        """Process a single markdown file."""
        try:
            rel_path = str(file_path.relative_to(self.docs_folder))

            # Read content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract title (first heading or filename)
            title = self._extract_title(content, file_path)

            # Extract headings
            headings = self._extract_headings(content)

            # Extract tables
            tables = self._extract_tables(content)

            # Extract JSON blocks
            json_blocks = self._extract_json_blocks(content)

            # Extract links
            links = self._extract_links(content)

            # Create document
            doc = MarkdownDocument(
                path=rel_path,
                title=title,
                content=content,
                headings=headings,
                tables=tables,
                json_blocks=json_blocks,
                links=links,
                metadata={
                    'size': len(content),
                    'has_tables': len(tables) > 0,
                    'has_json': len(json_blocks) > 0,
                    'num_headings': len(headings)
                }
            )

            return doc

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return None

    def _extract_title(self, content: str, file_path: Path) -> str:
        """Extract title from first heading or use filename."""
        # Look for first # heading
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Fallback to filename
        return file_path.stem.replace('-', ' ').replace('_', ' ').title()

    def _extract_headings(self, content: str) -> List[str]:
        """Extract all headings."""
        headings = []
        for match in self.heading_pattern.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append(f"{'  ' * (level-1)}{text}")
        return headings

    def _extract_tables(self, content: str) -> List[Dict[str, Any]]:
        """Extract markdown tables."""
        tables = []
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check if line is a table row
            if line.strip().startswith('|') and '|' in line:
                table_lines = [line]

                # Collect all consecutive table lines
                j = i + 1
                while j < len(lines) and lines[j].strip().startswith('|'):
                    table_lines.append(lines[j])
                    j += 1

                # Parse table
                if len(table_lines) >= 2:  # At least header + separator
                    table = self._parse_table(table_lines, i)
                    if table:
                        tables.append(table)

                i = j
            else:
                i += 1

        return tables

    def _parse_table(self, lines: List[str], start_line: int) -> Optional[Dict[str, Any]]:
        """Parse markdown table into structured format."""
        if len(lines) < 2:
            return None

        # Parse header
        header_cells = [cell.strip() for cell in lines[0].split('|') if cell.strip()]

        # Skip separator line (usually line 1)
        # Parse data rows
        rows = []
        for line in lines[2:]:  # Skip header and separator
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if len(cells) == len(header_cells):
                    row = dict(zip(header_cells, cells))
                    rows.append(row)

        if not rows:
            return None

        return {
            'headers': header_cells,
            'rows': rows,
            'raw_markdown': '\n'.join(lines),
            'start_line': start_line,
            'num_rows': len(rows)
        }

    def _extract_json_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract JSON code blocks."""
        json_blocks = []

        for i, match in enumerate(self.json_block_pattern.finditer(content)):
            json_text = match.group(1).strip()

            try:
                # Try to parse JSON
                parsed = json.loads(json_text)
                json_blocks.append({
                    'index': i,
                    'raw': json_text,
                    'parsed': parsed,
                    'type': self._detect_json_type(parsed)
                })
            except json.JSONDecodeError:
                # Invalid JSON, skip
                logger.debug(f"Invalid JSON block at index {i}")
                continue

        return json_blocks

    def _detect_json_type(self, parsed: Any) -> str:
        """Detect type of JSON content."""
        if isinstance(parsed, dict):
            # Check for common patterns
            if 'azure' in parsed or 'resourceType' in parsed:
                return 'config'
            elif 'parameters' in parsed or 'workflow' in parsed:
                return 'workflow'
            else:
                return 'object'
        elif isinstance(parsed, list):
            return 'array'
        else:
            return 'value'

    def _extract_links(self, content: str) -> List[str]:
        """Extract all markdown links."""
        links = []
        for match in self.link_pattern.finditer(content):
            url = match.group(2)
            links.append(url)
        return links

    async def _save_index(self, docs: List[MarkdownDocument]):
        """Save processed documents to index."""
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Create index structure
        index = {
            'documents': {doc.path: asdict(doc) for doc in docs},
            'stats': {
                'total_docs': len(docs),
                'total_tables': sum(len(doc.tables) for doc in docs),
                'total_json_blocks': sum(len(doc.json_blocks) for doc in docs),
                'docs_with_tables': sum(1 for doc in docs if doc.tables),
                'docs_with_json': sum(1 for doc in docs if doc.json_blocks)
            }
        }

        # Save to file
        index_file = self.index_path / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)

        logger.info(f"💾 Saved index to {index_file}")

        # Generate embeddings if enabled
        if os.getenv('ENABLE_EMBEDDINGS', 'true').lower() == 'true':
            await self._generate_embeddings(docs)

    async def _generate_embeddings(self, docs: List[MarkdownDocument]):
        """Generate embeddings for documents."""
        from search.embedders import create_embedder

        logger.info("🧮 Generating embeddings...")

        embedder = create_embedder(provider=os.getenv('EMBEDDING_PROVIDER', 'local'))

        # Prepare texts for embedding
        texts = [doc.content for doc in docs]

        # Generate in batches
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await embedder.embed_batch(batch)
            all_embeddings.extend(embeddings)

        # Save embeddings
        import numpy as np
        embeddings_array = np.array(all_embeddings)
        embeddings_file = self.index_path / "embeddings.npy"
        np.save(str(embeddings_file), embeddings_array)

        logger.info(f"✅ Generated {len(all_embeddings)} embeddings")


async def main():
    """CLI entry point."""
    import sys

    docs_folder = os.getenv('DOCS_FOLDER', './docs')
    index_path = Path('.index')

    indexer = SimpleDocIndexer(docs_folder, index_path)
    await indexer.build_index()


if __name__ == "__main__":
    asyncio.run(main())
