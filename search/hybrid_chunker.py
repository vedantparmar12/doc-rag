"""
HybridChunker implementation based on Docling 2.x.

Features:
- Hierarchical document chunking
- Preserves document structure (headings, lists, tables)
- Context-aware splitting
- Smart token counting
- Maintains semantic coherence
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of document content with metadata."""
    content: str
    metadata: Dict[str, Any]
    tokens: int
    chunk_id: str
    start_line: int
    end_line: int
    doc_path: str
    heading_hierarchy: List[str]  # Breadcrumb of headings
    chunk_type: str  # 'text', 'table', 'list', 'code'


class HybridChunker:
    """
    Advanced document chunker inspired by Docling's HybridChunker.

    Strategy:
    1. Parse document structure (headings, sections, tables)
    2. Split at natural boundaries (sections, paragraphs)
    3. Respect token limits while preserving context
    4. Maintain heading hierarchy for context
    5. Special handling for tables, lists, and code blocks
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        respect_structure: bool = True,
        merge_short_chunks: bool = True,
        min_chunk_tokens: int = 50
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.respect_structure = respect_structure
        self.merge_short_chunks = merge_short_chunks
        self.min_chunk_tokens = min_chunk_tokens

        # Regex patterns
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.table_pattern = re.compile(r'^\|.+\|$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        self.list_item_pattern = re.compile(r'^(\s*[-*+]|\s*\d+\.)\s+', re.MULTILINE)

    async def chunk_document(
        self,
        content: str,
        doc_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk a document into semantically coherent pieces.

        Args:
            content: Document content (markdown)
            doc_path: Path to document
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunks
        """
        if not content.strip():
            return []

        metadata = metadata or {}

        # Parse document structure
        structure = self._parse_structure(content)

        # Create initial chunks based on structure
        chunks = []

        if self.respect_structure:
            chunks = await self._chunk_by_structure(content, structure, doc_path, metadata)
        else:
            chunks = await self._chunk_by_tokens(content, doc_path, metadata)

        # Merge short chunks if enabled
        if self.merge_short_chunks:
            chunks = self._merge_short_chunks(chunks)

        # Add overlap for context
        chunks = self._add_overlap(chunks, content)

        # Assign chunk IDs
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{doc_path}:chunk_{i}"

        logger.debug(f"Created {len(chunks)} chunks for {doc_path}")

        return chunks

    def _parse_structure(self, content: str) -> Dict[str, Any]:
        """Parse document structure (headings, sections)."""
        lines = content.split('\n')

        structure = {
            'headings': [],
            'tables': [],
            'code_blocks': [],
            'lists': []
        }

        current_heading_stack = []
        line_num = 0

        for i, line in enumerate(lines):
            # Detect headings
            heading_match = self.heading_pattern.match(line)
            if heading_match:
                level = len(heading_match.group(1))  # Count #'s
                text = heading_match.group(2).strip()

                # Update heading stack
                while current_heading_stack and current_heading_stack[-1]['level'] >= level:
                    current_heading_stack.pop()

                heading_info = {
                    'level': level,
                    'text': text,
                    'line': i
                }

                current_heading_stack.append(heading_info)
                structure['headings'].append({
                    'line': i,
                    'level': level,
                    'text': text,
                    'hierarchy': [h['text'] for h in current_heading_stack]
                })

            # Detect tables
            elif self.table_pattern.match(line):
                if not structure['tables'] or structure['tables'][-1]['end_line'] != i - 1:
                    structure['tables'].append({
                        'start_line': i,
                        'end_line': i
                    })
                else:
                    structure['tables'][-1]['end_line'] = i

            # Detect lists
            elif self.list_item_pattern.match(line):
                if not structure['lists'] or structure['lists'][-1]['end_line'] != i - 1:
                    structure['lists'].append({
                        'start_line': i,
                        'end_line': i
                    })
                else:
                    structure['lists'][-1]['end_line'] = i

        # Detect code blocks
        for match in self.code_block_pattern.finditer(content):
            start = content[:match.start()].count('\n')
            end = content[:match.end()].count('\n')
            structure['code_blocks'].append({
                'start_line': start,
                'end_line': end
            })

        return structure

    async def _chunk_by_structure(
        self,
        content: str,
        structure: Dict[str, Any],
        doc_path: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk document respecting its structure."""
        lines = content.split('\n')
        chunks = []

        # Get section boundaries (headings)
        section_boundaries = [h['line'] for h in structure['headings']]
        section_boundaries.append(len(lines))  # End of document

        current_heading_hierarchy = []

        for i in range(len(section_boundaries) - 1):
            start_line = section_boundaries[i]
            end_line = section_boundaries[i + 1]

            # Get heading info
            heading_info = next(
                (h for h in structure['headings'] if h['line'] == start_line),
                None
            )

            if heading_info:
                current_heading_hierarchy = heading_info['hierarchy']

            # Extract section content
            section_content = '\n'.join(lines[start_line:end_line])

            # Estimate tokens
            tokens = self._estimate_tokens(section_content)

            # If section is too large, split further
            if tokens > self.max_tokens:
                # Split by paragraphs
                sub_chunks = await self._split_large_section(
                    section_content,
                    start_line,
                    doc_path,
                    current_heading_hierarchy,
                    metadata
                )
                chunks.extend(sub_chunks)
            else:
                # Keep as single chunk
                chunk = Chunk(
                    content=section_content,
                    metadata=metadata.copy(),
                    tokens=tokens,
                    chunk_id="",  # Assigned later
                    start_line=start_line,
                    end_line=end_line,
                    doc_path=doc_path,
                    heading_hierarchy=current_heading_hierarchy.copy(),
                    chunk_type=self._detect_chunk_type(section_content, structure, start_line, end_line)
                )
                chunks.append(chunk)

        return chunks

    async def _split_large_section(
        self,
        content: str,
        start_line: int,
        doc_path: str,
        heading_hierarchy: List[str],
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Split large section into smaller chunks."""
        chunks = []

        # Split by paragraphs (double newline)
        paragraphs = re.split(r'\n\n+', content)

        current_chunk_content = ""
        current_chunk_tokens = 0
        chunk_start_line = start_line

        for paragraph in paragraphs:
            paragraph_tokens = self._estimate_tokens(paragraph)

            # Check if adding this paragraph exceeds limit
            if current_chunk_tokens + paragraph_tokens > self.max_tokens and current_chunk_content:
                # Save current chunk
                chunks.append(Chunk(
                    content=current_chunk_content.strip(),
                    metadata=metadata.copy(),
                    tokens=current_chunk_tokens,
                    chunk_id="",
                    start_line=chunk_start_line,
                    end_line=chunk_start_line + current_chunk_content.count('\n'),
                    doc_path=doc_path,
                    heading_hierarchy=heading_hierarchy.copy(),
                    chunk_type='text'
                ))

                # Start new chunk
                current_chunk_content = paragraph
                current_chunk_tokens = paragraph_tokens
                chunk_start_line += current_chunk_content.count('\n') + 1
            else:
                # Add to current chunk
                if current_chunk_content:
                    current_chunk_content += "\n\n" + paragraph
                else:
                    current_chunk_content = paragraph
                current_chunk_tokens += paragraph_tokens

        # Add final chunk
        if current_chunk_content:
            chunks.append(Chunk(
                content=current_chunk_content.strip(),
                metadata=metadata.copy(),
                tokens=current_chunk_tokens,
                chunk_id="",
                start_line=chunk_start_line,
                end_line=chunk_start_line + current_chunk_content.count('\n'),
                doc_path=doc_path,
                heading_hierarchy=heading_hierarchy.copy(),
                chunk_type='text'
            ))

        return chunks

    async def _chunk_by_tokens(
        self,
        content: str,
        doc_path: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Simple token-based chunking (fallback)."""
        chunks = []
        lines = content.split('\n')

        current_chunk_lines = []
        current_tokens = 0
        start_line = 0

        for i, line in enumerate(lines):
            line_tokens = self._estimate_tokens(line)

            if current_tokens + line_tokens > self.max_tokens and current_chunk_lines:
                # Save chunk
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append(Chunk(
                    content=chunk_content,
                    metadata=metadata.copy(),
                    tokens=current_tokens,
                    chunk_id="",
                    start_line=start_line,
                    end_line=i - 1,
                    doc_path=doc_path,
                    heading_hierarchy=[],
                    chunk_type='text'
                ))

                # Start new chunk
                current_chunk_lines = [line]
                current_tokens = line_tokens
                start_line = i
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens

        # Final chunk
        if current_chunk_lines:
            chunks.append(Chunk(
                content='\n'.join(current_chunk_lines),
                metadata=metadata.copy(),
                tokens=current_tokens,
                chunk_id="",
                start_line=start_line,
                end_line=len(lines) - 1,
                doc_path=doc_path,
                heading_hierarchy=[],
                chunk_type='text'
            ))

        return chunks

    def _merge_short_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks that are too short."""
        if not chunks:
            return chunks

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            # Check if current chunk is too short
            if current.tokens < self.min_chunk_tokens:
                # Check if we can merge with next
                combined_tokens = current.tokens + next_chunk.tokens

                if combined_tokens <= self.max_tokens:
                    # Merge
                    current = Chunk(
                        content=current.content + "\n\n" + next_chunk.content,
                        metadata=current.metadata,
                        tokens=combined_tokens,
                        chunk_id=current.chunk_id,
                        start_line=current.start_line,
                        end_line=next_chunk.end_line,
                        doc_path=current.doc_path,
                        heading_hierarchy=current.heading_hierarchy,
                        chunk_type=current.chunk_type
                    )
                    continue

            # Can't merge, save current
            merged.append(current)
            current = next_chunk

        # Add final chunk
        merged.append(current)

        return merged

    def _add_overlap(self, chunks: List[Chunk], full_content: str) -> List[Chunk]:
        """Add overlapping context between chunks."""
        if not chunks or self.overlap_tokens == 0:
            return chunks

        lines = full_content.split('\n')

        for i in range(1, len(chunks)):
            # Get overlap from previous chunk
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            # Calculate overlap lines
            overlap_line_count = min(
                self.overlap_tokens // 10,  # Rough estimate: 10 tokens per line
                3  # Max 3 lines overlap
            )

            # Get last N lines from previous chunk
            overlap_start = max(0, prev_chunk.end_line - overlap_line_count + 1)
            overlap_end = prev_chunk.end_line + 1

            if overlap_start < overlap_end and overlap_end <= len(lines):
                overlap_content = '\n'.join(lines[overlap_start:overlap_end])

                # Prepend to current chunk
                curr_chunk.content = overlap_content + "\n\n" + curr_chunk.content
                curr_chunk.tokens += self._estimate_tokens(overlap_content)

        return chunks

    def _detect_chunk_type(
        self,
        content: str,
        structure: Dict[str, Any],
        start_line: int,
        end_line: int
    ) -> str:
        """Detect the type of chunk based on content."""
        # Check if chunk contains table
        for table in structure['tables']:
            if table['start_line'] >= start_line and table['end_line'] <= end_line:
                return 'table'

        # Check if chunk contains code block
        for code in structure['code_blocks']:
            if code['start_line'] >= start_line and code['end_line'] <= end_line:
                return 'code'

        # Check if chunk is a list
        for lst in structure['lists']:
            if lst['start_line'] >= start_line and lst['end_line'] <= end_line:
                return 'list'

        return 'text'

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Rule of thumb:
        - English: ~4 characters per token
        - Code: ~3.5 characters per token
        - Markdown: account for special characters
        """
        if not text:
            return 0

        # Rough estimation
        char_count = len(text)
        word_count = len(text.split())

        # Average between character-based and word-based estimates
        char_estimate = char_count / 4
        word_estimate = word_count * 1.3  # Words are ~1.3 tokens on average

        return int((char_estimate + word_estimate) / 2)
