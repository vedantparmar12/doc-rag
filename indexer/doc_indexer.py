"""
Document indexer for building searchable index.
NOW WITH: Batch processing, table extraction, image OCR, link tracking.
"""

import os
import re
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

import numpy as np

from .table_extractor import MarkdownTableExtractor
from .image_ocr import ImageProcessor
from .link_extractor import LinkExtractor

logger = logging.getLogger(__name__)


@dataclass
class ProcessingBatch:
    """Batch of files to process together."""
    files: List[Path]
    batch_id: int
    total_batches: int


class AdvancedChunkingConfig:
    """Configuration for advanced Docling chunking."""

    def __init__(
        self,
        max_tokens: int = 512,
        heading_as_metadata: bool = True,
        include_xml_tags: bool = True,
        merge_peers: bool = True,
        include_captions: bool = True,
        include_page_footer: bool = False
    ):
        """
        Initialize chunking config.

        Args:
            max_tokens: Maximum tokens per chunk (default: 512)
            heading_as_metadata: Include heading hierarchy in metadata
            include_xml_tags: Preserve document structure tags
            merge_peers: Merge small adjacent chunks
            include_captions: Include image/table captions
            include_page_footer: Include page footers (usually false)
        """
        self.max_tokens = max_tokens
        self.heading_as_metadata = heading_as_metadata
        self.include_xml_tags = include_xml_tags
        self.merge_peers = merge_peers
        self.include_captions = include_captions
        self.include_page_footer = include_page_footer


class DocIndexer:
    """Build and manage document index with ALL optimizations."""

    def __init__(
        self,
        docs_folder: Path,
        index_path: Path,
        enable_embeddings: bool = True,
        enable_vlm: bool = False,
        enable_table_extraction: bool = True,
        enable_image_ocr: bool = True,
        enable_link_tracking: bool = True,
        ocr_backend: str = "auto",
        max_workers: int = None,
        chunking_config: Optional[AdvancedChunkingConfig] = None
    ):
        self.docs_folder = docs_folder
        self.index_path = index_path
        self.enable_embeddings = enable_embeddings
        self.enable_vlm = enable_vlm
        self.enable_table_extraction = enable_table_extraction
        self.enable_image_ocr = enable_image_ocr
        self.enable_link_tracking = enable_link_tracking
        self.chunking_config = chunking_config or AdvancedChunkingConfig()

        # Parallel processing
        self.max_workers = max_workers or self._detect_optimal_workers()

        # Ensure index directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Index data
        self.file_index: Dict[str, Dict[str, Any]] = {}
        self.embeddings_list: List[np.ndarray] = []
        self.chunk_metadata_list: List[Dict[str, Any]] = []

        # Feature processors
        self.table_extractor = MarkdownTableExtractor() if enable_table_extraction else None
        self.image_processor = ImageProcessor(
            enable_ocr=enable_image_ocr,
            ocr_backend=ocr_backend
        ) if enable_image_ocr else None
        self.link_extractor = LinkExtractor(docs_folder) if enable_link_tracking else None

        # Embedder (lazy load)
        self._embedder = None
        self._docling_converter = None
        self._chunker = None
        self._tokenizer = None

    def _detect_optimal_workers(self) -> int:
        """Auto-detect optimal worker count for CPU."""
        import os
        cpu_count = os.cpu_count() or 4
        # Leave 1-2 cores for system, use rest for processing
        return max(1, cpu_count - 1)

    async def build_index(self):
        """Build complete index with BATCH PROCESSING."""
        logger.info(f"Building index for {self.docs_folder}")
        logger.info(f"Parallel workers: {self.max_workers}")
        logger.info(f"Features enabled:")
        logger.info(f"  - Table extraction: {self.enable_table_extraction}")
        logger.info(f"  - Image OCR: {self.enable_image_ocr}")
        logger.info(f"  - Link tracking: {self.enable_link_tracking}")
        logger.info(f"  - Embeddings: {self.enable_embeddings}")

        start_time = datetime.now()

        # Find all markdown files
        md_files = list(self.docs_folder.rglob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files")

        if not md_files:
            logger.warning("No markdown files found!")
            return

        # Create batches for parallel processing
        batch_size = 10  # Files per batch
        batches = self._create_batches(md_files, batch_size)

        # Process batches in parallel
        processed_count = 0
        total_tables = 0
        total_images = 0
        total_links = 0

        for batch in batches:
            results = await self._process_batch_parallel(batch)

            # Add successful results to index
            for rel_path, metadata in results.items():
                self.file_index[rel_path] = metadata

                # Track stats
                total_tables += metadata.get('table_count', 0)
                total_images += metadata.get('image_count', 0)
                total_links += metadata.get('link_stats', {}).get('total', 0)

            processed_count += len(results)
            progress = (processed_count / len(md_files)) * 100
            logger.info(f"Progress: {processed_count}/{len(md_files)} ({progress:.1f}%)")

        # Generate embeddings in parallel
        if self.enable_embeddings:
            logger.info("Generating embeddings in parallel...")
            await self._generate_embeddings_parallel()

        # Save index
        await self._save_index()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✓ Index built in {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
        logger.info(f"  Files indexed: {len(self.file_index)}")
        logger.info(f"  Tables extracted: {total_tables}")
        logger.info(f"  Images processed: {total_images}")
        logger.info(f"  Links tracked: {total_links}")
        if self.embeddings_list:
            logger.info(f"  Embeddings generated: {len(self.embeddings_list)}")

    def _create_batches(
        self,
        files: List[Path],
        batch_size: int
    ) -> List[ProcessingBatch]:
        """Split files into processing batches."""
        batches = []
        total_batches = (len(files) + batch_size - 1) // batch_size

        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            batches.append(ProcessingBatch(
                files=batch_files,
                batch_id=i // batch_size,
                total_batches=total_batches
            ))

        return batches

    async def _process_batch_parallel(
        self,
        batch: ProcessingBatch
    ) -> Dict[str, Dict[str, Any]]:
        """Process a batch of files in parallel."""
        # Create tasks for all files in batch
        tasks = [
            self._process_file_safe(file_path)
            for file_path in batch.files
        ]

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        batch_index = {}
        for file_path, result in zip(batch.files, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {file_path}: {result}")
                continue

            rel_path = str(file_path.relative_to(self.docs_folder))
            batch_index[rel_path] = result

        return batch_index

    async def _process_file_safe(self, file_path: Path) -> Dict[str, Any]:
        """Process file with error handling."""
        try:
            return await self._process_file(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            raise

    async def _process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single markdown file with ALL features."""
        metadata = {
            'path': str(file_path.relative_to(self.docs_folder)),
            'modified': file_path.stat().st_mtime,
            'size': file_path.stat().st_size
        }

        # Read content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        metadata['content'] = content

        # Extract title
        metadata['title'] = self._extract_title(content, file_path)

        # Extract headings
        metadata['headings'] = self._extract_headings(content)

        # Extract team/category from path
        path_parts = Path(metadata['path']).parts
        metadata['team'] = path_parts[0] if len(path_parts) > 0 else 'general'
        metadata['category'] = path_parts[1] if len(path_parts) > 1 else 'general'

        # FEATURE 1: Extract tables
        if self.enable_table_extraction and self.table_extractor:
            tables = self.table_extractor.extract_tables(content, str(file_path))
            metadata['tables'] = [table.to_dict() for table in tables]
            metadata['table_count'] = len(tables)

            # Append table text to content for better search
            if tables:
                table_texts = [table.to_text() for table in tables]
                metadata['table_text'] = "\n\n".join(table_texts)
                content += "\n\n" + metadata['table_text']

        # FEATURE 2: Process images with OCR
        image_refs = self._find_images(original_content)
        if image_refs:
            if self.enable_image_ocr and self.image_processor:
                enriched_images = await self.image_processor.process_images(
                    file_path,
                    image_refs
                )
                metadata['images'] = enriched_images
                metadata['has_images'] = len(enriched_images) > 0
                metadata['image_count'] = len(enriched_images)

                # Collect OCR text for search
                ocr_texts = []
                for img in enriched_images:
                    if img.get('ocr_text'):
                        alt_text = img.get('alt', 'Image')
                        ocr_texts.append(f"{alt_text}: {img['ocr_text']}")

                if ocr_texts:
                    metadata['image_ocr_text'] = '\n\n'.join(ocr_texts)
                    content += '\n\n' + metadata['image_ocr_text']
            else:
                # Basic image info without OCR
                metadata['images'] = image_refs
                metadata['has_images'] = len(image_refs) > 0
                metadata['image_count'] = len(image_refs)

        # FEATURE 3: Extract and validate links
        if self.enable_link_tracking and self.link_extractor:
            links = self.link_extractor.extract_links(original_content, file_path)
            metadata['links'] = [link.to_dict() for link in links]
            metadata['related_files'] = self.link_extractor.get_related_files(links)
            metadata['link_stats'] = self.link_extractor.get_link_stats(links)

        # Update content with all enrichments
        metadata['content'] = content

        # Generate chunks with advanced chunker
        chunks_data = await self._chunk_content(content, metadata)
        metadata['chunks'] = chunks_data

        return metadata

    async def _chunk_content(
        self,
        content: str,
        file_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Chunk content using advanced chunking.

        Returns list of chunk metadata with:
        - content: Chunk text
        - heading_hierarchy: List of parent headings
        - chunk_index: Position in document
        - token_count: Estimated tokens
        """
        chunks = []

        try:
            # Initialize chunker if needed
            if self._chunker is None:
                await self._init_advanced_chunker()

            # Use advanced markdown chunking with heading context
            chunks = self._chunk_markdown_with_context(content, file_metadata)

        except Exception as e:
            logger.error(f"Advanced chunking failed: {e}, using simple chunking")
            chunks = self._simple_chunk(content, file_metadata)

        return chunks

    async def _init_advanced_chunker(self):
        """Initialize advanced Docling chunker with proper serialization."""
        try:
            from transformers import AutoTokenizer
            from docling.chunking import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer
            )
            from .docling_serializers import create_serializer_provider

            # Initialize tokenizer (Docling-compatible wrapper)
            model_id = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Initializing tokenizer: {model_id}")

            hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer)

            # Create enhanced serializer provider
            serializer_provider = create_serializer_provider(
                mode="enhanced",
                use_markdown_tables=True,
                use_picture_annotations=self.enable_vlm,
                image_placeholder="*[Image]*"
            )

            # Create chunker with advanced options
            self._chunker = HybridChunker(
                tokenizer=self._tokenizer,
                max_tokens=self.chunking_config.max_tokens,
                merge_peers=self.chunking_config.merge_peers,
                serializer_provider=serializer_provider
            )

            logger.info(f"Advanced chunker initialized")

        except ImportError as e:
            logger.warning(f"Could not initialize advanced chunker: {e}")
            logger.warning("Falling back to simple chunking")
            self._chunker = None

    def _chunk_markdown_with_context(
        self,
        content: str,
        file_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Chunk markdown while preserving heading hierarchy.

        Each chunk includes:
        - The chunk content
        - Parent heading hierarchy
        - Metadata for better retrieval
        """
        chunks = []
        lines = content.split('\n')

        current_chunk = []
        current_headings = []  # Stack of current heading hierarchy
        chunk_index = 0

        max_chars = self.chunking_config.max_tokens * 4  # Rough estimate

        for line in lines:
            # Check if line is a heading
            if line.strip().startswith('#'):
                heading_level = len(line) - len(line.lstrip('#'))
                heading_text = line.lstrip('#').strip()

                # Update heading hierarchy
                current_headings = [
                    h for h in current_headings
                    if h['level'] < heading_level
                ]
                current_headings.append({
                    'level': heading_level,
                    'text': heading_text
                })

                # Start new chunk at major headings (h1, h2)
                if heading_level <= 2 and current_chunk:
                    chunks.append(self._create_chunk_metadata(
                        '\n'.join(current_chunk),
                        current_headings[:-1],
                        chunk_index,
                        file_metadata
                    ))
                    chunk_index += 1
                    current_chunk = []

            current_chunk.append(line)

            # Check if chunk is getting too large
            if len('\n'.join(current_chunk)) > max_chars:
                chunks.append(self._create_chunk_metadata(
                    '\n'.join(current_chunk),
                    current_headings,
                    chunk_index,
                    file_metadata
                ))
                chunk_index += 1
                current_chunk = []

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk_metadata(
                '\n'.join(current_chunk),
                current_headings,
                chunk_index,
                file_metadata
            ))

        return chunks

    def _create_chunk_metadata(
        self,
        chunk_content: str,
        heading_hierarchy: List[Dict[str, Any]],
        chunk_index: int,
        file_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create metadata for a chunk."""
        heading_path = [h['text'] for h in heading_hierarchy]
        token_count = len(chunk_content) // 4  # Rough estimate

        return {
            'content': chunk_content.strip(),
            'heading_hierarchy': heading_path,
            'chunk_index': chunk_index,
            'token_count': token_count,
            'file_path': file_metadata['path'],
            'file_title': file_metadata['title']
        }

    def _simple_chunk(
        self,
        content: str,
        file_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Simple fallback chunking."""
        max_chars = self.chunking_config.max_tokens * 4
        chunks = []
        chunk_index = 0

        for i in range(0, len(content), max_chars):
            chunk_content = content[i:i+max_chars]
            chunks.append({
                'content': chunk_content.strip(),
                'heading_hierarchy': [],
                'chunk_index': chunk_index,
                'token_count': len(chunk_content) // 4,
                'file_path': file_metadata['path'],
                'file_title': file_metadata['title']
            })
            chunk_index += 1

        return chunks

    def _extract_title(self, content: str, file_path: Path) -> str:
        """Extract title from markdown."""
        lines = content.split('\n')
        for line in lines[:20]:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()

        return file_path.stem.replace('-', ' ').replace('_', ' ').title()

    def _extract_headings(self, content: str) -> List[str]:
        """Extract all headings from markdown."""
        headings = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                heading = re.sub(r'^#+\s*', '', line)
                headings.append(heading)

        return headings

    def _find_images(self, content: str) -> List[Dict[str, str]]:
        """Find image references in markdown."""
        images = []

        # Pattern: ![alt text](image path)
        pattern = r'!\[(.*?)\]\((.*?)\)'
        matches = re.findall(pattern, content)

        for alt_text, image_path in matches:
            images.append({
                'alt': alt_text,
                'path': image_path
            })

        # Pattern: <img src="...">
        html_pattern = r'<img[^>]*src=["\']([^"\']*)["\'][^>]*>'
        html_matches = re.findall(html_pattern, content)

        for image_path in html_matches:
            images.append({
                'alt': '',
                'path': image_path
            })

        return images

    async def _generate_embeddings_parallel(self):
        """Generate embeddings for all documents in parallel (BATCH MODE)."""
        logger.info("Generating embeddings in parallel...")

        # Prepare contents
        contents = []
        paths = []
        for path, metadata in self.file_index.items():
            if metadata.get('content'):
                contents.append(metadata['content'])
                paths.append(path)

        if not contents:
            logger.warning("No content to embed")
            return

        # Generate embeddings in batches (MUCH FASTER!)
        batch_size = 32
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i+batch_size]

            embeddings = await self._embed_batch(batch_contents)
            self.embeddings_list.extend(embeddings)

            logger.info(f"Generated {len(self.embeddings_list)}/{len(contents)} embeddings")

    async def _embed_batch(self, contents: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple contents at once (BATCH EMBEDDING)."""
        if self._embedder is None:
            from search.embedders import create_embedder
            provider = os.getenv("EMBEDDING_PROVIDER", "local")
            self._embedder = create_embedder(provider=provider)

        # Truncate contents
        truncated = [c[:2000] if len(c) > 2000 else c for c in contents]

        # Batch embedding (much faster than one-by-one!)
        embeddings = await self._embedder.embed_batch(truncated)
        return [np.array(emb) for emb in embeddings]

    async def _save_index(self):
        """Save index to disk."""
        # Save metadata index
        index_file = self.index_path / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({
                'files': self.file_index,
                'created': datetime.now().isoformat(),
                'docs_count': len(self.file_index),
                'features': {
                    'table_extraction': self.enable_table_extraction,
                    'image_ocr': self.enable_image_ocr,
                    'link_tracking': self.enable_link_tracking,
                    'embeddings': self.enable_embeddings
                }
            }, f, indent=2)

        logger.info(f"Saved index to {index_file}")

        # Save embeddings if available
        if self.embeddings_list:
            embeddings_array = np.array(self.embeddings_list)
            embeddings_file = self.index_path / "embeddings.npy"
            np.save(str(embeddings_file), embeddings_array)
            logger.info(f"Saved {len(self.embeddings_list)} embeddings to {embeddings_file}")


async def main():
    """CLI for building index."""
    import argparse

    parser = argparse.ArgumentParser(description="Build documentation index with optimizations")
    parser.add_argument("--docs", default=os.getenv("DOCS_FOLDER", "docs"),
                       help="Documentation folder")
    parser.add_argument("--index", default=".index", help="Index output path")
    parser.add_argument("--mode", choices=['fast', 'full'], default='full',
                       help="Fast (metadata only) or full (with embeddings)")
    parser.add_argument("--enable-vlm", action='store_true',
                       help="Enable image understanding with Docling VLM")
    parser.add_argument("--enable-table-extraction", action='store_true', default=True,
                       help="Extract and index tables")
    parser.add_argument("--enable-image-ocr", action='store_true', default=True,
                       help="Extract text from images using OCR")
    parser.add_argument("--enable-link-tracking", action='store_true', default=True,
                       help="Track and validate internal/external links")
    parser.add_argument("--ocr-backend", default="auto",
                       choices=['auto', 'rapidocr', 'easyocr', 'tesseract'],
                       help="OCR backend to use")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Number of parallel workers (default: auto-detect)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create indexer with ALL features
    indexer = DocIndexer(
        docs_folder=Path(args.docs),
        index_path=Path(args.index),
        enable_embeddings=(args.mode == 'full'),
        enable_vlm=args.enable_vlm,
        enable_table_extraction=args.enable_table_extraction,
        enable_image_ocr=args.enable_image_ocr,
        enable_link_tracking=args.enable_link_tracking,
        ocr_backend=args.ocr_backend,
        max_workers=args.max_workers
    )

    # Build index
    await indexer.build_index()

    print("\n✓ Index built successfully!")
    print(f"  Files indexed: {len(indexer.file_index)}")
    if indexer.embeddings_list:
        print(f"  Embeddings generated: {len(indexer.embeddings_list)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
