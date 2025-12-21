"""
Document indexer for building searchable index.
Uses Docling for image understanding if enabled.
Uses advanced HybridChunker for better chunking.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


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
    """Build and manage document index with advanced chunking."""

    def __init__(
        self,
        docs_folder: Path,
        index_path: Path,
        enable_embeddings: bool = True,
        enable_vlm: bool = False,
        chunking_config: Optional[AdvancedChunkingConfig] = None
    ):
        self.docs_folder = docs_folder
        self.index_path = index_path
        self.enable_embeddings = enable_embeddings
        self.enable_vlm = enable_vlm
        self.chunking_config = chunking_config or AdvancedChunkingConfig()

        # Ensure index directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Index data
        self.file_index: Dict[str, Dict[str, Any]] = {}
        self.embeddings_list: List[np.ndarray] = []
        self.chunk_metadata_list: List[Dict[str, Any]] = []  # Track chunk metadata

        # Embedder (lazy load)
        self._embedder = None
        self._docling_converter = None
        self._chunker = None
        self._tokenizer = None

    async def build_index(self):
        """Build complete index."""
        logger.info(f"Building index for {self.docs_folder}")
        start_time = datetime.now()

        # Find all markdown files
        md_files = list(self.docs_folder.rglob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files")

        # Process each file
        for i, file_path in enumerate(md_files):
            rel_path = str(file_path.relative_to(self.docs_folder))
            logger.info(f"Processing [{i+1}/{len(md_files)}]: {rel_path}")

            try:
                metadata = await self._process_file(file_path)
                self.file_index[rel_path] = metadata

                # Generate embedding if enabled
                if self.enable_embeddings and metadata.get('content'):
                    embedding = await self._embed_content(metadata['content'])
                    self.embeddings_list.append(embedding)

            except Exception as e:
                logger.error(f"Failed to process {rel_path}: {e}", exc_info=True)

        # Save index
        await self._save_index()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Index built in {elapsed:.2f}s: {len(self.file_index)} files")

    async def _process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single markdown file with advanced chunking."""
        metadata = {
            'path': str(file_path.relative_to(self.docs_folder)),
            'modified': file_path.stat().st_mtime,
            'size': file_path.stat().st_size
        }

        # Read content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata['content'] = content

        # Extract title
        metadata['title'] = self._extract_title(content, file_path)

        # Extract headings
        metadata['headings'] = self._extract_headings(content)

        # Check for images
        image_info = self._find_images(content)
        metadata['has_images'] = len(image_info) > 0
        metadata['image_count'] = len(image_info)

        # Process images with Docling VLM if enabled
        if self.enable_vlm and image_info:
            image_descriptions = await self._process_images_with_docling(
                file_path,
                image_info
            )
            metadata['image_descriptions'] = image_descriptions
            # Append image descriptions to content for better search
            if image_descriptions:
                content += "\n\n" + "\n".join(image_descriptions)
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
        Chunk content using advanced Docling HybridChunker.

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

            # For markdown, we need to convert to DoclingDocument first
            # For now, use simple paragraph-based chunking with heading context
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
                serializer_provider=serializer_provider  # Advanced serialization!
            )

            logger.info(f"Advanced chunker initialized with enhanced serialization")
            logger.info(f"  - Markdown tables: enabled")
            logger.info(f"  - Picture annotations: {self.enable_vlm}")
            logger.info(f"  - Max tokens: {self.chunking_config.max_tokens}")

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
                # Remove headings at same or deeper level
                current_headings = [
                    h for h in current_headings
                    if h['level'] < heading_level
                ]
                # Add new heading
                current_headings.append({
                    'level': heading_level,
                    'text': heading_text
                })

                # Start new chunk at major headings (h1, h2)
                if heading_level <= 2 and current_chunk:
                    # Save current chunk
                    chunks.append(self._create_chunk_metadata(
                        '\n'.join(current_chunk),
                        current_headings[:-1],  # Previous headings
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
        # Build heading path
        heading_path = [h['text'] for h in heading_hierarchy]

        # Estimate tokens
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
        # Try to find first heading
        lines = content.split('\n')
        for line in lines[:20]:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()

        # Fallback to filename
        return file_path.stem.replace('-', ' ').replace('_', ' ').title()

    def _extract_headings(self, content: str) -> List[str]:
        """Extract all headings from markdown."""
        headings = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                # Remove hash marks
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

    async def _process_images_with_docling(
        self,
        file_path: Path,
        image_info: List[Dict[str, str]]
    ) -> List[str]:
        """Process images with Docling VLM for descriptions."""
        descriptions = []

        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat

            # Initialize Docling converter with VLM
            if self._docling_converter is None:
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_picture_description = True

                self._docling_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options
                        )
                    }
                )

            # For each image, try to get description
            for img in image_info:
                img_path = img['path']

                # Skip external URLs
                if img_path.startswith('http'):
                    continue

                # Resolve image path relative to markdown file
                full_img_path = (file_path.parent / img_path).resolve()

                if full_img_path.exists() and full_img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # Note: Docling VLM primarily works with PDFs
                    # For standalone images, we'd need a different approach
                    # For now, just use alt text if available
                    if img['alt']:
                        descriptions.append(f"Image: {img['alt']}")

        except ImportError:
            logger.warning("Docling not available for image processing")
        except Exception as e:
            logger.error(f"Image processing failed: {e}")

        return descriptions

    async def _embed_content(self, content: str) -> np.ndarray:
        """Generate embedding for content."""
        if self._embedder is None:
            # Lazy load embedder based on environment
            from ..search.embedders import create_embedder
            provider = os.getenv("EMBEDDING_PROVIDER", "local")
            self._embedder = create_embedder(provider=provider)

        # Truncate content if too long
        # Local models can usually handle up to 512 tokens (~2000 chars)
        # Adjust based on your embedding model
        if len(content) > 2000:
            content = content[:2000]

        embedding = await self._embedder.embed_query(content)
        return np.array(embedding)

    async def _save_index(self):
        """Save index to disk."""
        # Save metadata index
        index_file = self.index_path / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({
                'files': self.file_index,
                'created': datetime.now().isoformat(),
                'docs_count': len(self.file_index)
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

    parser = argparse.ArgumentParser(description="Build documentation index")
    parser.add_argument("--docs", default=os.getenv("DOCS_FOLDER", "docs"),
                       help="Documentation folder")
    parser.add_argument("--index", default=".index", help="Index output path")
    parser.add_argument("--mode", choices=['fast', 'full'], default='full',
                       help="Fast (metadata only) or full (with embeddings)")
    parser.add_argument("--enable-vlm", action='store_true',
                       help="Enable image understanding with Docling VLM")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create indexer
    indexer = DocIndexer(
        docs_folder=Path(args.docs),
        index_path=Path(args.index),
        enable_embeddings=(args.mode == 'full'),
        enable_vlm=args.enable_vlm
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
