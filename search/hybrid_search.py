"""
Hybrid search engine combining fast text search and semantic search.
"""

import os
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with metadata."""
    path: str
    title: str
    excerpt: str
    score: float
    team: str
    category: str
    match_type: str  # 'exact', 'fuzzy', 'semantic'
    headings: List[str]
    has_images: bool = False
    image_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HybridSearchEngine:
    """
    Hybrid search engine for documentation with reranking.

    Strategy:
    1. Fast text search (ripgrep/grep) for keyword matches
    2. Fuzzy matching for typos and variations
    3. Semantic search (embeddings) for conceptual queries
    4. Reranking (optional) for better result quality
    5. Folder-aware filtering by team/category
    """

    def __init__(
        self,
        docs_folder: Path,
        index_path: Path,
        enable_semantic: bool = True,
        enable_vlm: bool = False,
        enable_reranking: bool = False
    ):
        self.docs_folder = docs_folder
        self.index_path = index_path
        self.enable_semantic = enable_semantic
        self.enable_vlm = enable_vlm
        self.enable_reranking = enable_reranking

        # Index data
        self.file_index: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.file_paths: List[str] = []

        # Embedder (lazy load)
        self._embedder = None
        self._reranker = None

    async def initialize(self):
        """Initialize or load index."""
        logger.info("Initializing search engine...")

        # Try to load existing index
        if self.index_path.exists():
            logger.info(f"Loading index from {self.index_path}")
            await self._load_index()
        else:
            logger.info("No index found, building new index...")
            await self._build_index()

        logger.info(f"Index loaded: {len(self.file_index)} files")

    async def _build_index(self):
        """Build search index from markdown files."""
        from indexer.doc_indexer import DocIndexer

        indexer = DocIndexer(
            docs_folder=self.docs_folder,
            index_path=self.index_path,
            enable_embeddings=self.enable_semantic,
            enable_vlm=self.enable_vlm
        )

        await indexer.build_index()
        await self._load_index()

    async def _load_index(self):
        """Load index from disk."""
        index_file = self.index_path / "index.json"

        if not index_file.exists():
            logger.warning("Index file not found, building new index")
            await self._build_index()
            return

        with open(index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.file_index = data.get("files", {})
        self.file_paths = list(self.file_index.keys())

        # Load embeddings if available
        if self.enable_semantic:
            embeddings_file = self.index_path / "embeddings.npy"
            if embeddings_file.exists():
                self.embeddings = np.load(str(embeddings_file))
                logger.info(f"Loaded embeddings: {self.embeddings.shape}")

    async def fast_search(
        self,
        query: str,
        folder: Optional[str] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Fast keyword-based search using multiple strategies with optional reranking.

        1. Exact filename match
        2. Ripgrep content search
        3. Fuzzy matching on titles
        4. Reranking (if enabled)
        """
        # Get more candidates if reranking is enabled
        candidate_limit = limit * 5 if self.enable_reranking else limit * 2
        results = []

        # Strategy 1: Exact path/filename match
        exact_matches = self._find_exact_matches(query, folder)
        results.extend(exact_matches)

        # Strategy 2: Ripgrep content search (fast!)
        if len(results) < candidate_limit:
            grep_matches = await self._ripgrep_search(query, folder)
            results.extend(grep_matches)

        # Strategy 3: Fuzzy title matching
        if len(results) < candidate_limit:
            fuzzy_matches = self._fuzzy_title_search(query, folder)
            results.extend(fuzzy_matches)

        # Remove duplicates and sort by score
        seen_paths = set()
        unique_results = []
        for result in results:
            if result.path not in seen_paths:
                seen_paths.add(result.path)
                unique_results.append(result)

        unique_results.sort(key=lambda x: x.score, reverse=True)

        # Rerank if enabled
        if self.enable_reranking and len(unique_results) > limit:
            unique_results = await self._rerank_results(query, unique_results, limit)

        return unique_results[:limit]

    def _find_exact_matches(
        self,
        query: str,
        folder: Optional[str] = None
    ) -> List[SearchResult]:
        """Find files with exact query match in path or title."""
        results = []
        query_lower = query.lower()

        for file_path, metadata in self.file_index.items():
            # Apply folder filter
            if folder and not file_path.startswith(folder):
                continue

            # Check path contains query
            if query_lower in file_path.lower():
                results.append(self._create_result(
                    file_path,
                    metadata,
                    score=100.0,
                    match_type='exact',
                    excerpt=f"File path matches: {file_path}"
                ))

            # Check title contains query
            elif query_lower in metadata.get('title', '').lower():
                results.append(self._create_result(
                    file_path,
                    metadata,
                    score=95.0,
                    match_type='exact',
                    excerpt=f"Title: {metadata['title']}"
                ))

        return results

    async def _ripgrep_search(
        self,
        query: str,
        folder: Optional[str] = None
    ) -> List[SearchResult]:
        """Fast content search using ripgrep."""
        results = []

        # Try ripgrep first (much faster)
        rg_available = self._check_ripgrep()

        if rg_available:
            results = await self._ripgrep_exec(query, folder)
        else:
            # Fallback to Python-based search
            results = self._python_grep(query, folder)

        return results

    def _check_ripgrep(self) -> bool:
        """Check if ripgrep is available."""
        try:
            subprocess.run(['rg', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    async def _ripgrep_exec(
        self,
        query: str,
        folder: Optional[str] = None
    ) -> List[SearchResult]:
        """Execute ripgrep search."""
        search_path = self.docs_folder / folder if folder else self.docs_folder

        try:
            # Run ripgrep with context
            cmd = [
                'rg',
                '--type', 'md',
                '--context', '2',
                '--max-count', '3',
                '--line-number',
                '--heading',
                '--color', 'never',
                '--case-insensitive',
                query,
                str(search_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse ripgrep output
            return self._parse_ripgrep_output(result.stdout, query)

        except subprocess.TimeoutExpired:
            logger.warning("Ripgrep search timed out")
            return []
        except Exception as e:
            logger.error(f"Ripgrep search failed: {e}")
            return []

    def _parse_ripgrep_output(self, output: str, query: str) -> List[SearchResult]:
        """Parse ripgrep output into search results."""
        results = []
        current_file = None
        current_lines = []

        for line in output.split('\n'):
            if not line.strip():
                continue

            # File path line (no line number)
            if not re.match(r'^\d+[:-]', line):
                if current_file and current_lines:
                    # Create result for previous file
                    results.append(self._create_grep_result(
                        current_file,
                        current_lines,
                        query
                    ))

                # Start new file
                current_file = line.strip()
                current_lines = []
            else:
                # Content line
                current_lines.append(line)

        # Final file
        if current_file and current_lines:
            results.append(self._create_grep_result(
                current_file,
                current_lines,
                query
            ))

        return results

    def _create_grep_result(
        self,
        file_path: str,
        lines: List[str],
        query: str
    ) -> SearchResult:
        """Create search result from grep match."""
        # Convert absolute path to relative
        rel_path = str(Path(file_path).relative_to(self.docs_folder))

        # Get metadata if available
        metadata = self.file_index.get(rel_path, {})

        # Extract excerpt from matched lines
        excerpt_lines = []
        for line in lines[:5]:  # Max 5 lines
            # Remove line numbers
            content = re.sub(r'^\d+[:-]\s*', '', line)
            if content.strip():
                excerpt_lines.append(content.strip())

        excerpt = '\n'.join(excerpt_lines)

        # Score based on query frequency
        query_count = excerpt.lower().count(query.lower())
        score = min(90.0, 70.0 + query_count * 5)

        return self._create_result(
            rel_path,
            metadata,
            score=score,
            match_type='grep',
            excerpt=excerpt
        )

    def _python_grep(
        self,
        query: str,
        folder: Optional[str] = None
    ) -> List[SearchResult]:
        """Fallback Python-based content search."""
        results = []
        query_lower = query.lower()

        for file_path, metadata in self.file_index.items():
            # Apply folder filter
            if folder and not file_path.startswith(folder):
                continue

            # Search in content
            content = metadata.get('content', '')
            if query_lower in content.lower():
                # Find context around match
                excerpt = self._extract_context(content, query, max_chars=200)

                # Score based on query frequency
                query_count = content.lower().count(query_lower)
                score = min(85.0, 60.0 + query_count * 5)

                results.append(self._create_result(
                    file_path,
                    metadata,
                    score=score,
                    match_type='content',
                    excerpt=excerpt
                ))

        return results

    def _fuzzy_title_search(
        self,
        query: str,
        folder: Optional[str] = None
    ) -> List[SearchResult]:
        """Fuzzy match on file titles."""
        results = []

        # Collect titles
        titles_and_paths = []
        for file_path, metadata in self.file_index.items():
            if folder and not file_path.startswith(folder):
                continue
            titles_and_paths.append((metadata.get('title', ''), file_path))

        # Fuzzy match
        matches = process.extract(
            query,
            [t[0] for t in titles_and_paths],
            scorer=fuzz.WRatio,
            limit=5
        )

        for title, score, idx in matches:
            if score > 60:  # Threshold for fuzzy matches
                file_path = titles_and_paths[idx][1]
                metadata = self.file_index[file_path]

                results.append(self._create_result(
                    file_path,
                    metadata,
                    score=float(score * 0.8),  # Discount fuzzy scores
                    match_type='fuzzy',
                    excerpt=f"Fuzzy title match: {title}"
                ))

        return results

    async def semantic_search(
        self,
        query: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """Semantic search using embeddings with optional reranking."""
        if not self.enable_semantic or self.embeddings is None:
            logger.warning("Semantic search not enabled or embeddings not loaded")
            return []

        # Get query embedding
        query_embedding = await self._embed_query(query)

        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        similarities = similarities / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get more candidates if reranking is enabled
        candidate_limit = limit * 3 if self.enable_reranking else limit
        top_indices = np.argsort(similarities)[::-1][:candidate_limit]

        results = []
        for idx in top_indices:
            similarity = similarities[idx]

            if similarity < 0.6:  # Threshold
                continue

            file_path = self.file_paths[idx]
            metadata = self.file_index[file_path]

            results.append(self._create_result(
                file_path,
                metadata,
                score=float(similarity * 100),
                match_type='semantic',
                excerpt=self._extract_context(
                    metadata.get('content', ''),
                    query,
                    max_chars=200
                )
            ))

        # Rerank if enabled
        if self.enable_reranking and len(results) > limit:
            results = await self._rerank_results(query, results, limit)

        return results[:limit]

    async def _embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query."""
        if self._embedder is None:
            # Lazy load embedder based on environment
            from .embedders import create_embedder
            provider = os.getenv("EMBEDDING_PROVIDER", "local")
            self._embedder = create_embedder(provider=provider)

        embedding = await self._embedder.embed_query(query)
        return np.array(embedding)

    def _create_result(
        self,
        file_path: str,
        metadata: Dict[str, Any],
        score: float,
        match_type: str,
        excerpt: str
    ) -> SearchResult:
        """Create SearchResult from metadata."""
        # Extract team and category from path
        parts = Path(file_path).parts
        team = parts[0] if len(parts) > 0 else "unknown"
        category = parts[1] if len(parts) > 1 else "general"

        return SearchResult(
            path=file_path,
            title=metadata.get('title', Path(file_path).stem),
            excerpt=excerpt,
            score=score,
            team=team,
            category=category,
            match_type=match_type,
            headings=metadata.get('headings', []),
            has_images=metadata.get('has_images', False),
            image_count=metadata.get('image_count', 0)
        )

    async def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder for better quality."""
        if not results:
            return results

        # Initialize reranker if needed
        if self._reranker is None:
            from .reranker import create_reranker
            self._reranker = create_reranker(enabled=True)

        # Prepare candidates for reranking
        candidates = [
            (result.excerpt, result.score / 100.0, {
                'path': result.path,
                'title': result.title,
                'team': result.team,
                'category': result.category,
                'match_type': result.match_type,
                'headings': result.headings,
                'has_images': result.has_images,
                'image_count': result.image_count
            })
            for result in results
        ]

        # Rerank
        reranked = self._reranker.rerank(query, candidates, top_k=top_k)

        # Convert back to SearchResult
        reranked_results = []
        for item in reranked:
            reranked_results.append(SearchResult(
                path=item.metadata['path'],
                title=item.metadata['title'],
                excerpt=item.content,
                score=item.final_score * 100,  # Scale back to 0-100
                team=item.metadata['team'],
                category=item.metadata['category'],
                match_type=f"{item.metadata['match_type']}+reranked",
                headings=item.metadata['headings'],
                has_images=item.metadata['has_images'],
                image_count=item.metadata['image_count']
            ))

        return reranked_results

    def _extract_context(
        self,
        content: str,
        query: str,
        max_chars: int = 200
    ) -> str:
        """Extract context around query match."""
        query_lower = query.lower()
        content_lower = content.lower()

        # Find first occurrence
        idx = content_lower.find(query_lower)
        if idx == -1:
            # No match, return start of content
            return content[:max_chars] + "..."

        # Extract window around match
        start = max(0, idx - max_chars // 2)
        end = min(len(content), idx + len(query) + max_chars // 2)

        excerpt = content[start:end]

        # Add ellipsis
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."

        return excerpt
