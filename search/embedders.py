"""
Embedding providers for semantic search.
Supports multiple backends: OpenAI, HuggingFace (local), HuggingFace API
"""

import os
import logging
from typing import List, Optional
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Base class for embedders."""

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        pass


class LocalHuggingFaceEmbedder(BaseEmbedder):
    """
    Local HuggingFace embeddings using sentence-transformers.

    NO API KEY NEEDED - runs completely locally!
    Free and fast for most use cases.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedder.

        Args:
            model_name: HuggingFace model name
                - all-MiniLM-L6-v2 (default, 384 dim, fast)
                - all-mpnet-base-v2 (768 dim, better quality)
                - paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading local HuggingFace model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    async def embed_query(self, text: str) -> List[float]:
        """Embed single query."""
        # sentence-transformers is CPU-based, so we can run it directly
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (batched for efficiency)."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10
        )
        return embeddings.tolist()


class HuggingFaceAPIEmbedder(BaseEmbedder):
    """
    HuggingFace Inference API embeddings.

    Requires HF_API_KEY but has free tier!
    Get key from: https://huggingface.co/settings/tokens
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None
    ):
        """Initialize HuggingFace API embedder."""
        self.api_key = api_key or os.getenv("HF_API_KEY")
        if not self.api_key:
            raise ValueError("HF_API_KEY required for HuggingFace API")

        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"

        logger.info(f"Using HuggingFace API with model: {model_name}")

    async def embed_query(self, text: str) -> List[float]:
        """Embed single query via API."""
        import aiohttp

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": text}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise RuntimeError(f"HuggingFace API error: {error}")

                result = await response.json()
                # HF API returns nested array
                return result[0] if isinstance(result[0], list) else result

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts via API."""
        # HF API supports batching
        import aiohttp

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": texts}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise RuntimeError(f"HuggingFace API error: {error}")

                result = await response.json()
                return result


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embeddings (requires OpenAI API key).
    High quality but costs money.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """Initialize OpenAI embedder."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package not installed")

        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        logger.info(f"Using OpenAI embeddings: {model}")

    async def embed_query(self, text: str) -> List[float]:
        """Embed single query."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        # OpenAI supports batching up to 2048 texts
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]


def create_embedder(
    provider: str = "local",
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> BaseEmbedder:
    """
    Factory function to create embedder based on provider.

    Args:
        provider: "local", "huggingface", or "openai"
        model: Model name (provider-specific)
        api_key: API key if needed

    Returns:
        Embedder instance

    Examples:
        # Local HuggingFace (FREE, no API key)
        embedder = create_embedder("local")

        # HuggingFace API (has free tier)
        embedder = create_embedder("huggingface", api_key="hf_...")

        # OpenAI (costs money)
        embedder = create_embedder("openai", api_key="sk-...")
    """
    provider = provider.lower()

    if provider == "local":
        model = model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return LocalHuggingFaceEmbedder(model_name=model)

    elif provider == "huggingface" or provider == "hf":
        model = model or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        return HuggingFaceAPIEmbedder(model_name=model, api_key=api_key)

    elif provider == "openai":
        model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbedder(model=model, api_key=api_key)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Choose from: local, huggingface, openai"
        )


# Recommended configurations
EMBEDDING_CONFIGS = {
    "free_fast": {
        "provider": "local",
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "notes": "Fast, free, good for most cases"
    },
    "free_quality": {
        "provider": "local",
        "model": "all-mpnet-base-v2",
        "dimension": 768,
        "notes": "Higher quality, still free, slightly slower"
    },
    "multilingual": {
        "provider": "local",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "notes": "Supports 50+ languages"
    },
    "openai": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimension": 1536,
        "notes": "High quality, costs money (~$0.02 per 1M tokens)"
    }
}


def print_embedding_options():
    """Print available embedding configurations."""
    print("\n" + "="*60)
    print("AVAILABLE EMBEDDING CONFIGURATIONS")
    print("="*60)

    for name, config in EMBEDDING_CONFIGS.items():
        print(f"\n{name.upper()}:")
        print(f"  Provider: {config['provider']}")
        print(f"  Model: {config['model']}")
        print(f"  Dimension: {config['dimension']}")
        print(f"  Notes: {config['notes']}")

    print("\n" + "="*60)
    print("\nUsage:")
    print("  export EMBEDDING_PROVIDER=local")
    print("  export EMBEDDING_MODEL=all-MiniLM-L6-v2")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_embedding_options()
