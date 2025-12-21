"""
Advanced Docling serialization strategies.

Implements custom serializers following Docling's best practices:
- Custom table serialization (Markdown format)
- Picture annotation serialization
- Configurable chunking serialization
"""

import logging
from typing import Any

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownPictureSerializer,
    MarkdownTableSerializer,
    MarkdownParams,
)
from docling_core.types.doc.document import (
    DoclingDocument,
    PictureClassificationData,
    PictureDescriptionData,
    PictureMoleculeData,
    PictureItem,
)
from typing_extensions import override

logger = logging.getLogger(__name__)


class AnnotationPictureSerializer(MarkdownPictureSerializer):
    """
    Picture serializer that includes AI-generated annotations.

    This serializer extracts:
    - Picture descriptions (from VLM)
    - Picture classifications
    - Molecular structures (SMILES) if present

    Makes images searchable in RAG!
    """

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """Serialize picture with annotations."""
        text_parts: list[str] = []

        for annotation in item.annotations:
            if isinstance(annotation, PictureClassificationData):
                # Get predicted class (e.g., "diagram", "chart", "screenshot")
                predicted_class = (
                    annotation.predicted_classes[0].class_name
                    if annotation.predicted_classes
                    else None
                )
                if predicted_class is not None:
                    text_parts.append(f"**Image type:** {predicted_class}")

            elif isinstance(annotation, PictureMoleculeData):
                # Chemical molecule (SMILES notation)
                text_parts.append(f"**Chemical structure (SMILES):** {annotation.smi}")

            elif isinstance(annotation, PictureDescriptionData):
                # VLM-generated description
                text_parts.append(f"**Image description:** {annotation.text}")

        # Fallback if no annotations
        if not text_parts:
            text_parts.append("*[Image: No description available]*")

        text_res = "\n".join(text_parts)
        text_res = doc_serializer.post_process(text=text_res)

        return create_ser_result(text=text_res, span_source=item)


class EnhancedSerializerProvider(ChunkingSerializerProvider):
    """
    Enhanced serialization provider for better RAG quality.

    Features:
    - Markdown tables (better for LLMs than triplet notation)
    - Picture annotations (makes images searchable)
    - Custom image placeholder
    - Configurable parameters
    """

    def __init__(
        self,
        use_markdown_tables: bool = True,
        use_picture_annotations: bool = True,
        image_placeholder: str = "<!-- image -->",
    ):
        """
        Initialize serializer provider.

        Args:
            use_markdown_tables: Use Markdown table format (recommended)
            use_picture_annotations: Include picture annotations
            image_placeholder: Placeholder for images without annotations
        """
        self.use_markdown_tables = use_markdown_tables
        self.use_picture_annotations = use_picture_annotations
        self.image_placeholder = image_placeholder

    def get_serializer(self, doc: DoclingDocument):
        """Get configured serializer for document."""
        # Configure table serializer
        table_serializer = (
            MarkdownTableSerializer() if self.use_markdown_tables else None
        )

        # Configure picture serializer
        if self.use_picture_annotations:
            picture_serializer = AnnotationPictureSerializer()
        else:
            picture_serializer = None

        # Markdown parameters
        params = MarkdownParams(
            image_placeholder=self.image_placeholder,
        )

        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=table_serializer,
            picture_serializer=picture_serializer,
            params=params,
        )


class SimpleSerializerProvider(ChunkingSerializerProvider):
    """
    Simple serializer for maximum compatibility.

    Use when you don't need fancy formatting.
    """

    def get_serializer(self, doc: DoclingDocument):
        """Get basic serializer."""
        return ChunkingDocSerializer(doc=doc)


def create_serializer_provider(
    mode: str = "enhanced",
    **kwargs
) -> ChunkingSerializerProvider:
    """
    Factory function to create serializer provider.

    Args:
        mode: "enhanced" (recommended) or "simple"
        **kwargs: Additional parameters for EnhancedSerializerProvider

    Returns:
        Serializer provider instance

    Examples:
        # Enhanced mode (recommended for RAG)
        provider = create_serializer_provider("enhanced")

        # Enhanced with custom settings
        provider = create_serializer_provider(
            "enhanced",
            use_markdown_tables=True,
            use_picture_annotations=True
        )

        # Simple mode (basic)
        provider = create_serializer_provider("simple")
    """
    if mode == "enhanced":
        return EnhancedSerializerProvider(**kwargs)
    elif mode == "simple":
        return SimpleSerializerProvider()
    else:
        raise ValueError(f"Unknown serializer mode: {mode}")


# Configuration presets
SERIALIZER_CONFIGS = {
    "rag_optimized": {
        "mode": "enhanced",
        "use_markdown_tables": True,
        "use_picture_annotations": True,
        "image_placeholder": "*[Image]*",
        "notes": "Best for RAG - tables and images are searchable"
    },
    "markdown_compatible": {
        "mode": "enhanced",
        "use_markdown_tables": True,
        "use_picture_annotations": False,
        "image_placeholder": "<!-- image -->",
        "notes": "Clean Markdown without image descriptions"
    },
    "simple": {
        "mode": "simple",
        "notes": "Basic serialization, maximum compatibility"
    }
}


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SERIALIZER CONFIGURATIONS")
    print("="*60)

    for name, config in SERIALIZER_CONFIGS.items():
        print(f"\n{name.upper()}:")
        for key, value in config.items():
            if key != "notes":
                print(f"  {key}: {value}")
        print(f"  Notes: {config['notes']}")

    print("\n" + "="*60)
    print("\nUsage:")
    print("  from indexer.docling_serializers import create_serializer_provider")
    print("  provider = create_serializer_provider('enhanced')")
    print("="*60 + "\n")
