"""
Image OCR processing using local backends (no API keys required).
Supports RapidOCR, EasyOCR, and Tesseract.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class LocalOCREngine:
    """Local OCR engine without API dependencies."""

    def __init__(self, backend: str = "auto"):
        """
        Initialize OCR engine with specified backend.

        Args:
            backend: OCR backend - "rapidocr", "easyocr", "tesseract", or "auto"
        """
        self.backend = backend
        self._ocr_engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize OCR backend."""
        if self.backend == "auto":
            # Try backends in order of preference (speed + accuracy)
            for backend in ["rapidocr", "easyocr", "tesseract"]:
                if self._try_init_backend(backend):
                    self.backend = backend
                    logger.info(f"✓ Initialized OCR backend: {backend}")
                    return

            logger.warning("⚠ No OCR backend available - image text extraction disabled")
            logger.warning("  Install one: pip install rapidocr-onnxruntime OR easyocr OR pytesseract")
            self._ocr_engine = None
        else:
            if not self._try_init_backend(self.backend):
                logger.error(f"Failed to initialize OCR backend: {self.backend}")
                self._ocr_engine = None

    def _try_init_backend(self, backend: str) -> bool:
        """
        Try to initialize a specific OCR backend.

        Args:
            backend: Backend name

        Returns:
            True if successful, False otherwise
        """
        try:
            if backend == "rapidocr":
                from rapidocr_onnxruntime import RapidOCR
                self._ocr_engine = RapidOCR()
                logger.debug("RapidOCR initialized successfully")
                return True

            elif backend == "easyocr":
                import easyocr
                # Initialize with English, add more languages as needed
                self._ocr_engine = easyocr.Reader(['en'], gpu=False)
                logger.debug("EasyOCR initialized successfully")
                return True

            elif backend == "tesseract":
                import pytesseract
                # Test if tesseract is installed
                pytesseract.get_tesseract_version()
                self._ocr_engine = pytesseract
                logger.debug("Tesseract initialized successfully")
                return True

        except ImportError as e:
            logger.debug(f"Backend {backend} not available: {e}")
            return False
        except Exception as e:
            logger.debug(f"Failed to initialize {backend}: {e}")
            return False

        return False

    def extract_text(self, image_path: Path) -> Optional[str]:
        """
        Extract text from image using OCR.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text or None if OCR unavailable/failed
        """
        if self._ocr_engine is None:
            return None

        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Run OCR based on backend
            if self.backend == "rapidocr":
                result = self._ocr_engine(str(image_path))
                if result and len(result) > 0:
                    # RapidOCR returns list of (box, text, confidence)
                    texts = [item[1] for item in result if len(item) > 1]
                    extracted_text = ' '.join(texts)
                    return extracted_text.strip() if extracted_text else None

            elif self.backend == "easyocr":
                result = self._ocr_engine.readtext(np.array(image))
                # EasyOCR returns list of (box, text, confidence)
                texts = [item[1] for item in result]
                extracted_text = ' '.join(texts)
                return extracted_text.strip() if extracted_text else None

            elif self.backend == "tesseract":
                import pytesseract
                text = pytesseract.image_to_string(image)
                return text.strip() if text.strip() else None

        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return None

        return None


class ImageProcessor:
    """Process images referenced in markdown files."""

    def __init__(self, enable_ocr: bool = True, ocr_backend: str = "auto"):
        """
        Initialize image processor.

        Args:
            enable_ocr: Enable OCR text extraction
            ocr_backend: OCR backend to use
        """
        self.enable_ocr = enable_ocr
        self.ocr_engine = LocalOCREngine(backend=ocr_backend) if enable_ocr else None

    async def process_images(
        self,
        file_path: Path,
        image_refs: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Process all images referenced in a markdown file.

        Args:
            file_path: Path to markdown file
            image_refs: List of image references from markdown

        Returns:
            List of enriched image metadata with OCR text
        """
        enriched_images = []

        for img_ref in image_refs:
            img_data = await self._process_single_image(file_path, img_ref)
            if img_data:
                enriched_images.append(img_data)

        return enriched_images

    async def _process_single_image(
        self,
        markdown_path: Path,
        img_ref: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single image reference.

        Args:
            markdown_path: Path to the markdown file
            img_ref: Image reference dict with 'path' and 'alt'

        Returns:
            Enriched image metadata or None if processing fails
        """
        img_path_str = img_ref['path']

        # Handle external URLs (no OCR)
        if img_path_str.startswith('http://') or img_path_str.startswith('https://'):
            return {
                'path': img_path_str,
                'alt': img_ref.get('alt', ''),
                'type': 'external',
                'ocr_text': None
            }

        # Resolve local image path
        try:
            # Handle relative paths
            if not Path(img_path_str).is_absolute():
                img_path = (markdown_path.parent / img_path_str).resolve()
            else:
                img_path = Path(img_path_str)

            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                return None

        except Exception as e:
            logger.error(f"Invalid image path {img_path_str}: {e}")
            return None

        # Get image metadata
        try:
            image = Image.open(img_path)
            width, height = image.size
            format_type = image.format
        except Exception as e:
            logger.error(f"Could not open image {img_path}: {e}")
            return None

        # Run OCR if enabled
        ocr_text = None
        if self.enable_ocr and self.ocr_engine and self.ocr_engine._ocr_engine:
            try:
                ocr_text = self.ocr_engine.extract_text(img_path)
                if ocr_text:
                    logger.debug(f"OCR extracted {len(ocr_text)} chars from {img_path.name}")
            except Exception as e:
                logger.error(f"OCR failed for {img_path}: {e}")

        # Get relative path for storage
        try:
            relative_path = str(img_path.relative_to(markdown_path.parent))
        except ValueError:
            # If image is outside markdown directory
            relative_path = img_path.name

        return {
            'path': relative_path,
            'full_path': str(img_path),
            'alt': img_ref.get('alt', ''),
            'type': 'local',
            'width': width,
            'height': height,
            'format': format_type,
            'ocr_text': ocr_text,
            'size_bytes': img_path.stat().st_size,
            'has_text': bool(ocr_text)
        }


if __name__ == "__main__":
    # Test OCR on an image
    import sys
    import asyncio

    async def test_ocr(image_path: str):
        """Test OCR on a single image."""
        img_path = Path(image_path)

        if not img_path.exists():
            print(f"Image not found: {img_path}")
            return

        processor = ImageProcessor(enable_ocr=True, ocr_backend="auto")

        img_ref = {'path': str(img_path), 'alt': 'Test image'}
        result = await processor._process_single_image(img_path.parent / "dummy.md", img_ref)

        if result:
            print(f"\n✓ Image processed: {img_path.name}")
            print(f"  Size: {result['width']}x{result['height']}")
            print(f"  Format: {result['format']}")
            if result['ocr_text']:
                print(f"  OCR Text ({len(result['ocr_text'])} chars):")
                print(f"  {result['ocr_text'][:200]}...")
            else:
                print("  No text detected")
        else:
            print(f"✗ Failed to process image")

    if len(sys.argv) > 1:
        asyncio.run(test_ocr(sys.argv[1]))
    else:
        print("Usage: python image_ocr.py <image_path>")
