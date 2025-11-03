"""OCR service template for Milestone 6 deployment."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


@dataclass
class OCRPageResult:
    page_index: int
    has_text: bool
    confidence: float
    text: str
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OCRDocumentResult:
    source_path: Path
    media_type: str
    pages: List[OCRPageResult]
    raw_payload: Optional[Any] = None  # Store engine-specific details if needed.


class OCRService:
    """Stub class encapsulating text detection/extraction logic.

    Replace the TODO blocks with your chosen OCR backend (e.g., PaddleOCR, Azure,
    Google Vision, custom models). This design allows the router or UI layer to
    call `process_document` without knowing implementation details.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".pdf", ".docx"}

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        # TODO: Initialise OCR engines / API clients here.

    # --- Public API ---------------------------------------------------------

    def process_document(self, path: Union[str, Path]) -> OCRDocumentResult:
        path = Path(path)
        media_type = self._detect_media_type(path)
        pages = list(self._iter_pages(path))

        page_results: List[OCRPageResult] = []
        for page_idx, page_payload in enumerate(pages):
            has_text, confidence = self._detect_text(page_payload)
            text = self._extract_text(page_payload) if has_text else ""
            language = self._detect_language(text) if text else None
            page_results.append(
                OCRPageResult(
                    page_index=page_idx,
                    has_text=has_text,
                    confidence=confidence,
                    text=text,
                    language=language,
                    metadata=self._collect_metadata(page_payload, text),
                )
            )

        return OCRDocumentResult(
            source_path=path,
            media_type=media_type,
            pages=page_results,
        )

    # --- Internal helpers (replace with real logic) ------------------------

    def _detect_media_type(self, path: Path) -> str:
        mime, _ = mimetypes.guess_type(path.name)
        return mime or "application/octet-stream"

    def _iter_pages(self, path: Path) -> Iterable[Any]:
        """Yield page payloads depending on file type.

        Replace with actual PDF/DOCX parsing (e.g., pdfplumber, python-docx) and
        image loading (Pillow, OpenCV). This stub simply yields the path itself.
        """
        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path}")
        # TODO: handle PDFs/DOCX by iterating over pages.
        yield path

    def _detect_text(self, page_payload: Any) -> tuple[bool, float]:
        """Return text-present flag and confidence."""
        # TODO: implement classification logic.
        return True, 0.0

    def _extract_text(self, page_payload: Any) -> str:
        """Run OCR extraction."""
        # TODO: integrate with OCR engine.
        return ""

    def _detect_language(self, text: str) -> Optional[str]:
        """Optional language detection hook."""
        # TODO: integrate with fastText/langdetect if needed.
        return None

    def _collect_metadata(self, page_payload: Any, text: str) -> Dict[str, Any]:
        """Capture engine-specific metadata (bounding boxes, confidence arrays, etc.)."""
        # TODO: populate metadata dictionary.
        return {}
