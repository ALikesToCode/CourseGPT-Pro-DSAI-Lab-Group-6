"""OCR service template for Milestone 6 deployment."""


from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import docx
import easyocr


# Initialize once (same behavior as milestone 5)
READER = easyocr.Reader(["en"], gpu=False)


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
    raw_payload: Optional[Any] = None


class OCRService:

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".pdf", ".docx"}

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    # ----------------------------------------------------------------
    # PUBLIC: Main entry point
    # ----------------------------------------------------------------
    def process_document(self, path: Union[str, Path]) -> OCRDocumentResult:
        path = Path(path)
        media_type = self._detect_media_type(path)
        pages = list(self._iter_pages(path))

        page_results: List[OCRPageResult] = []

        for page_idx, page_payload in enumerate(pages):
            # detect text + confidence
            has_text, confidence = self._detect_text(page_payload)

            # extract text
            text = self._extract_text(page_payload) if has_text else ""

            # detect language (optional)
            language = self._detect_language(text) if text else None

            # metadata: bounding boxes + confidence per word
            meta = self._collect_metadata(page_payload, text)

            page_results.append(
                OCRPageResult(
                    page_index=page_idx,
                    has_text=has_text,
                    confidence=confidence,
                    text=text,
                    language=language,
                    metadata=meta,
                )
            )

        return OCRDocumentResult(
            source_path=path,
            media_type=media_type,
            pages=page_results,
        )

    # ----------------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------------

    def _detect_media_type(self, path: Path) -> str:
        mime, _ = mimetypes.guess_type(path.name)
        return mime or "application/octet-stream"

    def _iter_pages(self, path: Path) -> Iterable[Any]:
        """
        Returns one payload per page.
        - PDF -> Each page as PIL Image
        - DOCX -> Each page (text only as a string, because DOCX has no true pages)
        - Images -> Single image
        """

        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path}")

        # ---------------- PDF ----------------
        if suffix == ".pdf":
            pages = convert_from_path(str(path), dpi=200)
            for p in pages:
                yield ("pdf", p)
            return

        # ---------------- DOCX ----------------
        if suffix == ".docx":
            doc = docx.Document(str(path))
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text)
            yield ("docx", full_text)
            return

        # ---------------- Images ----------------
        img = Image.open(str(path)).convert("RGB")
        yield ("image", img)

    # ----------------------------
    # Text detection
    # ----------------------------
    def _detect_text(self, payload: Any) -> tuple[bool, float]:
        kind, data = payload

        # DOCX already contains plain text
        if kind == "docx":
            has = len(data.strip()) > 0
            return has, 1.0 if has else 0.0

        # PDF / IMAGE
        if kind in ("pdf", "image"):
            np_img = np.array(data)
            try:
                result = READER.readtext(np_img, detail=1)
            except:
                return False, 0.0

            if not result:
                return False, 0.0

            confidences = [float(x[2]) for x in result]
            avg = sum(confidences) / len(confidences)
            return True, avg

        return False, 0.0

    # ----------------------------
    # Text extraction
    # ----------------------------
    def _extract_text(self, payload: Any) -> str:
        kind, data = payload

        # DOCX extracted directly
        if kind == "docx":
            return data

        # IMAGE / PDF
        np_img = np.array(data)

        try:
            result = READER.readtext(np_img, detail=1)
        except:
            return ""

        lines = []
        for item in result:
            if len(item) >= 2 and item[1]:
                lines.append(item[1])

        return "\n".join(lines)

    # ----------------------------
    # Optional language detection
    # ----------------------------
    def _detect_language(self, text: str) -> Optional[str]:
        try:
            from langdetect import detect
            return detect(text)
        except:
            return None

    # ----------------------------
    # Metadata
    # ----------------------------
    def _collect_metadata(self, payload: Any, text: str) -> Dict[str, Any]:
        kind, data = payload

        if kind == "docx":
            return {"length": len(text)}

        np_img = np.array(data)

        try:
            result = READER.readtext(np_img, detail=1)
        except:
            return {}

        metadata = []
        for bbox, word, conf in result:
            metadata.append({
                "bbox": bbox,
                "text": word,
                "confidence": float(conf),
            })

        return {"words": metadata}
