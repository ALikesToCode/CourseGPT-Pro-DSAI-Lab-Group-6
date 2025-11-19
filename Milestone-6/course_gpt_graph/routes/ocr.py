from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tempfile

from milestone_6.ocr_service.handler import OCRService

router = APIRouter(prefix="/ocr", tags=["ocr"])

ocr_service = OCRService()


@router.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Accepts a file, saves temporarily, runs OCR, returns results.
    """
    try:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".pdf", ".tiff", ".docx"}:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Use PNG/JPG/PDF/DOCX."
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        result = ocr_service.process_document(tmp_path)

        pages_output = []
        for p in result.pages:
            pages_output.append({
                "page_index": p.page_index,
                "has_text": p.has_text,
                "confidence": p.confidence,
                "text": p.text,
                "language": p.language,
                "metadata": p.metadata,
            })

        return {
            "source": str(result.source_path),
            "media_type": result.media_type,
            "pages": pages_output,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
