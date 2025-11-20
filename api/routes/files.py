from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from api.dependencies import get_r2_service
from api.services.r2_storage import R2StorageService

router = APIRouter(prefix="/files", tags=["files"])


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document to Cloudflare R2",
)
async def upload_file(
    file: UploadFile = File(...),
    prefix: Optional[str] = Form(default=None, description="Optional prefix/folder for the object key"),
    r2_service: R2StorageService = Depends(get_r2_service),
):
    """
    Streams the uploaded file to Cloudflare R2 so it can later be ingested by AI Search.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    safe_prefix = (prefix or "").strip().strip("/")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    extension = Path(file.filename).suffix
    generated_name = f"{timestamp}-{uuid4().hex}{extension}"
    key = f"{safe_prefix}/{generated_name}" if safe_prefix else generated_name

    result = r2_service.upload_fileobj(
        file.file,
        key=key,
        content_type=file.content_type,
        metadata={"original_filename": file.filename},
    )
    return {"message": "uploaded", "file": result}


@router.get("/", summary="List objects stored in Cloudflare R2")
async def list_files(
    prefix: Optional[str] = None,
    limit: int = 50,
    r2_service: R2StorageService = Depends(get_r2_service),
):
    if limit <= 0 or limit > 1000:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")

    files = r2_service.list_objects(prefix=prefix, max_items=limit)
    return {"files": files}


@router.get("/view/{object_key:path}", summary="Generate a temporary view URL for an object")
async def view_file(
    object_key: str,
    expires_in: int = Query(900, ge=60, le=3600, description="Seconds before the URL expires"),
    r2_service: R2StorageService = Depends(get_r2_service),
):
    if not object_key:
        raise HTTPException(status_code=400, detail="Object key is required")

    try:
        url = r2_service.generate_presigned_get_url(object_key, expires_in=expires_in)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"url": url, "expires_in": expires_in}


@router.delete("/{object_key:path}", summary="Delete an object from Cloudflare R2")
async def delete_file(
    object_key: str,
    r2_service: R2StorageService = Depends(get_r2_service),
):
    if not object_key:
        raise HTTPException(status_code=400, detail="Object key is required")
    r2_service.delete_object(object_key)
    return {"message": "deleted", "key": object_key}
