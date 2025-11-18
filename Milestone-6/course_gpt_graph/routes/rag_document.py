from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form

router = APIRouter()


@router.get("/rag/documents/{user_id}")
async def list_rag_documents(user_id: str):
    """List RAG documents for a given user id.

    Returns a list of document metadata for the provided `user_id`.
    """
    # TODO: implement listing of RAG documents by `user_id`
    pass


@router.post("/rag/documents/upload")
async def upload_rag_document(user_id: str = Form(...), file: UploadFile = File(...)):
    """Upload a RAG document (PDF) for a user.

    Expects `user_id` as form field and a PDF file in `file`.
    """
    # TODO: implement upload logic (validate PDF, store file, persist metadata)
    pass


@router.get("/rag/documents/{file_id}/view")
async def view_rag_document(file_id: str):
    """Return a presigned URL (or reference) for the given `file_id`.

    The implementation should generate and return a time-limited URL.
    """
    # TODO: implement presigned URL generation and return it
    pass


@router.delete("/rag/documents/{file_id}")
async def delete_rag_document(file_id: str):
    """Delete a RAG document by `file_id`.

    Should remove stored file and metadata.
    """
    # TODO: implement deletion of stored file and metadata
    pass
