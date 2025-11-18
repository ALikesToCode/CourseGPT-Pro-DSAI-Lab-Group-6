from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/", summary="Service health check")
async def health_check():
    """
    Simple health endpoint so upstream services can verify the API is running.
    """
    return {"status": "ok", "message": "CourseGPT graph service running"}

