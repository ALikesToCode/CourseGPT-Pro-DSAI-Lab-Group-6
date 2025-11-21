from pathlib import Path
import sys
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mangum import Mangum

# Ensure repository root is on sys.path so `api.*` imports resolve when running from /api
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from api.routes import ai_search_router, files_router, health_router, graph_call

app = FastAPI(
    title="CourseGPT Graph",
    description="CourseGPT graph service with Cloudflare R2 + AI Search integrations",
    version="0.2.0",
)

app.include_router(health_router)
app.include_router(files_router)
app.include_router(ai_search_router)
app.include_router(graph_call.router, prefix="/graph")

app.mount("/static", StaticFiles(directory=repo_root / "api" / "static"), name="static")
templates = Jinja2Templates(directory=repo_root / "api" / "templates")

# Expose AWS Lambda-compatible handler so Vercel's Python runtime can invoke
# the FastAPI app as a serverless function.
handler = Mangum(app)


@app.get("/")
async def get_chat_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
