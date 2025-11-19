
from fastapi import FastAPI

from routes import ai_search_router, files_router, health_router, graph_call

app = FastAPI(
    title="CourseGPT Graph",
    description="CourseGPT graph service with Cloudflare R2 + AI Search integrations",
    version="0.2.0",
)

app.include_router(health_router)
app.include_router(files_router)
app.include_router(ai_search_router)
app.include_router(graph_call.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
