
from fastapi import FastAPI

app = FastAPI(title="CourseGPT Graph")


@app.get("/")
async def read_root():
    return {"status": "ok", "message": "CourseGPT running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
