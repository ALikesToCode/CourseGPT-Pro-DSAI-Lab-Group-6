from typing import Optional, Any, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from langchain.messages import HumanMessage

from course_gpt_graph.graph.states.main_state import CourseGPTState
from course_gpt_graph.graph.graph import graph as course_graph

router = APIRouter()


def _get_state_field(result_state, field):
    if isinstance(result_state, dict):
        return result_state.get(field)
    return getattr(result_state, field, None)


def _extract_latest_message(messages):
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if hasattr(last, "content"):
            return last.content
        if isinstance(last, dict):
            return last.get("content") or last.get("text") or str(last)
        return str(last)
    if isinstance(messages, str):
        return messages
    return None


@router.post("/chat")
async def graph_ask(
    prompt: str = Form(...),
    thread_id: str = Form(...),
    user_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    """Accepts form-data (multipart) with an optional file upload.

    Fields:
    - `prompt`: the user prompt
    - `thread_id`: thread identifier
    - `user_id`: user identifier
    - `file`: optional file uploaded with the request
    """

    # TODO: fetch RAG content based on user_id or on uploaded file
    # TODO: process the uploaded file if provided using OCR module

    try:
        # if a file was uploaded, enforce PDF-only and attach bytes+metadata
        if file is not None:
            filename = (file.filename or "").lower()
            is_pdf = (file.content_type == "application/pdf") or filename.endswith(".pdf")
            if not is_pdf:
                raise HTTPException(status_code=400, detail="Only PDF file uploads are accepted")

        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                # TODO: include RAG content here if available
                # TODO: process the textual content extracted from the uploaded file
            }
        }

        # invoke the compiled state graph with a HumanMessage
        result_state = course_graph.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )

        messages = _get_state_field(result_state, "messages")
        latest_message = _extract_latest_message(messages)

        return {"latest_message": latest_message}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
