import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

import config
import feedback_db
from rag_engine import ask, get_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Chroma collection...")
    app.state.collection = get_collection()
    feedback_db.init_db()
    logger.info("Collection ready.")
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(...)
    content: str = Field(..., min_length=1, max_length=8000)


class RegeneratePayload(BaseModel):
    reason: Literal["not_relevant", "too_complicated", "too_short", "other"] = Field(...)
    detail: str | None = Field(None, max_length=1000)
    previous_answer: str = Field(..., min_length=1, max_length=8000)
    previous_interaction_id: str | None = Field(None, max_length=100)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    grade: int = Field(...)
    session_id: str = Field(..., min_length=1, max_length=100)
    history: list[ChatTurn] = Field(default_factory=list, max_length=config.MAX_HISTORY_TURNS)
    regenerate: RegeneratePayload | None = None

    @field_validator("grade")
    @classmethod
    def grade_must_be_valid(cls, v: int) -> int:
        if v not in config.GRADE_SOURCES:
            raise ValueError(f"grade must be one of {config.VALID_GRADES}")
        return v


class FeedbackRequest(BaseModel):
    interaction_id: str = Field(..., min_length=1)
    vote: Literal["up", "down"] = Field(...)
    reason: str | None = Field(None, max_length=100)
    detail: str | None = Field(None, max_length=1000)


@app.get("/")
def index(request: Request):
    static_version = int(
        max(
            (BASE_DIR / "static" / "app.js").stat().st_mtime,
            (BASE_DIR / "static" / "style.css").stat().st_mtime,
        )
    )
    return templates.TemplateResponse(request, "index.html", {"static_version": static_version})


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/grades")
def get_grades():
    return {"grades": config.VALID_GRADES}


@app.post("/api/ask")
def api_ask(payload: AskRequest, request: Request):
    question = payload.question.strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "Question cannot be empty."})

    history = [
        {**turn.model_dump(), "content": turn.content[: config.MAX_HISTORY_CONTENT_LEN]}
        for turn in payload.history
    ]

    regenerate = payload.regenerate.model_dump() if payload.regenerate else None

    try:
        result = ask(
            question,
            request.app.state.collection,
            grade=payload.grade,
            history=history,
            regenerate=regenerate,
        )
    except Exception:
        logger.exception("ask() failed for question: %r (grade=%s)", question, payload.grade)
        return JSONResponse(
            status_code=503,
            content={"error": "Sorry, I couldn't reach the textbook assistant right now. Please try again."},
        )

    response_category = result.get("response_category")
    interaction_id = feedback_db.log_interaction(
        session_id=payload.session_id,
        grade=payload.grade,
        question=question,
        answer=result["answer"],
        response_category=response_category,
        sources=result.get("sources", []),
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
        regenerated_from_id=regenerate.get("previous_interaction_id") if regenerate else None,
    )
    result["interaction_id"] = interaction_id if not response_category else None

    return result


@app.post("/api/feedback")
def api_feedback(payload: FeedbackRequest):
    found = feedback_db.record_feedback(
        payload.interaction_id, payload.vote, reason=payload.reason, detail=payload.detail
    )
    if not found:
        return JSONResponse(status_code=404, content={"error": "Interaction not found."})
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "Something went wrong. Please try again."})


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
