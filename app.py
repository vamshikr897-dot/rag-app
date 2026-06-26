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
from rag_engine import ask, get_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Chroma collection...")
    app.state.collection = get_collection()
    logger.info("Collection ready.")
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(...)
    content: str = Field(..., min_length=1, max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    grade: int = Field(...)
    history: list[ChatTurn] = Field(default_factory=list, max_length=config.MAX_HISTORY_TURNS)

    @field_validator("grade")
    @classmethod
    def grade_must_be_valid(cls, v: int) -> int:
        if v not in config.GRADE_SOURCES:
            raise ValueError(f"grade must be one of {config.VALID_GRADES}")
        return v


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


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

    try:
        result = ask(question, request.app.state.collection, grade=payload.grade, history=history)
    except Exception:
        logger.exception("ask() failed for question: %r (grade=%s)", question, payload.grade)
        return JSONResponse(
            status_code=503,
            content={"error": "Sorry, I couldn't reach the textbook assistant right now. Please try again."},
        )

    return result


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "Something went wrong. Please try again."})


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
