# Curiosity Coach

A grade-aware RAG study assistant for NCERT Science (Grades 6-10) — ask a question, get an answer grounded in your actual textbook, with sources.

## Features

- **Grade-aware retrieval** — a single Chroma vector store holds all five grades, filtered by grade at query time so each student only gets content from their own book.
- **Conversational memory** — short-term chat history is sent with each question; an LLM-based query condenser resolves follow-ups like "explain that more" without losing the original wording for the answer itself.
- **Chapter tools** — "What's in Chapter 6?" triggers a metadata-filtered fetch across the whole chapter (not just top-3 similarity search) with a deterministic heading; asking about an invalid chapter, or for a full chapter list, returns a generated directory instead of a generic refusal.
- **Cross-grade awareness** — if a question matches content from a *different* grade's book, the app says so directly ("you'll cover this in Grade 9" / "this was covered in Grade 8") instead of just failing.
- **Query-intent classification** — typos get a "did you mean...?" suggestion, vague questions get clarifying options, off-topic questions get a friendly redirect, gibberish and inappropriate input are handled distinctly — only genuinely unanswerable in-scope questions fall back to the plain refusal message.
- **Math rendering** — LaTeX-formatted formulas in answers (`\(...\)`, `\[...\]`) render as proper typeset math via a self-hosted KaTeX.
- **Source attribution** — answers cite chapter + page, grouped and page-range-collapsed per chapter (no duplicate per-page spam for long chapter summaries).
- **Feedback loop** — every answer gets a thumbs up/down. A thumbs down asks why (not relevant, too complicated, too short, other) and automatically regenerates the answer using that signal (e.g. pulling more chunks, simplifying, adding detail), capped at 2 retries per question. Every interaction is logged to a local SQLite DB (`feedback.db`) for analytics, independent of what's shown in the UI.
- **Remembers your grade** — the selected grade persists across reloads (`localStorage`), so a student isn't asked to re-pick it every visit.
- **Voice input** — a microphone button next to the send button transcribes speech live into the question box (via the browser's native Web Speech API), so students can ask questions out loud instead of typing. No server-side audio processing or API key involved.

## Tech stack

- **Backend**: FastAPI, ChromaDB (vector store), `sentence-transformers` (embeddings), Ollama-hosted LLM (`gpt-oss:120b` by default) for generation.
- **Ingestion**: `pypdf` for text extraction, a sentence-aware chunker, per-chapter metadata tagging.
- **Feedback/analytics**: SQLite (`feedback.db`), no ORM — a small, dependency-free store for interaction + feedback logging.
- **Frontend**: vanilla HTML/CSS/JS — no build step, no framework. KaTeX is vendored locally under `static/vendor/`.

## Project structure

```
config.py       Central configuration (paths, chunking, model, thresholds)
ingest.py       PDF -> chunks -> Chroma vector store (run once per grade)
rag_engine.py   Retrieval + prompting + LLM calls + query-intent handling
app.py          FastAPI routes
feedback_db.py  SQLite-backed interaction logging + thumbs up/down feedback
cli.py          Minimal terminal client (ungraded, for quick local testing)
templates/      Jinja2 HTML shell
static/         Frontend JS/CSS + vendored KaTeX
```

## Setup

1. **Clone and create a virtual environment**
   ```bash
   git clone <your-repo-url>
   cd rag-app
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   pip install -r requirements.txt
   ```

2. **Configure your API key**
   ```bash
   copy .env.example .env       # Windows
   cp .env.example .env         # macOS/Linux
   ```
   Edit `.env` and set `OLLAMA_API_KEY` (from [ollama.com](https://ollama.com)).

3. **Add textbook PDFs**

   This repo does **not** include any NCERT textbook PDFs or the generated vector database — see [Content & Copyright](#content--copyright) below. Download the official NCERT Science PDFs for whichever grade(s) you want from [ncert.nic.in](https://ncert.nic.in), and place them under:
   ```
   data/grade6/   data/grade7/   data/grade8/   data/grade9/   data/grade10/
   ```
   (Each grade's folder is independently configurable via env vars — see `.env.example` — if you'd rather keep PDFs elsewhere.)

4. **Run ingestion** (once per grade you've added)
   ```bash
   python ingest.py --grade 8
   ```

5. **Start the app**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:8000`.

## Content & Copyright

NCERT textbooks are not included in this repository — only the code that processes them. `chroma_db/` (the generated vector store, which would contain extracted textbook text) is gitignored and must be built locally via `ingest.py` from your own legally-obtained copies of the textbooks. Source PDFs are expected to live under `data/gradeN/`, which is also gitignored.

## Known limitations

- English-only; no support for regional-language NCERT editions.
- Single shared embedding model/collection for all grades — works well at this corpus size but isn't optimized for much larger content sets.
- No automated test suite yet.
- Voice input relies on the browser's built-in `SpeechRecognition` API, which Firefox desktop doesn't support — the mic button is automatically hidden there, with typing as the fallback.

## License

MIT — see [LICENSE](LICENSE).
