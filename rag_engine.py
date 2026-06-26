import json
import re

import chromadb
from better_profanity import profanity
from chromadb.utils import embedding_functions
from ollama import Client

import config

profanity.load_censor_words()

CHAPTER_REFERENCE_RE = re.compile(r"\bchapter\s+(\d+)\b", re.IGNORECASE)

CHAPTER_LIST_INTENT_RE = re.compile(
    r"\b(all|which|what|how many|list)\b[^.?!]{0,20}\bchapters?\b"
    r"|\bchapters?\b[^.?!]{0,20}\b(covered|available|curriculum|book|list)\b",
    re.IGNORECASE,
)

EXAM_CHEATING_RE = re.compile(
    r"\b(answers?|solutions?)\s+(for|to)\s+(my|the)\s+(exam|test|quiz)\b"
    r"|\bcheat(ing)?\s+(on|in)\s+(my|the|an?)\s+(exam|test|quiz)\b",
    re.IGNORECASE,
)

DIAGRAM_REFERENCE_RE = re.compile(
    r"\b(figure|diagram)\b|\b(this|the)\s+(picture|image|photo)\s+(shows?|depicts?)\b",
    re.IGNORECASE,
)

SAFETY_NET_RESPONSE = (
    "Let's keep things friendly and focused on learning! I'm not able to help with that, "
    "but I'm happy to help you understand any Science topic — what would you like to explore?"
)

DIAGRAM_FALLBACK_MESSAGE = (
    "I can only read text from your textbook, not see diagrams, figures, or pictures. "
    "Try describing what you see, or ask me about the underlying concept instead!"
)

GIBBERISH_MESSAGE = "Hmm, that doesn't look like a real question to me! Could you try typing it again?"

OFF_TOPIC_FALLBACK_MESSAGES = {
    "off_topic_casual": "Ha, I'd love to chat about that, but my circuits are wired for Science only! Got a Science question for me?",
    "personal_chitchat": "I'm Curiosity Coach, your Science study buddy! What would you like to learn about today?",
}

CLASSIFICATION_CATEGORIES = {
    "ambiguous",
    "typo",
    "gibberish",
    "off_topic_academic",
    "off_topic_casual",
    "personal_chitchat",
    "genuinely_not_found",
}

# Kept deliberately narrow (category + two short structural fields, no free-form "answer" text):
# testing showed the model frequently DROPS the required "category" field when also asked to write
# a creative answer in the same JSON response. Splitting classification from creative text generation
# (see _generate_redirect_line) made category-field compliance reliable.
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": sorted(CLASSIFICATION_CATEGORIES)},
        "suggestions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "subject_guess": {"type": "string"},
    },
    "required": ["category", "suggestions", "subject_guess"],
}

CLASSIFICATION_SYSTEM_PROMPT = """A student in Grade {grade} asked a Science study assistant a question, and \
the assistant could not find an answer in the Grade {grade} NCERT Science textbook. Classify the original \
question into exactly one category below, and fill "suggestions"/"subject_guess" only where noted (otherwise \
leave them as [] / ""). Respond with ONLY a JSON object matching the schema -- no extra text, no explanation.

CATEGORIES (pick exactly one):
- "ambiguous": vague or underspecified (e.g. "what is the formula", "explain the chapter" with no number, \
"tell me about it" with no context). Set "suggestions" to 2-3 concrete, specific rephrased questions a Grade \
{grade} Science student might have actually meant.
- "typo": a recognizable Science term or question with a spelling mistake (e.g. "fotosynthesis", "nwtons \
laws"). Set "suggestions" to a list with EXACTLY ONE item: the question rewritten with correct spelling.
- "gibberish": not real words, or empty/meaningless (e.g. "asdkjhasdkj", "???", random keysmash).
- "off_topic_academic": a genuine school question but for a DIFFERENT subject than Science -- Maths, Social \
Studies/History/Geography/Civics, English/Language, Computer Science, etc. Set "subject_guess" to the likely \
subject name (e.g. "Maths", "Social Studies", "English").
- "off_topic_casual": about something outside school entirely -- movies, sports, games, celebrities, personal \
opinions, jokes, etc.
- "personal_chitchat": small talk directed at the assistant itself -- greetings, "who are you", "how are you", \
"are you a robot", etc.
- "genuinely_not_found": the question IS a fair, specific, in-scope Science question for roughly this grade \
level, but the textbook truly doesn't seem to cover it, and none of the above categories fit.

If you are unsure between two categories, prefer "genuinely_not_found" over guessing wrong with confidence."""

REDIRECT_SYSTEM_PROMPTS = {
    "off_topic_casual": """You are Curiosity Coach, a friendly Science study buddy for a Grade {grade} student. \
The student just asked something off-topic (not about school Science). Write ONE short, warm, playful sentence \
that gently redirects them back to asking a Science question -- light humor tailored to what they asked about \
is good, but never mock or scold the student. Respond with ONLY that one sentence, nothing else.""",
    "personal_chitchat": """You are Curiosity Coach, a friendly Science study buddy for a Grade {grade} student. \
The student just said something conversational directed at you (a greeting, asking who/what you are, etc). \
Write ONE short, warm, in-character reply, ending with a gentle nudge to ask a Science question. Respond with \
ONLY that one reply, nothing else.""",
}

CONDENSE_SYSTEM_PROMPT = """Given a short recent conversation and a new follow-up question, decide if the \
follow-up depends on the prior conversation (e.g. uses pronouns, says "explain more", "what about...", \
or omits a subject already discussed).

If it depends on the prior conversation, rewrite it as a standalone question that includes the missing \
context, using only information already present in the conversation. If it does NOT depend on the prior \
conversation (it's already a self-contained question), return it completely unchanged.

Respond with ONLY the resulting question text, nothing else."""

GRADE_BOOK_NAMES = {
    6: 'NCERT "Curiosity: Textbook of Science for Grade 6"',
    7: 'NCERT "Curiosity: Textbook of Science for Grade 7"',
    8: 'NCERT "Curiosity: Textbook of Science for Grade 8"',
    9: 'NCERT "Curiosity: Textbook of Science for Grade 9"',
    10: 'the NCERT "Science" textbook for Class 10',
}

REFUSAL_STRING = "I couldn't find that in your textbook, but it's a great question — try asking your teacher!"

# A short, structurally distinct marker is far more reliably reproduced by the model than a full
# natural-language sentence (LLM sampling can vary punctuation/quote style call to call, breaking
# exact-string matching on the user-facing refusal text). The actual REFUSAL_STRING is substituted
# in by ask() in Python whenever this marker is detected and no better fallback applies.
NOT_FOUND_MARKER = "NOT_FOUND"


def build_system_prompt(grade: int = None) -> str:
    if grade is not None and grade in GRADE_BOOK_NAMES:
        book_ref = f"a Grade {grade} student using {GRADE_BOOK_NAMES[grade]}"
    else:
        book_ref = "a student using their NCERT Science textbook"

    return f"""You are Curiosity Coach, a friendly study companion for {book_ref}. \
Answer questions using only the textbook excerpts provided below.

RULES:
- Only answer based on the provided context. Do not use outside knowledge.
- If the answer is not in the context, respond with exactly this and nothing else: {NOT_FOUND_MARKER}
- Do not cite, mention, or reference chapter numbers, chapter titles, or page numbers anywhere in your answer text — the app displays sources separately. Just answer the question directly.
- Keep answers clear and age-appropriate, with examples where helpful."""

_client = Client(
    host=config.OLLAMA_HOST,
    headers={"Authorization": f"Bearer {config.OLLAMA_API_KEY}"},
)


def get_collection(chunk_size: int = None, chunk_overlap: int = None):
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )
    collection_name = config.collection_name_for(chunk_size, chunk_overlap)
    return client.get_collection(name=collection_name, embedding_function=ef)


def detect_chapter_reference(query: str) -> int | None:
    match = CHAPTER_REFERENCE_RE.search(query)
    return int(match.group(1)) if match else None


def condense_query(query: str, history: list[dict]) -> str:
    if not history:
        return query

    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    try:
        response = _client.chat(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
                {"role": "user", "content": f"CONVERSATION:\n{history_text}\n\nFOLLOW-UP QUESTION:\n{query}"},
            ],
        )
        rewritten = (response.message.content or "").strip()
        return rewritten if rewritten else query
    except Exception:
        return query


def retrieve_chapter_context(
    collection, grade: int, chapter_number: int, max_chunks: int = None
) -> list[dict]:
    max_chunks = max_chunks or config.CHAPTER_CHUNK_CAP
    results = collection.get(
        where={"$and": [{"grade": grade}, {"chapter_number": chapter_number}]},
        include=["documents", "metadatas"],
    )
    documents = results["documents"]
    metadatas = results["metadatas"]
    if not documents:
        return []

    paired = list(zip(documents, metadatas))
    # One representative chunk per page gives breadth across the whole chapter
    # instead of depth on just the first few pages.
    representative = [(d, m) for d, m in paired if m.get("chunk_index") == 0]
    pool = representative if representative else paired

    if len(pool) > max_chunks:
        step = len(pool) / max_chunks
        pool = [pool[int(i * step)] for i in range(max_chunks)]

    pool.sort(key=lambda pair: pair[1]["page"])
    return [{"text": text, **meta} for text, meta in pool]


def get_chapter_directory(collection, grade: int) -> list[dict]:
    results = collection.get(where={"grade": grade}, include=["metadatas"])
    metadatas = results["metadatas"]

    seen = {}
    for meta in metadatas:
        number = meta.get("chapter_number")
        if number == -1:
            continue
        if number not in seen:
            seen[number] = meta.get("chapter_title")

    return [{"number": n, "title": seen[n]} for n in sorted(seen)]


def format_chapter_directory_answer(grade: int, chapters: list[dict], invalid_chapter: int = None) -> str:
    if invalid_chapter is not None:
        intro = (
            f"Chapter {invalid_chapter} doesn't exist for Grade {grade} -- "
            f"there are {len(chapters)} chapters in this book. Here they are:"
        )
    else:
        intro = f"Here are all {len(chapters)} chapters for Grade {grade}:"

    lines = [intro, ""]
    lines.extend(f"{c['number']}. {c['title']}" for c in chapters)
    return "\n".join(lines)


def retrieve_context(query: str, collection, top_k: int = None, grade: int = None) -> list[dict]:
    top_k = top_k or config.TOP_K
    query_kwargs = {"query_texts": [query], "n_results": top_k}
    if grade is not None:
        query_kwargs["where"] = {"$and": [{"grade": grade}, {"chapter_number": {"$ne": -1}}]}
    else:
        query_kwargs["where"] = {"chapter_number": {"$ne": -1}}
    results = collection.query(**query_kwargs)

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    for text, meta in zip(documents, metadatas):
        chunks.append({"text": text, **meta})
    return chunks


def build_prompt(query: str, chunks: list[dict]) -> str:
    labeled_chunks = []
    for chunk in chunks:
        if chunk["chapter_number"] != -1:
            label = f"[Chapter {chunk['chapter_number']}: {chunk['chapter_title']}, p.{chunk['page']}]"
        else:
            label = f"[{chunk['chapter_title']}, p.{chunk['page']}]"
        labeled_chunks.append(f"{label}\n{chunk['text']}")

    context = "\n\n---\n\n".join(labeled_chunks)
    return f"""TEXTBOOK CONTEXT:
{context}

STUDENT QUESTION:
{query}

Answer based only on the textbook context above."""


def _directory_response(
    answer: str, suggestions: list[str] | None = None, response_category: str | None = None
) -> dict:
    return {
        "answer": answer,
        "sources": [],
        "chapter_heading": None,
        "suggestions": suggestions or [],
        "response_category": response_category,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _check_safety_net(query: str) -> dict | None:
    if profanity.contains_profanity(query) or EXAM_CHEATING_RE.search(query):
        return _directory_response(SAFETY_NET_RESPONSE, response_category="safety")
    return None


def _cross_grade_lookup(query: str, collection, student_grade: int) -> dict | None:
    results = collection.query(
        query_texts=[query],
        n_results=config.CROSS_GRADE_TOP_K,
        where={"$and": [{"grade": {"$ne": student_grade}}, {"chapter_number": {"$ne": -1}}]},
    )
    distances = results["distances"][0]
    if not distances or distances[0] >= config.CROSS_GRADE_DISTANCE_THRESHOLD:
        return None

    meta = results["metadatas"][0][0]
    other_grade = meta["grade"]
    chapter_title = meta["chapter_title"]

    if other_grade > student_grade:
        answer = (
            f"This isn't part of the Grade {student_grade} curriculum yet, but you'll cover it in Grade "
            f"{other_grade} (in the chapter on {chapter_title})! Keep that curiosity going — "
            f"you'll get there soon."
        )
    else:
        answer = (
            f"This isn't part of the Grade {student_grade} curriculum, but it was already covered in Grade "
            f"{other_grade} (in the chapter on {chapter_title}) — a great one to revisit if you'd like a "
            f"refresher!"
        )
    return _directory_response(answer, response_category="cross_grade")


def _classify_refusal(query: str, grade: int) -> dict | None:
    try:
        response = _client.chat(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT.format(grade=grade)},
                {"role": "user", "content": query},
            ],
            format=CLASSIFICATION_SCHEMA,
        )
        parsed = json.loads(response.message.content)
    except Exception:
        return None

    category = parsed.get("category")
    if category not in CLASSIFICATION_CATEGORIES:
        return None

    return {
        "category": category,
        "suggestions": parsed.get("suggestions") or [],
        "subject_guess": parsed.get("subject_guess") or "",
    }


def _generate_redirect_line(category: str, query: str, grade: int) -> str:
    try:
        response = _client.chat(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": REDIRECT_SYSTEM_PROMPTS[category].format(grade=grade)},
                {"role": "user", "content": query},
            ],
        )
        text = (response.message.content or "").strip()
        return text if text else OFF_TOPIC_FALLBACK_MESSAGES[category]
    except Exception:
        return OFF_TOPIC_FALLBACK_MESSAGES[category]


def _build_fallback_response(category: str, parsed: dict, query: str, grade: int) -> dict | None:
    suggestions = parsed.get("suggestions") or []

    if category == "ambiguous":
        if not suggestions:
            return None
        answer = "I want to make sure I answer the right question — did you mean one of these?"
        return _directory_response(answer, suggestions=suggestions[:3], response_category=category)

    if category == "typo":
        if not suggestions:
            return None
        return _directory_response(
            f"Did you mean: {suggestions[0]}?", suggestions=suggestions[:1], response_category=category
        )

    if category == "gibberish":
        return _directory_response(GIBBERISH_MESSAGE, response_category=category)

    if category == "off_topic_academic":
        subject = parsed.get("subject_guess") or "a different subject"
        answer = f"That sounds like a {subject} question — I can only help with Science for now!"
        return _directory_response(answer, response_category=category)

    if category in ("off_topic_casual", "personal_chitchat"):
        answer = _generate_redirect_line(category, query, grade)
        return _directory_response(answer, response_category=category)

    return None  # genuinely_not_found or unrecognized category


def _handle_refusal(query: str, collection, grade: int) -> dict | None:
    cross_grade = _cross_grade_lookup(query, collection, grade)
    if cross_grade is not None:
        return cross_grade

    parsed = _classify_refusal(query, grade)
    if parsed is None:
        return None

    return _build_fallback_response(parsed["category"], parsed, query, grade)


def _answer_with_llm(query: str, chunks: list[dict], grade: int, history: list[dict], chapter_heading: str | None) -> dict:
    prompt = build_prompt(query, chunks)
    system_prompt = build_system_prompt(grade)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = _client.chat(model=config.MODEL, messages=messages)

    answer = response.message.content
    sources = [
        {
            "chapter_number": c["chapter_number"] if c["chapter_number"] != -1 else None,
            "chapter_title": c["chapter_title"],
            "page": c["page"],
            "snippet": c["text"][:200],
        }
        for c in chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "chapter_heading": chapter_heading,
        "input_tokens": response.prompt_eval_count,
        "output_tokens": response.eval_count,
    }


def ask(query: str, collection=None, grade: int = None, history: list[dict] | None = None) -> dict:
    if collection is None:
        collection = get_collection()
    history = history or []

    safety_response = _check_safety_net(query)
    if safety_response is not None:
        return safety_response

    if DIAGRAM_REFERENCE_RE.search(query):
        return _directory_response(DIAGRAM_FALLBACK_MESSAGE, response_category="diagram_reference")

    chapter_number = detect_chapter_reference(query) if grade is not None else None

    if chapter_number is not None:
        chapter_chunks = retrieve_chapter_context(collection, grade, chapter_number)
        if chapter_chunks:
            chapter_heading = f"Chapter {chapter_chunks[0]['chapter_number']}: {chapter_chunks[0]['chapter_title']}"
            return _answer_with_llm(query, chapter_chunks, grade, history, chapter_heading)

        chapters = get_chapter_directory(collection, grade)
        answer = format_chapter_directory_answer(grade, chapters, invalid_chapter=chapter_number)
        return _directory_response(answer)

    if grade is not None and CHAPTER_LIST_INTENT_RE.search(query):
        chapters = get_chapter_directory(collection, grade)
        answer = format_chapter_directory_answer(grade, chapters)
        return _directory_response(answer)

    retrieval_query = condense_query(query, history) if history else query
    chunks = retrieve_context(retrieval_query, collection, grade=grade)
    result = _answer_with_llm(query, chunks, grade, history, chapter_heading=None)

    if result["answer"].strip() == NOT_FOUND_MARKER:
        fallback = _handle_refusal(query, collection, grade) if grade is not None else None
        if fallback is not None:
            return fallback
        result["answer"] = REFUSAL_STRING

    return result
