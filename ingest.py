import argparse
import re
from pathlib import Path

import chromadb
import pypdf.filters
from chromadb.utils import embedding_functions
from pypdf import PdfReader

import config

# Some textbook pages (large embedded diagrams) decompress past pypdf's default
# 75MB zip-bomb guard; these are trusted, manually-sourced NCERT PDFs, so raise it.
pypdf.filters.ZLIB_MAX_OUTPUT_LENGTH = 200_000_000

# Running header on chapter pages looks like "Chapter 5<sep>Exploring Forces",
# where <sep> is a decorative glyph (varies by PDF font encoding) between thin spaces.
# It shows up on page 1 for chapter 1, page 2+ for later chapters (page 1 there is a
# title-splash page with no "Chapter N" text at all) -- so scan the first few pages.
# Exclude InDesign production-stamp footers like "Chapter 5.indd   62Chapter 5.indd..."
# which also start with "Chapter <N>" but are immediately followed by ".indd".
CHAPTER_HEADING_RE = re.compile(r"Chapter\s+(\d+)(?!\.indd)[^\w\n]+([^\n]+)")
CHAPTER_SCAN_PAGES = 3

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

BATCH_SIZE = 100


def discover_pdfs(source_dir: str) -> list[Path]:
    return sorted(Path(source_dir).glob("*.pdf"))


def extract_chapter_info(path: Path) -> dict:
    if path.name in config.CHAPTER_OVERRIDES:
        return config.CHAPTER_OVERRIDES[path.name]

    reader = PdfReader(str(path))
    for page in reader.pages[:CHAPTER_SCAN_PAGES]:
        text = page.extract_text() or ""
        match = CHAPTER_HEADING_RE.search(text)
        if match:
            return {"number": int(match.group(1)), "title": match.group(2).strip()}

    return {"number": None, "title": path.stem}


def extract_pages(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return SENTENCE_SPLIT_RE.split(normalized)


def chunk_page(text: str, size: int, overlap: int) -> list[str]:
    sentences = split_sentences(text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(sentence) > size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(sentence):
                end = start + size
                chunks.append(sentence[start:end])
                start = end - overlap
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= size:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail} {sentence}".strip()

    if current:
        chunks.append(current)
    return chunks


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )


def build_or_update_vector_store(
    chunk_size: int, chunk_overlap: int, pdf_source_dir: str, grade: int, subject: str
) -> dict:
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection_name = config.collection_name_for(chunk_size, chunk_overlap)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
    )

    pdf_paths = discover_pdfs(pdf_source_dir)
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in {pdf_source_dir}")

    chapter_counts: dict[str, int] = {}

    documents, metadatas, ids = [], [], []

    def flush():
        if not documents:
            return
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        documents.clear()
        metadatas.clear()
        ids.clear()

    for path in pdf_paths:
        chapter_info = extract_chapter_info(path)
        chapter_label = (
            f"Chapter {chapter_info['number']}: {chapter_info['title']}"
            if chapter_info["number"] is not None
            else chapter_info["title"]
        )
        file_stem = path.stem
        chunk_count_for_file = 0

        for page_num, page_text in extract_pages(path):
            for chunk_index, chunk_text in enumerate(
                chunk_page(page_text, chunk_size, chunk_overlap)
            ):
                documents.append(chunk_text)
                metadatas.append(
                    {
                        "source_file": path.name,
                        "chapter_number": chapter_info["number"] if chapter_info["number"] is not None else -1,
                        "chapter_title": chapter_info["title"],
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "grade": grade,
                        "subject": subject,
                    }
                )
                ids.append(f"{file_stem}_pg{page_num}_ck{chunk_index}")
                chunk_count_for_file += 1

                if len(documents) >= BATCH_SIZE:
                    flush()

        chapter_counts[chapter_label] = chunk_count_for_file

    flush()

    return {
        "collection_name": collection_name,
        "total_chunks": sum(chapter_counts.values()),
        "chapter_counts": chapter_counts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--grade", type=int, default=8, choices=sorted(config.GRADE_SOURCES))
    args = parser.parse_args()

    pdf_source_dir = config.GRADE_SOURCES[args.grade]

    print(f"Ingesting PDFs from {pdf_source_dir}")
    print(f"grade={args.grade} subject={config.SUBJECT}")
    print(f"chunk_size={args.chunk_size} chunk_overlap={args.chunk_overlap}")

    summary = build_or_update_vector_store(
        args.chunk_size, args.chunk_overlap, pdf_source_dir, args.grade, config.SUBJECT
    )

    print(f"\nCollection: {summary['collection_name']}")
    print(f"Total chunks: {summary['total_chunks']}")
    print("\nPer-chapter chunk counts:")
    for label, count in summary["chapter_counts"].items():
        print(f"  {label}: {count}")
