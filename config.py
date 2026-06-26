import os
from dotenv import load_dotenv

load_dotenv()

PDF_SOURCE_DIR = os.getenv("PDF_SOURCE_DIR", "./data/grade8")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "science_grade8")

SUBJECT = "science"

# Each grade's PDF folder is independently overridable via its own env var, defaulting
# to a repo-local ./data/gradeN convention so a fresh clone works without editing this
# file -- just drop each grade's NCERT PDFs into the matching folder.
GRADE_SOURCES = {
    6: os.getenv("GRADE_6_SOURCE_DIR", "./data/grade6"),
    7: os.getenv("GRADE_7_SOURCE_DIR", "./data/grade7"),
    8: PDF_SOURCE_DIR,
    9: os.getenv("GRADE_9_SOURCE_DIR", "./data/grade9"),
    10: os.getenv("GRADE_10_SOURCE_DIR", "./data/grade10"),
}
VALID_GRADES = sorted(GRADE_SOURCES.keys())

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

TOP_K = int(os.getenv("TOP_K", 3))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 10))
MAX_HISTORY_CONTENT_LEN = int(os.getenv("MAX_HISTORY_CONTENT_LEN", 4000))
CHAPTER_CHUNK_CAP = int(os.getenv("CHAPTER_CHUNK_CAP", 24))

CROSS_GRADE_DISTANCE_THRESHOLD = float(os.getenv("CROSS_GRADE_DISTANCE_THRESHOLD", 0.42))
CROSS_GRADE_TOP_K = int(os.getenv("CROSS_GRADE_TOP_K", 3))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
MODEL = os.getenv("MODEL", "gpt-oss:120b")

# Manual fallback for PDFs whose chapter heading can't be parsed automatically
# (e.g. front-matter files with no "Chapter N" text on page 1, or books whose
# page-1 running header doesn't match CHAPTER_HEADING_RE at all -- grades 6, 7,
# 9 and 10 each use a different page-1 layout than grade 8, so every chapter in
# those books is listed explicitly here, sourced from each book's own table of
# contents rather than the unreliable per-page header text).
CHAPTER_OVERRIDES = {
    "hecu1ps.pdf": {"number": None, "title": "Front Matter / Contents"},

    # Grade 6 (fecu1)
    "fecu101.pdf": {"number": 1, "title": "The Wonderful World of Science"},
    "fecu102.pdf": {"number": 2, "title": "Diversity in the Living World"},
    "fecu103.pdf": {"number": 3, "title": "Mindful Eating: A Path to a Healthy Body"},
    "fecu104.pdf": {"number": 4, "title": "Exploring Magnets"},
    "fecu105.pdf": {"number": 5, "title": "Measurement of Length and Motion"},
    "fecu106.pdf": {"number": 6, "title": "Materials Around Us"},
    "fecu107.pdf": {"number": 7, "title": "Temperature and its Measurement"},
    "fecu108.pdf": {"number": 8, "title": "A Journey through States of Water"},
    "fecu109.pdf": {"number": 9, "title": "Methods of Separation in Everyday Life"},
    "fecu110.pdf": {"number": 10, "title": "Living Creatures: Exploring their Characteristics"},
    "fecu111.pdf": {"number": 11, "title": "Nature's Treasures"},
    "fecu112.pdf": {"number": 12, "title": "Beyond Earth"},
    "fecu1ps.pdf": {"number": None, "title": "Front Matter / Contents"},

    # Grade 7 (gecu1)
    "gecu101.pdf": {"number": 1, "title": "The Ever-Evolving World of Science"},
    "gecu102.pdf": {"number": 2, "title": "Exploring Substances: Acidic, Basic, and Neutral"},
    "gecu103.pdf": {"number": 3, "title": "Electricity: Circuits and their Components"},
    "gecu104.pdf": {"number": 4, "title": "The World of Metals and Non-metals"},
    "gecu105.pdf": {"number": 5, "title": "Changes Around Us: Physical and Chemical"},
    "gecu106.pdf": {"number": 6, "title": "Adolescence: A Stage of Growth and Change"},
    "gecu107.pdf": {"number": 7, "title": "Heat Transfer in Nature"},
    "gecu108.pdf": {"number": 8, "title": "Measurement of Time and Motion"},
    "gecu109.pdf": {"number": 9, "title": "Life Processes in Animals"},
    "gecu110.pdf": {"number": 10, "title": "Life Processes in Plants"},
    "gecu111.pdf": {"number": 11, "title": "Light: Shadows and Reflections"},
    "gecu112.pdf": {"number": 12, "title": "Earth, Moon, and the Sun"},
    "gecu1ps.pdf": {"number": None, "title": "Front Matter / Contents"},

    # Grade 9 (iesc1)
    "iesc101.pdf": {"number": 1, "title": "Exploration: Entering the World of Secondary Science"},
    "iesc102.pdf": {"number": 2, "title": "Cell: The Building Block of Life"},
    "iesc103.pdf": {"number": 3, "title": "Tissues in Action"},
    "iesc104.pdf": {"number": 4, "title": "Describing Motion Around Us"},
    "iesc105.pdf": {"number": 5, "title": "Exploring Mixtures and their Separation"},
    "iesc106.pdf": {"number": 6, "title": "How Forces Affect Motion"},
    "iesc107.pdf": {"number": 7, "title": "Work, Energy, and Simple Machines"},
    "iesc108.pdf": {"number": 8, "title": "Journey Inside the Atom"},
    "iesc109.pdf": {"number": 9, "title": "Atomic Foundations of Matter"},
    "iesc110.pdf": {"number": 10, "title": "Sound Waves: Characteristics and Applications"},
    "iesc111.pdf": {"number": 11, "title": "Reproduction: How Life Continues"},
    "iesc112.pdf": {"number": 12, "title": "Patterns in Life: Diversity and Classification"},
    "iesc113.pdf": {"number": 13, "title": "Earth as a System: Energy, Matter, and Life"},
    "iesc1ps.pdf": {"number": None, "title": "Front Matter / Contents"},

    # Grade 10 (jesc1)
    "jesc101.pdf": {"number": 1, "title": "Chemical Reactions and Equations"},
    "jesc102.pdf": {"number": 2, "title": "Acids, Bases and Salts"},
    "jesc103.pdf": {"number": 3, "title": "Metals and Non-metals"},
    "jesc104.pdf": {"number": 4, "title": "Carbon and its Compounds"},
    "jesc105.pdf": {"number": 5, "title": "Life Processes"},
    "jesc106.pdf": {"number": 6, "title": "Control and Coordination"},
    "jesc107.pdf": {"number": 7, "title": "How do Organisms Reproduce?"},
    "jesc108.pdf": {"number": 8, "title": "Heredity"},
    "jesc109.pdf": {"number": 9, "title": "Light – Reflection and Refraction"},
    "jesc110.pdf": {"number": 10, "title": "The Human Eye and the Colourful World"},
    "jesc111.pdf": {"number": 11, "title": "Electricity"},
    "jesc112.pdf": {"number": 12, "title": "Magnetic Effects of Electric Current"},
    "jesc113.pdf": {"number": 13, "title": "Our Environment"},
    "jesc1ps.pdf": {"number": None, "title": "Front Matter / Contents"},
    "jesc1an.pdf": {"number": None, "title": "Answer Key"},
}


def collection_name_for(chunk_size: int, chunk_overlap: int) -> str:
    return f"{COLLECTION_NAME}_cs{chunk_size}_ov{chunk_overlap}"
