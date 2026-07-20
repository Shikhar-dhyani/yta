"""Central configuration, sourced from environment variables (and .env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Anchor .env and the default DB to the project root (the folder containing
# the yta package), so the CLI and the API server share one database no
# matter which directory they are launched from.
_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(_ROOT / ".env")
load_dotenv()  # a .env in the current directory can add missing values


# Path to the SQLite database file. A relative value (e.g. the default
# "transcripts.db" in .env) is resolved against the project root, so every
# entry point uses the same database regardless of working directory.
DB_PATH = str((_ROOT / os.getenv("YTA_DB_PATH", "transcripts.db")).resolve())

# Where downloaded embedding models are cached (survives temp cleanup).
MODELS_CACHE = _ROOT / ".models_cache"

# Local sentence-transformers model used to embed chunks and queries.
# paraphrase-multilingual-MiniLM-L12-v2 (~470MB) handles Hindi + 50 other
# languages and is cross-lingual: an English question matches Hindi passages.
# For an English-only library, all-MiniLM-L6-v2 (~90MB) is smaller and faster.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Optional cross-encoder reranker for search results (empty = disabled).
# e.g. YTA_RERANKER=jinaai/jina-reranker-v2-base-multilingual
RERANKER_MODEL = os.getenv("YTA_RERANKER", "")

# Gemini settings for the summary add-on.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Chunking: transcript snippets are grouped into chunks of roughly this many
# characters so that each chunk is a meaningful, searchable passage while still
# carrying a precise start timestamp.
CHUNK_TARGET_CHARS = int(os.getenv("YTA_CHUNK_TARGET_CHARS", "500"))
