"""SQLite storage for videos, transcript chunks and their embeddings.

Schema:
  videos  — one row per video (id, title, url, source, added_at)
  chunks  — timestamped transcript passages, each with an embedding blob

Embeddings are stored as raw float32 bytes; for a small personal DB a brute-
force cosine scan over a few thousand chunks is fast (<10ms), so no external
vector store is needed.
"""

import math
import re
import sqlite3
from dataclasses import dataclass

import numpy as np

from . import config

# Weight of exact-word matches relative to cosine similarity in search().
# Rare terms (names, jargon) get most of this; common words nearly none.
KEYWORD_BOOST = 0.3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id       TEXT PRIMARY KEY,          -- YouTube id, or 'manual:<slug>'
    title    TEXT NOT NULL,
    url      TEXT,                      -- NULL for manual imports without a link
    source   TEXT NOT NULL DEFAULT 'youtube',  -- 'youtube' | 'manual'
    added_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id  TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start     REAL,                     -- seconds; NULL if unknown (manual text)
    end       REAL,
    text      TEXT NOT NULL,
    embedding BLOB NOT NULL             -- float32 vector bytes
);

CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS channels (
    id          TEXT PRIMARY KEY,      -- YouTube channel id (UC...)
    name        TEXT NOT NULL,
    url         TEXT NOT NULL,         -- canonical channel URL
    added_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_synced TEXT
);

-- Memory of per-video sync outcomes that should not be retried immediately
-- (e.g. "no captions"): new uploads often gain auto-captions hours later,
-- so attempts expire and the video is eventually tried again.
CREATE TABLE IF NOT EXISTS sync_attempts (
    video_id     TEXT PRIMARY KEY,
    reason       TEXT NOT NULL,
    attempted_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass
class SearchHit:
    video_id: str
    video_title: str
    video_url: str | None
    start: float | None
    end: float | None
    text: str
    score: float


def connect(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
    )
    conn.commit()


def check_embedding_model(
    conn: sqlite3.Connection, model_name: str, claim: bool = False
) -> None:
    """Refuse to mix embeddings from different models in one database.

    Vectors from different models are incomparable: search would silently
    return garbage (same dimensions) or crash (different dimensions).
    With claim=True, an unclaimed database is stamped with this model.
    """
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'embedding_model'"
    ).fetchone()
    stored = row[0] if row else None
    if stored is None:
        if claim:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('embedding_model', ?)",
                (model_name,),
            )
        return
    if stored != model_name:
        raise RuntimeError(
            f"This database was built with embedding model {stored!r}, but the "
            f"configured model is {model_name!r}. Embeddings from different "
            f"models cannot be mixed. Either set EMBEDDING_MODEL={stored} in "
            f".env, or delete the database and re-add your videos with the "
            f"new model."
        )


def upsert_video(
    conn: sqlite3.Connection,
    video_id: str,
    title: str,
    url: str | None,
    source: str,
) -> None:
    """Insert or replace a video, clearing any previous chunks (re-ingest)."""
    conn.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
    conn.execute(
        "INSERT OR REPLACE INTO videos (id, title, url, source) VALUES (?, ?, ?, ?)",
        (video_id, title, url, source),
    )


def insert_chunks(
    conn: sqlite3.Connection,
    video_id: str,
    chunks: list[dict],
    embeddings: np.ndarray,
) -> None:
    """Store chunks with their embeddings. chunks[i] -> embeddings[i]."""
    rows = [
        (
            video_id,
            c.get("start"),
            c.get("end"),
            c["text"],
            embeddings[i].astype(np.float32).tobytes(),
        )
        for i, c in enumerate(chunks)
    ]
    conn.executemany(
        "INSERT INTO chunks (video_id, start, end, text, embedding) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def indexed_video_ids(conn: sqlite3.Connection) -> set[str]:
    """Ids of videos that are fully indexed (have at least one chunk)."""
    cur = conn.execute("SELECT DISTINCT video_id FROM chunks")
    return {row[0] for row in cur.fetchall()}


def list_videos(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        """SELECT v.id, v.title, v.url, v.source, v.added_at, COUNT(c.id) AS chunks
           FROM videos v LEFT JOIN chunks c ON c.video_id = v.id
           GROUP BY v.id ORDER BY v.added_at DESC"""
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def delete_video(conn: sqlite3.Connection, video_id: str) -> bool:
    cur = conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    conn.commit()
    return cur.rowcount > 0


def add_channel(conn: sqlite3.Connection, channel_id: str, name: str, url: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO channels (id, name, url) VALUES (?, ?, ?)",
        (channel_id, name, url),
    )
    conn.commit()


def list_channels(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        "SELECT id, name, url, added_at, last_synced FROM channels ORDER BY added_at"
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def delete_channel(conn: sqlite3.Connection, channel_id: str) -> bool:
    cur = conn.execute("DELETE FROM channels WHERE id = ?", (channel_id,))
    conn.commit()
    return cur.rowcount > 0


def touch_channel_synced(conn: sqlite3.Connection, channel_id: str) -> None:
    conn.execute(
        "UPDATE channels SET last_synced = datetime('now') WHERE id = ?",
        (channel_id,),
    )
    conn.commit()


def record_sync_attempt(conn: sqlite3.Connection, video_id: str, reason: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO sync_attempts (video_id, reason, attempted_at) "
        "VALUES (?, ?, datetime('now'))",
        (video_id, reason),
    )
    conn.commit()


def clear_sync_attempts(conn: sqlite3.Connection, video_ids: list[str]) -> None:
    """Forget skip records for videos that have since been indexed."""
    conn.executemany(
        "DELETE FROM sync_attempts WHERE video_id = ?",
        [(vid,) for vid in video_ids],
    )
    conn.commit()


def recently_attempted_ids(conn: sqlite3.Connection, within_days: int) -> set[str]:
    cur = conn.execute(
        "SELECT video_id FROM sync_attempts "
        "WHERE attempted_at > datetime('now', ?)",
        (f"-{within_days} days",),
    )
    return {row[0] for row in cur.fetchall()}


def _keyword_bonus(query_text: str, texts: list[str]) -> np.ndarray:
    """Rarity-weighted exact-word bonus per chunk, in [0, KEYWORD_BOOST].

    Semantic embeddings are weak on rare proper nouns ("kavana"): similar-
    sounding words score alike. A literal occurrence of a rare query word is
    strong evidence, so it is rewarded in proportion to its rarity (IDF-style);
    ubiquitous words contribute nearly nothing.
    """
    tokens = set(re.findall(r"[^\W\d_]{3,}", query_text.lower()))
    n = len(texts)
    if not tokens or n < 2:
        return np.zeros(n)

    lowered = [t.lower() for t in texts]
    weights = {}
    for tok in tokens:
        df = sum(1 for txt in lowered if tok in txt)
        if df:
            weights[tok] = math.log((n + 1) / (df + 1)) / math.log(n + 1)
    if not weights:
        return np.zeros(n)

    total = sum(weights.values())
    bonus = np.array(
        [sum(w for tok, w in weights.items() if tok in txt) for txt in lowered]
    )
    return KEYWORD_BOOST * bonus / total


def search(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    top_k: int = 5,
    query_text: str | None = None,
) -> list[SearchHit]:
    """Hybrid search: cosine similarity plus an exact-word rarity bonus."""
    cur = conn.execute(
        """SELECT c.text, c.start, c.end, c.embedding,
                  v.id, v.title, v.url
           FROM chunks c JOIN videos v ON v.id = c.video_id"""
    )
    rows = cur.fetchall()
    if not rows:
        return []

    matrix = np.frombuffer(b"".join(r[3] for r in rows), dtype=np.float32).reshape(
        len(rows), -1
    )
    q = query_embedding.astype(np.float32)
    # Embeddings are L2-normalized at encode time, so dot product == cosine.
    scores = matrix @ q
    if query_text:
        scores = scores + _keyword_bonus(query_text, [r[0] for r in rows])

    order = np.argsort(scores)[::-1][:top_k]
    hits = []
    for i in order:
        text, start, end, _, vid, title, url = rows[int(i)]
        hits.append(
            SearchHit(
                video_id=vid,
                video_title=title,
                video_url=url,
                start=start,
                end=end,
                text=text,
                score=float(scores[int(i)]),
            )
        )
    return hits
