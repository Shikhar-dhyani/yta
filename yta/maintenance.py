"""Library maintenance: rebuild chunks/embeddings without losing timestamps.

Used when the embedding model or backend changes, or after an ingestion
improvement, so the whole library can be re-processed in place. Each stored
chunk's timestamp is re-anchored into the text before re-chunking — rebuilds
must never lose timing information.
"""

from . import db, utils
from .ingest import _chunk_plain_text, _drop_noise_chunks
from .search import Embedder


def reindex_library(
    embedder: Embedder,
    db_path: str | None = None,
    progress=None,
) -> dict:
    """Re-chunk and re-embed every video from its stored chunks.

    Timestamps are preserved by prefixing each chunk's text with its stored
    start time. The database's embedding-model stamp is updated to the given
    embedder, so this is the sanctioned way to switch models.
    """
    conn = db.connect(db_path)
    try:
        videos = conn.execute(
            "SELECT id, title, url, source FROM videos ORDER BY added_at"
        ).fetchall()
        sources = {}
        for vid, title, url, source in videos:
            rows = conn.execute(
                "SELECT text, start FROM chunks WHERE video_id = ? ORDER BY id",
                (vid,),
            ).fetchall()
            lines = [
                f"[{utils.format_timestamp(start)}] {text}"
                if start is not None else text
                for text, start in rows
            ]
            sources[vid] = (title, url, source, "\n".join(lines))
        # Rebuilding everything under the new embedder: update the stamp.
        db.set_meta(conn, "embedding_model", embedder.model_name)
    finally:
        conn.close()

    rebuilt = []
    for i, (vid, (title, url, source, text)) in enumerate(sources.items(), 1):
        chunks = _drop_noise_chunks(_chunk_plain_text(text))
        if not chunks:
            continue
        embeddings = embedder.encode([c["text"] for c in chunks])
        conn = db.connect(db_path)
        try:
            db.upsert_video(conn, vid, title, url, source)
            db.insert_chunks(conn, vid, chunks, embeddings)
        finally:
            conn.close()
        rebuilt.append({"video_id": vid, "title": title, "chunks": len(chunks)})
        if progress:
            progress(i, len(sources), title)
    return {"videos": rebuilt}
