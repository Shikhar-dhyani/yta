"""Ingestion: fetch transcripts from YouTube or import manual text, then chunk.

Chunks are groups of consecutive transcript snippets merged until they reach
~CHUNK_TARGET_CHARS. Each chunk keeps the start time of its first snippet and
the end time of its last, so search results deep-link to the right moment.
"""

import os
import re
import urllib.request
from datetime import datetime, timezone
from xml.etree.ElementTree import ParseError

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    YouTubeRequestFailed,
)

from . import config, db, utils
from .search import Embedder

# Errors that mean "YouTube is rate-limiting this network", not "this video
# has no transcript". ParseError: a throttled endpoint returns an empty 200.
BLOCK_ERRORS = (RequestBlocked, IpBlocked, YouTubeRequestFailed, ParseError)

# Chunks with fewer meaningful words than this are noise ("[Music] this")
# that embeds badly and pollutes search results.
MIN_CHUNK_WORDS = 5


def _drop_noise_chunks(chunks: list[dict]) -> list[dict]:
    """Remove chunks too short to be a meaningful, searchable passage."""

    def words(c: dict) -> int:
        return len(re.sub(r"\[[^\]]*\]", "", c["text"]).split())

    return [c for c in chunks if words(c) >= MIN_CHUNK_WORDS]


def record_block(db_path: str | None = None) -> None:
    """Remember when YouTube last blocked us, so status/UI can tell the user."""
    conn = db.connect(db_path)
    try:
        db.set_meta(
            conn, "last_blocked_at",
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        )
    finally:
        conn.close()


def _make_api() -> YouTubeTranscriptApi:
    """Build the transcript client, routed through YTA_PROXY_URL if set.

    A proxy gives transcript requests a different IP — the escape hatch when
    YouTube rate-limits your network. Format: http://user:pass@host:port
    """
    proxy = os.getenv("YTA_PROXY_URL")
    if proxy:
        from youtube_transcript_api.proxies import GenericProxyConfig

        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(http_url=proxy, https_url=proxy)
        )
    return YouTubeTranscriptApi()


def _fetch_title(video_id: str) -> str:
    """Get the video title via YouTube's public oEmbed endpoint (no API key)."""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        import json

        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.load(resp).get("title") or video_id
    except Exception:
        return video_id  # title is nice-to-have; never fail ingestion over it


def _fetch_transcript(video_id: str, languages: list[str]) -> tuple[list[dict], str]:
    """Return ([{'text', 'start', 'duration'}, ...], language_code).

    Tries the requested languages first; if none match, falls back to any
    available transcript (manually created preferred over auto-generated).
    """
    api = _make_api()
    try:
        fetched = api.fetch(video_id, languages=languages)
    except NoTranscriptFound:
        transcript_list = api.list(video_id)
        available = sorted(transcript_list, key=lambda t: t.is_generated)
        if not available:
            raise
        fetched = available[0].fetch()
    snippets = [
        {"text": s.text, "start": s.start, "duration": s.duration}
        for s in fetched
    ]
    return snippets, fetched.language_code


def _chunk_snippets(snippets: list[dict]) -> list[dict]:
    """Merge consecutive timestamped snippets into ~CHUNK_TARGET_CHARS chunks."""
    chunks: list[dict] = []
    buf: list[str] = []
    start: float | None = None
    end: float | None = None

    for s in snippets:
        text = re.sub(r"\s+", " ", s["text"]).strip()
        if not text:
            continue
        if start is None:
            start = s["start"]
        end = s["start"] + s.get("duration", 0)
        buf.append(text)
        if sum(len(t) + 1 for t in buf) >= config.CHUNK_TARGET_CHARS:
            chunks.append({"text": " ".join(buf), "start": start, "end": end})
            buf, start, end = [], None, None

    if buf:
        chunks.append({"text": " ".join(buf), "start": start, "end": end})
    return chunks


_SUB_TIMECODE_RE = re.compile(
    r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[.,](\d{3})"
)


def _parse_subtitle(text: str) -> list[dict] | None:
    """Parse SRT / WebVTT subtitle content into timestamped snippets.

    Returns None when the text is not a subtitle file (no '-->' cues), so the
    caller can fall back to plain-text handling.
    """
    if "-->" not in text:
        return None

    snippets: list[dict] = []
    start: float | None = None
    lines: list[str] = []

    def flush():
        nonlocal start, lines
        joined = " ".join(lines).strip()
        # Auto-generated captions often repeat the previous cue's text in a
        # rolling window; drop exact consecutive duplicates.
        is_dup = snippets and snippets[-1]["text"] == joined
        if start is not None and joined and not is_dup:
            snippets.append({"text": joined, "start": start, "duration": 0})
        start, lines = None, []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if "-->" in line:
            flush()
            m = _SUB_TIMECODE_RE.search(line.split("-->")[0])
            if m:
                h, mnt, sec, ms = m.groups()
                start = int(h or 0) * 3600 + int(mnt) * 60 + int(sec) + int(ms) / 1000
            continue
        if line.upper().startswith(("WEBVTT", "NOTE", "KIND:", "LANGUAGE:")):
            continue
        if line.isdigit() and start is None:
            continue  # SRT cue counter
        if start is not None:
            lines.append(re.sub(r"<[^>]*>", "", line))  # strip VTT inline tags
    flush()
    return snippets or None


# YouTube's transcript panel sometimes copies timestamps in the spoken-out
# accessibility form, glued straight into the text:
#   "22 minutes, 53 secondsWhen he entered the room..."
#   "1 hour, 46 minutes, 48 secondsposition right..."
_VERBOSE_TS_RE = re.compile(
    r"(?:(\d+)\s+hours?,\s+)?(\d+)\s+minutes?,\s+(\d+)\s+seconds?\s*"
)


def _normalize_verbose_timestamps(text: str) -> str:
    """Turn spoken-out timestamps into '[H:MM:SS]' line anchors."""

    def repl(m: re.Match) -> str:
        h, mnt, sec = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
        total = h * 3600 + mnt * 60 + sec
        hh, rem = divmod(total, 3600)
        mm, ss = divmod(rem, 60)
        stamp = f"{hh}:{mm:02d}:{ss:02d}" if hh else f"{mm}:{ss:02d}"
        return f"\n[{stamp}] "

    return _VERBOSE_TS_RE.sub(repl, text)


def _chunk_plain_text(text: str) -> list[dict]:
    """Chunk manual text. Honors '[H:MM:SS] ...' / 'M:SS ...' line prefixes.

    Lines with a leading timestamp become anchored chunks; untimestamped text
    is grouped into ~CHUNK_TARGET_CHARS chunks with start=None.
    """
    text = _normalize_verbose_timestamps(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    timestamped = [(utils.parse_timestamp(ln), ln) for ln in lines]

    if any(ts is not None for ts, _ in timestamped):
        # Treat each timestamped line as a snippet; carry timestamps through.
        # YouTube's own transcript panel copies timestamps on their own line
        # ("0:05" then the text below) — such lines stamp the next text line
        # instead of becoming junk text themselves.
        snippets = []
        pending_ts = None
        for ts, ln in timestamped:
            clean = re.sub(r"^\s*\[?\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]?\s*", "", ln)
            if ts is not None and not clean:
                pending_ts = ts
                continue
            if ts is None and pending_ts is not None:
                ts = pending_ts
            pending_ts = None
            snippets.append({"text": clean or ln, "start": ts, "duration": 0})
        # Untimestamped lines inherit the previous line's timestamp.
        last = None
        for s in snippets:
            if s["start"] is None:
                s["start"] = last
            else:
                last = s["start"]
        # _chunk_snippets requires numeric starts; fall back to 0 for leaders.
        for s in snippets:
            if s["start"] is None:
                s["start"] = 0.0
        return _chunk_snippets(snippets)

    # No timestamps at all: chunk by sentences/length with unknown times.
    words = " ".join(lines).split()
    chunks, buf, size = [], [], 0
    for w in words:
        buf.append(w)
        size += len(w) + 1
        if size >= config.CHUNK_TARGET_CHARS:
            chunks.append({"text": " ".join(buf), "start": None, "end": None})
            buf, size = [], 0
    if buf:
        chunks.append({"text": " ".join(buf), "start": None, "end": None})
    return chunks


def add_youtube_video(
    url_or_id: str,
    embedder: Embedder,
    languages: list[str] | None = None,
    title: str | None = None,
    db_path: str | None = None,
) -> dict:
    """Fetch transcript + title from YouTube, chunk, embed, and store."""
    video_id = utils.extract_video_id(url_or_id)
    try:
        snippets, language = _fetch_transcript(video_id, languages or ["en"])
    except BLOCK_ERRORS:
        record_block(db_path)  # single adds should show up in status too
        raise
    chunks = _drop_noise_chunks(_chunk_snippets(snippets))
    if not chunks:
        raise ValueError(f"Transcript for {video_id} is empty.")

    title = title or _fetch_title(video_id)
    embeddings = embedder.encode([c["text"] for c in chunks])

    conn = db.connect(db_path)
    try:
        db.check_embedding_model(conn, embedder.model_name, claim=True)
        db.upsert_video(conn, video_id, title, utils.watch_url(video_id), "youtube")
        db.insert_chunks(conn, video_id, chunks, embeddings)
    finally:
        conn.close()
    return {
        "video_id": video_id,
        "title": title,
        "chunks": len(chunks),
        "language": language,
    }


def add_manual_transcript(
    text: str,
    title: str,
    embedder: Embedder,
    url: str | None = None,
    db_path: str | None = None,
) -> dict:
    """Import a transcript: plain text, '[M:SS] line' text, SRT, or WebVTT."""
    if not text.strip():
        raise ValueError("Transcript text is empty.")

    subtitle_snippets = _parse_subtitle(text)

    # Stable id: derived from the URL when it's a YouTube link, else a slug.
    try:
        video_id = utils.extract_video_id(url) if url else None
    except ValueError:
        video_id = None
    if video_id is None:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
        video_id = f"manual:{slug or 'untitled'}"

    if subtitle_snippets is not None:
        chunks = _chunk_snippets(subtitle_snippets)
    else:
        chunks = _chunk_plain_text(text)
    chunks = _drop_noise_chunks(chunks)
    if not chunks:
        raise ValueError("No usable text found in the transcript.")
    embeddings = embedder.encode([c["text"] for c in chunks])

    conn = db.connect(db_path)
    try:
        db.check_embedding_model(conn, embedder.model_name, claim=True)
        db.upsert_video(conn, video_id, title, url, "manual")
        db.insert_chunks(conn, video_id, chunks, embeddings)
    finally:
        conn.close()
    return {"video_id": video_id, "title": title, "chunks": len(chunks)}
