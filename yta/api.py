"""FastAPI HTTP API (single-user, local).

Run with:  uvicorn yta.api:app --reload

Endpoints:
  GET  /                                                       — web UI
  POST /videos            {"url": "..."}                      — index a YouTube video
  POST /playlists         {"url": "..."}                      — index a whole playlist
  POST /videos/manual     {"title": "...", "text": "...", "url": "..."}
  GET  /videos                                                 — list indexed videos
  DELETE /videos/{id}                                          — remove a video
  POST /channels          {"url": "..."}                      — follow a channel
  GET  /channels                                               — list followed channels
  DELETE /channels/{id}                                        — unfollow (keeps videos)
  POST /sync                                                   — index missing videos now
  GET  /ask?q=...&top_k=5&summary=false                        — search (+ summary)

While the server runs, followed channels are synced automatically every
YTA_SYNC_INTERVAL_MINUTES (default 360; set 0 to disable).
"""

import asyncio
import contextlib
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import db
from .search import Embedder
from .search import ask as _ask

SYNC_INTERVAL_MINUTES = int(os.getenv("YTA_SYNC_INTERVAL_MINUTES", "360"))

# How long after a block we keep warning the user (matches YouTube's typical
# cool-down of roughly an hour).
BLOCK_COOLDOWN_MINUTES = 60

# One YouTube import at a time — playlist imports and syncs (manual or
# automatic) share this lock so they never fetch concurrently, which would
# double the request rate and trigger rate limits faster.
_sync_lock = threading.Lock()

logger = logging.getLogger("yta")


def _minutes_since_block() -> int | None:
    """Minutes since YouTube last blocked us, or None if never/unparseable."""
    conn = db.connect()
    try:
        last_blocked = db.get_meta(conn, "last_blocked_at")
    finally:
        conn.close()
    if not last_blocked:
        return None
    try:
        then = datetime.strptime(last_blocked, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return int((datetime.now(timezone.utc) - then).total_seconds() // 60)


def _run_sync() -> dict:
    from . import channels

    if not _sync_lock.acquire(blocking=False):
        raise RuntimeError("Another import or sync is already running.")
    try:
        return channels.sync_channels(_embedder)
    finally:
        _sync_lock.release()


async def _auto_sync_loop():
    """Sync followed channels periodically while the server is up.

    Respects the rate-limit cool-down: while YouTube is blocking this network,
    automatic passes are postponed — hitting a blocked endpoint only restarts
    the block clock.
    """
    await asyncio.sleep(60)  # let the app settle before the first pass
    while True:
        mins = _minutes_since_block()
        if mins is not None and mins < BLOCK_COOLDOWN_MINUTES:
            logger.info("auto-sync postponed: YouTube rate-limit cool-down (%s min left)",
                        BLOCK_COOLDOWN_MINUTES - mins)
            await asyncio.sleep(15 * 60)
            continue
        conn = db.connect()
        try:
            has_channels = bool(db.list_channels(conn))
        finally:
            conn.close()
        if has_channels:
            try:
                result = await asyncio.to_thread(_run_sync)
                added = sum(len(c["added"]) for c in result["channels"])
                logger.info("auto-sync done: %s new videos indexed%s", added,
                            f" — {result['aborted']}" if result["aborted"] else "")
            except Exception as e:
                logger.warning("auto-sync failed: %s", e)
        await asyncio.sleep(SYNC_INTERVAL_MINUTES * 60)


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    task = None
    if SYNC_INTERVAL_MINUTES > 0:
        task = asyncio.create_task(_auto_sync_loop())
    yield
    if task:
        task.cancel()


app = FastAPI(
    title="YouTube Transcript Answer",
    description="Ask questions; get timestamped answers from indexed transcripts.",
    version="0.1.0",
    lifespan=_lifespan,
)

# One embedder shared across requests; the model loads lazily on first use.
_embedder = Embedder()

_STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(_STATIC / "index.html")


class AddVideoRequest(BaseModel):
    url: str
    languages: list[str] = ["en"]
    title: str | None = None


class ManualImportRequest(BaseModel):
    title: str
    text: str
    url: str | None = None


class AddPlaylistRequest(BaseModel):
    url: str
    languages: list[str] = ["en"]
    refresh: bool = False


@app.post("/videos")
def add_video(req: AddVideoRequest):
    from . import ingest

    try:
        return ingest.add_youtube_video(
            req.url, _embedder, languages=req.languages, title=req.title
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/playlists")
def add_playlist(req: AddPlaylistRequest):
    """Index every video in a playlist/channel. Skips videos without captions.

    Note: runs synchronously — long playlists take a while. The response
    reports what was added and what was skipped (with reasons).
    """
    from . import playlist

    if not _sync_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Another import or sync is already running. "
                   "Wait for it to finish and try again.",
        )
    try:
        return playlist.add_playlist(
            req.url, _embedder, languages=req.languages, refresh=req.refresh
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        _sync_lock.release()


@app.post("/videos/manual")
def import_manual(req: ManualImportRequest):
    from . import ingest

    try:
        return ingest.add_manual_transcript(
            req.text, title=req.title, embedder=_embedder, url=req.url
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/videos")
def list_videos():
    conn = db.connect()
    try:
        return db.list_videos(conn)
    finally:
        conn.close()


@app.delete("/videos/{video_id}")
def delete_video(video_id: str):
    conn = db.connect()
    try:
        if not db.delete_video(conn, video_id):
            raise HTTPException(status_code=404, detail=f"No video {video_id!r}")
    finally:
        conn.close()
    return {"deleted": video_id}


@app.get("/status")
def status():
    """Library counts plus YouTube block state, for the UI status banner."""
    conn = db.connect()
    try:
        videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        channels = conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0]
        last_blocked = db.get_meta(conn, "last_blocked_at")
    finally:
        conn.close()

    blocked_minutes_ago = _minutes_since_block()

    return {
        "videos": videos,
        "chunks": chunks,
        "channels": channels,
        "last_blocked_at": last_blocked,
        "blocked_minutes_ago": blocked_minutes_ago,
        "blocked_recently": (
            blocked_minutes_ago is not None
            and blocked_minutes_ago < BLOCK_COOLDOWN_MINUTES
        ),
        "cooldown_minutes": BLOCK_COOLDOWN_MINUTES,
        "sync_running": _sync_lock.locked(),
    }


class FollowChannelRequest(BaseModel):
    url: str


@app.post("/channels")
def follow_channel(req: FollowChannelRequest):
    from . import channels

    try:
        return channels.follow_channel(req.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/channels")
def list_channels():
    conn = db.connect()
    try:
        return db.list_channels(conn)
    finally:
        conn.close()


@app.delete("/channels/{channel_id}")
def unfollow_channel(channel_id: str):
    conn = db.connect()
    try:
        if not db.delete_channel(conn, channel_id):
            raise HTTPException(status_code=404, detail=f"Not following {channel_id!r}")
    finally:
        conn.close()
    return {"unfollowed": channel_id}


@app.post("/sync")
def sync_now():
    """Index missing videos from all followed channels. Synchronous."""
    try:
        return _run_sync()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/ask")
def ask(
    q: str = Query(..., description="Your question"),
    top_k: int = Query(5, ge=1, le=25),
    summary: bool = Query(False, description="Also generate a Gemini summary"),
):
    results = _ask(q, _embedder, top_k=top_k)
    payload = {"question": q, "results": results}

    if summary:
        from .summarize import summarize

        try:
            payload["summary"] = summarize(q, results)
        except Exception as e:
            payload["summary_error"] = str(e)

    return payload
