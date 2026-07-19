"""Playlist / channel import: enumerate videos with yt-dlp, ingest each one.

Videos that genuinely lack captions are skipped with a note. Rate-limit blocks
from YouTube are retried with backoff — and if YouTube keeps blocking, the run
aborts early with clear guidance instead of "skipping" every remaining video.
"""

import os
import time

from . import db, ingest
from .ingest import BLOCK_ERRORS as _BLOCK_ERRORS
from .ingest import record_block as _record_block
from .search import Embedder

# Politeness delay between transcript fetches, so a 30-video playlist doesn't
# look like a bot burst to YouTube.
DELAY_BETWEEN_VIDEOS = float(os.getenv("YTA_PLAYLIST_DELAY", "3.0"))

# Backoff schedule (seconds) when YouTube blocks a request. Long on purpose:
# hammering a throttled endpoint prolongs the throttle.
BLOCK_RETRY_WAITS = [60]

# Abort the whole run after this many consecutive block failures.
MAX_CONSECUTIVE_BLOCKS = 3


def enumerate_videos(playlist_url: str) -> list[dict]:
    """Return [{'id', 'title'}, ...] for a playlist or channel URL.

    Uses yt-dlp in flat mode: fetches only the listing, not the videos.
    """
    from yt_dlp import YoutubeDL

    opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    entries = info.get("entries") or []
    videos, seen = [], set()
    for e in entries:
        if not e:
            continue
        vid = e.get("id")
        # Flat channel listings can nest tabs (e.g. Videos/Shorts); recurse one level.
        if e.get("_type") == "playlist" and e.get("entries"):
            for s in e["entries"]:
                if s and s.get("id") and s["id"] not in seen:
                    seen.add(s["id"])
                    videos.append({"id": s["id"], "title": s.get("title") or s["id"]})
        elif vid and len(vid) == 11 and vid not in seen:
            seen.add(vid)
            videos.append({"id": vid, "title": e.get("title") or vid})
    return videos


def _add_with_retry(video_id: str, embedder, languages, db_path):
    """Ingest one video, retrying with backoff if YouTube blocks the request.

    Returns (info, None) on success, (None, reason) on a genuine skip.
    Raises the block error if retries are exhausted (caller decides to abort).
    """
    for attempt, wait in enumerate([0] + BLOCK_RETRY_WAITS):
        if wait:
            time.sleep(wait)
        try:
            info = ingest.add_youtube_video(
                video_id, embedder, languages=languages, db_path=db_path
            )
            return info, None
        except _BLOCK_ERRORS:
            if attempt == len(BLOCK_RETRY_WAITS):
                raise  # retries exhausted; caller counts consecutive blocks
        except Exception as e:
            reason = str(e).strip().splitlines()[0][:160] or type(e).__name__
            return None, reason
    return None, "unreachable"


def add_playlist(
    playlist_url: str,
    embedder: Embedder,
    languages: list[str] | None = None,
    db_path: str | None = None,
    progress=None,
    refresh: bool = False,
) -> dict:
    """Ingest every video in a playlist. Returns per-video results.

    Videos already in the database are skipped without touching YouTube, so
    re-running after a partial (rate-limited) run only fetches what's missing.
    Pass refresh=True to re-fetch everything.
    """
    videos = enumerate_videos(playlist_url)
    if not videos:
        raise ValueError("No videos found at that URL (is it a playlist/channel link?)")
    return import_videos(
        videos, embedder, languages=languages, db_path=db_path,
        progress=progress, refresh=refresh,
    )


def import_videos(
    videos: list[dict],
    embedder: Embedder,
    languages: list[str] | None = None,
    db_path: str | None = None,
    progress=None,
    refresh: bool = False,
) -> dict:
    """Ingest a list of {'id', 'title'} videos with pacing and block handling.

    Shared by playlist import and channel sync. Skipped entries carry a
    "blocked" flag so callers can tell rate-limit skips (worth retrying soon)
    from genuine ones like missing captions (worth remembering).

    progress: optional callable(index, total, video, status_message)
    called after each video — used by the CLI for live output.
    """
    existing: set[str] = set()
    if not refresh:
        conn = db.connect(db_path)
        try:
            existing = db.indexed_video_ids(conn)
        finally:
            conn.close()

    added, skipped, already = [], [], []
    aborted = None
    consecutive_blocks = 0
    fetched_any = False
    attempted = 0
    for i, v in enumerate(videos, 1):
        attempted = i
        if v["id"] in existing:
            already.append(v)
            if progress:
                progress(i, len(videos), v, "already indexed, skipped fetch")
            continue
        if fetched_any:
            time.sleep(DELAY_BETWEEN_VIDEOS)
        fetched_any = True
        try:
            info, reason = _add_with_retry(v["id"], embedder, languages, db_path)
        except _BLOCK_ERRORS:
            consecutive_blocks += 1
            _record_block(db_path)
            skipped.append(
                {"video_id": v["id"], "title": v["title"],
                 "reason": "blocked by YouTube (rate limit)", "blocked": True}
            )
            if progress:
                progress(i, len(videos), v, "blocked by YouTube, retries exhausted")
            if consecutive_blocks >= MAX_CONSECUTIVE_BLOCKS:
                aborted = (
                    f"Stopped after {i} of {len(videos)} videos: YouTube is "
                    f"rate-limiting transcript requests from this network. "
                    f"Wait an hour without importing anything (every attempt "
                    f"while blocked restarts the clock), then re-run — "
                    f"indexed videos are skipped, so each run makes progress."
                )
                break
            continue

        # Either outcome here means YouTube actually responded (a "no
        # captions" skip is a real answer), so the block streak is over.
        consecutive_blocks = 0
        if info:
            added.append(info)
            msg = f"{info['chunks']} chunks, language: {info['language']}"
        else:
            skipped.append(
                {"video_id": v["id"], "title": v["title"],
                 "reason": reason, "blocked": False}
            )
            msg = f"skipped ({reason})"
        if progress:
            progress(i, len(videos), v, msg)

    if added:
        # A successful add supersedes any old "skipped" memory for the video.
        conn = db.connect(db_path)
        try:
            db.clear_sync_attempts(conn, [a["video_id"] for a in added])
        finally:
            conn.close()

    return {
        "total": len(videos),
        "added": added,
        "already_indexed": [v["id"] for v in already],
        "skipped": skipped,
        "not_attempted": len(videos) - attempted if aborted else 0,
        "aborted": aborted,
    }
