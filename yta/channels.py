"""Followed channels: keep a channel's videos — current and future — indexed.

Follow a channel once; every sync enumerates its uploads and imports whatever
is missing from the index. Videos that failed for a real reason (usually "no
captions yet") are remembered and retried only after SKIP_RETRY_DAYS, since
new uploads often gain auto-captions a few hours after publishing.
"""

import os

from . import db
from .playlist import enumerate_videos, import_videos
from .search import Embedder

# Days to wait before retrying a video that was skipped for a genuine reason.
SKIP_RETRY_DAYS = int(os.getenv("YTA_SKIP_RETRY_DAYS", "7"))


def resolve_channel(url: str) -> dict:
    """Resolve any channel URL form (@handle, /channel/UC..., /c/...) to
    {'id', 'name', 'url'} using yt-dlp metadata (one listing request)."""
    from yt_dlp import YoutubeDL

    opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "playlist_items": "0",  # metadata only, no entries
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    channel_id = info.get("channel_id") or info.get("id")
    name = info.get("channel") or info.get("title") or url
    canonical = info.get("channel_url") or info.get("webpage_url") or url
    if not channel_id:
        raise ValueError(f"Could not resolve a YouTube channel from: {url}")
    return {"id": channel_id, "name": name, "url": canonical}


def follow_channel(url: str, db_path: str | None = None) -> dict:
    """Add a channel to the followed list. Indexing happens on sync."""
    info = resolve_channel(url)
    conn = db.connect(db_path)
    try:
        db.add_channel(conn, info["id"], info["name"], info["url"])
    finally:
        conn.close()
    return info


def sync_channels(
    embedder: Embedder,
    db_path: str | None = None,
    progress=None,
    channel_id: str | None = None,
) -> dict:
    """Bring followed channels up to date: import any videos not yet indexed.

    Returns {"channels": [per-channel result], "aborted": msg | None}.
    Stops early if YouTube rate-limits — the block is network-wide, so
    continuing with the next channel would only prolong it.
    """
    conn = db.connect(db_path)
    try:
        channels = db.list_channels(conn)
        recent = db.recently_attempted_ids(conn, SKIP_RETRY_DAYS)
        indexed = db.indexed_video_ids(conn)
    finally:
        conn.close()

    if channel_id is not None:
        channels = [c for c in channels if c["id"] == channel_id]
        if not channels:
            raise ValueError(f"Not following any channel with id {channel_id!r}")
    if not channels:
        raise ValueError("No followed channels. Follow one first (yta channel add <url>).")

    report, aborted = [], None
    for ch in channels:
        videos = enumerate_videos(ch["url"].rstrip("/") + "/videos")
        missing = [
            v for v in videos
            if v["id"] not in indexed and v["id"] not in recent
        ]
        entry = {
            "channel": ch["name"],
            "channel_id": ch["id"],
            "videos_on_channel": len(videos),
            "new": len(missing),
            "added": [],
            "skipped": [],
        }

        if missing:
            result = import_videos(
                missing, embedder, db_path=db_path, progress=progress
            )
            entry["added"] = result["added"]
            entry["skipped"] = result["skipped"]
            # Remember genuine skips so they cool off; blocked ones stay
            # eligible — they say nothing about the video itself.
            conn = db.connect(db_path)
            try:
                for s in result["skipped"]:
                    if not s.get("blocked"):
                        db.record_sync_attempt(conn, s["video_id"], s["reason"])
                indexed |= {a["video_id"] for a in result["added"]}
            finally:
                conn.close()
            if result["aborted"]:
                aborted = result["aborted"]

        conn = db.connect(db_path)
        try:
            db.touch_channel_synced(conn, ch["id"])
        finally:
            conn.close()
        report.append(entry)
        if aborted:
            break

    return {"channels": report, "aborted": aborted}
