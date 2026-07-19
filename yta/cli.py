"""Command-line interface.

Usage:
  python -m yta add <youtube-url-or-id> [--lang en hi] [--title "..."]
  python -m yta add-playlist <playlist-or-channel-url> [--lang en hi]
  python -m yta channel add <channel-url>     (follow; 'list'/'remove' too)
  python -m yta sync                          (index missing videos from followed channels)
  python -m yta import <file.txt> --title "..." [--url <link>]
  python -m yta ask "question" [--top-k 5] [--summary]
  python -m yta list
  python -m yta delete <video-id>
"""

import argparse
import sys
from pathlib import Path

from . import db
from .search import Embedder
from .search import ask as _ask

# Windows consoles often default to a legacy codepage (cp1252) that cannot
# print non-Latin transcript text (e.g. Devanagari). Force UTF-8 output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _cmd_add(args) -> int:
    from . import ingest

    info = ingest.add_youtube_video(
        args.video,
        Embedder(),
        languages=args.lang,
        title=args.title,
    )
    print(
        f"Added: {info['title']} ({info['video_id']}) - "
        f"{info['chunks']} chunks, language: {info['language']}"
    )
    return 0


def _cmd_add_playlist(args) -> int:
    from . import playlist

    def progress(i, total, video, msg):
        print(f"[{i}/{total}] {video['title'][:60]} - {msg}", flush=True)

    result = playlist.add_playlist(
        args.url, Embedder(), languages=args.lang, progress=progress,
        refresh=args.refresh,
    )
    print(
        f"\nDone: {len(result['added'])} added, "
        f"{len(result['already_indexed'])} already indexed, "
        f"{len(result['skipped'])} skipped, "
        f"{result['not_attempted']} not attempted, {result['total']} total."
    )
    if result["skipped"]:
        print("\nSkipped:")
        for s in result["skipped"]:
            print(f"  {s['video_id']}  {s['title'][:60]}\n    -> {s['reason']}")
    if result["aborted"]:
        print(f"\nWARNING: {result['aborted']}", file=sys.stderr)
        return 1
    return 0


def _cmd_channel_add(args) -> int:
    from . import channels

    info = channels.follow_channel(args.url)
    print(f"Following: {info['name']} ({info['id']})")
    print("Run 'yta sync' to index its videos now; the API server also "
          "syncs followed channels automatically while it runs.")
    return 0


def _cmd_channel_list(_args) -> int:
    conn = db.connect()
    try:
        chans = db.list_channels(conn)
    finally:
        conn.close()
    if not chans:
        print("Not following any channels. Add one: yta channel add <url>")
        return 0
    for c in chans:
        synced = c["last_synced"] or "never"
        print(f"{c['id']:<26} {c['name']}  (last synced: {synced})")
    return 0


def _cmd_channel_remove(args) -> int:
    conn = db.connect()
    try:
        ok = db.delete_channel(conn, args.channel_id)
    finally:
        conn.close()
    print("Unfollowed. Its indexed videos stay searchable."
          if ok else f"Not following any channel with id {args.channel_id!r}.")
    return 0 if ok else 1


def _cmd_sync(_args) -> int:
    from . import channels

    def progress(i, total, video, msg):
        print(f"[{i}/{total}] {video['title'][:60]} - {msg}", flush=True)

    result = channels.sync_channels(Embedder(), progress=progress)
    for ch in result["channels"]:
        print(
            f"\n{ch['channel']}: {ch['videos_on_channel']} videos on channel, "
            f"{ch['new']} new, {len(ch['added'])} added, "
            f"{len(ch['skipped'])} skipped."
        )
        for s in ch["skipped"]:
            print(f"  skipped {s['video_id']}  {s['title'][:56]}\n    -> {s['reason']}")
    if result["aborted"]:
        print(f"\nWARNING: {result['aborted']}", file=sys.stderr)
        return 1
    return 0


def _cmd_import(args) -> int:
    from . import ingest

    text = Path(args.file).read_text(encoding="utf-8")
    info = ingest.add_manual_transcript(
        text, title=args.title, embedder=Embedder(), url=args.url
    )
    print(f"Imported: {info['title']} ({info['video_id']}) — {info['chunks']} chunks")
    return 0


def _cmd_ask(args) -> int:
    results = _ask(args.question, Embedder(), top_k=args.top_k)
    if not results:
        print("No transcripts in the database yet. Add one with: python -m yta add <url>")
        return 1

    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r['timestamp']}] {r['video_title']}  (score {r['score']:.2f})")
        if r["link"]:
            print(f"   {r['link']}")
        print(f"   \"{r['text'][:220]}{'...' if len(r['text']) > 220 else ''}\"")

    if args.summary:
        from .summarize import summarize

        print("\n--- Summary (Gemini) ---")
        try:
            print(summarize(args.question, results))
        except Exception as e:
            print(f"Summary unavailable: {e}", file=sys.stderr)
            return 1
    return 0


def _cmd_status(_args) -> int:
    from datetime import datetime, timezone

    conn = db.connect()
    try:
        videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        chans = db.list_channels(conn)
        last_blocked = db.get_meta(conn, "last_blocked_at")
    finally:
        conn.close()

    print(f"Library: {videos} videos, {chunks} chunks, {len(chans)} followed channel(s)")
    try:
        then = datetime.strptime(last_blocked or "", "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        then = None
    if then is not None:
        mins = int((datetime.now(timezone.utc) - then).total_seconds() // 60)
        if mins < 60:
            print(
                f"YouTube block: fetching was blocked {mins} min ago. Wait "
                f"~{60 - mins} more min without fetch attempts before adding/"
                f"syncing. Searching and manual import (yta import) work fine."
            )
        else:
            print(f"YouTube block: last seen {mins} min ago — likely cleared.")
    else:
        print("YouTube block: none recorded.")
    return 0


def _cmd_list(_args) -> int:
    conn = db.connect()
    try:
        videos = db.list_videos(conn)
    finally:
        conn.close()
    if not videos:
        print("Database is empty.")
        return 0
    for v in videos:
        print(f"{v['id']:<44} {v['chunks']:>4} chunks  {v['title']}")
    return 0


def _cmd_delete(args) -> int:
    conn = db.connect()
    try:
        ok = db.delete_video(conn, args.video_id)
    finally:
        conn.close()
    print("Deleted." if ok else f"No video with id {args.video_id!r}.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yta", description="Search YouTube transcripts for timestamped answers."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add", help="Fetch and index a YouTube video's transcript")
    p.add_argument("video", help="YouTube URL or 11-char video id")
    p.add_argument("--lang", nargs="+", default=["en"], help="Preferred transcript languages")
    p.add_argument("--title", help="Override the auto-fetched title")
    p.set_defaults(func=_cmd_add)

    p = sub.add_parser("add-playlist", help="Index every video in a playlist/channel")
    p.add_argument("url", help="YouTube playlist or channel URL")
    p.add_argument("--lang", nargs="+", default=["en"], help="Preferred transcript languages")
    p.add_argument("--refresh", action="store_true",
                   help="Re-fetch videos that are already indexed")
    p.set_defaults(func=_cmd_add_playlist)

    p = sub.add_parser("channel", help="Follow channels so their videos stay indexed")
    chsub = p.add_subparsers(dest="channel_command", required=True)
    pc = chsub.add_parser("add", help="Follow a channel")
    pc.add_argument("url", help="Channel URL (e.g. https://www.youtube.com/@handle)")
    pc.set_defaults(func=_cmd_channel_add)
    pc = chsub.add_parser("list", help="List followed channels")
    pc.set_defaults(func=_cmd_channel_list)
    pc = chsub.add_parser("remove", help="Unfollow a channel (keeps its videos)")
    pc.add_argument("channel_id")
    pc.set_defaults(func=_cmd_channel_remove)

    p = sub.add_parser("sync", help="Index missing videos from followed channels")
    p.set_defaults(func=_cmd_sync)

    p = sub.add_parser("import", help="Import a transcript from a text file")
    p.add_argument("file", help="Path to a .txt transcript (optionally '[MM:SS] text' lines)")
    p.add_argument("--title", required=True, help="Video/transcript title")
    p.add_argument("--url", help="Optional video link")
    p.set_defaults(func=_cmd_import)

    p = sub.add_parser("ask", help="Ask a question; get timestamped matches")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=5, help="Number of results (default 5)")
    p.add_argument("--summary", action="store_true", help="Also generate a Gemini summary")
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("list", help="List indexed videos")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("status", help="Library counts and YouTube block state")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("delete", help="Remove a video and its chunks")
    p.add_argument("video_id")
    p.set_defaults(func=_cmd_delete)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
