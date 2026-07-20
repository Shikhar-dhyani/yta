# YouTube Transcript Answer

Ask a question and get the **exact moment** in your indexed YouTube videos that answers it — timestamp, video title, and a deep link (`&t=123s`). Optionally, a **Gemini-generated summary** of the answer.

## How it works

1. **Ingest** — transcripts are fetched from YouTube (or imported from text files), split into ~500-char timestamped chunks.
2. **Embed** — each chunk is embedded locally with `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`: free, offline, no API key, understands Hindi + 50 other languages, and cross-lingual — ask in English, match Hindi transcripts).
3. **Store** — videos, chunks, and embedding vectors live in a single SQLite file (`transcripts.db`).
4. **Ask** — your question is embedded and matched against all chunks by cosine similarity; the top hits come back with timestamps and links.
5. **Summarize (optional)** — the top hits are sent to Gemini to write a concise, cited answer.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
copy .env.example .env   # then put your GEMINI_API_KEY in .env (only needed for summaries)
```

Installing the package adds a `yta` command, so `yta ask "..."` works from any directory (`python -m yta ...` works too). The database always lives in the project root regardless of where you run from.

> First run downloads the embedding model (~470 MB) from Hugging Face; after that everything except the summary works offline.

> **Windows Smart App Control:** if Windows blocks PyTorch's unsigned DLLs ("Application Control policy has blocked this file"), the app automatically switches to an ONNX Runtime backend (Microsoft-signed, same model, same quality). The database records which backend built it, so the two are never mixed; set `EMBEDDING_MODEL=onnx:paraphrase-multilingual-MiniLM-L12-v2` to pin the ONNX backend explicitly. For an English-only library you can set `EMBEDDING_MODEL=all-MiniLM-L6-v2` in `.env` (~90 MB, faster) — but re-add all videos after switching models, since embeddings from different models don't mix.

## CLI usage

```powershell
# Index a YouTube video (auto-fetches transcript + title)
python -m yta add "https://www.youtube.com/watch?v=VIDEO_ID"
python -m yta add VIDEO_ID --lang en hi          # preferred caption languages

# Index every video in a playlist or channel (skips videos without captions)
python -m yta add-playlist "https://www.youtube.com/playlist?list=PLAYLIST_ID"
python -m yta add-playlist "..." --refresh    # re-fetch videos already indexed

# Follow channels: their videos — including future uploads — stay indexed
python -m yta channel add "https://www.youtube.com/@handle"
python -m yta sync                            # index whatever is missing right now
python -m yta channel list
python -m yta channel remove CHANNEL_ID       # unfollow (keeps indexed videos)

# Import a transcript manually — .srt / .vtt subtitle files, or plain text
# (optionally with lines like "[12:34] some text"). Works even while YouTube
# fetching is rate-limited. The web UI can bulk-upload many files at once.
python -m yta import talk.srt --title "My Talk" --url "https://youtu.be/VIDEO_ID"

# Library counts + whether YouTube is currently rate-limiting this network
python -m yta status

# Ask a question
python -m yta ask "what did they say about pricing?"
python -m yta ask "what did they say about pricing?" --summary   # + Gemini answer

# Manage
python -m yta list
python -m yta delete VIDEO_ID
```

Example output:

```
1. [12:41] How We Price Our Product  (score 0.62)
   https://www.youtube.com/watch?v=abc123def45&t=761s
   "...so the way we thought about pricing was to anchor on value..."
```

## Web UI + HTTP API

```powershell
uvicorn yta.api:app --reload
```

- **Web UI** at http://127.0.0.1:8000 — search with timestamped results and seek-position markers, Gemini summaries, and library management (add video / playlist / manual import, remove). Shareable links: `/?q=your+question` runs a search, `/#library` opens the library.
- Interactive API docs at http://127.0.0.1:8000/docs

| Method | Path             | Body / Query                              | Purpose                    |
|--------|------------------|-------------------------------------------|----------------------------|
| POST   | `/videos`        | `{"url": "...", "languages": ["en"]}`     | Index a YouTube video      |
| POST   | `/playlists`     | `{"url": "...", "languages": ["en"]}`     | Index a playlist/channel   |
| POST   | `/videos/manual` | `{"title": "...", "text": "...", "url"?}` | Import a manual transcript |
| GET    | `/videos`        |                                           | List indexed videos        |
| DELETE | `/videos/{id}`   |                                           | Remove a video             |
| GET    | `/ask`           | `?q=...&top_k=5&summary=true`             | Search (+ Gemini summary)  |

## Project layout

```
yta/
  config.py     # env-based settings (.env supported)
  utils.py      # YouTube id parsing, timestamps, deep links
  db.py         # SQLite schema + cosine-similarity search
  ingest.py     # YouTube fetch / manual import + chunking
  playlist.py   # playlist/channel enumeration (yt-dlp) + batch ingest
  search.py     # embedding model wrapper + ask()
  static/       # web UI (single self-contained page)
  summarize.py  # Gemini summary add-on
  cli.py        # command-line interface (python -m yta ...)
  api.py        # FastAPI app (uvicorn yta.api:app)
```

## Followed channels & automatic updates

This is a single-user tool: everything runs locally against one SQLite file, and the web UI binds to your machine only.

- **Follow a channel** (UI Library tab or `yta channel add <url>`). Following stores the channel; indexing happens on sync.
- **Sync** compares the channel's uploads against your index and fetches only what's missing — new uploads included. Trigger it three ways: the **Sync now** button in the UI, `yta sync` in the terminal, or automatically: while the API server is running it syncs all followed channels every 6 hours (`YTA_SYNC_INTERVAL_MINUTES`, 0 disables).
- Videos skipped for a real reason (usually "no captions yet") are remembered and retried after `YTA_SKIP_RETRY_DAYS` (default 7) — new uploads often gain auto-captions a few hours after publishing, so they get picked up on a later sync.
- Rate-limit blocks are never memorized; those videos stay eligible for the very next sync.
- For fully unattended updates without keeping the server running, schedule `yta sync` in Windows Task Scheduler.

## Tests

```powershell
python tests/run_tests.py
```

Fully offline — mocks YouTube and the embedding model, uses a throwaway database.

## Notes & limits

- Search is hybrid: embedding cosine similarity plus a rarity-weighted exact-word bonus (so names and jargon rank correctly), optionally sharpened by a cross-encoder reranker (`YTA_RERANKER`). The brute-force scan is plenty fast for a personal DB (thousands of chunks); past ~100k chunks, swap in FAISS or sqlite-vec.
- Switching `EMBEDDING_MODEL` (or the reranker finding better/worse) never mixes vector spaces: the DB records which model built it, and `yta reindex` rebuilds all chunks/embeddings under the new model while preserving timestamps.
- YouTube fetching needs the video to have captions (manual or auto-generated). If none of your preferred `--lang` languages exist, it automatically falls back to whatever language the video has (manual captions preferred over auto-generated). For videos without any captions, use `import` with your own transcript.
- Re-adding a video replaces its previous chunks (safe to re-run).
- When YouTube rate-limits your network, the web UI shows a banner with how long ago the block happened and when it should clear (`yta status` shows the same in the terminal). Searching and manual import keep working throughout.
- YouTube rate-limits bursts of transcript requests. Playlist import paces itself (`YTA_PLAYLIST_DELAY` seconds between videos, default 3), retries blocked requests once after 60s, and aborts with a clear message if the block persists. Already-indexed videos are never re-fetched (use `--refresh` to force), so re-running a partially imported playlist continues where it left off.
