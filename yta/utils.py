"""Small helpers: YouTube ID parsing, timestamp formatting and links."""

import re
from urllib.parse import parse_qs, urlparse

_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Return the 11-char YouTube video id from a URL or a bare id.

    Accepts full watch URLs, youtu.be short links, embed URLs, or the id itself.
    """
    s = url_or_id.strip()
    if _YT_ID_RE.match(s):
        return s

    parsed = urlparse(s)
    host = parsed.netloc.lower()

    if "youtu.be" in host:
        candidate = parsed.path.lstrip("/").split("/")[0]
        if _YT_ID_RE.match(candidate):
            return candidate

    if "youtube.com" in host:
        # /watch?v=ID
        qs = parse_qs(parsed.query)
        if "v" in qs and _YT_ID_RE.match(qs["v"][0]):
            return qs["v"][0]
        # /embed/ID or /shorts/ID or /v/ID
        parts = [p for p in parsed.path.split("/") if p]
        for i, part in enumerate(parts):
            if part in ("embed", "shorts", "v") and i + 1 < len(parts):
                candidate = parts[i + 1]
                if _YT_ID_RE.match(candidate):
                    return candidate

    raise ValueError(f"Could not extract a YouTube video id from: {url_or_id!r}")


def watch_url(video_id: str, start_seconds: float | None = None) -> str:
    """Build a youtube.com watch URL, optionally deep-linked to a timestamp."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    if start_seconds is not None:
        url += f"&t={int(start_seconds)}s"
    return url


def format_timestamp(seconds: float | None) -> str:
    """Format seconds as H:MM:SS or M:SS. Returns '--:--' when unknown."""
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def parse_timestamp(text: str) -> float | None:
    """Parse a leading timestamp like '1:23', '01:23', '[1:02:03]' -> seconds.

    Returns None if the text does not start with a recognizable timestamp.
    """
    m = re.match(r"^\s*\[?\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*\]?", text)
    if not m:
        return None
    a, b, c = m.group(1), m.group(2), m.group(3)
    if c is not None:
        return int(a) * 3600 + int(b) * 60 + int(c)
    return int(a) * 60 + int(b)
