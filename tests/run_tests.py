"""Offline test suite — no YouTube requests, no embedding-model downloads.

Run:  python tests/run_tests.py
"""

import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_TMP = tempfile.mkdtemp(prefix="yta-tests-")
os.environ["YTA_DB_PATH"] = os.path.join(_TMP, "test.db")

from unittest.mock import patch  # noqa: E402
from xml.etree.ElementTree import ParseError  # noqa: E402

import numpy as np  # noqa: E402
from youtube_transcript_api._errors import IpBlocked  # noqa: E402

import yta.playlist as pl  # noqa: E402
from yta import channels as ch  # noqa: E402
from yta import db, ingest, search, utils

DB = os.environ["YTA_DB_PATH"]


class FakeEmbedder:
    """Deterministic 4-dim embedder so tests never download a model."""

    model_name = "fake"

    def encode(self, texts):
        out = []
        for t in texts:
            v = np.array(
                [len(t), t.count("pricing") * 10, t.count("hiring") * 10, 1.0],
                dtype=np.float32,
            )
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def _add_ok(vid, *_args, **_kwargs):
    conn = db.connect(DB)
    db.check_embedding_model(conn, "fake", claim=True)
    db.upsert_video(conn, vid, "t", None, "youtube")
    db.insert_chunks(
        conn, vid, [{"text": "x", "start": 0, "end": 1}],
        np.zeros((1, 4), dtype="float32"),
    )
    conn.close()
    return {"video_id": vid, "title": "t", "chunks": 1, "language": "en"}


FAKE = [{"id": f"vid{i:02d}xxxxxxx"[:11], "title": f"V{i}"} for i in range(6)]
passed = 0


def ok(msg):
    global passed
    passed += 1
    print(f"{passed:>2}. {msg}")


def test_utils():
    assert utils.extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=5") == "dQw4w9WgXcQ"
    assert utils.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert utils.extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert utils.format_timestamp(3725) == "1:02:05"
    assert utils.format_timestamp(None) == "--:--"
    assert utils.parse_timestamp("[1:02:03] x") == 3723
    assert utils.parse_timestamp("no stamp") is None
    ok("utils: id extraction, timestamp format/parse")


def test_ingest_and_ask():
    emb = FakeEmbedder()
    ingest.add_manual_transcript(
        "[0:10] pricing pricing pricing\n[2:30] hiring hiring hiring",
        title="T", embedder=emb, url="https://youtu.be/dQw4w9WgXcQ", db_path=DB,
    )
    r = search.ask("about pricing", emb, db_path=DB)
    assert r and "t=" in r[0]["link"] and r[0]["video_duration"] is not None
    ok("manual import + semantic ask: timestamp, deep link, duration")


def test_subtitles():
    srt = ("1\n00:00:01,000 --> 00:00:04,000\nfirst cue\n\n"
           "2\n00:02:10,500 --> 00:02:14,000\nsecond cue\n")
    s = ingest._parse_subtitle(srt)
    assert s[0]["start"] == 1.0 and s[1]["start"] == 130.5
    vtt = ("WEBVTT\n\n00:01.000 --> 00:04.000\n<c>hi</c> there\n\n"
           "00:04.000 --> 00:07.000\nhi there\n")
    s = ingest._parse_subtitle(vtt)
    assert len(s) == 1 and s[0]["text"] == "hi there"  # dup dropped, tags stripped
    assert ingest._parse_subtitle("plain text, no cues") is None
    # YouTube transcript-panel pastes: verbose "22 minutes, 48 seconds" stamps
    verbose = ("22 minutes, 48 secondsOne god brother named Kavana told us. "
               "1 hour, 2 minutes, 5 secondslater he was smiling.")
    chunks = ingest._chunk_plain_text(verbose)
    assert chunks[0]["start"] == 22 * 60 + 48
    assert "seconds" not in chunks[0]["text"]
    # ...and numeric timestamp-on-own-line pastes
    chunks = ingest._chunk_plain_text("0:05\nhello world\n2:10\nmore text")
    assert chunks[0]["start"] == 5 and "0:05" not in chunks[0]["text"]
    ok("SRT/VTT parsing + YouTube-panel pastes (verbose & numeric stamps)")


def test_keyword_bonus():
    texts = [
        "the main problem in Kaluga is what Brahmanas fell",   # similar-sounding
        "one of his god brothers his holiness Kavana told us", # exact rare match
        "everybody was dancing and sweating at the festival",
    ]
    bonus = db._keyword_bonus("which video contains about kavana", texts)
    assert bonus[1] > bonus[0] and bonus[1] > bonus[2], bonus
    assert bonus.max() <= db.KEYWORD_BOOST + 1e-9
    # no query words present anywhere -> all zeros
    assert db._keyword_bonus("zzz qqq", texts).sum() == 0
    ok("hybrid search: rare exact word outranks similar-sounding text")


def test_model_guard():
    class EmbB(FakeEmbedder):
        model_name = "other"

    try:
        search.ask("q", EmbB(), db_path=DB)
        raise AssertionError("guard should trip")
    except RuntimeError as e:
        assert "EMBEDDING_MODEL" in str(e)
    ok("embedding-model guard trips on mismatch")


def test_import_block_abort():
    with patch.object(pl, "time"), \
         patch.object(pl.ingest, "add_youtube_video", side_effect=IpBlocked("x")):
        res = pl.import_videos(FAKE, embedder=None, db_path=DB)
    assert res["aborted"] and len(res["skipped"]) == 3 and res["not_attempted"] == 3
    ok("import: abort after 3 consecutive blocks, partial results kept")


def test_import_disguised_block():
    with patch.object(pl, "time"), \
         patch.object(pl.ingest, "add_youtube_video",
                      side_effect=ParseError("no element found")):
        res = pl.import_videos(FAKE, embedder=None, db_path=DB)
    assert res["aborted"]
    ok("import: disguised block (empty-response ParseError) recognized")


def test_import_genuine_skips():
    with patch.object(pl, "time"), \
         patch.object(pl.ingest, "add_youtube_video",
                      side_effect=ValueError("No transcripts")):
        res = pl.import_videos(FAKE, embedder=None, db_path=DB)
    assert not res["aborted"] and len(res["skipped"]) == 6
    assert all(s["blocked"] is False for s in res["skipped"])
    ok("import: genuine no-caption skips never abort")


def test_skip_memory_cleared_on_success():
    conn = db.connect(DB)
    db.record_sync_attempt(conn, "retryvidxxx", "no captions")
    conn.close()
    with patch.object(pl, "time"), \
         patch.object(pl.ingest, "add_youtube_video", side_effect=_add_ok):
        pl.import_videos([{"id": "retryvidxxx", "title": "R"}],
                         embedder=None, db_path=DB)
    conn = db.connect(DB)
    assert "retryvidxxx" not in db.recently_attempted_ids(conn, 7)
    conn.close()
    ok("successful add clears its old skip record")


def test_channel_sync():
    fake_ch = {"id": "UCx", "name": "Chan", "url": "https://youtube.com/channel/UCx"}
    with patch.object(ch, "resolve_channel", return_value=fake_ch):
        ch.follow_channel("u", db_path=DB)
    with patch.object(pl, "time"), \
         patch.object(ch, "enumerate_videos", return_value=FAKE[:3]), \
         patch.object(pl.ingest, "add_youtube_video", side_effect=_add_ok):
        r1 = ch.sync_channels(embedder=None, db_path=DB)
    assert len(r1["channels"][0]["added"]) == 3
    with patch.object(ch, "enumerate_videos", return_value=FAKE[:3]), \
         patch.object(pl.ingest, "add_youtube_video", side_effect=AssertionError):
        r2 = ch.sync_channels(embedder=None, db_path=DB)
    assert r2["channels"][0]["new"] == 0
    ok("channel sync: adds missing videos, re-sync fetches nothing")


def test_api():
    from fastapi.testclient import TestClient

    from yta import api

    api._embedder = FakeEmbedder()
    with TestClient(api.app) as c:
        assert "<title>YTA" in c.get("/").text
        st = c.get("/status").json()
        assert {"videos", "blocked_recently", "sync_running"} <= set(st)
        assert c.get("/ask", params={"q": "pricing"}).json()["results"]
        assert c.get("/videos").status_code == 200
        assert c.get("/channels").status_code == 200
        # one-at-a-time import lock
        api._sync_lock.acquire()
        try:
            assert c.post("/playlists", json={"url": "x"}).status_code == 409
            assert c.post("/sync").status_code == 409
        finally:
            api._sync_lock.release()
    ok("API: UI, status, ask, listings, one-at-a-time import lock")


def test_single_add_records_block():
    from yta import api

    conn = db.connect(DB)
    conn.execute("DELETE FROM meta WHERE key = 'last_blocked_at'")
    conn.commit()
    conn.close()
    with patch.object(ingest, "_fetch_transcript", side_effect=IpBlocked("x")):
        try:
            ingest.add_youtube_video("dQw4w9WgXcQ", FakeEmbedder(), db_path=DB)
            raise AssertionError("should re-raise the block")
        except IpBlocked:
            pass
    assert api._minutes_since_block() == 0
    ok("single-video add records the block for status/UI")


def test_status_robustness():
    from yta import api

    conn = db.connect(DB)
    db.set_meta(conn, "last_blocked_at", "garbage")
    conn.close()
    assert api._minutes_since_block() is None
    ok("status: malformed block timestamp degrades gracefully")


if __name__ == "__main__":
    # Explicit order: later groups build on earlier ones (seeded DB, etc.).
    for fn in [
        test_utils,
        test_ingest_and_ask,
        test_subtitles,
        test_keyword_bonus,
        test_model_guard,
        test_import_block_abort,
        test_import_disguised_block,
        test_import_genuine_skips,
        test_skip_memory_cleared_on_success,
        test_channel_sync,
        test_api,
        test_single_add_records_block,
        test_status_robustness,
    ]:
        fn()
    print(f"\n{passed} test groups: ALL PASS")
