"""Embedding model wrapper + question answering over the chunk store."""

import re
from dataclasses import asdict

import numpy as np

from . import config, db, utils


class Embedder:
    """Lazy-loading embedding model with two backends.

    Primary: sentence-transformers (PyTorch). Fallback: fastembed (ONNX
    Runtime, Microsoft-signed DLLs) — used automatically when PyTorch cannot
    load, e.g. when Windows Smart App Control blocks its unsigned binaries.

    The two backends produce slightly different vectors, so the model name
    is prefixed with "onnx:" for the fastembed backend — the database guard
    then refuses to mix them. Set EMBEDDING_MODEL=onnx:<name> to force the
    ONNX backend explicitly.
    """

    def __init__(self, model_name: str | None = None):
        raw = model_name or config.EMBEDDING_MODEL
        self._forced_onnx = raw.startswith("onnx:")
        self._base = raw.removeprefix("onnx:")
        self._backend: str | None = "onnx" if self._forced_onnx else None
        self._model = None

    def _detect_backend(self) -> str:
        if self._backend is None:
            try:
                import sentence_transformers  # noqa: F401  (imports torch)

                self._backend = "torch"
            except Exception:
                self._backend = "onnx"
        return self._backend

    @property
    def model_name(self) -> str:
        """Backend-qualified name; what the database guard stores."""
        if self._detect_backend() == "onnx":
            return f"onnx:{self._base}"
        return self._base

    def _load(self):
        if self._model is None:
            if self._detect_backend() == "torch":
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._base)
            else:
                from fastembed import TextEmbedding

                name = (
                    self._base if "/" in self._base
                    else f"sentence-transformers/{self._base}"
                )
                self._model = TextEmbedding(
                    name, cache_dir=str(config.MODELS_CACHE)
                )
        return self._model

    def _prep_texts(self, texts: list[str], is_query: bool) -> list[str]:
        """Apply model-required prefixes.

        E5-family models are trained with asymmetric "query: " / "passage: "
        prefixes; skipping them costs significant retrieval quality. Neither
        backend adds them automatically.
        """
        if "e5" in self._base.lower():
            prefix = "query: " if is_query else "passage: "
            return [prefix + t for t in texts]
        return texts

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode texts to L2-normalized float32 vectors."""
        model = self._load()
        texts = self._prep_texts(texts, is_query)
        if self._backend == "torch":
            return model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
        vecs = np.array(list(model.embed(texts)), dtype=np.float32)
        # fastembed does not normalize; our cosine search requires unit norms.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)


class Reranker:
    """Lazy cross-encoder that re-scores (question, chunk) pairs.

    A cross-encoder reads the question and passage together, so it separates
    right from wrong answers far better than embedding distance alone. Only
    the top retrieval candidates are re-scored, keeping queries fast.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.RERANKER_MODEL
        self._model = None

    def scores(self, question: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(
                self.model_name, cache_dir=str(config.MODELS_CACHE)
            )
        return list(self._model.rerank(question, texts))


def _rerank_hits(reranker, question: str, hits: list, top_k: int) -> list:
    """Reorder hits by cross-encoder score; scores become 0..1 (sigmoid)."""
    if not hits:
        return hits
    raw = reranker.scores(question, [h.text for h in hits])
    for h, s in zip(hits, raw, strict=True):
        h.score = float(1 / (1 + np.exp(-s)))
    return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]


def _snippet(text: str, question: str, width: int = 240) -> str:
    """A window of the chunk centered on the first query-word match.

    Chunks run ~500 chars; showing only the head can hide the very words
    that made the chunk match. Falls back to the head when nothing matches.
    """
    lowered = text.lower()
    positions = [
        p for tok in re.findall(r"[^\W\d_]{3,}", question.lower())
        if (p := lowered.find(tok)) >= 0
    ]
    if not positions or min(positions) < width // 3:
        return text[:width] + ("…" if len(text) > width else "")
    start = min(positions) - width // 3
    end = min(len(text), start + width)
    return "…" + text[start:end] + ("…" if end < len(text) else "")


def ask(
    question: str,
    embedder: Embedder,
    top_k: int = 5,
    db_path: str | None = None,
    reranker: Reranker | None = None,
) -> list[dict]:
    """Semantic search: return matching moments with timestamp + link.

    With a reranker configured (YTA_RERANKER), a wider candidate set is
    retrieved and re-scored by the cross-encoder before returning top_k.
    """
    if reranker is None and config.RERANKER_MODEL:
        reranker = Reranker()

    conn = db.connect(db_path)
    try:
        if conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0:
            return []  # empty library: skip loading the embedding model
        db.check_embedding_model(conn, embedder.model_name)
        q_vec = embedder.encode([question], is_query=True)[0]
        candidates = max(20, top_k * 4) if reranker else top_k
        hits = db.search(conn, q_vec, top_k=candidates, query_text=question)
        if reranker:
            hits = _rerank_hits(reranker, question, hits, top_k)
        # Approximate each video's duration (last chunk end) so callers can
        # show where in the video a moment falls.
        durations = dict(
            conn.execute("SELECT video_id, MAX(end) FROM chunks GROUP BY video_id")
        )
    finally:
        conn.close()

    results = []
    for h in hits:
        d = asdict(h)
        d["timestamp"] = utils.format_timestamp(h.start)
        d["video_duration"] = durations.get(h.video_id)
        d["snippet"] = _snippet(h.text, question)
        if h.video_url and h.start is not None and h.video_id.count(":") == 0:
            d["link"] = utils.watch_url(h.video_id, h.start)
        else:
            d["link"] = h.video_url
        results.append(d)
    return results
