"""Embedding model wrapper + question answering over the chunk store."""

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

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to L2-normalized float32 vectors."""
        model = self._load()
        if self._backend == "torch":
            return model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
        vecs = np.array(list(model.embed(texts)), dtype=np.float32)
        # fastembed does not normalize; our cosine search requires unit norms.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)


def ask(
    question: str,
    embedder: Embedder,
    top_k: int = 5,
    db_path: str | None = None,
) -> list[dict]:
    """Semantic search: return matching moments with timestamp + link."""
    conn = db.connect(db_path)
    try:
        if conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0:
            return []  # empty library: skip loading the embedding model
        db.check_embedding_model(conn, embedder.model_name)
        q_vec = embedder.encode([question])[0]
        hits = db.search(conn, q_vec, top_k=top_k, query_text=question)
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
        if h.video_url and h.start is not None and h.video_id.count(":") == 0:
            d["link"] = utils.watch_url(h.video_id, h.start)
        else:
            d["link"] = h.video_url
        results.append(d)
    return results
