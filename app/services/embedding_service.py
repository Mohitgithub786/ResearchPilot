"""
Configurable embedding provider factory with process-level singleton caching.

ResearchPilot uses LangChain's Embeddings interface so providers can be swapped
without changing ingestion or retrieval code.

Supported providers:
- openai:       OpenAIEmbeddings            (production, API-based)
- huggingface:  HuggingFaceEmbeddings       (local, zero cost, privacy-friendly)
- test:         TestKeywordEmbeddings       (deterministic, no external deps)

Set EMBEDDING_PROVIDER in .env to select a provider.
Set HF_EMBEDDING_MODEL in .env to override the default HuggingFace model.

Singleton guarantee
-------------------
`get_embedding_model()` is decorated with `@lru_cache`. Python's lru_cache is
thread-safe: the GIL ensures the decorated function body executes exactly once
per unique argument tuple even under concurrent calls. Subsequent calls return
the already-constructed instance from the cache in O(1) with no model I/O.

This means one model instance per worker process, which is the correct scope
for Uvicorn (single-process dev) and Gunicorn+Uvicorn (multi-process prod).

Testing
-------
Call `get_embedding_model.cache_clear()` in test fixtures to force fresh
construction. monkeypatch EMBEDDING_PROVIDER=test before calling the factory
to avoid loading any real model during unit tests.
"""

from app.core.logging import get_logger
import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.core.config import HF_EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Test stub — fully deterministic, no external deps
# ---------------------------------------------------------------------------

class TestKeywordEmbeddings(Embeddings):
    """
    Deterministic test embeddings based on keyword presence.

    Used only when EMBEDDING_PROVIDER=test so pytest can verify retrieval
    without calling external APIs or loading PyTorch.
    """

    KEYWORDS = (
        "positional",
        "encoding",
        "attention",
        "transformer",
        "database",
        "sqlite",
        "vector",
    )

    def _vectorize(self, text: str) -> list[float]:
        text_lower = text.lower()
        vector = [1.0 if keyword in text_lower else 0.0 for keyword in self.KEYWORDS]
        vector.append(min(len(text) / 1000.0, 1.0))
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """
    Return the configured embedding model, loading it exactly once per process.

    The @lru_cache decorator makes this function a process-level singleton.
    The model is loaded the first time this function is called; every subsequent
    call returns the cached instance immediately (no disk I/O, no RAM spike).

    Changing EMBEDDING_PROVIDER at runtime has no effect after the first call.
    To use a different provider, restart the worker process.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        logger.info("Loading embedding model: OpenAI / %s", OPENAI_EMBEDDING_MODEL)
        model = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )
        logger.info("Embedding model loaded successfully (OpenAI).")
        return model

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model: HuggingFace / %s", HF_EMBEDDING_MODEL)
        model = HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL,
            # encode_kwargs controls the underlying sentence-transformers call
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully (HuggingFace / %s).", HF_EMBEDDING_MODEL)
        return model

    if provider == "test":
        logger.info("Loading embedding model: TestKeywordEmbeddings (stub).")
        return TestKeywordEmbeddings()

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER '{provider}'. "
        "Supported values: openai, huggingface, test."
    )


def get_embedding_dimension() -> int:
    """
    Return the vector dimension produced by the active embedding model.

    Probes the model by embedding a single character and measuring the output
    length.  Because get_embedding_model() is an @lru_cache singleton, the
    model is already loaded when this function is first called from
    chroma_service; the probe itself is essentially free (one tiny forward
    pass or API call).

    The result is cached in a module-level variable after the first call so
    subsequent accesses are O(1) dict lookups with no model interaction.
    """
    global _cached_dim
    if _cached_dim is None:
        sample = get_embedding_model().embed_query("x")
        _cached_dim = len(sample)
        logger.info("Detected embedding dimension: %d", _cached_dim)
    return _cached_dim


# Module-level cache for the detected dimension (reset alongside lru_cache in tests)
_cached_dim: int | None = None
