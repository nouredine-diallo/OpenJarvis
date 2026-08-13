"""Tests for the embeddings abstraction layer."""

from __future__ import annotations

import pytest

st = pytest.importorskip("sentence_transformers")

from openjarvis.tools.storage.embeddings import (  # noqa: E402
    Embedder,
    SentenceTransformerEmbedder,
)


@pytest.fixture()
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


def test_produces_vectors(embedder: SentenceTransformerEmbedder):
    """embed() returns a numpy array with one row per input."""
    import numpy as np

    vecs = embedder.embed(["hello world"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 1


def test_correct_dimension(
    embedder: SentenceTransformerEmbedder,
):
    """Embedding dimension matches the declared dim()."""
    vecs = embedder.embed(["test"])
    assert vecs.shape[1] == embedder.dim()


def test_batch(embedder: SentenceTransformerEmbedder):
    """Batch of texts produces matching number of vectors."""
    texts = ["one", "two", "three"]
    vecs = embedder.embed(texts)
    assert vecs.shape[0] == 3
    assert vecs.shape[1] == embedder.dim()


def test_empty_input(embedder: SentenceTransformerEmbedder):
    """Empty list produces an empty array."""
    import numpy as np

    vecs = embedder.embed([])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 0


def test_missing_dep(monkeypatch: pytest.MonkeyPatch):
    """Import error is raised with a helpful message."""
    import builtins

    real_import = builtins.__import__

    def _block_st(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "sentence_transformers":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_st)
    with pytest.raises(ImportError, match="sentence-transformers"):
        SentenceTransformerEmbedder()


def test_embedder_abc_cannot_instantiate():
    """Embedder ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Embedder()  # type: ignore[abstract]


fastembed = pytest.importorskip("fastembed")

from openjarvis.tools.storage.embeddings import FastEmbedEmbedder  # noqa: E402


class TestFastEmbedEmbedder:
    """FastEmbedEmbedder is the default for the semantic memory layer
    (Brique 2) -- Ollama/Gemma were abandoned on this machine (OOM, see
    PLAN.md D9), so these tests exercise the real model, not a mock: the
    RAM-discipline behavior (load-on-use, release-after-use) is the actual
    thing worth verifying, and mocking it away would test nothing."""

    def test_produces_normalized_vectors(self):
        import numpy as np

        embedder = FastEmbedEmbedder()
        vecs = embedder.embed(["hello world"])
        assert vecs.shape == (1, embedder.dim())
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_empty_input(self):
        import numpy as np

        embedder = FastEmbedEmbedder()
        vecs = embedder.embed([])
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape[0] == 0

    def test_dim_matches_embed_output_without_forcing_a_second_load(self):
        embedder = FastEmbedEmbedder()
        vecs = embedder.embed(["probe"])
        assert embedder.dim() == vecs.shape[1]

    def test_model_not_resident_between_calls(self):
        """Brique 2 decision D1: never hold the ~1 GB model in RAM between
        calls -- load on use, release right after. Verified via the actual
        internal state, not just behavior, since that's the property being
        guaranteed."""
        embedder = FastEmbedEmbedder()
        assert embedder._model is None
        embedder.embed(["hello"])
        assert embedder._model is None  # released, not cached on the instance

    def test_missing_dep(self, monkeypatch: pytest.MonkeyPatch):
        import builtins

        real_import = builtins.__import__

        def _block_fastembed(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "fastembed":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_fastembed)
        with pytest.raises(ImportError, match="fastembed"):
            FastEmbedEmbedder().embed(["x"])
