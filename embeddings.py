"""Dense semantic index for the SHL catalogue (FAISS + Sentence Transformers).

Optional: the service runs on pure-Python BM25 by default. Install the extras
and set ``SHL_USE_EMBEDDINGS=1`` to blend a dense semantic score into retrieval.

    pip install faiss-cpu sentence-transformers

The index is cached under ``.cache/embeddings/`` and rebuilt automatically when
the catalogue or model name changes. Sanity-check with::

    python embeddings.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from catalog import load_catalog

MODEL_NAME = os.getenv("SHL_EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
CACHE_DIR = os.getenv(
    "SHL_EMBEDDINGS_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "embeddings"),
)
SAMPLE_QUERY = "Java backend developer with SQL"

_model = None


def _deps():
    """Return ``(numpy, faiss, SentenceTransformer)`` or ``None`` if missing."""
    try:
        import numpy as np
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    return np, faiss, SentenceTransformer


def available() -> bool:
    return _deps() is not None


def get_model():
    """Load and memoise the Sentence Transformers model."""
    global _model
    deps = _deps()
    if deps is None:
        return None
    _, _, SentenceTransformer = deps
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def doc_text(item: Dict[str, Any]) -> str:
    keys = item.get("keys") or []
    return f"{item.get('name', '')}. {item.get('description', '')} {' '.join(keys)}".strip()


def _fingerprint(catalog: List[Dict[str, Any]]) -> str:
    digest = hashlib.sha256(MODEL_NAME.encode("utf-8"))
    for item in catalog:
        digest.update(item.get("name", "").encode("utf-8", "ignore"))
        digest.update(b"|")
        digest.update(item.get("description", "").encode("utf-8", "ignore"))
    return digest.hexdigest()[:16]


class DenseIndex:
    """A FAISS inner-product index over normalised document embeddings."""

    def __init__(self, catalog: List[Dict[str, Any]], vectors: Any, index: Any,
                 np: Any) -> None:
        self.catalog = catalog
        self.vectors = vectors
        self.index = index
        self._np = np

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, catalog: Optional[List[Dict[str, Any]]] = None) -> Optional["DenseIndex"]:
        deps = _deps()
        if deps is None:
            return None
        np, faiss, _ = deps
        catalog = catalog if catalog is not None else load_catalog()
        model = get_model()
        if model is None:
            return None
        vectors = np.asarray(
            model.encode([doc_text(it) for it in catalog],
                         normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(catalog, vectors, index, np)

    def save(self, directory: Optional[str] = None) -> str:
        directory = directory or CACHE_DIR
        os.makedirs(directory, exist_ok=True)
        self._np.save(os.path.join(directory, "vectors.npy"), self.vectors)
        meta = {
            "model": MODEL_NAME,
            "fingerprint": _fingerprint(self.catalog),
            "count": len(self.catalog),
            "dim": int(self.vectors.shape[1]),
        }
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        return directory

    @classmethod
    def load_or_build(cls, catalog: Optional[List[Dict[str, Any]]] = None,
                      directory: Optional[str] = None) -> Optional["DenseIndex"]:
        deps = _deps()
        if deps is None:
            return None
        np, faiss, _ = deps
        catalog = catalog if catalog is not None else load_catalog()
        directory = directory or CACHE_DIR
        meta_path = os.path.join(directory, "meta.json")
        vec_path = os.path.join(directory, "vectors.npy")
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if (meta.get("model") == MODEL_NAME
                    and meta.get("fingerprint") == _fingerprint(catalog)
                    and meta.get("count") == len(catalog)):
                vectors = np.asarray(np.load(vec_path), dtype="float32")
                index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)
                return cls(catalog, vectors, index, np)
        except Exception:
            pass
        built = cls.build(catalog)
        if built is not None:
            try:
                built.save(directory)
            except Exception:
                pass
        return built

    # -- query ------------------------------------------------------------
    def encode(self, texts: List[str]) -> Any:
        model = get_model()
        return self._np.asarray(
            model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )

    def similarities(self, query: str) -> Any:
        """Cosine similarity of ``query`` against every catalog item (via FAISS)."""
        n = self.index.ntotal
        if n == 0:
            return self._np.zeros(0, dtype="float32")
        query_vec = self.encode([query])
        scores, ids = self.index.search(query_vec, n)
        out = self._np.zeros(n, dtype="float32")
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                out[idx] = score
        return out

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Top-``k`` ``(catalog_index, score)`` pairs for ``query``."""
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(self.encode([query]), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def sample_search(index: Optional[DenseIndex]) -> None:
    if index is None:
        print("[embeddings] unavailable; install faiss-cpu sentence-transformers")
        return
    print(f"\nTop results for: {SAMPLE_QUERY}\n")
    for i, score in index.search(SAMPLE_QUERY, 5):
        print(f"  {score:.3f}  {index.catalog[i]['name']}")


if __name__ == "__main__":
    if not available():
        print("[embeddings] missing deps. install: pip install faiss-cpu sentence-transformers")
        sys.exit(1)
    sample_search(DenseIndex.load_or_build())
