"""Tests for the optional FAISS + Sentence Transformers dense index."""
import os
import tempfile
import unittest
from unittest import mock

from tests import REPO_ROOT  # noqa: F401

import embeddings as emb
import retrieval


def _item(name, description, keys):
    return {
        "name": name, "description": description, "keys": keys,
        "url": name.lower().replace(" ", "-"), "test_type": "K",
        "test_types": ["K"], "job_levels": [], "duration": "",
        "remote": False, "adaptive": False,
    }


@unittest.skipUnless(emb.available(), "faiss-cpu / sentence-transformers not installed")
class DenseIndexTests(unittest.TestCase):
    CATALOG = [
        _item("Java Programming (New)", "Test of Java coding and objects.",
              ["Knowledge & Skills"]),
        _item("Customer Service Simulation",
              "Contact centre scenarios and customer interactions.",
              ["Simulations"]),
        _item("Nursing Essentials", "Clinical healthcare and patient knowledge.",
              ["Knowledge & Skills"]),
    ]

    def test_build_and_search(self):
        index = emb.DenseIndex.build(self.CATALOG)
        self.assertEqual(index.index.ntotal, 3)
        hits = index.search("Java developer coding", k=1)
        self.assertEqual(hits[0][0], 0)

    def test_similarities_cover_every_item(self):
        index = emb.DenseIndex.build(self.CATALOG)
        sims = index.similarities("nursing")
        self.assertEqual(len(sims), 3)
        self.assertEqual(int(sims.argmax()), 2)

    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            emb.DenseIndex.build(self.CATALOG).save(directory)
            self.assertTrue(os.path.exists(os.path.join(directory, "vectors.npy")))
            loaded = emb.DenseIndex.load_or_build(self.CATALOG, directory=directory)
            self.assertEqual(loaded.index.ntotal, 3)

    def test_dense_scores_actually_change_ranking(self):
        """The dense component must measurably affect blended scores."""
        index = emb.DenseIndex.build(self.CATALOG)
        retriever = retrieval.Retriever(catalog=self.CATALOG)
        retriever._dense = index
        tokens = "nursing clinical patient care"
        base = retriever.bm25(retrieval.tokenize(tokens))
        blended = retriever._blend_dense(tokens, base, weight=0.5)
        self.assertNotEqual([round(x, 6) for x in base],
                            [round(x, 6) for x in blended])
        # The nursing item should win in the dense-only ranking.
        self.assertEqual(int(index.similarities(tokens).argmax()), 2)


class MissingDependencyTests(unittest.TestCase):
    def test_unavailable_deps_return_none(self):
        with mock.patch.object(emb, "_deps", lambda: None):
            self.assertFalse(emb.available())
            self.assertIsNone(emb.DenseIndex.build([_item("X", "y", ["K"])]))

    def test_retriever_falls_back_when_dense_unavailable(self):
        os.environ["SHL_USE_EMBEDDINGS"] = "1"
        original = emb.available
        try:
            emb.available = lambda: False
            retriever = retrieval.Retriever(catalog=retrieval.load_catalog()[:3])
            self.assertIsNone(retriever._dense)
            # Keyword search still works.
            self.assertIsInstance(retriever.search("java", k=1), list)
        finally:
            os.environ.pop("SHL_USE_EMBEDDINGS", None)
            emb.available = original

    def test_dense_disabled_by_default(self):
        os.environ.pop("SHL_USE_EMBEDDINGS", None)
        retriever = retrieval.Retriever(catalog=retrieval.load_catalog()[:3])
        self.assertIsNone(retriever._dense)


class DenseBlendTests(unittest.TestCase):
    def test_rank_applies_dense_blend(self):
        """Recommendation path (``rank``) must use the dense blend too."""
        import retrieval

        calls = []

        class FakeDense:
            def similarities(self, query):
                calls.append(query)
                return [0.5, 0.1]

        retriever = retrieval.Retriever(catalog=retrieval.load_catalog()[:2])
        retriever._dense = FakeDense()
        ranked = retriever.rank("java backend")
        self.assertEqual(calls, ["java backend"])
        self.assertEqual(len(ranked), 2)
        # Blended scores must differ from the pure-BM25 ordering check: the
        # dense component contributes a normalised term to every score.
        blended = {item["name"]: score for item, score in ranked}
        self.assertNotEqual(round(sum(blended.values()), 6), 0.0)

    def test_retriever_loads_and_uses_dense_when_enabled(self):
        import retrieval

        calls = []

        class FakeDense:
            def similarities(self, query):
                calls.append(query)
                return [0.0] * 2

        original_available = emb.available
        original_load = emb.DenseIndex.load_or_build
        os.environ["SHL_USE_EMBEDDINGS"] = "1"
        try:
            emb.available = lambda: True
            emb.DenseIndex.load_or_build = classmethod(
                lambda cls, catalog: FakeDense())
            retriever = retrieval.Retriever(catalog=retrieval.load_catalog()[:2])
            self.assertIsNotNone(retriever._dense)
            retriever.search("java", k=1)
            self.assertEqual(calls, ["java"])
        finally:
            os.environ.pop("SHL_USE_EMBEDDINGS", None)
            emb.available = original_available
            emb.DenseIndex.load_or_build = original_load


if __name__ == "__main__":
    unittest.main()
