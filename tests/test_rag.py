"""Tests for the retrieval-augmented generation loop."""
import unittest
from unittest import mock

from tests import REPO_ROOT  # noqa: F401
from tests import user

import rag


class _FakeClient:
    def __init__(self, data, enabled=True):
        self.data = data
        self.enabled = enabled

    def complete_json(self, *args, **kwargs):
        return self.data


class RagGenerateTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            {"name": "Java (New)", "url": "u1", "test_type": "K"},
            {"name": "SQL (New)", "url": "u2", "test_type": "K"},
            {"name": "OPQ", "url": "u3", "test_type": "P"},
        ]

    def test_disabled_client_returns_base(self):
        reply, picked = rag.generate(
            _FakeClient(None, enabled=False), [user("hi")], "base",
            self.docs, "recommend")
        self.assertEqual(reply, "base")
        self.assertEqual(picked, self.docs)

    def test_enabled_client_rewrites_and_reorders(self):
        client = _FakeClient({"reply": "RAG reply",
                              "selected": ["SQL (New)", "Java (New)"]})
        reply, picked = rag.generate(client, [user("hi")], "base",
                                     self.docs, "recommend")
        self.assertEqual(reply, "RAG reply")
        self.assertEqual([p["name"] for p in picked], ["SQL (New)", "Java (New)"])

    def test_unknown_selected_names_are_ignored(self):
        client = _FakeClient({"reply": "hi", "selected": ["Made Up Test"]})
        reply, picked = rag.generate(client, [user("hi")], "base",
                                     self.docs, "recommend")
        self.assertEqual(reply, "hi")
        self.assertEqual(picked, self.docs)

    def test_build_prompt_lists_candidates_and_cap(self):
        prompt = rag.build_prompt([user("hire java")], ["A", "B"], "recommend", 3)
        self.assertIn("- A", prompt)
        self.assertIn("- B", prompt)
        self.assertIn("max 3", prompt)


class RagEndToEndTests(unittest.TestCase):
    def test_answer_retrieves_and_falls_back_without_llm(self):
        import llm

        with mock.patch.object(llm, "get_client",
                               lambda: _FakeClient(None, enabled=False)):
            reply, docs = rag.answer("Hiring a Java developer with SQL", k=5)
        self.assertTrue(docs)
        for item in docs:
            self.assertIn("url", item)
        self.assertIsInstance(reply, str)

    def test_answer_uses_llm_selection_end_to_end(self):
        import llm

        def fake_client():
            return _FakeClient({"reply": "ok", "selected": []})

        with mock.patch.object(llm, "get_client", fake_client):
            reply, docs = rag.answer("Hiring a Java developer with SQL", k=5)
        self.assertEqual(reply, "ok")
        self.assertTrue(docs)


if __name__ == "__main__":
    unittest.main()
