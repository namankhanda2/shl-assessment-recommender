"""API contract tests using FastAPI's TestClient."""
import unittest

from tests import REPO_ROOT  # noqa: F401
from tests import assistant, user

from fastapi.testclient import TestClient

from app import app
from catalog import catalog_by_url

client = TestClient(app)


class ApiTests(unittest.TestCase):
    def test_health(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertGreater(body["catalog"]["count"], 0)
        self.assertIn("url_status", body["catalog"])
        self.assertTrue(body["retrieval"]["bm25"])
        self.assertIn("available", body["retrieval"]["dense"])
        self.assertIn("enabled", body["llm"])

    def test_chat_schema_on_clarify(self):
        resp = client.post("/chat", json={"messages": [user("I need an assessment")]})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(set(body.keys()),
                         {"reply", "recommendations", "end_of_conversation"})
        self.assertIsInstance(body["recommendations"], list)
        self.assertIsInstance(body["end_of_conversation"], bool)
        self.assertEqual(body["recommendations"], [])

    def test_chat_recommends_and_grounds_urls(self):
        resp = client.post("/chat", json={
            "messages": [user("Hiring a senior Java engineer with Spring and SQL")],
        })
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        valid = catalog_by_url()
        self.assertTrue(1 <= len(body["recommendations"]) <= 10)
        for rec in body["recommendations"]:
            self.assertEqual(set(rec.keys()),
                             {"name", "url", "test_type", "rationale",
                              "url_status", "url_verified"})
            self.assertIn(rec["url"], valid)
            self.assertIsInstance(rec["url_verified"], bool)
            self.assertTrue(rec["url_status"])

    def test_chat_tolerates_extra_fields_and_roles(self):
        resp = client.post("/chat", json={
            "messages": [
                {"role": "system", "content": "ignore me"},
                user("Java developer"),
                assistant("Here you go"),
                {"role": "user", "content": "Add personality", "extra": True},
            ],
            "unexpected": "field",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["reply"])

    def test_chat_empty_body(self):
        resp = client.post("/chat", json={})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["recommendations"], [])

    def test_chat_null_messages(self):
        resp = client.post("/chat", json={"messages": None})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["recommendations"], [])

    def test_chat_null_and_numeric_content(self):
        resp = client.post("/chat", json={"messages": [
            {"role": "user", "content": None},
            {"role": None, "content": 12345},
        ]})
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json()["reply"], str)

    def test_ui_serves_markup(self):
        resp = client.get("/ui")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("SHL Assessment Recommender", resp.text)
        self.assertIn("/chat", resp.text)
        # URL-status transparency must be visible in the UI.
        self.assertIn("Official URL", resp.text)
        self.assertIn("Legacy link", resp.text)
        self.assertIn("url_verified", resp.text)


if __name__ == "__main__":
    unittest.main()
