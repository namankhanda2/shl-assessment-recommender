"""Tests for the Streamlit frontend's API client (frontend/api_client.py)."""
import json
import os
import unittest
from unittest import mock

import httpx

from tests import REPO_ROOT  # noqa: F401  (ensures repo root is importable)

from frontend import api_client


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class PayloadTests(unittest.TestCase):
    def test_build_payload_matches_backend_schema(self):
        payload = api_client.build_payload("  Senior Python developer  ", 5)
        self.assertEqual(
            payload,
            {
                "messages": [{"role": "user", "content": "Senior Python developer"}],
                "k": 5,
            },
        )

    def test_blank_query_raises_readable_error(self):
        with self.assertRaises(api_client.ApiError):
            api_client.build_payload("   ", 5)

    def test_count_is_clamped_to_backend_range(self):
        self.assertEqual(api_client.build_payload("x", 0)["k"], 1)
        self.assertEqual(api_client.build_payload("x", 99)["k"], 10)
        self.assertEqual(api_client.build_payload("x", "nonsense")["k"], 10)

    def test_resolve_base_url_precedence_and_trim(self):
        self.assertEqual(
            api_client.resolve_base_url("https://api.example.com/"),
            "https://api.example.com",
        )
        with mock.patch.dict(os.environ, {"SHL_API_URL": "https://env.example.com/"}):
            self.assertEqual(api_client.resolve_base_url(), "https://env.example.com")
            self.assertEqual(
                api_client.resolve_base_url("https://arg.example.com"),
                "https://arg.example.com",
            )

    def test_test_type_label(self):
        self.assertEqual(api_client.test_type_label("k"), "Knowledge & Skills")
        self.assertEqual(api_client.test_type_label("A"), "Ability & Aptitude")
        self.assertEqual(api_client.test_type_label(None), "—")


class RequestTests(unittest.TestCase):
    def test_success_posts_to_chat_and_parses_body(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "reply": "Here you go",
                    "recommendations": [
                        {
                            "name": "Python (New)",
                            "url": "https://www.shl.com/x",
                            "test_type": "K",
                            "rationale": "Relevant to Python.",
                            "url_status": "generic_redirect",
                            "url_verified": False,
                        }
                    ],
                    "end_of_conversation": False,
                },
            )

        data = api_client.request_recommendations(
            "https://api.test", "python dev", 3, client=_client(handler)
        )
        self.assertTrue(seen["url"].endswith("/chat"))
        self.assertEqual(seen["body"]["k"], 3)
        self.assertEqual(seen["body"]["messages"][0]["content"], "python dev")
        self.assertEqual(data["recommendations"][0]["name"], "Python (New)")

    def test_http_error_becomes_readable_api_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"detail": "messages required"})

        with self.assertRaises(api_client.ApiError) as ctx:
            api_client.request_recommendations(
                "https://api.test", "x", 5, client=_client(handler)
            )
        self.assertIn("422", str(ctx.exception))
        self.assertIn("messages required", str(ctx.exception))

    def test_connection_error_becomes_readable_api_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        with self.assertRaises(api_client.ApiError) as ctx:
            api_client.request_recommendations(
                "https://api.test", "x", 5, client=_client(handler)
            )
        self.assertIn("Could not connect", str(ctx.exception))

    def test_unexpected_shape_is_rejected(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": True})

        with self.assertRaises(api_client.ApiError):
            api_client.request_recommendations(
                "https://api.test", "x", 5, client=_client(handler)
            )

    def test_health_success_and_failure(self):
        def ok(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "ok", "catalog": {"count": 377}})

        health = api_client.check_health("https://api.test", client=_client(ok))
        self.assertEqual(health["status"], "ok")

        def bad(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="down")

        with self.assertRaises(api_client.ApiError):
            api_client.check_health("https://api.test", client=_client(bad))


if __name__ == "__main__":
    unittest.main()
