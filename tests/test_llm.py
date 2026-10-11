"""Tests for the optional LLM layer and its graceful fallback."""
import os
import unittest
from unittest import mock

from tests import REPO_ROOT  # noqa: F401
from tests import assistant, user

import agent
import llm

PROVIDER_KEYS = [
    "SHL_LLM_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
    "OPENROUTER_API_KEY", "TOGETHER_API_KEY",
    "SHL_LLM_BASE_URL", "SHL_LLM_MODEL", "SHL_LLM_TIMEOUT",
]


class _FakeClient:
    enabled = True

    def __init__(self, data):
        self.data = data

    def complete_json(self, *args, **kwargs):
        return self.data


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _chat_payload(content):
    return {"choices": [{"message": {"content": content}}]}


class LlmLayerTests(unittest.TestCase):
    def setUp(self):
        self._orig = agent.get_client

    def tearDown(self):
        agent.get_client = self._orig

    def test_disabled_client_keeps_deterministic_output(self):
        history = [user("Hiring a Java developer")]
        result = agent.respond(history)
        self.assertTrue(result["recommendations"])
        for rec in result["recommendations"]:
            self.assertTrue(rec["url"].startswith("http"))

    def test_llm_reply_and_selection_are_used(self):
        result = agent.respond([user("Hiring a Java developer")])
        recs = result["recommendations"]
        self.assertTrue(recs)
        agent.get_client = lambda: _FakeClient(
            {"reply": "LLM reply", "selected": [recs[1]["name"], recs[0]["name"]]}
        )
        reply, picked = agent._llm_refine(
            [user("Hiring a Java developer")], "base", recs, "recommend")
        self.assertEqual(reply, "LLM reply")
        self.assertEqual(picked[0]["name"], recs[1]["name"])

    def test_unknown_selected_names_are_ignored(self):
        result = agent.respond([user("Hiring a Java developer")])
        recs = result["recommendations"]
        agent.get_client = lambda: _FakeClient(
            {"reply": "hi", "selected": ["Totally Made Up Test"]}
        )
        reply, picked = agent._llm_refine(
            [user("Hiring a Java developer")], "base", recs, "recommend")
        self.assertEqual(reply, "hi")
        self.assertEqual(picked, recs)

    def test_failed_llm_falls_back(self):
        result = agent.respond([user("Hiring a Java developer")])
        recs = result["recommendations"]
        agent.get_client = lambda: _FakeClient(None)
        reply, picked = agent._llm_refine(
            [user("Hiring a Java developer")], "base", recs, "recommend")
        self.assertEqual(reply, "base")
        self.assertEqual(picked, recs)

    def test_parse_json_variants(self):
        self.assertEqual(llm._parse_json('{"a": 1}'), {"a": 1})
        self.assertEqual(llm._parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(llm._parse_json('noise {"a": 1} tail'), {"a": 1})
        self.assertIsNone(llm._parse_json("not json"))
        self.assertIsNone(llm._parse_json("[1, 2, 3]"))


class LlmProviderConfigTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in PROVIDER_KEYS}
        for key in PROVIDER_KEYS:
            os.environ.pop(key, None)
        llm.reset_client()

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        llm.reset_client()

    def test_disabled_without_any_key(self):
        self.assertFalse(llm.LLMClient().enabled)

    def test_openai_key_is_detected(self):
        os.environ["OPENAI_API_KEY"] = "sk-test"
        client = llm.LLMClient()
        self.assertTrue(client.enabled)
        self.assertEqual(client.provider, "OPENAI_API_KEY")
        self.assertEqual(client.base_url, "https://api.openai.com/v1")
        self.assertEqual(client.model, "gpt-4o-mini")

    def test_generic_key_with_overrides(self):
        os.environ["SHL_LLM_API_KEY"] = "secret"
        os.environ["SHL_LLM_BASE_URL"] = "http://localhost:1234/v1/"
        os.environ["SHL_LLM_MODEL"] = "local-model"
        client = llm.LLMClient()
        self.assertTrue(client.enabled)
        self.assertEqual(client.provider, "SHL_LLM_API_KEY")
        self.assertEqual(client.base_url, "http://localhost:1234/v1")
        self.assertEqual(client.model, "local-model")

    def test_generic_key_has_priority(self):
        os.environ["SHL_LLM_API_KEY"] = "secret"
        os.environ["OPENAI_API_KEY"] = "sk-test"
        self.assertEqual(llm.LLMClient().provider, "SHL_LLM_API_KEY")

    def test_reset_client_rereads_environment(self):
        self.assertFalse(llm.get_client().enabled)
        os.environ["OPENAI_API_KEY"] = "sk-test"
        llm.reset_client()
        self.assertTrue(llm.get_client().enabled)
        os.environ.pop("OPENAI_API_KEY", None)
        llm.reset_client()
        self.assertFalse(llm.get_client().enabled)


class LlmClientHttpTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in PROVIDER_KEYS}
        for key in PROVIDER_KEYS:
            os.environ.pop(key, None)
        os.environ["OPENAI_API_KEY"] = "sk-test"
        self.client = llm.LLMClient()

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        llm.reset_client()

    def test_complete_posts_to_chat_completions(self):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append({"url": url, "headers": headers, "json": json})
            return _FakeResponse(_chat_payload("hello world"))

        with mock.patch.object(llm.httpx, "post", fake_post):
            out = self.client.complete("system", "user")

        self.assertEqual(out, "hello world")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["url"].endswith("/chat/completions"))
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(calls[0]["json"]["model"], "gpt-4o-mini")
        self.assertEqual(calls[0]["json"]["messages"][0]["content"], "system")
        self.assertNotIn("response_format", calls[0]["json"])

    def test_complete_json_parses_model_output(self):
        def fake_post(*args, **kwargs):
            return _FakeResponse(_chat_payload('```json\n{"reply": "hi"}\n```'))

        with mock.patch.object(llm.httpx, "post", fake_post):
            out = self.client.complete_json("system", "user")
        self.assertEqual(out, {"reply": "hi"})

    def test_json_mode_falls_back_without_response_format(self):
        payloads = []

        def fake_post(url, headers=None, json=None, timeout=None):
            payloads.append(json)
            if len(payloads) == 1:
                raise RuntimeError("response_format not supported")
            return _FakeResponse(_chat_payload('{"selected": []}'))

        with mock.patch.object(llm.httpx, "post", fake_post):
            out = self.client.complete("system", "user", json_mode=True)

        self.assertEqual(len(payloads), 2)
        self.assertIn("response_format", payloads[0])
        self.assertNotIn("response_format", payloads[1])
        self.assertEqual(out, '{"selected": []}')

    def test_complete_error_returns_none_and_records_error(self):
        def boom(*args, **kwargs):
            raise ConnectionError("network down")

        with mock.patch.object(llm.httpx, "post", boom):
            out = self.client.complete("system", "user")
        self.assertIsNone(out)
        self.assertIn("network down", self.client.last_error or "")


if __name__ == "__main__":
    unittest.main()
