"""Opt-in live LLM integration test.

Skipped by default. It only runs when ``SHL_LLM_INTEGRATION=1`` is set *and* a
real provider key is available (from the environment or a local ``.env``). It
makes exactly one real completion request and never prints the key.

    SHL_LLM_INTEGRATION=1 pytest tests/test_llm_integration.py -q
"""
import os
import unittest

from tests import REPO_ROOT  # noqa: F401

import llm

PROVIDER_KEYS = [
    "SHL_LLM_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
    "OPENROUTER_API_KEY", "TOGETHER_API_KEY",
    "SHL_LLM_BASE_URL", "SHL_LLM_MODEL", "SHL_LLM_TIMEOUT",
]


def _integration_enabled() -> bool:
    return os.getenv("SHL_LLM_INTEGRATION", "").lower() in {"1", "true", "yes"}


@unittest.skipUnless(_integration_enabled(),
                     "set SHL_LLM_INTEGRATION=1 with valid credentials to run")
class LiveLlmIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in PROVIDER_KEYS}
        for key in PROVIDER_KEYS:
            os.environ.pop(key, None)
        llm._load_dotenv()  # restore any real values from the local .env file
        self.client = llm.LLMClient()

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        llm.reset_client()

    def test_single_real_completion(self):
        if not self.client.enabled:
            self.skipTest("no provider credentials configured")
        out = self.client.complete(
            "You are terse.", "Reply with the single word: ok",
            max_tokens=256, timeout=30,
        )
        self.assertIsNotNone(out, f"LLM call failed: {self.client.last_error}")
        self.assertTrue(out and out.strip())


if __name__ == "__main__":
    unittest.main()
