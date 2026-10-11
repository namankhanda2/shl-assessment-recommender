"""Shared helpers for the SHL recommender tests."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Keep the suite hermetic: never let a real key (from the environment or a local
# .env) trigger live LLM calls during tests. Setting these to empty strings
# stops llm._load_dotenv() from populating them (it only fills *missing* keys)
# and LLMClient treats empty values as unconfigured. Tests that need a client
# inject a fake, or set their own key explicitly.
for _provider_key in (
    "SHL_LLM_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
    "OPENROUTER_API_KEY", "TOGETHER_API_KEY",
    "SHL_LLM_BASE_URL", "SHL_LLM_MODEL",
):
    os.environ[_provider_key] = ""


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}
