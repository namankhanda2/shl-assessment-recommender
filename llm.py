"""Optional LLM client (OpenAI-compatible chat completions).

The service works fully without an LLM: every request falls back to the
deterministic catalogue-grounded planner in ``agent.py``. Configure one of the
supported keys to enable an LLM-authored reply / re-ranking.

Required: **one** OpenAI-compatible API key. The simplest is ``OPENAI_API_KEY``.
Any of these are auto-detected (first match wins):

    SHL_LLM_API_KEY       generic; pair with SHL_LLM_BASE_URL / SHL_LLM_MODEL
    OPENAI_API_KEY        https://api.openai.com/v1           gpt-4o-mini
    GROQ_API_KEY          https://api.groq.com/openai/v1      openai/gpt-oss-120b
    OPENROUTER_API_KEY    https://openrouter.ai/api/v1        openai/gpt-4o-mini
    TOGETHER_API_KEY      https://api.together.xyz/v1         llama-3.3-70b

Optional overrides:
    SHL_LLM_BASE_URL      force a specific OpenAI-compatible endpoint
    SHL_LLM_MODEL         force a specific model id
    SHL_LLM_TIMEOUT       request timeout in seconds (default 20)

Values may also be placed in a ``.env`` file next to this module; real
environment variables always take precedence. Run ``python llm.py`` to print the
resolved configuration without revealing the key.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:  # httpx is a FastAPI/TestClient dependency, but stay import-safe.
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

DEFAULT_TIMEOUT = 20.0


def _load_dotenv() -> None:
    """Populate missing environment variables from a local ``.env`` file."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except OSError:  # pragma: no cover - unreadable .env is non-fatal
        pass


_load_dotenv()


# (env var, base_url, default_model)
_PROVIDERS: List[Tuple[str, Optional[str], str]] = [
    ("SHL_LLM_API_KEY", None, "gpt-4o-mini"),
    ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini"),
    ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
    ("TOGETHER_API_KEY", "https://api.together.xyz/v1",
     "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
]


class LLMClient:
    def __init__(self) -> None:
        self.api_key: Optional[str] = None
        self.base_url: Optional[str] = None
        self.model: Optional[str] = None
        self.provider: Optional[str] = None
        self.last_error: Optional[str] = None
        self._configure()

    def _configure(self) -> None:
        self.api_key = self.base_url = self.model = self.provider = None
        explicit_base = os.getenv("SHL_LLM_BASE_URL")
        explicit_model = os.getenv("SHL_LLM_MODEL")
        try:
            self.timeout = float(os.getenv("SHL_LLM_TIMEOUT", DEFAULT_TIMEOUT))
        except ValueError:
            self.timeout = DEFAULT_TIMEOUT
        for env_key, base, model in _PROVIDERS:
            key = os.getenv(env_key)
            if key:
                self.api_key = key
                self.base_url = (explicit_base or base
                                 or "https://api.openai.com/v1").rstrip("/")
                self.model = explicit_model or model
                self.provider = env_key
                break

    def reload(self) -> "LLMClient":
        """Re-read environment variables (and ``.env``)."""
        _load_dotenv()
        self._configure()
        return self

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and httpx is not None)

    def _post(self, payload: Dict[str, Any], timeout: float) -> Optional[str]:
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - network dependent
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def complete(self, system: str, user: str, *, json_mode: bool = False,
                 max_tokens: int = 700, temperature: float = 0.2,
                 timeout: Optional[float] = None) -> Optional[str]:
        if not self.enabled:
            return None
        base_payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        timeout = timeout if timeout is not None else self.timeout
        if not json_mode:
            return self._post(base_payload, timeout)

        payload = {**base_payload, "response_format": {"type": "json_object"}}
        content = self._post(payload, timeout)
        if content is None:
            # Some OpenAI-compatible servers reject ``response_format``; retry
            # once without it and let the parser recover JSON from prose.
            content = self._post(base_payload, timeout)
        return content

    def complete_json(self, system: str, user: str, *, max_tokens: int = 700,
                      timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        raw = self.complete(system, user, json_mode=True, max_tokens=max_tokens,
                            timeout=timeout)
        if not raw:
            return None
        return _parse_json(raw)


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else None
            except Exception:
                return None
        return None


_default_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def reset_client() -> None:
    """Drop the cached client so new environment variables take effect."""
    global _default_client
    _default_client = None


def _describe() -> str:
    client = LLMClient()
    if not client.enabled:
        return (
            "LLM disabled (deterministic fallback active).\n"
            "Set an OpenAI-compatible key to enable it, e.g.:\n"
            "    export OPENAI_API_KEY=sk-...\n"
            "Other options: GROQ_API_KEY, OPENROUTER_API_KEY, TOGETHER_API_KEY,\n"
            "or SHL_LLM_API_KEY with SHL_LLM_BASE_URL and SHL_LLM_MODEL."
        )
    return (
        "LLM enabled.\n"
        f"  provider : {client.provider}\n"
        f"  base_url : {client.base_url}\n"
        f"  model    : {client.model}\n"
        f"  timeout  : {client.timeout:g}s\n"
        "  api_key  : set (hidden)"
    )


if __name__ == "__main__":  # pragma: no cover - manual diagnostic
    print(_describe())
