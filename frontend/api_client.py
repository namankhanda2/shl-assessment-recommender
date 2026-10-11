"""Thin, Streamlit-free client for the SHL recommender HTTP API.

Used by ``frontend/streamlit_app.py`` but deliberately free of any Streamlit
import so it can be unit-tested and reused. It never reads the catalog directly:
every recommendation comes from the deployed ``POST /chat`` endpoint, so the
backend stays the single source of truth and its behaviour is unchanged.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 60.0
MIN_RECOMMENDATIONS = 1
MAX_RECOMMENDATIONS = 10

# Mirrors catalog.TEST_TYPE_LABELS (kept local so the frontend has no backend
# import and can run against a remote deployment).
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


class ApiError(RuntimeError):
    """A human-readable error intended to be surfaced directly in the UI."""


def resolve_base_url(explicit: Optional[str] = None) -> str:
    """Resolve the API base URL from an argument, ``SHL_API_URL`` or the default."""
    url = (explicit or os.getenv("SHL_API_URL") or DEFAULT_API_URL).strip()
    return url.rstrip("/") or DEFAULT_API_URL


def clamp_count(k: Any) -> int:
    """Coerce a user-supplied count into the backend's 1..10 range."""
    try:
        value = int(k)
    except (TypeError, ValueError):
        value = MAX_RECOMMENDATIONS
    return max(MIN_RECOMMENDATIONS, min(value, MAX_RECOMMENDATIONS))


def build_payload(query: str, k: Any = MAX_RECOMMENDATIONS) -> Dict[str, Any]:
    """Build the exact ``POST /chat`` body the backend expects."""
    text = (query or "").strip()
    if not text:
        raise ApiError("Please enter a job description or hiring query.")
    return {
        "messages": [{"role": "user", "content": text}],
        "k": clamp_count(k),
    }


def test_type_label(code: Optional[str]) -> str:
    code = (code or "").strip().upper()
    return TEST_TYPE_LABELS.get(code, code or "—")


def _extract_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return (response.text or "").strip()[:200]
    if isinstance(body, dict):
        return str(body.get("detail") or body.get("message") or body)[:300]
    return str(body)[:300]


def request_recommendations(base_url: Optional[str], query: str,
                            k: Any = MAX_RECOMMENDATIONS, *,
                            timeout: float = DEFAULT_TIMEOUT,
                            client: Optional[httpx.Client] = None) -> Dict[str, Any]:
    """Call ``POST /chat`` and return the parsed JSON body.

    Raises :class:`ApiError` with a readable message for every failure mode
    (unreachable host, timeout, HTTP >= 400, non-JSON or unexpected shape).
    """
    base = resolve_base_url(base_url)
    payload = build_payload(query, k)
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        response = client.post(f"{base}/chat", json=payload)
    except httpx.ConnectError as exc:
        raise ApiError(
            f"Could not connect to the API at {base}. Check the URL and that "
            "the service is running."
        ) from exc
    except httpx.TimeoutException as exc:
        raise ApiError("The API request timed out. Please try again.") from exc
    except httpx.HTTPError as exc:
        raise ApiError(f"Network error contacting the API: {exc}") from exc
    finally:
        if owns_client:
            client.close()

    if response.status_code >= 400:
        raise ApiError(
            f"The API returned HTTP {response.status_code}: "
            f"{_extract_detail(response)}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise ApiError("The API returned a response that was not valid JSON.") from exc
    if not isinstance(data, dict) or "recommendations" not in data:
        raise ApiError("The API returned an unexpected response shape.")
    return data


def check_health(base_url: Optional[str], *, timeout: float = 10.0,
                 client: Optional[httpx.Client] = None) -> Dict[str, Any]:
    """Call ``GET /health``; raises :class:`ApiError` when unreachable."""
    base = resolve_base_url(base_url)
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        response = client.get(f"{base}/health")
    except httpx.HTTPError as exc:
        raise ApiError(f"Could not reach {base}/health: {exc}") from exc
    finally:
        if owns_client:
            client.close()
    if response.status_code >= 400:
        raise ApiError(f"Health check failed (HTTP {response.status_code}).")
    try:
        return response.json()
    except ValueError as exc:
        raise ApiError("Health check returned non-JSON.") from exc
