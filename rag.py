"""Retrieval-augmented generation over the SHL catalogue.

Retrieval (BM25, optionally blended with the FAISS dense index) lives in
``retrieval.py`` and is driven by ``agent.py``. This module is the generation
half of the RAG loop: it turns the retrieved, catalogue-grounded candidates
into a natural-language reply using the optional LLM client from ``llm.py``.

Grounding guarantee: the model may only *select* from the candidate names it is
given, so it can never invent products or URLs. When no LLM is configured, or
the call fails, the deterministic ``base_reply`` and candidate order are
returned unchanged.

Run the full loop for a sample query::

    python rag.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

SYSTEM_PROMPT = (
    "You are an SHL assessment advisor. You must only discuss assessments from "
    "the provided candidate list. Never invent product names or URLs. "
    "Name assessments in your reply only if you also list them in selected. "
    'Return strict JSON: {"reply": string, "selected": [candidate names]}.'
)


def _role(message: Dict[str, Any]) -> str:
    value = message.get("role", "user")
    return ("" if value is None else str(value)).lower()


def _content(message: Dict[str, Any]) -> str:
    value = message.get("content", "")
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def build_prompt(messages: List[Dict[str, Any]], candidates: List[str],
                 intent: str, max_selected: int) -> str:
    conversation = "\n".join(
        f"{_role(m)}: {_content(m)}" for m in messages[-8:]
    )
    listing = "\n".join(f"- {name}" for name in candidates)
    return (
        f"Conversation:\n{conversation}\n\nIntent: {intent}\n"
        f"Candidates (choose only from these, max {max_selected}):\n{listing}\n\n"
        f"Select the best 1-{max_selected} candidates and list them in \"selected\". "
        "Write one short, helpful reply that names only the assessments you "
        "selected."
    )


def generate(client: Any, messages: List[Dict[str, Any]], base_reply: str,
             recommendations: List[Dict[str, Any]], intent: str, *,
             max_selected: int = 10,
             fill_to_max: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    """Generate a grounded reply; fall back to ``base_reply`` when unavailable.

    When ``fill_to_max`` is set the returned list keeps the requested shortlist
    size: the model's preferred order first, then the remaining grounded
    candidates. It never adds anything outside the retrieved catalogue set.
    """
    if client is None or not getattr(client, "enabled", False) or not recommendations:
        return base_reply, recommendations

    candidates = [item["name"] for item in recommendations]
    prompt = build_prompt(messages, candidates, intent, max_selected)
    data = client.complete_json(SYSTEM_PROMPT, prompt, max_tokens=500)
    if not data:
        return base_reply, recommendations

    reply = data.get("reply") or base_reply
    chosen = data.get("selected")
    if isinstance(chosen, list) and chosen:
        by_name = {item["name"].lower(): item for item in recommendations}
        picked = []
        seen = set()
        for name in chosen:
            if not isinstance(name, str) or name.lower() not in by_name:
                continue
            item = by_name[name.lower()]
            if item["url"] in seen:
                continue
            seen.add(item["url"])
            picked.append(item)
        if picked:
            if fill_to_max:
                # Preserve the requested shortlist size: the model's preferred
                # order first, then the remaining grounded candidates.
                remainder = [it for it in recommendations if it["url"] not in seen]
                picked = picked + remainder
            return str(reply), picked[:max_selected]
    return str(reply), recommendations


def retrieve(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """Retrieve the top-``k`` catalogue items for a query."""
    from retrieval import default_retriever

    return default_retriever().search(query, k=k)


def answer(query: str, k: int = 10) -> Tuple[str, List[Dict[str, Any]]]:
    """End-to-end RAG: retrieve candidates, then generate a grounded reply."""
    from llm import get_client

    docs = retrieve(query, k=k)
    base_reply = f"Here are {len(docs)} SHL assessments that fit your needs."
    return generate(get_client(), [{"role": "user", "content": query}],
                    base_reply, docs, "recommend", max_selected=k)


if __name__ == "__main__":  # pragma: no cover - manual demo
    from llm import _describe

    sample = "Hiring a mid-level Java developer who works with SQL and stakeholders"
    reply, chosen = answer(sample)
    print(f"Query: {sample}\n")
    print(_describe())
    print(f"\nReply: {reply}\n")
    for item in chosen:
        print(f"  [{item['test_type']}] {item['name']}\n        {item['url']}")
