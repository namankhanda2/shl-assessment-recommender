"""Conversational planner for the SHL assessment recommender.

The agent is stateless: every call receives the full message history. It
implements four behaviours required by the SHL brief:

* clarify vague requests before recommending,
* recommend 1-10 catalog-grounded assessments,
* refine when the user changes constraints, and
* compare named assessments from catalog data.

It also refuses out-of-scope / prompt-injection requests and honours an 8-message
turn cap. Recommendations are always resolved back to catalogue objects, so
names and URLs can never be hallucinated.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import rag
from catalog import TEST_TYPE_LABELS, load_catalog, normalize_url
from llm import get_client
from retrieval import (
    DROP_RE,
    SKILL_TERMS,
    default_retriever,
    detect_seniority,
    extract_skills,
    find_mentioned_items,
    has_cognitive_signal,
    has_personality_signal,
    has_situational_signal,
    is_specific,
    tokenize,
)

MAX_RECOMMENDATIONS = 10
TURN_CAP = 8            # hard cap on total messages in a conversation
FORCE_FINAL_USER_TURNS = 4   # stop clarifying after this many user turns

INJECTION_RE = re.compile(
    r"ignore (the |all )?(previous|prior|above|earlier) instructions|"
    r"disregard .{0,20}instructions|system prompt|"
    r"you are now|act as (a|an)|jailbreak|developer mode|"
    r"reveal .{0,20}(prompt|instructions)|pretend (to be|you are)",
    re.IGNORECASE,
)

LEGAL_RE = re.compile(
    r"\blawsuit\b|\bsue\b|discriminat|eeoc|gdpr|\bada\b.*compliance|"
    r"is it legal|legally|legal advice|compliance question",
    re.IGNORECASE,
)

HIRING_ADVICE_RE = re.compile(
    r"how (do|should) i interview|interview questions?|"
    r"salary|compensation|pay range|negotiat|"
    r"\bvisa\b|sponsorship|work permit|background check|drug test|"
    r"\bresume\b|curriculum vitae|\bcv\b|offer letter|onboarding",
    re.IGNORECASE,
)

OFF_TOPIC_RE = re.compile(
    r"weather|recipe|cook|president|election|stock (price|market)|"
    r"write (me )?(a )?(poem|story|essay|song)|"
    r"\bjoke\b|\bsing\b|\bmovie\b|"
    r"who (is|was) .{0,30}\?|what is the capital|translate|"
    r"what.{0,10}(meaning of life|time is it)",
    re.IGNORECASE,
)

COMPARE_RE = re.compile(
    r"\b(compare|comparison|difference|differ|versus|vs\.?|contrast|"
    r"which is better|better than)\b",
    re.IGNORECASE,
)

CONFIRM_RE = re.compile(
    r"^\s*(yes|yeah|yep|sure|perfect|great|thanks|thank you|that works|"
    r"looks good|confirmed|approve[d]?|ok|okay|correct|done|good)\b",
    re.IGNORECASE,
)

ASSESSMENT_HINT_RE = re.compile(
    r"assessment|test|recommend|shl|catalog|shortlist|benchmark|questionnaire",
    re.IGNORECASE,
)

DEFAULT_SHORTLIST_NAMES = [
    "Occupational Personality Questionnaire OPQ32r",
    "SHL Verify Interactive G+",
    "SHL Verify Interactive – Numerical Reasoning",
    "Verify - Verbal Ability - Next Generation",
]


# --------------------------------------------------------------------------- #
# message helpers
# --------------------------------------------------------------------------- #
def _role(msg: Dict[str, Any]) -> str:
    value = msg.get("role", "user")
    return ("" if value is None else str(value)).lower()


def _content(msg: Dict[str, Any]) -> str:
    value = msg.get("content", "")
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def user_contents(messages: List[Dict[str, Any]]) -> List[str]:
    return [_content(m) for m in messages if _role(m) == "user" and _content(m).strip()]


def _last_user(messages: List[Dict[str, Any]]) -> str:
    contents = user_contents(messages)
    return contents[-1] if contents else ""


# --------------------------------------------------------------------------- #
# constraint tracking (stateless, re-derived every turn)
# --------------------------------------------------------------------------- #
class Constraints:
    def __init__(self) -> None:
        self.active_skills: set = set()
        self.dropped_skills: set = set()
        self.personality = False
        self.cognitive = False
        self.situational = False
        self.seniority: List[str] = []
        self.job_description = ""

    def apply(self, text: str) -> None:
        low = (text or "").lower()
        present = extract_skills(low)
        dropping = bool(DROP_RE.search(low))
        if dropping:
            self.active_skills -= present
            self.dropped_skills |= present
        else:
            self.active_skills |= present
        p = has_personality_signal(low)
        c = has_cognitive_signal(low)
        s = has_situational_signal(low)
        if dropping:
            if p and "personality" in present:
                self.personality = False
            if c and ("cognitive" in present or "reasoning" in present):
                self.cognitive = False
            if s:
                self.situational = False
        else:
            self.personality |= p
            self.cognitive |= c
            self.situational |= s
        for lvl in detect_seniority(low):
            if lvl not in self.seniority:
                self.seniority.append(lvl)
        if re.search(r"job description|responsibilities|requirements|jd\b", low):
            self.job_description = (self.job_description + " " + text).strip()


def build_constraints(messages: List[Dict[str, Any]]) -> Constraints:
    c = Constraints()
    for content in user_contents(messages):
        c.apply(content)
    return c


def build_query(messages: List[Dict[str, Any]], c: Constraints) -> str:
    aggregate = " ".join(user_contents(messages))
    tokens = [t for t in tokenize(aggregate) if t not in c.dropped_skills]
    parts = [" ".join(tokens)]
    parts.extend(sorted(c.active_skills))
    if c.personality:
        parts.append("personality behavior occupational questionnaire")
    if c.cognitive:
        parts.append("cognitive reasoning ability aptitude numerical verbal")
    if c.situational:
        parts.append("situational judgment biodata scenario")
    if c.job_description:
        parts.append(c.job_description)
    return " ".join(p for p in parts if p).strip()


# --------------------------------------------------------------------------- #
# recommendation
# --------------------------------------------------------------------------- #
def _resolve_names(names: List[str], catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for target in names:
        needle = target.lower()
        for item in catalog:
            if item["name"].lower() == needle:
                out.append(item)
                break
    return out


def select_recommendations(query: str, c: Constraints,
                           k: int = MAX_RECOMMENDATIONS) -> List[Dict[str, Any]]:
    catalog = load_catalog()
    retriever = default_retriever()
    ranked = retriever.rank(query) if query.strip() else [(it, 0.0) for it in catalog]
    # Multiple catalogue records can share a current URL (e.g. the OPQ/MQ report
    # families all map to the same product page), so keep only the best-ranked
    # record per URL to avoid duplicate recommendations.
    positive = []
    seen_urls = set()
    for it, sc in ranked:
        if sc <= 0:
            continue
        key = normalize_url(it["url"])
        if key in seen_urls:
            continue
        seen_urls.add(key)
        positive.append((it, sc))
    if not positive:
        return _resolve_names(DEFAULT_SHORTLIST_NAMES, catalog)[:k]

    selected = [it for it, _ in positive[:k]]
    score = {it["url"]: sc for it, sc in positive}

    desired: List[str] = []
    if c.active_skills:
        desired.append("K")
    if c.personality:
        desired.append("P")
    if c.cognitive:
        desired.append("A")
    if c.situational:
        desired.append("B")

    for t in desired:
        if any(t in it["test_types"] for it in selected):
            continue
        best = next((it for it, _ in positive if t in it["test_types"]
                     and it not in selected), None)
        if best is not None:
            if len(selected) >= k:
                selected[-1] = best
            else:
                selected.append(best)

    selected.sort(key=lambda it: score.get(it["url"], 0.0), reverse=True)
    return selected[:k]


# --------------------------------------------------------------------------- #
# behaviours
# --------------------------------------------------------------------------- #
def _refusal_reply(kind: str) -> str:
    if kind == "injection":
        return ("I can only help with selecting SHL assessments and I can't follow "
                "instructions that try to change my role or reveal internal prompts. "
                "Tell me the role or skills you're hiring for and I'll suggest SHL "
                "assessments.")
    return ("I can only help with selecting SHL assessments from the SHL catalogue, "
            "so I can't help with legal questions, compensation, or general hiring "
            "advice. Tell me the role and skills you're hiring for and I'll recommend "
            "relevant SHL assessments.")


def _clarify_reply(c: Constraints) -> str:
    if not c.active_skills:
        return ("Happy to help. What role or skill area are you hiring for? For "
                "example: a Java backend developer, a graduate analyst, or a "
                "customer-service representative.")
    if not c.seniority:
        return ("Got it. What seniority level are you hiring for, and is this for "
                "selection or development?")
    return ("Could you share a bit more about the role — the key skills or the "
            "seniority level you're targeting?")


def _compare_reply(items: List[Dict[str, Any]]) -> str:
    parts = ["Here is a comparison based strictly on the SHL catalogue:"]
    for item in items:
        types = ", ".join(TEST_TYPE_LABELS.get(t, t) for t in item["test_types"])
        levels = ", ".join(item["job_levels"][:4]) or "not specified"
        duration = item["duration"] or "not specified"
        desc = item["description"] or "No description available."
        parts.append(
            f"**{item['name']}** — type: {types}; level: {levels}; "
            f"duration: {duration}. {desc}"
        )
    return "\n\n".join(parts)


def _join(parts: List[str]) -> str:
    parts = [str(p) for p in parts if p]
    if len(parts) <= 1:
        return "".join(parts)
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _format_duration(duration: Any) -> str:
    value = str(duration or "").strip()
    if not value:
        return ""
    if value.lower() in {"untimed", "n/a", "-", "unknown"}:
        return "untimed"
    return f"takes about {value}"


def recommendation_rationale(item: Dict[str, Any], c: "Constraints") -> str:
    """A short, catalogue-grounded reason for recommending ``item``.

    Only facts already present in the catalog record (and the extracted
    constraints) are used, so nothing is invented. A skill is cited only when it
    actually appears in the record's own name/description/keys.
    """
    label = TEST_TYPE_LABELS.get(item.get("test_type", ""),
                                 item.get("test_type", "")) or "SHL"
    text = (
        f"{item.get('name', '')} {item.get('description', '')} "
        f"{' '.join(item.get('keys') or [])}"
    ).lower()
    matched = sorted(s for s in c.active_skills if s and s in text)
    reason = f"{label} assessment"
    if matched:
        reason += " relevant to " + _join(matched[:4])
    levels = [str(level) for level in (item.get("job_levels") or [])]
    if levels:
        reason += f"; suitable for {_join(levels[:2])} level"
    duration = _format_duration(item.get("duration"))
    if duration:
        reason += f"; {duration}"
    return reason + "."


def _recommend_reply(count: int, c: Constraints) -> str:
    focus = []
    if c.active_skills:
        focus.append(" & ".join(sorted(c.active_skills)))
    if c.personality:
        focus.append("personality/behaviour")
    if c.cognitive:
        focus.append("cognitive ability")
    if c.situational:
        focus.append("situational judgement")
    label = ", ".join(focus) if focus else "your requirements"
    return (f"Here are {count} SHL assessments that fit {label}. "
            "Tell me if you'd like me to add or remove any.")


def _llm_refine(messages: List[Dict[str, Any]], base_reply: str,
                recommendations: List[Dict[str, Any]], intent: str,
                k: int = MAX_RECOMMENDATIONS) -> Tuple[str, List[Dict[str, Any]]]:
    """Optionally let an LLM polish the reply / re-rank candidates.

    Delegates to ``rag.generate``; recommendations stay catalogue-grounded (the
    model may only choose from the candidate names we give it, and any unknown
    name is ignored). Falls back to the deterministic reply when no LLM is set.
    """
    return rag.generate(get_client(), messages, base_reply, recommendations,
                        intent, max_selected=k, fill_to_max=True)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def respond(messages: Optional[List[Dict[str, Any]]],
            k: int = MAX_RECOMMENDATIONS) -> Dict[str, Any]:
    # Honour the caller's requested shortlist size exactly (clamped to 1..10).
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = MAX_RECOMMENDATIONS
    k = max(1, min(k, MAX_RECOMMENDATIONS))
    messages = [m for m in (messages or []) if isinstance(m, dict)]
    users = user_contents(messages)
    c = build_constraints(messages)

    if not users:
        return {
            "reply": ("Tell me about the role you're hiring for and I'll recommend "
                      "relevant SHL assessments."),
            "recommendations": [],
            "end_of_conversation": False,
        }

    last = _last_user(messages)
    total = len(messages)
    aggregate = " ".join(users)
    at_cap = total >= TURN_CAP or len(users) >= FORCE_FINAL_USER_TURNS
    last_assistant_had_recs = any(
        _role(m) == "assistant"
        and ("recommend" in _content(m).lower()
             or "assessment" in _content(m).lower())
        for m in messages[:-1]
    )

    # 1. Prompt injection -> always refuse.
    if INJECTION_RE.search(last):
        return {"reply": _refusal_reply("injection"),
                "recommendations": [], "end_of_conversation": False}

    # 2. Scope guardrails (legal / general hiring advice / off-topic).
    assessment_hint = bool(ASSESSMENT_HINT_RE.search(last))
    if LEGAL_RE.search(last) and not assessment_hint:
        return {"reply": _refusal_reply("scope"),
                "recommendations": [], "end_of_conversation": False}
    if HIRING_ADVICE_RE.search(last) and not assessment_hint:
        return {"reply": _refusal_reply("scope"),
                "recommendations": [], "end_of_conversation": False}
    if OFF_TOPIC_RE.search(last) and not (extract_skills(last) or assessment_hint):
        return {"reply": _refusal_reply("scope"),
                "recommendations": [], "end_of_conversation": False}

    # 3. Comparison.
    if COMPARE_RE.search(last):
        mentioned = find_mentioned_items(aggregate)
        if len(mentioned) >= 2:
            return {"reply": _compare_reply(mentioned[:4]),
                    "recommendations": [], "end_of_conversation": False}
        return {"reply": ("Sure — which two SHL assessments would you like me to "
                          "compare? For example: OPQ32r vs Verify G+."),
                "recommendations": [], "end_of_conversation": False}

    # 4. Recommendation when we have enough context (or the cap forces us).
    enough_context = is_specific(aggregate) and bool(
        c.active_skills or c.job_description or c.personality
        or c.cognitive or c.situational
    )
    confirmed = bool(CONFIRM_RE.match(last)) and last_assistant_had_recs

    if enough_context or at_cap:
        query = build_query(messages, c)
        recs = select_recommendations(query, c, k=k)
        reply = _recommend_reply(len(recs), c)
        reply, recs = _llm_refine(messages, reply, recs, "recommend", k)
        return {
            "reply": reply,
            "recommendations": [
                {
                    "name": r["name"],
                    "url": r["url"],
                    "test_type": r["test_type"],
                    "rationale": recommendation_rationale(r, c),
                }
                for r in recs
            ],
            "end_of_conversation": bool(at_cap or confirmed),
        }

    # 5. Otherwise clarify once.
    return {"reply": _clarify_reply(c), "recommendations": [],
            "end_of_conversation": False}
