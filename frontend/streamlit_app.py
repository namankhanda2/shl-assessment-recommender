"""Streamlit frontend for the SHL Talent Assessment Advisor.

A standalone recruiter dashboard that talks to the FastAPI backend over
``POST /chat``. The backend and its API contract are unchanged — this app only
reads the JSON it returns. No relevance scores or assessment details are
invented client-side.

Run locally (defaults to http://localhost:8000)::

    streamlit run frontend/streamlit_app.py

Point it at the deployed Render service::

    SHL_API_URL="https://<your-service>.onrender.com" \
        streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import html
import os
import sys

import streamlit as st

# Allow running via "streamlit run frontend/streamlit_app.py" from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.api_client import (  # noqa: E402
    ApiError,
    DEFAULT_API_URL,
    MAX_RECOMMENDATIONS,
    MIN_RECOMMENDATIONS,
    check_health,
    request_recommendations,
    resolve_base_url,
    test_type_label,
)

st.set_page_config(
    page_title="SHL Talent Assessment Advisor",
    layout="wide",
)

EXPERIENCE_LEVELS = [
    "Not specified",
    "Internship",
    "Entry level",
    "Graduate",
    "Mid level",
    "Senior",
    "Lead / Manager",
    "Director / Executive",
]

# Example hiring profiles used to autofill the workspace. Values are plain role
# descriptions only — nothing here is sent except as normal query text.
PROFILES = [
    {
        "label": "Senior Python backend",
        "role": "Senior Backend Engineer",
        "experience": "Senior",
        "skills": "Python, REST APIs, SQL, AWS, Docker",
        "jd": (
            "We are hiring a senior backend engineer to design and scale "
            "Python services. The role owns REST APIs, works with SQL "
            "databases and deploys to AWS. Strong coding, problem solving and "
            "collaboration are essential."
        ),
    },
    {
        "label": "Graduate analyst",
        "role": "Graduate Business Analyst",
        "experience": "Graduate",
        "skills": "Numerical reasoning, verbal reasoning, attention to detail",
        "jd": (
            "Graduate analyst role supporting reporting and analysis. The "
            "successful candidate interprets numerical data, reasons with "
            "written information and communicates findings to stakeholders."
        ),
    },
    {
        "label": "Contact centre agent",
        "role": "Customer Service Representative",
        "experience": "Entry level",
        "skills": "Communication, customer focus, resilience",
        "jd": (
            "Front-line customer service role in a busy contact centre. "
            "Handles inbound calls and chats, resolves customer issues and "
            "works to clear service targets."
        ),
    },
    {
        "label": "Registered nurse",
        "role": "Registered Nurse",
        "experience": "Mid level",
        "skills": "Clinical care, patient safety, situational judgement",
        "jd": (
            "Registered nurse providing clinical patient care on a hospital "
            "ward. Requires sound clinical judgement, safe practice and "
            "effective teamwork under pressure."
        ),
    },
]

CSS = """
<style>
:root {
    --shl-navy: #0b1f3a;
    --shl-navy-2: #12315c;
    --shl-blue: #1f6feb;
    --shl-blue-dark: #14509e;
    --shl-bg: #f5f7fb;
    --shl-card: #ffffff;
    --shl-border: #e2e8f0;
    --shl-text: #0f172a;
    --shl-muted: #64748b;
}

.stApp {
    background: var(--shl-bg);
    color: var(--shl-text);
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
        Roboto, Helvetica, Arial, sans-serif;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

/* ---------- Header ---------- */
.shl-header {
    background: linear-gradient(135deg, #0b1f3a 0%, #12315c 58%, #1f6feb 145%);
    color: #ffffff;
    padding: 30px 34px;
    border-radius: 18px;
    margin-bottom: 24px;
    box-shadow: 0 12px 34px rgba(11, 31, 58, 0.20);
}
.shl-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    font-weight: 600;
    color: #8fb4f0;
    margin-bottom: 10px;
}
.shl-header h1 {
    font-size: 2rem;
    line-height: 1.15;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 8px 0;
}
.shl-header p {
    margin: 0;
    color: #c9d8f2;
    font-size: 1rem;
    max-width: 720px;
    line-height: 1.5;
}

/* ---------- Section labels ---------- */
.shl-section {
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.74rem;
    font-weight: 700;
    color: var(--shl-muted);
    margin: 4px 0 8px 0;
}

/* ---------- Cards ---------- */
.shl-cards { display: flex; flex-direction: column; gap: 14px; }
.shl-card {
    background: var(--shl-card);
    border: 1px solid var(--shl-border);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}
.shl-card:hover {
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.09);
    border-color: #cdd9ec;
}
.shl-card-head { display: flex; gap: 14px; align-items: flex-start; }
.shl-card-index {
    flex: 0 0 auto;
    width: 34px; height: 34px;
    border-radius: 9px;
    background: #eef4ff;
    color: var(--shl-blue-dark);
    font-weight: 700;
    font-size: 0.9rem;
    display: flex; align-items: center; justify-content: center;
}
.shl-card-main { flex: 1 1 auto; min-width: 0; }
.shl-card-name {
    font-size: 1.08rem;
    font-weight: 650;
    color: var(--shl-navy);
    line-height: 1.35;
    overflow-wrap: anywhere;
    word-break: break-word;
}
.shl-card-meta {
    display: flex; flex-wrap: wrap; gap: 8px;
    margin-top: 8px;
}
.shl-chip {
    display: inline-block;
    background: #eef2f8;
    color: #33507a;
    border: 1px solid #dbe4f2;
    border-radius: 999px;
    padding: 2px 11px;
    font-size: 0.76rem;
    font-weight: 600;
}
.shl-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 2px 11px;
    font-size: 0.76rem;
    font-weight: 600;
    border: 1px solid transparent;
}
.shl-badge.verified { background: #e7f6ec; color: #17663a; border-color: #b7e4c7; }
.shl-badge.legacy   { background: #fff4e5; color: #8a4b00; border-color: #ffd8a8; }
.shl-badge.unknown  { background: #eef2f7; color: #475569; border-color: #dbe2ea; }
.shl-card-rationale {
    color: #334155;
    line-height: 1.55;
    margin: 12px 0 12px 48px;
    overflow-wrap: anywhere;
    word-break: break-word;
}
.shl-card-foot {
    display: flex; flex-wrap: wrap; gap: 10px 16px;
    align-items: center;
    margin-left: 48px;
    padding-top: 12px;
    border-top: 1px solid #eef2f7;
}
.shl-open {
    color: var(--shl-blue);
    font-weight: 650;
    text-decoration: none;
    font-size: 0.9rem;
    white-space: nowrap;
}
.shl-open:hover { text-decoration: underline; }
.shl-url {
    color: #94a3b8;
    font-size: 0.76rem;
    word-break: break-all;
    overflow-wrap: anywhere;
    min-width: 0;
}

/* ---------- Summary ---------- */
.shl-summary {
    display: flex; flex-wrap: wrap; gap: 26px; align-items: center;
    background: var(--shl-card);
    border: 1px solid var(--shl-border);
    border-radius: 14px;
    padding: 16px 22px;
    margin: 4px 0 20px 0;
}
.shl-stat { display: flex; flex-direction: column; }
.shl-stat .num {
    font-size: 1.7rem; font-weight: 700; color: var(--shl-navy);
    line-height: 1.1;
}
.shl-stat .lbl {
    font-size: 0.74rem; color: var(--shl-muted);
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px;
}

/* ---------- Advisor note ---------- */
.shl-note {
    background: #f0f6ff;
    border: 1px solid #d3e3ff;
    border-left: 4px solid var(--shl-blue);
    border-radius: 10px;
    padding: 14px 18px;
    color: #1e3a5f;
    line-height: 1.55;
    margin-bottom: 20px;
    overflow-wrap: anywhere;
}
.shl-note .lbl {
    display: block;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--shl-blue-dark);
    margin-bottom: 6px;
}

/* ---------- Empty state ---------- */
.shl-empty {
    border: 1px dashed #cbd5e1;
    border-radius: 14px;
    padding: 40px 24px;
    text-align: center;
    color: var(--shl-muted);
    background: #fbfcfe;
    line-height: 1.6;
}
.shl-empty strong { color: var(--shl-navy); }

/* ---------- Widgets ---------- */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}
.stButton > button[kind="primary"] {
    background: var(--shl-blue);
    border-color: var(--shl-blue);
}
.stButton > button[kind="primary"]:hover {
    background: var(--shl-blue-dark);
    border-color: var(--shl-blue-dark);
}
div[data-baseweb="input"] input,
.stTextArea textarea,
div[data-baseweb="select"] > div {
    border-radius: 10px !important;
}

@media (max-width: 640px) {
    .shl-header { padding: 22px 20px; }
    .shl-header h1 { font-size: 1.5rem; }
    .shl-card-rationale,
    .shl-card-foot { margin-left: 0; }
}
</style>
"""


def _secret_api_url() -> str:
    try:
        return st.secrets.get("SHL_API_URL", "") or ""
    except Exception:
        return ""


def _initial_base_url() -> str:
    return resolve_base_url(_secret_api_url() or os.getenv("SHL_API_URL"))


def _inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _render_header() -> None:
    st.markdown(
        """
        <div class="shl-header">
            <div class="shl-eyebrow">SHL Talent Assessment Advisor</div>
            <h1>Find the right assessments for every role</h1>
            <p>Describe the role you are hiring for and get grounded SHL
            assessment recommendations, each linking to its SHL catalog page
            with a URL verification status.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _combine_query(role: str, experience: str, skills: str, jd: str) -> str:
    """Assemble the single user message from the structured workspace fields."""
    parts = []
    if role.strip():
        parts.append(f"Job role: {role.strip()}")
    if experience and experience != "Not specified":
        parts.append(f"Experience level: {experience}")
    if skills.strip():
        parts.append(f"Required skills: {skills.strip()}")
    header = "\n".join(parts)
    body = jd.strip()
    if header and body:
        return f"{header}\n\nJob description:\n{body}"
    return header or body


def _verification(rec: dict) -> tuple[str, str]:
    """Return (badge_css_class, badge_text) from API-provided fields only."""
    verified = rec.get("url_verified")
    status = rec.get("url_status")
    if verified is True:
        return "verified", "Verified official URL"
    if verified is False:
        text = "Legacy link"
        if status:
            text = f"Legacy link · {status}"
        return "legacy", text
    return "unknown", "Verification unavailable"


def _safe_url(url) -> str:
    url = str(url or "").strip()
    if url.startswith("http://") or url.startswith("https://"):
        return html.escape(url, quote=True)
    return ""


def _card_html(index: int, rec: dict) -> str:
    name = html.escape(str(rec.get("name") or "Unnamed assessment"))
    rationale = html.escape(str(rec.get("rationale") or "")).strip()
    code = html.escape(str(rec.get("test_type") or "—"))
    label = html.escape(test_type_label(rec.get("test_type")))
    badge_class, badge_text = _verification(rec)
    badge_text = html.escape(badge_text)

    url = _safe_url(rec.get("url"))
    raw_url = str(rec.get("url") or "")
    raw_url_html = html.escape(raw_url)

    rationale_html = (
        f'<div class="shl-card-rationale">{rationale}</div>' if rationale else ""
    )
    if url:
        foot = (
            '<div class="shl-card-foot">'
            f'<a class="shl-open" href="{url}" target="_blank" '
            'rel="noopener noreferrer">Open assessment &rarr;</a>'
            f'<span class="shl-url">{raw_url_html}</span>'
            "</div>"
        )
    elif raw_url:
        foot = (
            '<div class="shl-card-foot">'
            f'<span class="shl-url">{raw_url_html}</span>'
            "</div>"
        )
    else:
        foot = ""

    return (
        '<div class="shl-card">'
        '<div class="shl-card-head">'
        f'<div class="shl-card-index">{index:02d}</div>'
        '<div class="shl-card-main">'
        f'<div class="shl-card-name">{name}</div>'
        '<div class="shl-card-meta">'
        f'<span class="shl-chip">{code} · {label}</span>'
        f'<span class="shl-badge {badge_class}">{badge_text}</span>'
        "</div>"
        "</div>"
        "</div>"
        f"{rationale_html}"
        f"{foot}"
        "</div>"
    )


def _render_summary(recs: list) -> None:
    verified = sum(1 for r in recs if isinstance(r, dict) and r.get("url_verified") is True)
    flagged = sum(1 for r in recs if isinstance(r, dict) and r.get("url_verified") is False)
    unknown = len(recs) - verified - flagged

    stats = [
        (str(len(recs)), "Recommendations"),
        (str(verified), "Verified official"),
        (str(flagged), "Flagged legacy"),
    ]
    if unknown > 0:
        stats.append((str(unknown), "Status unavailable"))

    cells = "".join(
        f'<div class="shl-stat"><span class="num">{num}</span>'
        f'<span class="lbl">{lbl}</span></div>'
        for num, lbl in stats
    )
    st.markdown(f'<div class="shl-summary">{cells}</div>', unsafe_allow_html=True)


def _render_results(data: dict, query: str) -> None:
    recs = [r for r in (data.get("recommendations") or []) if isinstance(r, dict)]

    reply = data.get("reply")
    if reply:
        note = html.escape(str(reply)).replace("\n", "<br>")
        st.markdown(
            f'<div class="shl-note"><span class="lbl">Advisor summary</span>{note}</div>',
            unsafe_allow_html=True,
        )

    if not recs:
        st.markdown(
            '<div class="shl-empty"><strong>No recommendations returned.</strong><br>'
            "Try adding more detail such as specific skills, seniority or the "
            "type of assessment you need, then generate again.</div>",
            unsafe_allow_html=True,
        )
        return

    _render_summary(recs)
    st.markdown(
        '<div class="shl-section">Recommended assessments</div>',
        unsafe_allow_html=True,
    )
    cards = "".join(_card_html(i, r) for i, r in enumerate(recs, start=1))
    st.markdown(f'<div class="shl-cards">{cards}</div>', unsafe_allow_html=True)

    with st.expander("API response (debug)"):
        st.json(data)


def _render_sidebar() -> tuple[str, int]:
    with st.sidebar:
        st.markdown(
            '<div class="shl-section">API connection</div>', unsafe_allow_html=True
        )
        if "api_url" not in st.session_state:
            st.session_state.api_url = _initial_base_url()
        base_url = st.text_input(
            "API base URL",
            key="api_url",
            help="Your Render service URL, e.g. https://your-service.onrender.com",
        )
        if st.button("Test connection", use_container_width=True):
            with st.spinner("Checking the API..."):
                try:
                    health = check_health(base_url)
                    st.success("Connected")
                    catalog = health.get("catalog", {})
                    if isinstance(catalog, dict) and catalog:
                        st.caption(
                            f"Catalog: {catalog.get('count', '?')} assessments"
                        )
                    llm = health.get("llm", {})
                    if isinstance(llm, dict) and llm:
                        st.caption(
                            "LLM: "
                            + (
                                "enabled"
                                if llm.get("enabled")
                                else "deterministic fallback"
                            )
                        )
                except ApiError as exc:
                    st.error(str(exc))

        st.markdown(
            '<div class="shl-section" style="margin-top:18px;">Search settings</div>',
            unsafe_allow_html=True,
        )
        count = st.slider(
            "Number of recommendations",
            min_value=MIN_RECOMMENDATIONS,
            max_value=MAX_RECOMMENDATIONS,
            value=5,
        )
        st.caption(
            f"Default host: `{DEFAULT_API_URL}` (override with the `SHL_API_URL` "
            "env var)."
        )
    return base_url, count


def _render_workspace() -> None:
    st.markdown(
        '<div class="shl-section">Example hiring profiles</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(PROFILES))
    for col, profile in zip(cols, PROFILES):
        with col:
            if st.button(profile["label"], use_container_width=True):
                st.session_state.jd_input = profile["jd"]
                st.session_state.role_input = profile["role"]
                st.session_state.exp_input = profile["experience"]
                st.session_state.skills_input = profile["skills"]
                st.rerun()


def main() -> None:
    _inject_css()
    _render_header()

    base_url, count = _render_sidebar()

    _render_workspace()

    with st.form("requirements"):
        st.markdown(
            '<div class="shl-section">Job requirements workspace</div>',
            unsafe_allow_html=True,
        )
        jd = st.text_area(
            "Job description",
            key="jd_input",
            height=210,
            placeholder=(
                "Paste the job description or describe the role, "
                "responsibilities and must-have skills..."
            ),
        )
        c_role, c_exp, c_skills = st.columns([1, 1, 1])
        with c_role:
            role = st.text_input(
                "Job role (optional)", key="role_input", placeholder="e.g. Backend Engineer"
            )
        with c_exp:
            experience = st.selectbox(
                "Experience level (optional)", EXPERIENCE_LEVELS, key="exp_input"
            )
        with c_skills:
            skills = st.text_input(
                "Required skills (optional)",
                key="skills_input",
                placeholder="e.g. Python, SQL, AWS",
            )
        submitted = st.form_submit_button(
            "Generate recommendations", type="primary", use_container_width=True
        )

    if submitted:
        query = _combine_query(role, experience, skills, jd)
        if not query:
            st.error(
                "Please provide a job description or at least the job role or "
                "required skills before generating recommendations."
            )
        else:
            with st.spinner("Analysing requirements and matching assessments..."):
                try:
                    data = request_recommendations(base_url, query, count)
                except ApiError as exc:
                    st.session_state.pop("result", None)
                    st.error(str(exc))
                else:
                    st.session_state.result = {"data": data, "query": query}
                    st.session_state.pop("result_error", None)

    result = st.session_state.get("result")
    if result:
        _render_results(result["data"], result["query"])
    elif not submitted:
        st.markdown(
            '<div class="shl-empty">'
            "<strong>Start by describing the role.</strong><br>"
            "Use an example profile above, or enter the job description and any "
            "optional role details, choose how many recommendations you need, "
            "then select <em>Generate recommendations</em>."
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
