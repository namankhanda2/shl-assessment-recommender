"""FastAPI entrypoint for the SHL conversational assessment recommender.

Endpoints
---------
GET  /health  -> {"status": "ok"}
POST /chat    -> {"reply": str, "recommendations": [...], "end_of_conversation": bool}

The request carries the full, stateless conversation history:

    {"messages": [{"role": "user", "content": "..."}, ...]}
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agent import MAX_RECOMMENDATIONS, respond
from catalog import find_record, load_catalog, url_status_counts
from llm import get_client

logger = logging.getLogger("shl.recommender")

# Soft cap: very long histories are trimmed to the most recent messages rather
# than rejected, so the stateless harness keeps working.
MAX_MESSAGES = 100


@asynccontextmanager
async def lifespan(_: FastAPI):
    catalog = load_catalog()
    logger.info("Loaded %d SHL assessments", len(catalog))
    yield


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)


class Message(BaseModel):
    role: Any = "user"
    content: Any = ""

    model_config = {"extra": "ignore"}


class ChatRequest(BaseModel):
    messages: Optional[List[Message]] = Field(default_factory=list)
    # Optional, backward-compatible requested shortlist size. Several aliases
    # are accepted so any harness spelling ("k", "count", "n", "limit", ...)
    # works; unknown extra fields are still ignored.
    k: Optional[int] = None
    count: Optional[int] = None
    n: Optional[int] = None
    limit: Optional[int] = None
    num_recommendations: Optional[int] = None
    max_recommendations: Optional[int] = None

    model_config = {"extra": "ignore"}

    def requested_k(self) -> int:
        for value in (self.k, self.count, self.n, self.limit,
                      self.num_recommendations, self.max_recommendations):
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    break
        return MAX_RECOMMENDATIONS


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str
    rationale: Optional[str] = None
    # Additive, backward-compatible URL transparency fields (existing clients
    # that ignore unknown fields are unaffected). ``url_verified`` is True only
    # for a live/mapped official page; a flagged legacy link is never claimed as
    # verified.
    url_status: Optional[str] = None
    url_verified: Optional[bool] = None


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "SHL Assessment Recommender",
        "endpoints": ["/health", "/chat", "/ui", "/docs"],
        "assessments_loaded": len(load_catalog()),
    }


_UI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SHL Assessment Recommender</title>
<style>
  :root {
    --ink: #14202b; --muted: #5b6b7a; --line: #dfe6ec; --bg: #f5f8fa;
    --card: #ffffff; --brand: #004b6b; --brand-2: #00a0b0;
    --ok-bg: #e3f6ec; --ok-ink: #0f7a45; --warn-bg: #fff3e0; --warn-ink: #a35b00;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         line-height: 1.55; }
  header { background: linear-gradient(100deg, var(--brand), var(--brand-2));
           color: #fff; padding: 1.4rem 1.2rem; }
  header .wrap { max-width: 900px; margin: 0 auto; }
  header h1 { margin: 0; font-size: 1.45rem; letter-spacing: .2px; }
  header p { margin: .35rem 0 0; opacity: .92; font-size: .95rem; }
  main { max-width: 900px; margin: 0 auto; padding: 1.2rem; }
  .panel { background: var(--card); border: 1px solid var(--line);
           border-radius: 14px; padding: 1rem; box-shadow: 0 1px 2px rgba(20,32,43,.05); }
  textarea { width: 100%; min-height: 86px; font: inherit; padding: .7rem .8rem;
             border-radius: 10px; border: 1px solid var(--line); resize: vertical; }
  textarea:focus { outline: 2px solid var(--brand-2); border-color: transparent; }
  .row { display: flex; gap: .6rem; align-items: center; flex-wrap: wrap;
         margin-top: .6rem; }
  button { font: inherit; padding: .55rem 1.25rem; border-radius: 10px; border: 0;
           background: var(--brand); color: #fff; cursor: pointer; }
  button:hover { background: #003b55; }
  button.ghost { background: transparent; color: var(--muted); border: 1px solid var(--line); }
  button:disabled { opacity: .55; cursor: default; }
  .hint { color: var(--muted); font-size: .82rem; }
  .chips { display: flex; gap: .45rem; flex-wrap: wrap; margin-top: .7rem; }
  .chip { border: 1px solid var(--line); background: #fff; color: var(--brand);
          border-radius: 999px; padding: .22rem .7rem; font-size: .82rem; cursor: pointer; }
  .chip:hover { border-color: var(--brand-2); }
  #status { color: var(--muted); min-height: 1.3em; margin: 1rem 2px .2rem; font-size: .9rem; }
  .bubble { border-radius: 12px; padding: .6rem .85rem; margin: .5rem 0; max-width: 85%;
            white-space: pre-wrap; }
  .bubble.user { background: #e8f1fb; margin-left: auto; }
  .bubble.assistant { background: var(--card); border: 1px solid var(--line); }
  h2.section { font-size: 1rem; margin: 1.3rem 0 .4rem; }
  ol#recs { list-style: none; padding: 0; margin: 0; counter-reset: rec; }
  .rec { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
         padding: .8rem .9rem; margin: .55rem 0; }
  .rec .top { display: flex; gap: .5rem; align-items: baseline; flex-wrap: wrap; }
  .rec .name { font-weight: 650; color: var(--brand); text-decoration: none; }
  .rec .name:hover { text-decoration: underline; }
  .badge { display: inline-block; padding: .05rem .5rem; border-radius: 999px;
           font-size: .74rem; font-weight: 600; }
  .badge.type { background: #eef2f5; color: var(--muted); }
  .badge.ok { background: var(--ok-bg); color: var(--ok-ink); }
  .badge.warn { background: var(--warn-bg); color: var(--warn-ink); }
  .why { color: var(--muted); font-size: .9rem; margin-top: .35rem; }
  footer { max-width: 900px; margin: 1rem auto 2.5rem; padding: 0 1.2rem;
           color: var(--muted); font-size: .82rem; }
  footer .legend { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: .5rem; }
  a { color: var(--brand); }
</style>
</head>
<body>
<header><div class="wrap">
  <h1>SHL Assessment Recommender</h1>
  <p>Grounded recommendations from the SHL Individual Test Solutions catalog —
     for selection by skills, seniority, personality and cognitive ability.</p>
</div></header>
<main>
  <div class="panel">
    <textarea id="q" placeholder="e.g. Hiring a senior Java backend engineer with Spring, SQL and stakeholder management"></textarea>
    <div class="row">
      <button id="btn">Get recommendations</button>
      <button id="clear" class="ghost">Clear</button>
      <span class="hint">Press Cmd/Ctrl + Enter to submit</span>
    </div>
    <div class="chips" id="chips"></div>
  </div>
  <div id="status"></div>
  <div id="conv"></div>
  <h2 class="section" id="recs-title" style="display:none">Recommended assessments</h2>
  <ol id="recs"></ol>
</main>
<footer>
  <div class="legend">
    <span><span class="badge ok">Official URL</span> verified live SHL detail page</span>
    <span><span class="badge warn">Legacy link</span> archived URL that redirects (not verified)</span>
  </div>
  Recommendations are grounded in the bundled catalog; assessment names and URLs
  are never invented. Legacy links are flagged, never presented as verified.
</footer>
<script>
const q = document.getElementById('q');
const btn = document.getElementById('btn');
const clearBtn = document.getElementById('clear');
const conv = document.getElementById('conv');
const recs = document.getElementById('recs');
const recsTitle = document.getElementById('recs-title');
const status = document.getElementById('status');
const chips = document.getElementById('chips');
const history = [];
const EXAMPLES = [
  'Hiring a senior Java backend engineer with Spring, SQL and stakeholder management',
  'Graduate analyst needing numerical and verbal reasoning',
  'Customer service representative for a contact centre',
  'Frontend developer with React and TypeScript',
];
EXAMPLES.forEach(text => {
  const c = document.createElement('span');
  c.className = 'chip'; c.textContent = text;
  c.onclick = () => { q.value = text; q.focus(); };
  chips.appendChild(c);
});
function badge(text, cls, title) {
  const s = document.createElement('span');
  s.className = 'badge ' + cls; s.textContent = text;
  if (title) s.title = title;
  return s;
}
function renderConversation() {
  conv.innerHTML = '';
  history.forEach(m => {
    const d = document.createElement('div');
    d.className = 'bubble ' + (m.role === 'user' ? 'user' : 'assistant');
    d.textContent = m.content;
    conv.appendChild(d);
  });
}
function renderRecommendations(list) {
  recs.innerHTML = '';
  recsTitle.style.display = list.length ? 'block' : 'none';
  list.forEach(r => {
    const li = document.createElement('li');
    li.className = 'rec';
    const top = document.createElement('div');
    top.className = 'top';
    const a = document.createElement('a');
    a.className = 'name'; a.href = r.url; a.target = '_blank';
    a.rel = 'noopener'; a.textContent = r.name;
    top.appendChild(a);
    top.appendChild(badge(r.test_type || '-', 'type', 'SHL test type'));
    if (r.url_verified) {
      top.appendChild(badge('Official URL', 'ok', 'Verified live SHL page'));
    } else {
      top.appendChild(badge('Legacy link', 'warn',
        'Archived URL (' + (r.url_status || 'unresolved') + ') — redirects, not verified'));
    }
    const why = document.createElement('div');
    why.className = 'why';
    why.textContent = r.rationale || '';
    li.appendChild(top); li.appendChild(why);
    recs.appendChild(li);
  });
}
btn.onclick = async () => {
  const text = q.value.trim();
  if (!text) return;
  btn.disabled = true;
  history.push({role: 'user', content: text});
  renderConversation();
  status.textContent = 'Thinking...';
  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({messages: history.map(m => ({role: m.role, content: m.content}))}),
    });
    const data = await resp.json();
    history.push({role: 'assistant', content: data.reply});
    renderConversation();
    renderRecommendations(data.recommendations || []);
    const n = (data.recommendations || []).length;
    status.textContent = n + ' recommendation(s)' +
      (data.end_of_conversation ? ' - conversation complete' : '');
    q.value = '';
  } catch (err) {
    status.textContent = 'Request failed: ' + err.message;
  } finally {
    btn.disabled = false;
    q.focus();
  }
};
clearBtn.onclick = () => {
  history.length = 0;
  conv.innerHTML = ''; recs.innerHTML = '';
  recsTitle.style.display = 'none'; status.textContent = '';
  q.value = ''; q.focus();
};
q.addEventListener('keydown', e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) btn.onclick(); });
</script>
</body>
</html>
"""


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    """A minimal, dependency-free web UI built on the same JSON /chat API."""
    return _UI_HTML


def _dense_status() -> Dict[str, Any]:
    try:
        import embeddings

        available = embeddings.available()
    except Exception:
        available = False
    enabled = os.getenv("SHL_USE_EMBEDDINGS", "0") in {"1", "true", "True"}
    return {"available": available, "enabled": enabled and available}


@app.get("/health")
def health() -> Dict[str, Any]:
    """Non-sensitive health/component availability for operators."""
    client = get_client()
    return {
        "status": "ok",
        "catalog": {
            "count": len(load_catalog()),
            "url_status": url_status_counts(),
        },
        "retrieval": {
            "bm25": True,
            "dense": _dense_status(),
        },
        "llm": {
            "enabled": bool(client.enabled),
            "provider": client.provider,
            "model": client.model,
            "base_url": client.base_url,
        },
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    messages = [m.model_dump() for m in (req.messages or [])]
    if len(messages) > MAX_MESSAGES:
        messages = messages[-MAX_MESSAGES:]
    k = max(1, min(req.requested_k(), MAX_RECOMMENDATIONS))
    try:
        result = respond(messages, k=k)
    except Exception:  # never fail the harness
        logger.exception("agent failed; returning graceful fallback")
        result = {
            "reply": ("Could you tell me the role or skills you're hiring for? "
                      "I'll then recommend relevant SHL assessments."),
            "recommendations": [],
            "end_of_conversation": False,
        }

    recs = []
    for r in result.get("recommendations", []):
        name, url = str(r.get("name", "")), str(r.get("url", ""))
        if not name or not url:
            continue
        record = find_record(name, url)
        recs.append(Recommendation(
            name=name,
            url=url,
            test_type=str(r.get("test_type", "")),
            rationale=(str(r["rationale"]) if r.get("rationale") else None),
            url_status=(record["url_status"] if record else None),
            url_verified=(record["url_verified"] if record else None),
        ))
    return ChatResponse(
        reply=str(result.get("reply", "")),
        recommendations=recs[:k],
        end_of_conversation=bool(result.get("end_of_conversation", False)),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
