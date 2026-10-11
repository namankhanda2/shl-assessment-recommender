# SHL Assessment Recommender

A stateless, production-style **Retrieval-Augmented Generation (RAG)** service that
recommends SHL assessments from a recruiter's job description. It combines
lexical retrieval (BM25), optional semantic retrieval (sentence-transformer
embeddings + FAISS), and an optional LLM, and exposes a small FastAPI contract.

The service is fully functional **without** any API key or ML dependency: it falls
back to a deterministic, catalog-grounded planner.

```
GET  /            service metadata
GET  /health      health + non-sensitive component status
POST /chat        stateless conversational recommendation
GET  /ui          light dependency-free web UI (same JSON API under the hood)
GET  /docs        OpenAPI UI
GET  /openapi.json
```

---

## 1. Architecture & RAG workflow

```
Recruiter job description / conversation
        │
        ▼
┌───────────────────────────── app.py (FastAPI) ─────────────────────────────┐
│  validate + normalise messages  →  agent.respond(messages)                 │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
┌────────────────────────────── agent.py ────────────────────────────────────┐
│  scope guards (injection / legal / off-topic)                              │
│  clarify when the brief is vague                                           │
│  compare named assessments from catalog data                               │
│  build constraints → build query → select_recommendations                  │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
┌──────────────────────────── retrieval.py ──────────────────────────────────┐
│  BM25 (pure Python, field-weighted)                                        │
│  + optional bounded FAISS tie-breaker when SHL_USE_EMBEDDINGS=1            │
│  → ranked, de-duplicated catalog records                                   │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
┌────────────────── rag.py  (generation half) ───────────────────────────────┐
│  build a grounded prompt with candidate names only                         │
│  optional LLM may re-rank / rewrite  →  selected names validated back to   │
│  catalog records (names/URLs can never be hallucinated)                    │
│  no key / error → deterministic base reply                                 │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
        catalog-grounded JSON: {reply,
          recommendations[{name,url,test_type,rationale,url_status,url_verified}],
          end_of_conversation}
```

Key guarantees:

- **Grounded**: every recommendation is resolved back to a catalog record; the
  LLM may only *select* from supplied candidate names. Unknown names are dropped.
- **De-duplicated**: at most one record per normalized URL (OPQ/MQ report
  families all map to one product page).
- **Deterministic fallback**: identical output for equivalent requests when no
  LLM/dense index is available.

## 2. Repository structure

```
app.py                     FastAPI entrypoint (/, /health, /chat, /ui)
agent.py                   planner: guards, clarify, refine, compare, recommend, rationale
rag.py                     generation half: prompt build + grounded LLM selection
retrieval.py               BM25 retriever + optional FAISS dense blend (shared by search & rank)
embeddings.py              FAISS + sentence-transformers index (cached)
llm.py                     optional OpenAI-compatible LLM client
catalog.py                 catalog loading, URL mapping, status + validation helpers + CLI
scraper.py                 live SHL catalog harvester + URL auditor (CLI)
evaluate.py                offline Recall@10 + behaviour probes
catalog.json               bundled 377-record catalog (source of truth)
catalog.enriched.json      normalised snapshot incl. url_status / legacy_url (derived)
catalog.scraped.json       verified records scraped from the live site
url_map.json               legacy → verified current detail page mapping
verification_report.json   per-URL audit (redirect + detail classification)
verification_sample.json   quick sampled audit (never overwrites the full report)
url_mapping_report.json    mapped vs unmappable summary
frontend/                  standalone Streamlit UI (talks to /chat over HTTP)
  streamlit_app.py         UI: query, count, loading/errors, result cards
  api_client.py            Streamlit-free HTTP client (payload build + parsing)
  requirements.txt         frontend-only deps (streamlit, httpx)
tests/                     pytest / unittest suite
  test_improvements.py     role ranking, rationale grounding, URL transparency
  test_frontend_api_client.py  frontend request/response + error handling
```

## 3. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional: dense retrieval
pip install numpy sentence-transformers faiss-cpu
```

## 4. Configuration (`.env`)

Copy `.env.example` to `.env` and set **one** OpenAI-compatible key. Real
environment variables always take precedence over `.env`. The key is never
printed or logged.

```bash
cp .env.example .env
# then edit .env, e.g. GROQ_API_KEY=...
```

See `.env.example` for all options. `.env` is git-ignored; only placeholders are
committed.

## 5. Running the API

```bash
python app.py
# or
uvicorn app:app --reload --port 8000
```

## 6. Testing the API

```bash
# health / component status
curl -s localhost:8000/health | python -m json.tool

# a recommendation request
curl -s localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hiring a Java backend developer with SQL and stakeholder skills"}]}' \
  | python -m json.tool
```

Open <http://localhost:8000/ui> for the web UI, or
<http://localhost:8000/docs> for the interactive OpenAPI docs.

The number of returned cards is **exact** when the catalog can satisfy it. Pass
an optional size with any of `k`, `count`, `n`, `limit`, `num_recommendations` or
`max_recommendations` (clamped to 1..10; default 10):

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Senior Python Backend Developer with REST API, SQL and AWS"}],"k":5}'
# -> exactly 5 recommendations, even after LLM refinement
```

The count is guaranteed to be exact only up to what the catalog genuinely
contains for the brief (e.g. a nursing query has one matching assessment); it is
never padded with invented items.

Each recommendation carries **additive** URL-transparency fields (existing
clients that ignore unknown keys are unaffected):

```json
{
  "name": "Nursing (New)",
  "url": "https://www.shl.com/products/product-catalog/view/nursing-new/",
  "test_type": "K",
  "rationale": "Knowledge & Skills assessment relevant to nurse; suitable for Graduate and Manager level; takes about 11 minutes.",
  "url_status": "category_redirect",
  "url_verified": false
}
```

`url_verified` is `true` only for a live/mapped official detail page
(`current` / `verified_mapped`); a flagged legacy link is **never** reported as
verified. The `/ui` page renders this as an *Official URL* vs *Legacy link*
badge.

## 6.1 Frontend (Streamlit UI)

A standalone Streamlit app (`frontend/streamlit_app.py`) provides a richer UI
than the built-in `/ui` page. It only calls the public `POST /chat` endpoint, so
the backend is unchanged. It lets you:

- enter a job description or hiring query;
- pick the number of recommendations (1–10, sent as `k`);
- see loading states and readable API errors (unreachable host, timeout,
  HTTP ≥ 400, bad response);
- view assessment name, test type, rationale and the **URL verification status**
  (✅ Official URL vs ⚠️ Legacy link) per result.

Install the frontend-only deps and run it:

```bash
pip install -r frontend/requirements.txt

# Point at the deployed Render service:
SHL_API_URL="https://<your-service>.onrender.com" \
  streamlit run frontend/streamlit_app.py
```

Without `SHL_API_URL` it defaults to `http://localhost:8000`; the sidebar also
lets you edit the base URL and run a connection check against `/health`.
`frontend/api_client.py` holds the request/response logic (no Streamlit import)
and is covered by `tests/test_frontend_api_client.py`.

## 7. Retrieval: BM25, embeddings, FAISS, hybrid

- **BM25** (`retrieval.py`): pure-Python, field-weighted (name ×3, keys ×2, job
  levels ×1.5, description ×1) with light name boosting. Always available.
- **Specificity-aware scoring**: generic role words (`developer`, `engineer`,
  `analyst`, …) and seniority words (`senior`, `lead`, …) are stripped from the
  extracted skills, excluded from the document model and get no name boost, so a
  query like *"Frontend developer with React and TypeScript"* ranks **ReactJS**
  above unrelated `.NET`/`SQL` items that merely share the word "Developer".
  A small synonym map (`nurse`↔`nursing`, `react`↔`reactjs`, `rest`↔`restful`,
  `api`↔`restful`/`webservice`, …) also lets plural, short or phrase forms match
  the catalog. Rationales cite a skill only when it appears in the record's own
  name/description/keys.
- **Tiered scoring** (`Retriever._scores`), strongest signal first:
  1. an explicit query skill found in the product **name** (`+6`); a skill found
     in the description/keys scores `+2`;
  2. a matching catalogue **job level** for the detected seniority (`+0.6`);
  3. a **technical preference** (`+2.5`) for technical assessments when the brief
     names a technical skill — so Python/SQL/AWS/Docker/Git/… genuinely present
     in the catalogue are prioritised, and nothing outside the catalogue is ever
     added;
  4. BM25 plus light name boost.
- **Dense** (`embeddings.py`): `all-MiniLM-L6-v2` sentence embeddings, L2-normalised
  and stored in a FAISS `IndexFlatIP`. Cosine similarity is computed for the query.
- **Hybrid blend**: when `SHL_USE_EMBEDDINGS=1`, dense similarity is min-max
  normalised to `[0, 1]` and **added** as a bounded tie-breaker, so a semantic hit
  can re-order items but can never override a strong explicit-skill match.

The index is built once and cached under `.cache/embeddings/` (keyed by model name
+ catalog fingerprint), so it is **not rebuilt per request**. Disabling the env
var (default) falls back to keyword-only retrieval.

The blend is applied through a single shared scoring path used by **both**
`Retriever.search()` and `Retriever.rank()` — `rank()` feeds
`agent.select_recommendations`, so enabling `SHL_USE_EMBEDDINGS` genuinely changes
what `/chat` recommends (verified: the two modes return different rankings).

```bash
# demo dense search
SHL_USE_EMBEDDINGS=1 python embeddings.py
```

## 8. Deterministic fallback

With no LLM key (or on any provider error) the deterministic planner in
`agent.py` produces the reply and recommendation order; `rag.generate` returns the
base result unchanged. This makes the service reproducible and testable offline.

## 9. Evaluations and integration tests

```bash
python evaluate.py                 # deterministic Recall@10 + behaviour probes
SHL_USE_EMBEDDINGS=1 python evaluate.py   # same probes with the dense blend on

python -m pytest -q                # full suite (hermetic, no network)
python -m unittest discover -s tests -t .

# one real LLM call (only when credentials are configured)
SHL_LLM_INTEGRATION=1 pytest tests/test_llm_integration.py -q
```

Latest results on this machine: 112 tests passed + 1 skipped (the opt-in LLM test);
deterministic Recall@10 0.967 with all 7 behaviour probes passing; with
`SHL_USE_EMBEDDINGS=1`, Recall@10 0.967 with all probes passing.

Shortlist size and URL rules are preserved: `/chat` returns exactly the requested
number of candidates (up to 10) even after an LLM re-orders them (the model's
preferred order first, then the remaining grounded candidates), and every
recommendation keeps its `url_status`/`url_verified` fields.

## 10. Catalog & URL verification status

The bundled catalog has **377** records. SHL retired the legacy
`/products/product-catalog/view/<slug>/` URLs; they now redirect. The audit
(`verification_report.json`) classifies every record, and `catalog.py` exposes the
result as `url_status`:

| status | meaning | count |
| --- | --- | --- |
| `verified_mapped` | legacy URL remapped to a verified current detail page | 40 |
| `category_redirect` | legacy URL redirects to an assessment category page | 198 |
| `generic_redirect` | legacy URL redirects to a generic page (e.g. `/products/`) | 139 |
| `current` | live at its own official detail page | 0 |
| `error` | unreachable during the audit | 0 |

- Mappings live in `url_map.json` and are derived **only** from audited redirect
  targets — URLs are never guessed.
- Only `current` / `verified_mapped` are treated as verified (`url_verified`).
- Unresolved records stay available for retrieval but keep their original URL and
  are flagged; their links are **not** presented as verified.
- Inspect or export the derived, URL-resolved catalog:

```bash
python catalog.py --status                 # counts by url_status
python catalog.py --export catalog.enriched.json
```

- Refresh the audit / mapping:

```bash
python scraper.py --verify --sample 20   # quick sample -> verification_sample.json
python scraper.py --verify               # full audit -> verification_report.json (slow)
python scraper.py --map                  # rebuild url_map.json from the audit
```

A sampled audit writes `verification_sample.json` and never overwrites the full
`verification_report.json`. Full auditing is **never** run automatically at
startup.

## 11. Known limitations

- The public SHL site exposes only 7 official assessment detail pages, so 337 of
  377 legacy records cannot be mapped to a dedicated current page; they are
  flagged `category_redirect` / `generic_redirect`.
- Dense retrieval needs `SHL_USE_EMBEDDINGS=1` and the optional dependencies; it
  is off by default so the base install stays light.
- The LLM is optional and never required for correctness.
- Legacy URLs remain in the catalog for retrieval coverage; consume `url_status`
  before displaying links.

## 12. Troubleshooting

- **LLM disabled**: run `python llm.py` to see the resolved provider/model without
  exposing the key; check `SHL_LLM_INTEGRATION` is unset for offline runs.
- **Dense index stale**: delete `.cache/embeddings/` and retry; it rebuilds from
  the current catalog fingerprint.
- **`SSL: CERTIFICATE_VERIFY_FAILED` on macOS**: use the bundled `requests`-based
  scraper (not stdlib `urllib`), or install certificates for Python.
- **Port in use**: `uvicorn app:app --port 8001`.
