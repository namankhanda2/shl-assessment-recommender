"""Hybrid retrieval over the SHL catalog.

The default retriever is pure Python (BM25 with field weighting) so the service
runs with no heavyweight dependencies. If ``sentence-transformers`` is installed
and ``SHL_USE_EMBEDDINGS=1`` is set, a dense semantic score is blended in.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Set, Tuple

from catalog import load_catalog

TOKEN_RE = re.compile(r"[^a-z0-9+#.]+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with", "at",
    "by", "is", "are", "am", "be", "been", "being", "was", "were", "do", "does",
    "did", "have", "has", "had", "i", "we", "you", "he", "she", "it", "they",
    "me", "us", "them", "my", "our", "your", "their", "this", "that", "these",
    "those", "as", "from", "about", "into", "over", "after", "before", "between",
    "need", "want", "looking", "hire", "hiring", "role", "job", "candidate",
    "candidates", "position", "please", "help", "find", "assessment",
    "assessments", "test", "tests", "some", "any", "get", "give", "show", "also",
    "would", "like", "can", "could", "should", "will", "just", "really",
}

# Generic-role tokens add almost no signal: they appear across many product
# names/descriptions and drown out the specific skill terms in a query. They are
# excluded from the document model and get no name-substring boost.
GENERIC_ROLE_TOKENS = {
    "developer", "engineer", "analyst", "manager", "specialist", "technician",
    "programmer", "consultant", "administrator", "executive", "supervisor",
    "director", "associate", "representative", "officer", "agent", "staff",
    "senior", "junior", "lead", "professional",
}

# Tokens that carry no role-specific signal (generic job titles + seniority
# words). They are stripped from extracted skills so they cannot pollute the
# query, the rationale, or the type-selection logic.
ROLE_NOISE: Set[str] = set(GENERIC_ROLE_TOKENS) | {
    "senior", "junior", "lead", "principal", "manager", "supervisor",
    "director", "head", "executive", "graduate", "intern", "internship",
    "mid", "entry",
}

# Technology / software skill tokens. When a query names a technical skill we
# prefer the genuinely technical assessments that exist in the catalogue (they
# are real products, never invented) over unrelated reports/behaviour tests.
TECH_TERMS: Set[str] = {
    "python", "java", "sql", "javascript", "typescript", "c", "c++", "c#",
    ".net", "php", "ruby", "golang", "go", "rust", "kotlin", "swift", "scala",
    "perl", "matlab", "r", "html", "html5", "css", "react", "reactjs",
    "angular", "angularjs", "vue", "node", "spring", "django", "flask", "aws",
    "azure", "gcp", "cloud", "docker", "kubernetes", "linux", "unix", "git",
    "hadoop", "spark", "kafka", "oracle", "mysql", "postgresql", "postgres",
    "mongodb", "nosql", "microservice", "microservices", "devops", "database",
    "backend", "frontend", "fullstack", "software", "programming", "coding",
    "api", "rest", "restful", "soap", "http", "web", "webservice",
    "network", "security", "cybersecurity", "testing", "qa",
    "automation", "selenium", "agile", "scrum", "shell", "bash", "powershell",
}

# Score weights (kept explicit so the tiering is obvious):
#   * an explicit skill in the product NAME is the strongest signal,
#   * an explicit skill in the description/keys is next,
#   * a technical item for a technical query is a small, positive preference,
#   * the dense (FAISS) term is a bounded tie-breaker that can never override
#     explicit skill evidence.
SKILL_NAME_WEIGHT = 6.0
SKILL_TEXT_WEIGHT = 2.0
TECH_PREF_BOOST = 2.5
DENSE_WEIGHT = 1.0

# Synonym expansion applied to query tokens so plural/inflected and short-vs-long
# forms ("nurses" vs the catalog's "nursing") still match the same documents.
SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "nurse": ("nursing",),
    "nursing": ("nurse",),
    "nurses": ("nursing", "nurse"),
    "react": ("reactjs",),
    "reactjs": ("react",),
    "angular": ("angularjs",),
    "angularjs": ("angular",),
    "rest": ("restful",),
    "restful": ("rest",),
    "api": ("restful", "webservice"),
    "webservice": ("web", "service"),
}

# Skill / domain terms used to decide whether a brief is specific enough and to
# boost matching catalog items by name.
SKILL_TERMS: Set[str] = {
    "java", "python", "sql", "javascript", "typescript", "c++", "c#", ".net",
    "php", "ruby", "golang", "go", "rust", "kotlin", "swift", "scala", "perl",
    "r", "matlab", "html", "css", "react", "angular", "vue", "node", "spring",
    "django", "flask", "aws", "azure", "gcp", "cloud", "docker", "kubernetes",
    "linux", "unix", "git", "hadoop", "spark", "kafka", "oracle", "mysql",
    "postgresql", "postgres", "mongodb", "nosql", "excel", "word", "powerpoint",
    "accounting", "accountant", "finance", "financial", "sales", "customer",
    "service", "call", "center", "contact", "leadership", "management", "manager",
    "communication", "teamwork", "stakeholder", "personality", "cognitive",
    "reasoning", "numerical", "verbal", "inductive", "deductive", "aptitude",
    "graduate", "entry", "senior", "junior", "mid", "director", "executive",
    "supervisor", "data", "analyst", "analysis", "testing", "qa", "automation",
    "selenium", "agile", "scrum", "project", "marketing", "hr", "human",
    "resources", "nursing", "nurse", "nurses", "clinical", "patient", "healthcare",
    "mechanical", "electrical", "civil",
    "reactjs", "angularjs", "frontend", "typescript",
    "api", "rest", "restful", "soap", "http", "web", "webservice",
    "network", "security", "cybersecurity", "devops", "helpdesk", "support",
    "administrative", "clerical", "retail", "hospitality", "safety", "banking",
    "insurance", "programming", "developer", "engineer", "software", "coding",
    "backend", "frontend", "fullstack", "database", "devops", "machine",
    "learning", "ai", "design", "ux", "ui", "writer", "legal", "paralegal",
    "receptionist", "warehouse", "logistics", "manufacturing", "operator",
    "technician", "account", "audit", "tax", "payroll", "procurement",
    "situational", "judgement", "judgment", "sjt", "biodata", "scenario",
    "screening", "competency", "competencies",
}

SENIORITY_TERMS: Dict[str, str] = {
    "entry": "Entry-Level", "entry-level": "Entry-Level", "junior": "Entry-Level",
    "graduate": "Graduate", "intern": "Entry-Level", "internship": "Entry-Level",
    "mid": "Mid-Professional", "mid-level": "Mid-Professional",
    "professional": "Professional Individual Contributor",
    "senior": "Professional Individual Contributor", "sr": "Professional Individual Contributor",
    "lead": "Professional Individual Contributor", "principal": "Professional Individual Contributor",
    "manager": "Manager", "supervisor": "Supervisor",
    "director": "Director", "head": "Director", "vp": "Executive",
    "executive": "Executive", "c-suite": "Executive", "cxo": "Executive",
}

PERSONALITY_TERMS = {
    "personality", "behaviour", "behavior", "culture", "fit", "stakeholder",
    "teamwork", "team", "collaboration", "collaborate", "communication",
    "leadership", "motivation", "interpersonal", "soft", "eq", "emotional",
    "integrity", "reliability", "values", "sjt", "situational", "judgment",
    "judgement", "biodata",
}

COGNITIVE_TERMS = {
    "cognitive", "reasoning", "numerical", "verbal", "inductive", "deductive",
    "aptitude", "ability", "iq", "analytical", "analytics", "problem", "solving",
    "aptitude", "graduate", "logic", "logical", "spatial", "checking",
}

SITUATIONAL_TERMS = {"situational", "sjt", "judgment", "judgement", "scenario"}

# Common acronyms / aliases mapped to substrings found in catalog names.
ALIASES: Dict[str, str] = {
    "opq": "opq",
    "opq32": "opq32",
    "opq32r": "opq32r",
    "gsa": "global skills",
    "mq": "motivation",
    "sjt": "situational",
    "verify": "verify",
    "gs+": "verify interactive g+",
    "g+": "verify interactive g+",
}

# Acronyms resolving to a specific canonical catalogue entry (preferred).
ALIAS_TO_NAME: Dict[str, str] = {
    "opq": "Occupational Personality Questionnaire OPQ32r",
    "opq32": "Occupational Personality Questionnaire OPQ32r",
    "opq32r": "Occupational Personality Questionnaire OPQ32r",
    "gsa": "Global Skills Development Report",
    "mq": "Motivation Questionnaire MQM5",
    "verify g+": "SHL Verify Interactive G+",
    "g+": "SHL Verify Interactive G+",
}

_FIELD_WEIGHTS = {"name": 3.0, "keys": 2.0, "job_levels": 1.5, "description": 1.0}

DROP_RE = re.compile(
    r"\b(drop|remove|without|exclude|excluding|forget|no longer|not needed|"
    r"don't need|do not need|instead of)\b"
)
ADD_RE = re.compile(r"\b(add|also|include|including|plus|as well as|need)\b")


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = TOKEN_RE.sub(" ", text)
    out = []
    for tok in text.split():
        tok = tok.strip(".")
        if not tok or tok in STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


def _expand(tokens: Iterable[str]) -> List[str]:
    """Expanded token list including synonym forms (keeps original order)."""
    out: List[str] = []
    for tok in tokens:
        out.append(tok)
        out.extend(SYNONYMS.get(tok, ()))
    return out


def _item_tokens(item: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    for field, weight in _FIELD_WEIGHTS.items():
        value = item.get(field)
        if isinstance(value, (list, tuple)):
            value = " ".join(str(v) for v in value)
        field_tokens = [
            t for t in tokenize(str(value or ""))
            if t not in GENERIC_ROLE_TOKENS
        ]
        reps = int(round(weight))
        for _ in range(max(1, reps)):
            tokens.extend(field_tokens)
    # Always include the single-letter test type and its label keywords.
    tokens.extend(tokenize(item.get("test_type", "")))
    return tokens


class Retriever:
    """BM25 retriever with light field weighting and optional dense blending."""

    def __init__(self, catalog: List[Dict[str, Any]] | None = None):
        self.catalog = catalog or load_catalog()
        self.k1 = 1.5
        self.b = 0.75
        self._tf: List[Counter] = []
        self._doc_len: List[int] = []
        self._df: Counter = Counter()
        self._avgdl = 1.0
        self._build()
        self._dense = None
        self._maybe_load_embeddings()

    def _build(self) -> None:
        for item in self.catalog:
            tokens = _item_tokens(item)
            tf = Counter(tokens)
            self._tf.append(tf)
            self._doc_len.append(sum(tf.values()) or 1)
            for tok in tf:
                self._df[tok] += 1
        n = len(self.catalog) or 1
        self._avgdl = sum(self._doc_len) / n
        self._n = n

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def bm25(self, query_tokens: Iterable[str]) -> List[float]:
        scores = [0.0] * len(self.catalog)
        for term in set(query_tokens):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i, tf in enumerate(self._tf):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avgdl)
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores

    def _name_boost(self, item: Dict[str, Any], query_tokens: Set[str],
                    raw_query: str) -> float:
        name = item["name"].lower()
        name_tokens = set(tokenize(name))
        boost = 0.0
        # Generic role / seniority words must not create name overlap, otherwise
        # "Informatica (Developer)" would match a plain "developer" query.
        query_tokens = {t for t in query_tokens if t not in ROLE_NOISE}
        overlap = name_tokens & query_tokens
        boost += 1.2 * len(overlap)
        # A specific skill term appearing in the product name is the strongest
        # signal. Generic role words (developer, analyst, ...) are deliberately
        # treated as weak, otherwise every "Informatica (Developer)" style item
        # outranks the genuinely matching "ReactJS" for a React role.
        for tok in query_tokens & SKILL_TERMS:
            if tok in GENERIC_ROLE_TOKENS:
                continue
            if tok in name:
                boost += 1.8
            else:
                # Prefix / suffix overlap, e.g. query "react" vs name "reactjs".
                for nt in name_tokens:
                    if nt.startswith(tok) and len(nt) - len(tok) <= 6:
                        boost += 1.0
                        break
                    if tok.startswith(nt) and len(tok) - len(nt) <= 6:
                        boost += 0.8
                        break
        if name and name in raw_query:
            boost += 6.0
        return boost

    def _scores(self, query: str) -> Optional[List[float]]:
        """Full-weighted score for every catalog item (BM25 + boosts + dense).

        A single code path shared by ``search`` and ``rank`` so the optional
        FAISS dense blend always affects recommendation ranking, not just the
        ``search`` helper. Returns ``None`` when the query has no tokens.
        """
        qtokens = _expand(tokenize(query))
        if not qtokens:
            return None
        qset = set(qtokens)
        raw = (query or "").lower()
        skills = {t for t in qset if t in SKILL_TERMS and t not in ROLE_NOISE}
        tech_query = skills & TECH_TERMS
        wanted_levels = set(detect_seniority(raw))
        scores = self.bm25(qtokens)
        for i, item in enumerate(self.catalog):
            score = scores[i]
            score += self._name_boost(item, qset, raw)
            # Explicit job-skill matches dominate the ranking.
            if skills:
                name_tokens = set(tokenize(item["name"]))
                haystack = set(tokenize(" ".join([
                    item["name"], item.get("description", ""),
                    " ".join(item.get("keys") or []),
                ])))
                for skill in skills:
                    forms = {skill} | set(SYNONYMS.get(skill, ()))
                    if name_tokens & forms:
                        score += SKILL_NAME_WEIGHT
                    elif haystack & forms:
                        score += SKILL_TEXT_WEIGHT
            # Seniority: reward matching catalogue job levels.
            if wanted_levels and wanted_levels & set(item.get("job_levels") or []):
                score += 0.6
            # For a technical brief, prefer the technical assessments that exist.
            if tech_query and set(tokenize(item["name"])) & TECH_TERMS:
                score += TECH_PREF_BOOST
            name = item["name"].lower()
            if "report" in name and "report" not in raw:
                score *= 0.45
            if "solution" in name and "solution" not in raw:
                score *= 0.6
            scores[i] = score
        if self._dense is not None:
            scores = self._blend_dense(query, scores, weight=DENSE_WEIGHT)
        return scores

    def search(self, query: str, k: int = 10,
               restrict_types: Iterable[str] | None = None) -> List[Dict[str, Any]]:
        scores = self._scores(query)
        if scores is None:
            return []
        if restrict_types:
            allowed = set(restrict_types)
            scores = [s if set(item["test_types"]) & allowed else -1.0
                      for s, item in zip(scores, self.catalog)]
        ranked = sorted(range(len(self.catalog)), key=lambda i: scores[i], reverse=True)
        return [self.catalog[i] for i in ranked[:k] if scores[i] > 0]

    def rank(self, query: str) -> List[Tuple[Dict[str, Any], float]]:
        """Return every catalog item with its score, best first."""
        scores = self._scores(query)
        if scores is None:
            return [(item, 0.0) for item in self.catalog]
        order = sorted(range(len(self.catalog)), key=lambda i: scores[i], reverse=True)
        return [(self.catalog[i], scores[i]) for i in order]

    # -- optional dense retrieval -----------------------------------------
    def _maybe_load_embeddings(self) -> None:
        """Build/load the FAISS index when ``SHL_USE_EMBEDDINGS=1`` is set."""
        if os.getenv("SHL_USE_EMBEDDINGS", "0") not in {"1", "true", "True"}:
            return
        try:
            import embeddings as emb

            if emb.available():
                self._dense = emb.DenseIndex.load_or_build(self.catalog)
        except Exception:
            self._dense = None

    def _blend_dense(self, query: str, base: List[float], weight: float) -> List[float]:
        """Add a *bounded* dense-similarity tie-breaker to the lexical scores.

        The dense term is min-max normalised to ``[0, weight]`` and added to the
        base score, so a semantic hit can re-order items but can never leapfrog a
        strong explicit-skill match (those carry much larger weights). When the
        lexical signal is empty the dense term still ranks semantically.
        """
        try:
            sims = [float(s) for s in self._dense.similarities(query)]
            lo = min(sims)
            span = (max(sims) - lo) or 1.0
            return [base[i] + weight * (sims[i] - lo) / span
                    for i in range(len(base))]
        except Exception:
            return base


def extract_skills(text: str) -> Set[str]:
    toks = set(tokenize(text))
    return {t for t in toks if t in SKILL_TERMS and t not in ROLE_NOISE}


def detect_seniority(text: str) -> List[str]:
    low = (text or "").lower()
    found: List[str] = []
    for term, level in SENIORITY_TERMS.items():
        if re.search(r"\b" + re.escape(term) + r"\b", low) and level not in found:
            found.append(level)
    return found


def has_personality_signal(text: str) -> bool:
    toks = set(tokenize(text))
    return bool(toks & PERSONALITY_TERMS)


def has_cognitive_signal(text: str) -> bool:
    toks = set(tokenize(text))
    return bool(toks & COGNITIVE_TERMS)


def has_situational_signal(text: str) -> bool:
    toks = set(tokenize(text))
    return bool(toks & SITUATIONAL_TERMS)


def is_specific(text: str) -> bool:
    """True when a message contains an actionable role/skill/domain signal."""
    low = (text or "").lower()
    if extract_skills(low):
        return True
    if re.search(r"job description|jd\b|responsibilities|requirements|skills? needed",
                 low):
        return True
    meaningful = [t for t in tokenize(low) if len(t) > 2]
    return len(meaningful) >= 4


DEFAULT = None


def default_retriever() -> Retriever:
    global DEFAULT
    if DEFAULT is None:
        DEFAULT = Retriever()
    return DEFAULT


def find_mentioned_items(text: str, catalog: List[Dict[str, Any]] | None = None,
                         limit: int = 5) -> List[Dict[str, Any]]:
    """Find catalog items explicitly named or aliased in the user's text."""
    catalog = catalog or load_catalog()
    low = (text or "").lower()
    if not low.strip():
        return []

    hits: List[Tuple[int, Dict[str, Any]]] = []
    seen = set()
    by_name = {it["name"].lower(): it for it in catalog}

    def add(item, rank):
        if item["url"] in seen:
            return
        seen.add(item["url"])
        hits.append((rank, item))

    # 0. Canonical acronym -> exact catalogue entry.
    for alias, target in ALIAS_TO_NAME.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", low):
            item = by_name.get(target.lower())
            if item is not None:
                add(item, 0)

    # 1. Full product name present verbatim.
    for item in catalog:
        name = item["name"].lower()
        if len(name) >= 4 and name in low:
            add(item, 1)

    # 2. Acronym / short-alias substring matches, shortest name first.
    for alias, needle in ALIASES.items():
        if not re.search(r"\b" + re.escape(alias) + r"\b", low):
            continue
        name_hits = [it for it in catalog if needle in it["name"].lower()]
        desc_hits = [it for it in catalog
                     if needle in it["description"].lower()
                     and needle not in it["name"].lower()]
        name_hits.sort(key=lambda it: len(it["name"]))
        for item in name_hits:
            add(item, 2)
        for item in desc_hits[:2]:
            add(item, 4)

    # 3. Distinctive token overlap (e.g. "java", "excel", "python").
    _generic = {"new", "report", "solution", "test", "level", "advanced",
                "essentials", "next", "generation", "and", "the"}
    for item in catalog:
        toks = [t for t in tokenize(item["name"])
                if len(t) > 2 and t not in _generic]
        if 1 <= len(toks) <= 2 and all(t in low for t in toks):
            add(item, 3)
    hits.sort(key=lambda pair: pair[0])
    return [item for _, item in hits[:limit]]
