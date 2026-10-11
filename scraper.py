"""Harvest the *current* SHL assessment catalog.

History: SHL's legacy ``/solutions/products/product-catalog/`` listing and its
``product-catalog/view/<slug>/`` detail URLs have been retired. Those detail
URLs now issue 301 redirects to generic pages, so scraping them yields nothing
useful. This scraper therefore works against the public site as it exists today:

1. Seed from the assessment listing pages under ``/products/assessments/``.
2. Discover ``/products/assessments/<category>/<slug>/`` detail pages from the
   listing pages and, optionally, from ``sitemap.xml``.
3. Extract ``name`` / ``description`` / ``test_type`` from each detail page.
4. Mark every record extracted from a live detail page as ``verified: True``.

The bundled 377-record ``catalog.json`` (the earlier full Individual Test
Solutions export) is preserved as the fallback and left untouched; verified
records are written to ``catalog.scraped.json``. ``--verify`` re-checks the
bundled URLs and writes ``verification_report.json`` distinguishing records
that still resolve to a live, specific page from those that redirect away.

Run with::

    python scraper.py                 # write catalog.scraped.json
    python scraper.py --print         # preview without writing
    python scraper.py --merge         # merge verified records into catalog.json
    python scraper.py --verify        # full audit -> verification_report.json
    python scraper.py --verify --sample 20   # quick sample audit
    python scraper.py --map           # derive url_map.json from the audit
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

REQUEST_TIMEOUT = 25
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

SITE = "https://www.shl.com"
SITEMAP = f"{SITE}/sitemap.xml"

# Public assessment listing pages that exist today. These act as crawl seeds.
LISTING_PAGES = [
    f"{SITE}/products/assessments/",
    f"{SITE}/products/assessments/skills-and-simulations/",
    f"{SITE}/products/assessments/skills-and-simulations/technical-skills/",
    f"{SITE}/products/assessments/skills-and-simulations/coding-simulations/",
    f"{SITE}/products/assessments/skills-and-simulations/business-skills/",
    f"{SITE}/products/assessments/skills-and-simulations/call-center-simulations/",
    f"{SITE}/products/assessments/skills-and-simulations/language-evaluation/",
    f"{SITE}/products/assessments/cognitive-assessments/",
    f"{SITE}/products/assessments/personality-assessment/",
    f"{SITE}/products/assessments/behavioral-assessments/",
    f"{SITE}/products/assessments/aptitude-tests/",
    f"{SITE}/products/assessments/psychometric-tests/",
    f"{SITE}/products/assessments/job-focused-assessments/",
]

# Category slug -> SHL single-letter test-type code.
CATEGORY_TEST_TYPE = {
    "cognitive-assessments": "A",
    "aptitude-tests": "A",
    "personality-assessment": "P",
    "psychometric-tests": "P",
    "behavioral-assessments": "B",
    "skills-and-simulations": "K",
    "technical-skills": "K",
    "business-skills": "K",
    "language-evaluation": "K",
    "job-focused-assessments": "K",
    "coding-simulations": "S",
    "call-center-simulations": "S",
    "assessment-and-development-centers": "E",
}

# Per-product overrides where the category default is misleading.
DETAIL_TYPE_OVERRIDES = {
    "global-skills-assessment-gsa": "C",
    "universal-competency-framework": "C",
    "situation-judgement-tests-sjt": "B",
    "realistic-job-and-culture-previews-rjp": "B",
    "shl-motivation-questionnaire-mq": "P",
    "shl-occupational-personality-questionnaire-opq": "P",
    "svar": "K",
}

# Slugs that look like detail pages but are actually navigation / categories.
IGNORE_SLUGS = {"all", "search", "search-results", "index.html", "products"}

# Minimum number of live detail pages before we trust the crawl over the bundle.
MIN_VERIFIED = 3


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable, no network)
# --------------------------------------------------------------------------- #
def strip_url(url: str) -> str:
    """Drop query string and fragment; normalise the trailing slash."""
    parts = urlparse(url)
    path = parts.path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((parts.scheme, parts.netloc, path, "", "", ""))


def is_assessment_url(url: str) -> bool:
    parts = urlparse(url)
    if parts.netloc and parts.netloc not in {"www.shl.com", "shl.com"}:
        return False
    return "/products/assessments/" in parts.path


def is_detail_url(url: str) -> bool:
    """True for ``/products/assessments/<category>/<slug>/`` product pages."""
    path = urlparse(url).path.rstrip("/")
    parts = [p for p in path.split("/") if p]
    if len(parts) < 4 or parts[0] != "products" or parts[1] != "assessments":
        return False
    slug = parts[-1]
    if slug in CATEGORY_TEST_TYPE or slug in IGNORE_SLUGS:
        return False
    return True


def infer_test_type(url: str) -> str:
    """Infer the SHL test-type code for a detail URL."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    slug = parts[-1] if parts else ""
    if slug in DETAIL_TYPE_OVERRIDES:
        return DETAIL_TYPE_OVERRIDES[slug]
    for seg in reversed(parts):
        if seg in CATEGORY_TEST_TYPE:
            return CATEGORY_TEST_TYPE[seg]
    return "K"


def _clean(text: Optional[str]) -> str:
    return " ".join((text or "").split())


def _meta(soup: Any, **attrs: str) -> Optional[str]:
    tag = soup.find("meta", attrs=attrs)
    if tag and tag.get("content"):
        return _clean(tag.get("content"))
    return None


def _ldjson_description(html: str) -> Optional[str]:
    for block in re.findall(
        r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S
    ):
        try:
            data = json.loads(block)
        except Exception:
            continue
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if isinstance(item, dict) and item.get("@type") in {
                "Product", "WebPage", "Article", "Service",
            }:
                desc = item.get("description")
                if desc:
                    return _clean(desc)
    return None


def extract_links(html: str, base_url: str = SITE) -> set:
    """All distinct ``/products/assessments/**`` URLs referenced by a page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out = set()
    for anchor in soup.find_all("a", href=True):
        url = urljoin(base_url, anchor["href"])
        if is_assessment_url(url):
            out.add(strip_url(url))
    for match in re.findall(r'https://www\.shl\.com/products/assessments/[^\s"\\<>]+', html):
        out.add(strip_url(match))
    return out


def parse_detail_page(html: str, url: str) -> Optional[Dict[str, Any]]:
    """Extract a single assessment record from a live detail page's HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    name = ""
    h1 = soup.find("h1")
    if h1:
        name = _clean(h1.get_text(" ", strip=True))
    if not name:
        name = _meta(soup, property="og:title") or ""
    if not name and soup.title:
        name = _clean(re.split(r"\s*\|\s*SHL\s*$", soup.title.get_text())[0])

    if not name:
        return None

    description = (
        _meta(soup, name="description")
        or _meta(soup, property="og:description")
        or _ldjson_description(html)
        or ""
    )
    if not description:
        for para in soup.find_all("p"):
            text = _clean(para.get_text(" ", strip=True))
            if len(text) >= 40:
                description = text
                break

    test_type = infer_test_type(url)
    return {
        "name": name,
        "url": strip_url(url),
        "test_type": test_type,
        "test_types": [test_type],
        "description": description,
        "job_levels": [],
        "duration": "",
        "remote": False,
        "adaptive": False,
        "keys": [],
        "verified": True,
        "source": "shl.com",
    }


def classify_redirect(requested: str, status: int, final: str) -> str:
    """Classify how a URL responded: ``ok``, ``redirected`` or ``error``.

    A ``redirected`` result means the requested specific page no longer exists
    and the server bounced us to a different path (e.g. the generic /products/
    or a category page).
    """
    if status != 200:
        return "error"
    req_path = urlparse(requested).path.rstrip("/")
    fin_path = urlparse(final or requested).path.rstrip("/")
    return "ok" if req_path == fin_path else "redirected"


# --------------------------------------------------------------------------- #
# Network layer
# --------------------------------------------------------------------------- #
def _session() -> Any:
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _requests_fetch(session: Any, url: str, retries: int = 2) -> Tuple[int, str, str]:
    """GET ``url`` with a timeout, transient-error retries and backoff."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if resp.status_code >= 500 and attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return resp.status_code, resp.text, resp.url
        except Exception as exc:  # network/timeout errors
            last_exc = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            break
    raise last_exc if last_exc else RuntimeError(f"failed to fetch {url}")


def discover_from_sitemap(fetch: Callable[[str], Tuple[int, str, str]]) -> List[str]:
    """Return assessment listing/detail URLs advertised in sitemap.xml."""
    try:
        status, body, _ = fetch(SITEMAP)
    except Exception:
        return []
    if status != 200:
        return []
    locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", body)
    children = [loc for loc in locs if loc.endswith(".xml")]
    urls: List[str] = []
    targets = [loc for loc in locs if "/products/assessments/" in loc]
    for child in children:
        if "en" not in child and "product" not in child:
            continue
        try:
            cstatus, cbody, _ = fetch(child)
        except Exception:
            continue
        if cstatus == 200:
            targets.extend(
                loc for loc in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", cbody)
                if "/products/assessments/" in loc
            )
    for url in targets:
        if is_assessment_url(url):
            urls.append(strip_url(url))
    return list(dict.fromkeys(urls))


def scrape_catalog(
    fetch: Optional[Callable[[str], Tuple[int, str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Crawl the live site and return verified assessment records.

    ``fetch`` maps a URL to ``(status, text, final_url)`` and is injectable so
    the crawl can be exercised offline in tests.
    """
    if fetch is None:
        session = _session()
        fetch = lambda u: _requests_fetch(session, u)  # noqa: E731

    candidates: Dict[str, None] = {}
    for seed in LISTING_PAGES:
        try:
            status, body, _ = fetch(seed)
        except Exception:
            continue
        if status != 200:
            continue
        for link in extract_links(body, seed):
            candidates.setdefault(link, None)
        time.sleep(0.2)

    for link in discover_from_sitemap(fetch):
        candidates.setdefault(link, None)

    records: List[Dict[str, Any]] = []
    seen_names = set()
    for url in candidates:
        if not is_detail_url(url):
            continue
        try:
            status, body, final = fetch(url)
        except Exception:
            continue
        if status != 200:
            continue
        record = parse_detail_page(body, final or url)
        if not record:
            continue
        key = record["name"].lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        records.append(record)
        time.sleep(0.2)

    return records


def classify_record(verdict: str, final_url: str) -> str:
    """Bucket a record by how its official URL resolved.

    * ``verified``  – still served at its own URL,
    * ``remapped``  – redirects to a specific current detail page,
    * ``category``  – redirects to a current assessment category page,
    * ``generic``   – redirects to a generic page (e.g. ``/products/``),
    * ``error``     – request failed.
    """
    if verdict == "ok":
        return "verified"
    if verdict == "error":
        return "error"
    if is_detail_url(final_url):
        return "remapped"
    if is_assessment_url(final_url):
        return "category"
    return "generic"


def verify_url(fetch: Callable[[str], Tuple[int, str, str]], url: str) -> Dict[str, Any]:
    try:
        status, _, final = fetch(url)
    except Exception as exc:  # pragma: no cover - network errors
        return {"url": url, "status": None, "final_url": url, "verdict": "error",
                "classification": "error", "final_is_detail": False,
                "final_is_assessment": False, "error": str(exc)}
    final = final or url
    verdict = classify_redirect(url, status, final)
    return {
        "url": url,
        "status": status,
        "final_url": final,
        "verdict": verdict,
        "classification": classify_record(verdict, final),
        "final_is_detail": is_detail_url(final),
        "final_is_assessment": is_assessment_url(final),
    }


def verify_records(
    records: List[Dict[str, Any]],
    fetch: Optional[Callable[[str], Tuple[int, str, str]]] = None,
    limit: Optional[int] = None,
    progress_every: Optional[int] = None,
) -> Dict[str, Any]:
    """Audit a list of records, splitting verified from unverified."""
    if fetch is None:
        session = _session()
        fetch = lambda u: _requests_fetch(session, u)  # noqa: E731

    checked = []
    selected = records[: limit or len(records)]
    for index, record in enumerate(selected, start=1):
        result = verify_url(fetch, record.get("url", ""))
        result["name"] = record.get("name", "")
        result["verified"] = result["verdict"] == "ok"
        checked.append(result)
        if progress_every and (index % progress_every == 0):
            print(f"[scraper] checked {index}/{len(selected)} URLs ...")
        time.sleep(0.1)

    verified = [r for r in checked if r["verified"]]
    verdicts: Dict[str, int] = {}
    classes: Dict[str, int] = {}
    for record in checked:
        verdicts[record["verdict"]] = verdicts.get(record["verdict"], 0) + 1
        cls = record.get("classification", "error")
        classes[cls] = classes.get(cls, 0) + 1
    return {
        "checked": len(checked),
        "verified": len(verified),
        "unverified": len(checked) - len(verified),
        "verdicts": verdicts,
        "classifications": classes,
        "records": checked,
    }


# --------------------------------------------------------------------------- #
# Legacy -> current URL mapping (derived from verified redirects)
# --------------------------------------------------------------------------- #
def build_url_map(records: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map legacy URLs to the *current* detail page they redirect to.

    Only records whose verified redirect target is a specific
    ``/products/assessments/<category>/<slug>`` detail page are mapped. Details
    come straight from the audited redirects in ``verification_report.json``, so
    no URL is ever guessed. Records that land on a category/generic page are
    omitted (they are reported as unmappable).
    """
    mapping: Dict[str, str] = {}
    for record in records:
        if not record.get("final_is_detail"):
            continue
        legacy = strip_url(record.get("url", ""))
        target = strip_url(record.get("final_url", ""))
        if legacy and target and legacy != target:
            mapping[legacy] = target
    return mapping


def build_url_mapping_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Human-readable summary of which legacy URLs could be remapped."""
    from catalog import classify_verification_record

    mapping = build_url_map(records)
    targets: Dict[str, int] = {}
    for target in mapping.values():
        targets[target] = targets.get(target, 0) + 1
    unmappable = [
        {
            "name": r.get("name", ""),
            "url": r.get("url", ""),
            "final_url": r.get("final_url", ""),
            "status": classify_verification_record(r),
        }
        for r in records
        if strip_url(r.get("url", "")) not in mapping
    ]
    status_counts: Dict[str, int] = {}
    for entry in unmappable:
        status_counts[entry["status"]] = status_counts.get(entry["status"], 0) + 1
    return {
        "source": "verification_report.json",
        "checked": len(records),
        "mapped": len(mapping),
        "unmappable": len(unmappable),
        "unmappable_status": status_counts,
        "targets": targets,
        "mappings": mapping,
        "unmappable_records": unmappable,
    }


# --------------------------------------------------------------------------- #
# Fallback + persistence
# --------------------------------------------------------------------------- #
def _repo_path(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def load_bundled() -> List[Dict[str, Any]]:
    path = _repo_path("catalog.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            fh.seek(0)
            return json.load(fh, strict=False)


def scrape_or_fallback(
    fetch: Optional[Callable[[str], Tuple[int, str, str]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return ``(records, report)`` preferring live verified records."""
    live = scrape_catalog(fetch=fetch)
    if len(live) >= MIN_VERIFIED:
        report = {
            "source": "shl.com",
            "verified": len(live),
            "unverified": 0,
            "used_fallback": False,
        }
        return live, report

    bundled = load_bundled()
    report = {
        "source": "bundled-legacy",
        "verified": 0,
        "unverified": len(bundled),
        "used_fallback": True,
        "discovered": len(live),
    }
    return bundled, report


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest the live SHL catalog")
    parser.add_argument("--print", action="store_true", dest="do_print")
    parser.add_argument("--merge", action="store_true",
                        help="merge verified records into catalog.json (deduped)")
    parser.add_argument("--verify", action="store_true",
                        help="audit bundled URLs -> verification_report.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="limit records checked during --verify")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="quick URL sample: audit only the first N records")
    parser.add_argument("--map", action="store_true",
                        help="derive url_map.json from verification_report.json")
    args = parser.parse_args()

    if args.map:
        report_path = _repo_path("verification_report.json")
        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        records = report.get("records", [])
        mapping = build_url_map(records)
        _write_json(_repo_path("url_map.json"), mapping)
        summary = build_url_mapping_report(records)
        summary["generated_at"] = _now()
        _write_json(_repo_path("url_mapping_report.json"), summary)
        print(f"[scraper] mapped {summary['mapped']} legacy URLs to current "
              f"detail pages; {summary['unmappable']} unmappable "
              f"-> url_map.json, url_mapping_report.json")
        return

    if args.verify:
        records = load_bundled()
        limit = args.sample if args.sample is not None else args.limit
        report = verify_records(records, limit=limit,
                                progress_every=25 if not limit else None)
        report["generated_at"] = _now()
        # A sampled audit must not overwrite the full report.
        name = ("verification_sample.json" if limit
                else "verification_report.json")
        out = _repo_path(name)
        _write_json(out, report)
        print(f"[scraper] verified {report['verified']}/{report['checked']} "
              f"bundled URLs; unverified {report['unverified']} -> {out}")
        return

    records, report = scrape_or_fallback()
    report["generated_at"] = _now()

    if args.do_print:
        print(json.dumps(records[:5], indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2))
        return

    if report["used_fallback"]:
        print(f"[scraper] live crawl found {report.get('discovered', 0)} detail "
              "pages; preserving bundled catalog.json as fallback")
        return

    out = _repo_path("catalog.scraped.json")
    _write_json(out, records)
    print(f"[scraper] wrote {len(records)} verified assessments to {out}")

    if args.merge:
        bundled = load_bundled()
        merged = {r["name"].strip().lower(): r for r in bundled if r.get("name")}
        for record in records:
            merged[record["name"].strip().lower()] = record
        _write_json(_repo_path("catalog.json"), list(merged.values()))
        print(f"[scraper] merged into catalog.json ({len(merged)} records total)")


if __name__ == "__main__":
    main()
