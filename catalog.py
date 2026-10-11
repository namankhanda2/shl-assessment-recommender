"""Catalog loading and normalization for the SHL recommender.

The catalog is a list of Individual Test Solutions with the fields:
    name, url, test_type, test_types, description, job_levels,
    duration, remote, adaptive, keys
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Set

CATALOG_FILENAME = "catalog.json"
SCRAPED_FILENAME = "catalog.scraped.json"
URL_MAP_FILENAME = "url_map.json"
VERIFICATION_FILENAME = "verification_report.json"

# url_status values for a catalog record. Only ``current`` and
# ``verified_mapped`` are considered verified; everything else is an honestly
# flagged limitation.
STATUS_CURRENT = "current"                  # live at its own official detail URL
STATUS_MAPPED = "verified_mapped"           # legacy URL remapped to a live detail page
STATUS_CATEGORY = "category_redirect"       # redirects to an assessment category page
STATUS_GENERIC = "generic_redirect"         # redirects to a generic page (e.g. /products/)
STATUS_ERROR = "error"                      # could not be reached during the audit
STATUS_UNVERIFIED = "legacy_unverified"     # unresolved; original URL retained

VERIFIED_STATUSES = frozenset({STATUS_CURRENT, STATUS_MAPPED})
UNVERIFIED_STATUSES = frozenset(
    {STATUS_CATEGORY, STATUS_GENERIC, STATUS_ERROR, STATUS_UNVERIFIED}
)

# Human readable labels for the single-letter test-type codes used by SHL.
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

_CATALOG: List[Dict[str, Any]] | None = None
_VERIFIED: List[Dict[str, Any]] | None = None
_URL_MAP: Dict[str, str] | None = None
_URL_STATUS: Dict[str, str] | None = None
_OFFICIAL_URLS: Set[str] | None = None


def normalize_url(url: str) -> str:
    """Canonical form for URL comparison (drop trailing slash, lowercase)."""
    return (url or "").strip().rstrip("/").lower()


def url_map_path() -> str:
    """Absolute path to the legacy->current URL map, overridable via SHL_URL_MAP."""
    env = os.getenv("SHL_URL_MAP")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, URL_MAP_FILENAME)


def load_url_map(force: bool = False) -> Dict[str, str]:
    """Legacy URL -> verified current detail URL (from ``url_map.json``)."""
    global _URL_MAP
    if _URL_MAP is not None and not force:
        return _URL_MAP
    path = url_map_path()
    if not os.path.exists(path):
        _URL_MAP = {}
        return _URL_MAP
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    mapping: Dict[str, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key and value:
                mapping[normalize_url(str(key))] = str(value).strip()
    _URL_MAP = mapping
    return mapping


def verification_path() -> str:
    """Path to the URL audit report, overridable via SHL_VERIFICATION_REPORT."""
    env = os.getenv("SHL_VERIFICATION_REPORT")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, VERIFICATION_FILENAME)


def classify_verification_record(record: Dict[str, Any]) -> str:
    """Map one audit record to a catalog ``url_status`` value."""
    if record.get("final_is_detail"):
        return STATUS_MAPPED
    verdict = record.get("verdict")
    if verdict == "error":
        return STATUS_ERROR
    if verdict == "ok":
        return STATUS_CURRENT if is_official_url(record.get("url", "")) else STATUS_UNVERIFIED
    if record.get("final_is_assessment"):
        return STATUS_CATEGORY
    return STATUS_GENERIC


def load_url_status(force: bool = False) -> Dict[str, str]:
    """Legacy URL -> audited status, derived from ``verification_report.json``.

    Empty when the report is unavailable; callers then fall back to
    ``legacy_unverified`` rather than guessing.
    """
    global _URL_STATUS
    if _URL_STATUS is not None and not force:
        return _URL_STATUS
    path = verification_path()
    if not os.path.exists(path):
        _URL_STATUS = {}
        return _URL_STATUS
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    records = raw.get("records", []) if isinstance(raw, dict) else raw
    statuses: Dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        url = record.get("url")
        if url:
            statuses[normalize_url(url)] = classify_verification_record(record)
    _URL_STATUS = statuses
    return statuses


def catalog_path() -> str:
    """Absolute path to the bundled catalog, overridable via SHL_CATALOG."""
    env = os.getenv("SHL_CATALOG")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, CATALOG_FILENAME)


def _normalize(item: Dict[str, Any]) -> Dict[str, Any]:
    name = str(item.get("name", "")).strip()
    url = str(item.get("url") or item.get("link") or "").strip()
    test_type = str(item.get("test_type") or "").strip().upper()[:1]
    test_types = item.get("test_types") or ([test_type] if test_type else [])
    test_types = [str(t).strip().upper()[:1] for t in test_types if str(t).strip()]
    if not test_type and test_types:
        test_type = test_types[0]
    keys = item.get("keys") or [
        TEST_TYPE_LABELS.get(t, t) for t in test_types
    ]
    verified = bool(item.get("verified", False))
    return {
        "name": name,
        "url": url,
        "test_type": test_type or "K",
        "test_types": test_types or [test_type or "K"],
        "description": " ".join(str(item.get("description", "")).split()),
        "job_levels": list(item.get("job_levels") or []),
        "duration": str(item.get("duration") or ""),
        "remote": bool(item.get("remote", False)),
        "adaptive": bool(item.get("adaptive", False)),
        "keys": list(keys),
        "verified": verified,
        "source": str(item.get("source") or ("shl.com" if verified else "bundled-legacy")),
        "legacy_url": url,
        "url_status": str(item.get("url_status") or STATUS_UNVERIFIED),
        "url_verified": bool(item.get("url_verified", False)),
    }


def official_detail_urls(force: bool = False) -> Set[str]:
    """Normalised URLs of the verified current SHL detail pages."""
    global _OFFICIAL_URLS
    if _OFFICIAL_URLS is not None and not force:
        return _OFFICIAL_URLS
    urls = {normalize_url(u) for u in load_url_map().values()}
    for record in load_verified_catalog():
        urls.add(normalize_url(record["url"]))
    _OFFICIAL_URLS = {u for u in urls if u}
    return _OFFICIAL_URLS


def _apply_url_map(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rewrite legacy URLs to their verified current detail pages in place.

    Only URLs backed by the audit are marked verified. Everything else keeps
    its original URL and is flagged with an honest, specific status
    (``category_redirect``, ``generic_redirect``, ``error`` or
    ``legacy_unverified``) so callers never present a legacy link as verified.
    """
    url_map = load_url_map()
    official = official_detail_urls()
    status_map = load_url_status()
    for record in records:
        legacy = record["url"]
        record["legacy_url"] = legacy
        key = normalize_url(legacy)
        mapped = url_map.get(key)
        if mapped:
            record["url"] = mapped
            status = STATUS_MAPPED
        elif key in official:
            status = STATUS_CURRENT
        else:
            status = status_map.get(key, STATUS_UNVERIFIED)
        record["url_status"] = status
        record["url_verified"] = status in VERIFIED_STATUSES
    return records


def load_catalog(force: bool = False) -> List[Dict[str, Any]]:
    """Load and cache the catalog, tolerating minor formatting issues."""
    global _CATALOG
    if _CATALOG is not None and not force:
        return _CATALOG

    path = catalog_path()
    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError:
            fh.seek(0)
            raw = json.load(fh, strict=False)

    if isinstance(raw, dict):
        raw = raw.get("products") or raw.get("assessments") or raw.get("catalog") or []

    catalog = [_normalize(item) for item in raw if isinstance(item, dict)]
    catalog = [c for c in catalog if c["name"] and c["url"]]
    _apply_url_map(catalog)
    _CATALOG = catalog
    return catalog


def load_verified_catalog(force: bool = False) -> List[Dict[str, Any]]:
    """Verified records scraped from the live SHL site (empty if unavailable).

    These come from ``catalog.scraped.json`` and are produced by ``scraper.py``.
    They are *not* merged into the main catalog automatically so the bundled
    377-record fallback stays the source of truth for recommendations; callers
    that want the fresher data can opt in explicitly.
    """
    global _VERIFIED
    if _VERIFIED is not None and not force:
        return _VERIFIED

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.getenv("SHL_SCRAPED_CATALOG") or os.path.join(here, SCRAPED_FILENAME)
    if not os.path.exists(path):
        _VERIFIED = []
        return _VERIFIED

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        raw = raw.get("products") or raw.get("assessments") or raw.get("records") or []

    records = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        item = {"verified": True, "source": "shl.com", **item}
        record = _normalize(item)
        if record["name"] and record["url"]:
            record["url_status"] = STATUS_CURRENT
            record["url_verified"] = True
            records.append(record)
    _VERIFIED = records
    return records


def catalog_by_url() -> Dict[str, Dict[str, Any]]:
    return {item["url"]: item for item in load_catalog()}


def is_official_url(url: str) -> bool:
    """True when ``url`` is a verified current SHL detail page."""
    return normalize_url(url) in official_detail_urls()


def url_status_counts() -> Dict[str, int]:
    """Count catalog records by ``url_status``."""
    counts: Dict[str, int] = {}
    for item in load_catalog():
        status = item["url_status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def validate_recommendations(
    recommendations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return a list of problems for recommendations not grounded in the catalog.

    Every recommended ``(name, url)`` pair must exist in the catalog; an empty
    list means all recommendations are valid and none were invented.
    """
    valid = {(item["name"], item["url"]) for item in load_catalog()}
    issues: List[Dict[str, Any]] = []
    for rec in recommendations:
        name, url = rec.get("name"), rec.get("url")
        if (name, url) not in valid:
            issues.append({"name": name, "url": url, "reason": "not in catalog"})
    return issues


def find_record(name: str, url: str) -> Dict[str, Any] | None:
    """Return the catalog record matching an exact ``(name, url)`` pair."""
    target = (str(name), str(url))
    for item in load_catalog():
        if (item["name"], item["url"]) == target:
            return item
    return None


def status_summary() -> Dict[str, Any]:
    """High-level catalog health: counts + verified/unverified totals."""
    catalog = load_catalog()
    return {
        "count": len(catalog),
        "url_status": url_status_counts(),
        "verified": sum(1 for c in catalog if c["url_verified"]),
        "flagged": sum(1 for c in catalog if not c["url_verified"]),
    }


def export_enriched(path: str = "catalog.enriched.json") -> str:
    """Persist the normalised + URL-resolved catalog for offline inspection.

    This is a derived snapshot of ``load_catalog()``: the same data the API and
    retrieval consume, including ``url_status`` / ``url_verified`` / ``legacy_url``.
    It does not replace the raw ``catalog.json`` source of truth.
    """
    keys = (
        "name", "url", "legacy_url", "url_status", "url_verified",
        "test_type", "test_types", "description", "job_levels", "duration",
        "remote", "adaptive", "keys", "source",
    )
    import os as _os

    out = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), path)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump([{k: c.get(k) for k in keys} for c in load_catalog()],
                  fh, ensure_ascii=False, indent=1)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Catalog health / export tool")
    parser.add_argument("--status", action="store_true",
                        help="print catalog + url_status summary")
    parser.add_argument("--export", metavar="PATH", nargs="?", const="catalog.enriched.json",
                        help="write the enriched catalog snapshot")
    args = parser.parse_args()

    if args.status:
        summary = status_summary()
        print(f"records: {summary['count']}  verified: {summary['verified']}  "
              f"flagged: {summary['flagged']}")
        for status, count in sorted(summary["url_status"].items()):
            print(f"  {status:20s} {count:4d}")
    if args.export:
        path = export_enriched(args.export)
        print(f"enriched catalog written to {path}")


if __name__ == "__main__":  # pragma: no cover - manual CLI
    main()
