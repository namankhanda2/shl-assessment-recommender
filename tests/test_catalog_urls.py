"""Regression tests for legacy->current SHL URL mapping and validation."""
import json
import os
import unittest

from tests import REPO_ROOT  # noqa: F401  (adds repo root to sys.path)
from tests import user

from agent import respond
from catalog import (
    STATUS_CURRENT,
    STATUS_MAPPED,
    STATUS_UNVERIFIED,
    UNVERIFIED_STATUSES,
    VERIFIED_STATUSES,
    is_official_url,
    load_catalog,
    load_url_map,
    normalize_url,
    official_detail_urls,
    url_status_counts,
    validate_recommendations,
)
import scraper

REPORT_PATH = os.path.join(REPO_ROOT, "verification_report.json")


def _report_records():
    with open(REPORT_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)["records"]


class UrlMapTests(unittest.TestCase):
    def test_targets_are_verified_detail_pages(self):
        official = official_detail_urls()
        for legacy, target in load_url_map().items():
            self.assertIn(normalize_url(target), official,
                          f"mapped target is not a current detail page: {target}")

    def test_mapping_matches_audited_redirects(self):
        report = {normalize_url(r["url"]): r for r in _report_records()}
        for legacy, target in load_url_map().items():
            self.assertIn(legacy, report, f"mapped key not in audit: {legacy}")
            record = report[legacy]
            self.assertTrue(record["final_is_detail"],
                            f"{legacy} did not redirect to a detail page")
            self.assertEqual(normalize_url(record["final_url"]), normalize_url(target),
                             f"mapping for {legacy} disagrees with the audit")

    def test_every_mapping_key_exists_in_catalog(self):
        legacy_urls = {normalize_url(c["legacy_url"]) for c in load_catalog()}
        for legacy in load_url_map():
            self.assertIn(legacy, legacy_urls,
                          f"mapped URL is not present in the catalog: {legacy}")


class CatalogStatusTests(unittest.TestCase):
    def test_every_record_has_a_known_status(self):
        catalog = load_catalog()
        statuses = {c["url_status"] for c in catalog}
        self.assertLessEqual(statuses, VERIFIED_STATUSES | UNVERIFIED_STATUSES)
        self.assertEqual(sum(url_status_counts().values()), len(catalog))
        # Each record is unambiguously verified or flagged.
        for c in catalog:
            self.assertEqual(c["url_verified"], c["url_status"] in VERIFIED_STATUSES)

    def test_mapped_records_point_to_official_pages(self):
        mapped = [c for c in load_catalog() if c["url_status"] == STATUS_MAPPED]
        self.assertTrue(mapped, "expected some legacy URLs to be remapped")
        for record in mapped:
            self.assertTrue(is_official_url(record["url"]))
            self.assertTrue(record["url_verified"])
            self.assertNotEqual(normalize_url(record["legacy_url"]),
                                normalize_url(record["url"]))

    def test_unmappable_records_are_flagged(self):
        flagged = [c for c in load_catalog()
                   if c["url_status"] in UNVERIFIED_STATUSES]
        self.assertTrue(flagged)
        for record in flagged:
            self.assertFalse(record["url_verified"])
            self.assertFalse(is_official_url(record["url"]))
            self.assertIn("product-catalog/view/", record["url"])

    def test_known_legacy_record_is_flagged(self):
        by_name = {c["name"]: c for c in load_catalog()}
        rec = by_name["Core Java (Advanced Level) (New)"]
        self.assertIn(rec["url_status"], UNVERIFIED_STATUSES)
        self.assertFalse(rec["url_verified"])
        self.assertIn("product-catalog/view/", rec["url"])

    def test_unverified_records_are_classified_by_redirect(self):
        counts = url_status_counts()
        self.assertGreater(counts.get("category_redirect", 0), 0)
        self.assertGreater(counts.get("generic_redirect", 0), 0)

    def test_known_mapable_record_is_remapped(self):
        by_name = {c["name"]: c for c in load_catalog()}
        rec = by_name["Motivation Questionnaire MQM5"]
        self.assertEqual(rec["url_status"], STATUS_MAPPED)
        self.assertTrue(is_official_url(rec["url"]))

    def test_opq_family_shares_one_official_page(self):
        opq = [c for c in load_catalog()
               if c["url_status"] == STATUS_MAPPED and c["name"].startswith("OPQ")]
        self.assertGreater(len(opq), 1)
        targets = {normalize_url(c["url"]) for c in opq}
        self.assertEqual(len(targets), 1)
        self.assertTrue(all("occupational-personality" in t for t in targets))


class RecommendationUrlTests(unittest.TestCase):
    QUERIES = [
        "Senior leadership role needing OPQ and personality insight",
        "Hiring a French speaking customer service representative",
        "Mid-level Python developer with AWS and SQL",
        "Graduate analyst needing numerical and verbal reasoning",
        "Java backend developer with SQL and Spring",
    ]

    def test_recommended_urls_are_grounded_and_unique(self):
        for query in self.QUERIES:
            recs = respond([user(query)])["recommendations"]
            self.assertEqual(validate_recommendations(recs), [],
                             f"ungrounded recommendation for: {query}")
            urls = [r["url"] for r in recs]
            self.assertEqual(len(urls), len(set(urls)),
                             f"duplicate URLs for: {query}")

    def test_every_recommended_url_is_official_or_explicitly_flagged(self):
        by_pair = {(c["name"], c["url"]): c for c in load_catalog()}
        for query in self.QUERIES:
            for rec in respond([user(query)])["recommendations"]:
                record = by_pair[(rec["name"], rec["url"])]
                self.assertTrue(
                    record["url_verified"]
                    or record["url_status"] in UNVERIFIED_STATUSES,
                    f"unclassified recommended URL: {rec['url']}",
                )

    def test_validate_recommendations_catches_invented_url(self):
        bad = [{"name": "Made Up Test", "url": "https://example.com/nope"}]
        issues = validate_recommendations(bad)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["reason"], "not in catalog")


class BuildUrlMapTests(unittest.TestCase):
    def test_only_detail_redirects_are_mapped(self):
        records = [
            {"name": "a", "url": "https://www.shl.com/legacy/a/",
             "final_url": "https://www.shl.com/products/assessments/x/y", "final_is_detail": True},
            {"name": "b", "url": "https://www.shl.com/legacy/b/",
             "final_url": "https://www.shl.com/products/", "final_is_detail": False},
            {"name": "c", "url": "https://www.shl.com/legacy/c/",
             "final_url": "https://www.shl.com/legacy/c/", "final_is_detail": False},
        ]
        mapping = scraper.build_url_map(records)
        self.assertEqual(list(mapping), ["https://www.shl.com/legacy/a"])
        self.assertEqual(mapping["https://www.shl.com/legacy/a"],
                         "https://www.shl.com/products/assessments/x/y")

    def test_mapping_report_counts(self):
        report = scraper.build_url_mapping_report(_report_records())
        self.assertEqual(report["mapped"], len(load_url_map()))
        self.assertEqual(report["mapped"] + report["unmappable"], report["checked"])
        for target in report["mappings"].values():
            self.assertIn(normalize_url(target), official_detail_urls())


if __name__ == "__main__":
    unittest.main()
