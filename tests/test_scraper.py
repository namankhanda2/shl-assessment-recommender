"""Tests for the SHL catalog scraper: extraction, redirects and fallback."""
import json
import os
import tempfile
import unittest

from tests import REPO_ROOT  # noqa: F401  (adds repo root to sys.path)

import catalog as catalog_module
import scraper
from scraper import (
    LISTING_PAGES,
    classify_redirect,
    extract_links,
    is_detail_url,
    parse_detail_page,
    scrape_catalog,
    scrape_or_fallback,
    infer_test_type,
    verify_records,
)

LISTING_HTML = """
<html><body>
  <a href="/products/assessments/">Assessments</a>
  <a href="/products/assessments/behavioral-assessments/">Behavioral</a>
  <a href="/products/assessments/behavioral-assessments/global-skills-assessment-gsa/">
     Global Skills</a>
  <a href="https://www.shl.com/products/assessments/skills-and-simulations/language-evaluation/svar/">
     SVAR</a>
  <a href="/solutions/talent-acquisition/graduate/">Graduate</a>
  <script type="application/ld+json">
    {"url": "https://www.shl.com/products/assessments/personality-assessment/shl-motivation-questionnaire-mq"}
  </script>
</body></html>
"""

DETAIL_HTML = """
<html><head>
  <title>Global Skills Assessment | Measure 96 Workplace Skills | SHL</title>
  <meta name="description"
        content="SHL's Global Skills Assessment measures 96 behavioral skills.">
</head><body>
  <h1>SHL Global Skills Assessment</h1>
  <p>Objective data for onboarding, upskilling and workforce planning.</p>
</body></html>
"""


def make_detail_fetch(url):
    """Return a plausible detail page for any detail URL."""
    name = url.rstrip("/").split("/")[-1].replace("-", " ").title()
    html = (
        "<html><head>"
        f"<title>{name} | SHL</title>"
        f"<meta name='description' content='About {name}.'>"
        f"</head><body><h1>{name}</h1></body></html>"
    )
    return 200, html, url


class UrlClassificationTests(unittest.TestCase):
    def test_is_detail_url_accepts_product_pages(self):
        self.assertTrue(is_detail_url(
            "https://www.shl.com/products/assessments/behavioral-assessments/"
            "global-skills-assessment-gsa/"))
        self.assertTrue(is_detail_url(
            "https://www.shl.com/products/assessments/skills-and-simulations/"
            "language-evaluation/svar/"))
        self.assertTrue(is_detail_url(
            "https://www.shl.com/products/assessments/personality-assessment/"
            "shl-occupational-personality-questionnaire-opq/"))

    def test_is_detail_url_rejects_categories_and_root(self):
        for url in [
            "https://www.shl.com/products/assessments/",
            "https://www.shl.com/products/assessments/behavioral-assessments/",
            "https://www.shl.com/products/assessments/skills-and-simulations/technical-skills/",
            "https://www.shl.com/solutions/talent-acquisition/graduate/",
        ]:
            self.assertFalse(is_detail_url(url), url)

    def test_test_type_inference(self):
        self.assertEqual(infer_test_type(
            "https://www.shl.com/products/assessments/behavioral-assessments/"
            "global-skills-assessment-gsa/"), "C")
        self.assertEqual(infer_test_type(
            "https://www.shl.com/products/assessments/behavioral-assessments/"
            "universal-competency-framework/"), "C")
        self.assertEqual(infer_test_type(
            "https://www.shl.com/products/assessments/behavioral-assessments/"
            "situation-judgement-tests-sjt/"), "B")
        self.assertEqual(infer_test_type(
            "https://www.shl.com/products/assessments/personality-assessment/"
            "shl-occupational-personality-questionnaire-opq/"), "P")
        self.assertEqual(infer_test_type(
            "https://www.shl.com/products/assessments/skills-and-simulations/"
            "coding-simulations/some-coding-test/"), "S")
        self.assertEqual(infer_test_type(
            "https://www.shl.com/products/assessments/cognitive-assessments/"
            "verify-g/"), "A")


class ExtractionTests(unittest.TestCase):
    def test_extract_links_only_returns_assessment_urls(self):
        links = extract_links(LISTING_HTML)
        self.assertIn(
            "https://www.shl.com/products/assessments/behavioral-assessments/"
            "global-skills-assessment-gsa", links)
        self.assertIn(
            "https://www.shl.com/products/assessments/skills-and-simulations/"
            "language-evaluation/svar", links)
        self.assertIn(
            "https://www.shl.com/products/assessments/personality-assessment/"
            "shl-motivation-questionnaire-mq", links)
        self.assertFalse(any("graduate" in u for u in links))

    def test_parse_detail_page_extracts_fields(self):
        record = parse_detail_page(
            DETAIL_HTML,
            "https://www.shl.com/products/assessments/behavioral-assessments/"
            "global-skills-assessment-gsa/")
        self.assertEqual(record["name"], "SHL Global Skills Assessment")
        self.assertIn("96 behavioral skills", record["description"])
        self.assertEqual(record["test_type"], "C")
        self.assertTrue(record["verified"])
        self.assertEqual(record["source"], "shl.com")
        self.assertFalse(record["url"].endswith("/"))

    def test_parse_detail_page_falls_back_to_title(self):
        html = "<html><head><title>Verify G+ | SHL</title></head><body></body></html>"
        record = parse_detail_page(
            html, "https://www.shl.com/products/assessments/cognitive-assessments/verify-g/")
        self.assertEqual(record["name"], "Verify G+")

    def test_parse_detail_page_without_name_returns_none(self):
        self.assertIsNone(parse_detail_page(
            "<html><body><p>nothing here</p></body></html>",
            "https://www.shl.com/products/assessments/cognitive-assessments/x/"))


class RedirectTests(unittest.TestCase):
    def test_same_path_is_ok(self):
        url = "https://www.shl.com/products/assessments/cognitive-assessments/verify-g/"
        self.assertEqual(classify_redirect(url, 200, url), "ok")
        self.assertEqual(classify_redirect(url, 200, url.rstrip("/")), "ok")

    def test_legacy_redirect_is_flagged(self):
        requested = "https://www.shl.com/products/product-catalog/view/net-mvc-new/"
        final = "https://www.shl.com/products/assessments/skills-and-simulations/technical-skills/"
        self.assertEqual(classify_redirect(requested, 200, final), "redirected")

    def test_non_200_is_error(self):
        url = "https://www.shl.com/products/assessments/cognitive-assessments/verify-g/"
        self.assertEqual(classify_redirect(url, 404, url), "error")


class CrawlAndFallbackTests(unittest.TestCase):
    def _live_fetch(self, url):
        if url in LISTING_PAGES:
            return 200, LISTING_HTML, url
        if is_detail_url(url):
            return make_detail_fetch(url)
        return 404, "", url

    def _dead_fetch(self, url):
        return 404, "", url

    def test_scrape_catalog_offline(self):
        records = scrape_catalog(fetch=self._live_fetch)
        self.assertGreaterEqual(len(records), 3)
        for record in records:
            self.assertTrue(record["verified"])
            self.assertEqual(record["source"], "shl.com")
            self.assertTrue(is_detail_url(record["url"]))

    def test_scrape_or_fallback_prefers_live(self):
        records, report = scrape_or_fallback(fetch=self._live_fetch)
        self.assertFalse(report["used_fallback"])
        self.assertTrue(records)
        self.assertTrue(all(r.get("verified") for r in records))

    def test_scrape_or_fallback_uses_bundled_when_dead(self):
        records, report = scrape_or_fallback(fetch=self._dead_fetch)
        self.assertTrue(report["used_fallback"])
        self.assertGreater(len(records), 100)
        self.assertTrue(all(not r.get("verified", False) for r in records))

    def test_verify_records_splits_verified_and_unverified(self):
        good = "https://www.shl.com/products/assessments/cognitive-assessments/verify-g/"
        bad = "https://www.shl.com/products/product-catalog/view/java-8-new/"

        def fetch(url):
            if url == good:
                return 200, "", good
            return 200, "", "https://www.shl.com/products/"

        report = verify_records(
            [{"name": "Good", "url": good}, {"name": "Bad", "url": bad}],
            fetch=fetch,
        )
        self.assertEqual(report["checked"], 2)
        self.assertEqual(report["verified"], 1)
        self.assertEqual(report["unverified"], 1)


class VerifiedCatalogTests(unittest.TestCase):
    def test_load_verified_catalog_reads_scraped_file(self):
        payload = [{
            "name": "Example Test",
            "url": "https://www.shl.com/products/assessments/cognitive-assessments/example-test",
            "test_type": "A",
            "description": "Example.",
        }]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(payload, fh)
            path = fh.name
        os.environ["SHL_SCRAPED_CATALOG"] = path
        try:
            records = catalog_module.load_verified_catalog(force=True)
        finally:
            os.environ.pop("SHL_SCRAPED_CATALOG", None)
            os.unlink(path)
            catalog_module.load_verified_catalog(force=True)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["verified"])
        self.assertEqual(records[0]["source"], "shl.com")


if __name__ == "__main__":
    unittest.main()
