"""Regression tests for role-specific ranking, rationales and URL transparency."""
import re
import unittest
from unittest import mock

from tests import REPO_ROOT  # noqa: F401  (adds repo root to sys.path)
from tests import user

from fastapi.testclient import TestClient

import agent
from agent import _format_duration, respond
from app import app
from catalog import (
    TEST_TYPE_LABELS,
    UNVERIFIED_STATUSES,
    VERIFIED_STATUSES,
    find_record,
    load_catalog,
)
import retrieval

client = TestClient(app)


class _FakeClient:
    enabled = True

    def __init__(self, data):
        self.data = data

    def complete_json(self, *args, **kwargs):
        return self.data


class RoleRankingTests(unittest.TestCase):
    def _names(self, query):
        return [r["name"].lower() for r in respond([user(query)])["recommendations"]]

    def _types(self, query):
        return [r["test_type"] for r in respond([user(query)])["recommendations"]]

    def test_nursing_role_is_matched(self):
        # Regression: "nurses" was not a recognised skill and returned nothing.
        names = self._names("Hiring nurses for clinical patient care")
        self.assertTrue(names)
        self.assertTrue(any("nursing" in n for n in names))

    def test_frontend_role_prefers_framework_over_unrelated(self):
        recs = respond([user("Frontend developer with React and TypeScript")])
        names = [r["name"].lower() for r in recs["recommendations"]]
        self.assertTrue(names)
        self.assertIn("react", names[0])
        self.assertTrue(any("angular" in n for n in names[:3]))
        for bad in ("sql server", "asp.net", "informatica"):
            self.assertNotIn(bad, " | ".join(names[:3]))

    def test_customer_service_role_includes_simulation_or_situational(self):
        types = self._types("Customer service representative for a contact centre")
        names = self._names("Customer service representative for a contact centre")
        self.assertTrue(
            any(t in {"S", "B"} for t in types)
            or any("simulation" in n for n in names)
        )

    def test_graduate_analyst_includes_cognitive_ability(self):
        self.assertIn("A", self._types("Graduate analyst needing numerical and verbal reasoning"))

    def test_leadership_role_includes_personality(self):
        self.assertIn("P", self._types("Senior leadership role needing personality insight"))

    def test_specific_skill_outranks_generic_role_word(self):
        order = [it["name"] for it, _ in retrieval.Retriever().rank("React frontend developer")]
        self.assertLess(order.index("ReactJS (New)"),
                        order.index("Informatica (Developer) (New)"))


class RationaleTests(unittest.TestCase):
    def _recs(self, query):
        recs = respond([user(query)])["recommendations"]
        self.assertTrue(recs)
        return query, recs

    def test_rationale_names_the_test_type_label(self):
        for query in ("Hiring nurses for clinical patient care",
                      "Python developer with SQL and AWS"):
            _, recs = self._recs(query)
            for rec in recs:
                label = TEST_TYPE_LABELS.get(rec["test_type"], rec["test_type"])
                self.assertIn(label, rec["rationale"])

    def test_cited_skills_are_grounded_in_the_record(self):
        for query in ("Hiring nurses for clinical patient care",
                      "Customer service representative for a contact centre",
                      "Frontend developer with React and TypeScript"):
            _, recs = self._recs(query)
            for rec in recs:
                record = find_record(rec["name"], rec["url"])
                text = (f"{record['name']} {record.get('description', '')} "
                        f"{' '.join(record.get('keys') or [])}").lower()
                match = re.search(r"relevant to (.*?)(?:;|\.|$)", rec["rationale"])
                if not match:
                    continue
                for skill in re.split(r", | and ", match.group(1)):
                    skill = skill.strip()
                    if skill:
                        self.assertIn(skill, text,
                                      f"{skill!r} cited but not in record for {query}")

    def test_duration_formatting(self):
        self.assertEqual(_format_duration("11 minutes"), "takes about 11 minutes")
        self.assertEqual(_format_duration("Untimed"), "untimed")
        self.assertEqual(_format_duration(""), "")


class UrlTransparencyTests(unittest.TestCase):
    def _recs(self, query):
        resp = client.post("/chat", json={"messages": [user(query)]})
        self.assertEqual(resp.status_code, 200)
        return resp.json()["recommendations"]

    def test_verified_and_flagged_are_distinguished(self):
        # OPQ family resolves to a verified current detail page.
        opq = self._recs("Senior leadership role needing OPQ personality insight")
        verified = [r for r in opq if r["url_verified"]]
        self.assertTrue(verified, "expected at least one verified recommendation")
        for rec in verified:
            self.assertIn(rec["url_status"], VERIFIED_STATUSES)

        # Nursing keeps its archived legacy URL and must be flagged.
        nursing = self._recs("Hiring nurses for clinical patient care")
        flagged = [r for r in nursing if not r["url_verified"]]
        self.assertTrue(flagged)
        for rec in flagged:
            self.assertIn(rec["url_status"], UNVERIFIED_STATUSES)

    def test_url_status_is_never_contradictory(self):
        for query in ("Hiring nurses for clinical patient care",
                      "Mid-level Python developer with AWS and SQL",
                      "Customer service representative for a contact centre"):
            for rec in self._recs(query):
                if rec["url_verified"]:
                    self.assertIn(rec["url_status"], VERIFIED_STATUSES)
                else:
                    self.assertIn(rec["url_status"], UNVERIFIED_STATUSES)

    def test_every_status_maps_to_a_catalog_record(self):
        catalog = load_catalog()
        statuses = {(c["name"], c["url"]): c["url_status"] for c in catalog}
        for rec in self._recs("Java backend developer with SQL and Spring"):
            self.assertEqual(rec["url_status"], statuses[(rec["name"], rec["url"])])


class SeniorPythonBackendRegressionTests(unittest.TestCase):
    QUERY = "Senior Python Backend Developer"

    def _recs(self, query=None):
        return respond([user(query or self.QUERY)])["recommendations"]

    def test_python_and_technical_items_rank_first(self):
        recs = self._recs()
        names = [r["name"].lower() for r in recs]
        self.assertEqual(len(recs), 10)                  # count preserved
        self.assertEqual(names[0], "python (new)")       # explicit skill wins
        # No unrelated reasoning / personality / report filler.
        for rec in recs:
            self.assertIn(rec["test_type"], {"K", "S"},
                          f"unexpected non-technical type for {rec['name']}")

    def test_sql_and_aws_surface_when_requested(self):
        recs = self._recs("Senior Python Backend Developer with SQL and AWS")
        names = [r["name"].lower() for r in recs]
        self.assertIn("python (new)", names)
        self.assertTrue(any("aws" in n for n in names))
        self.assertTrue(any("sql" in n for n in names))
        self.assertEqual(len(recs), 10)

    def test_explicit_skill_outranks_dense_similarity(self):
        catalog = load_catalog()
        retriever = retrieval.Retriever(catalog=catalog)
        names = [it["name"] for it in catalog]
        python_i = names.index("Python (New)")
        report = "Sales Transformation 1.0 - Individual Contributor"
        report_i = names.index(report)

        class AdversarialDense:
            def similarities(self, query):
                sims = [0.0] * len(catalog)
                sims[report_i] = 1.0            # dense loves the wrong item
                return sims

        retriever._dense = AdversarialDense()
        order = [it["name"] for it, _ in retriever.rank(self.QUERY)]
        self.assertLess(order.index("Python (New)"), order.index(report))

    def test_api_preserves_count_and_url_rules(self):
        resp = client.post("/chat", json={"messages": [user(self.QUERY)]})
        self.assertEqual(resp.status_code, 200)
        recs = resp.json()["recommendations"]
        self.assertEqual(len(recs), 10)
        self.assertEqual(recs[0]["name"], "Python (New)")
        for rec in recs:
            self.assertIsInstance(rec["url_verified"], bool)
            if rec["url_verified"]:
                self.assertIn(rec["url_status"], VERIFIED_STATUSES)
            else:
                self.assertIn(rec["url_status"], UNVERIFIED_STATUSES)


class ExactCountTests(unittest.TestCase):
    QUERY = "Senior Python Backend Developer with REST API, SQL and AWS"

    def test_agent_returns_exact_requested_count(self):
        for k in (1, 3, 5, 8, 10):
            recs = respond([user(self.QUERY)], k=k)["recommendations"]
            self.assertEqual(len(recs), k, f"expected {k} recommendations")

    def test_count_is_clamped_to_valid_range(self):
        self.assertEqual(len(respond([user(self.QUERY)], k=0)["recommendations"]), 1)
        self.assertEqual(len(respond([user(self.QUERY)], k=50)["recommendations"]), 10)

    def test_count_is_exact_after_llm_refinement(self):
        recs = respond([user(self.QUERY)], k=5)["recommendations"]
        # Model picks a single favourite and returns it first; the shortlist must
        # still contain exactly the requested five, all grounded.
        fake = _FakeClient({"reply": "ok", "selected": [recs[2]["name"]]})
        with mock.patch.object(agent, "get_client", lambda: fake):
            _reply, picked = agent._llm_refine(
                [user(self.QUERY)], "base", recs, "recommend", 5)
        self.assertEqual(len(picked), 5)
        self.assertEqual(picked[0]["name"], recs[2]["name"])

    def test_api_honours_k_alias(self):
        for payload, expected in (
            ({"messages": [user(self.QUERY)], "k": 5}, 5),
            ({"messages": [user(self.QUERY)], "count": 4}, 4),
            ({"messages": [user(self.QUERY)], "limit": 7}, 7),
            ({"messages": [user(self.QUERY)]}, 10),
        ):
            resp = client.post("/chat", json=payload)
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(len(resp.json()["recommendations"]), expected, payload)

    def test_api_count_is_exact_after_llm_refinement(self):
        recs = client.post("/chat", json={"messages": [user(self.QUERY)], "k": 5})
        chosen = recs.json()["recommendations"][3]["name"]
        fake = _FakeClient({"reply": "ok", "selected": [chosen]})
        with mock.patch.object(agent, "get_client", lambda: fake):
            resp = client.post("/chat", json={"messages": [user(self.QUERY)], "k": 5})
        body = resp.json()["recommendations"]
        self.assertEqual(len(body), 5)
        self.assertEqual(body[0]["name"], chosen)


class TechnicalRelevanceTests(unittest.TestCase):
    QUERY = "Senior Python Backend Developer with REST API, SQL and AWS"

    def _recs(self):
        return respond([user(self.QUERY)])["recommendations"]

    def test_core_skills_and_rest_are_all_present(self):
        names = " | ".join(r["name"].lower() for r in self._recs())
        self.assertIn("python", names)
        self.assertIn("aws", names)
        self.assertIn("sql", names)
        self.assertIn("rest", names)          # RESTful Web Services (New)

    def test_no_reasoning_or_personality_filler(self):
        for rec in self._recs():
            self.assertIn(rec["test_type"], {"K", "S"},
                          f"unrelated non-technical item: {rec['name']}")

    def test_rest_is_in_the_top_tier(self):
        order = [it["name"] for it, _ in retrieval.Retriever().rank(self.QUERY)]
        self.assertLess(order.index("RESTful Web Services (New)"), 5)


class GroundingAndDedupTests(unittest.TestCase):
    QUERIES = [
        "Senior Python Backend Developer with REST API, SQL and AWS",
        "Hiring nurses for clinical patient care",
        "Customer service representative for a contact centre",
    ]

    def test_no_duplicate_urls(self):
        for q in self.QUERIES:
            recs = respond([user(q)], k=10)["recommendations"]
            urls = [r["url"] for r in recs]
            self.assertEqual(len(urls), len(set(urls)), q)

    def test_every_recommendation_is_catalog_grounded(self):
        valid = {(it["name"], it["url"]): it for it in load_catalog()}
        by_url = {it["url"]: it for it in load_catalog()}
        for q in self.QUERIES:
            for rec in respond([user(q)], k=10)["recommendations"]:
                self.assertIn((rec["name"], rec["url"]), valid, q)
                self.assertEqual(by_url[rec["url"]]["name"], rec["name"])

    def test_api_url_badges_are_consistent(self):
        for q in self.QUERIES:
            resp = client.post("/chat", json={"messages": [user(q)], "k": 6})
            recs = resp.json()["recommendations"]
            # Exact count is bounded by what the catalog actually contains
            # (never padded with invented items).
            self.assertTrue(1 <= len(recs) <= 6)
            for rec in recs:
                self.assertIsInstance(rec["url_verified"], bool)
                if rec["url_verified"]:
                    self.assertIn(rec["url_status"], VERIFIED_STATUSES)
                else:
                    self.assertIn(rec["url_status"], UNVERIFIED_STATUSES)


if __name__ == "__main__":
    unittest.main()
