"""Evaluation thresholds wired into the automated test suite."""
import unittest

from tests import REPO_ROOT  # noqa: F401

import evaluate


class EvaluationTests(unittest.TestCase):
    def test_mean_recall_at_10(self):
        mean = evaluate.evaluate_recall()
        self.assertGreaterEqual(mean, 0.8,
                                f"mean Recall@10 too low: {mean}")

    def test_behaviour_probes_all_pass(self):
        passed, total = evaluate.behaviour_probes()
        self.assertEqual(passed, total,
                         f"behaviour probes failed: {passed}/{total}")


if __name__ == "__main__":
    unittest.main()
