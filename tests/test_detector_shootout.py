"""Focused regressions for the bounded detector comparison path."""
import unittest

from analytics_lab.detector_shootout import (
    DETECTOR_CANDIDATES,
    choose_material_candidate,
    count_atss_people,
    count_ssd_people,
)


class DetectorShootoutTests(unittest.TestCase):
    def test_candidate_set_is_bounded_and_pinned(self):
        self.assertEqual(len(DETECTOR_CANDIDATES), 3)
        self.assertEqual(len({item.name for item in DETECTOR_CANDIDATES}), 3)
        for candidate in DETECTOR_CANDIDATES:
            self.assertEqual(candidate.xml.license_id, "Apache-2.0")
            self.assertEqual(candidate.weights.license_id, "Apache-2.0")
            self.assertTrue(candidate.xml.sha384)
            self.assertTrue(candidate.weights.sha384)
            self.assertLess(candidate.xml.size_bytes, 8 * 1024 * 1024)
            self.assertLess(candidate.weights.size_bytes, 8 * 1024 * 1024)

    def test_ssd_person_label_is_candidate_specific(self):
        rows = [
            [0, 0, 0.90, 0.1, 0.1, 0.5, 0.5],
            [0, 1, 0.80, 0.2, 0.2, 0.6, 0.6],
            [0, 0, 0.49, 0.3, 0.3, 0.7, 0.7],
            [-1, 0, 1.0, 0, 0, 1, 1],
        ]
        self.assertEqual(count_ssd_people(rows, person_label=0), 1)
        self.assertEqual(count_ssd_people(rows, person_label=1), 1)

    def test_atss_counts_only_confident_person_rows(self):
        boxes = [
            [0, 0, 10, 10, 0.75],
            [0, 0, 10, 10, 0.49],
            [0, 0, 10, 10, 0.99],
        ]
        labels = [1, 1, 0]
        self.assertEqual(count_atss_people(boxes, labels), 1)

    def test_material_replacement_requires_recall_gain_and_negative_coverage(self):
        results = {
            "baseline": {
                "positive_during_coverage": 0.22,
                "negative_coverage": 1.0,
                "artifact_bytes": 2_000_000,
                "detector_fps": 50.0,
            },
            "small-good": {
                "positive_during_coverage": 0.70,
                "negative_coverage": 0.95,
                "artifact_bytes": 4_000_000,
                "detector_fps": 35.0,
            },
            "fast-but-weak": {
                "positive_during_coverage": 0.40,
                "negative_coverage": 1.0,
                "artifact_bytes": 3_000_000,
                "detector_fps": 80.0,
            },
            "large-good": {
                "positive_during_coverage": 0.90,
                "negative_coverage": 1.0,
                "artifact_bytes": 6_000_000,
                "detector_fps": 25.0,
            },
        }
        self.assertEqual(choose_material_candidate(results, baseline="baseline"), "small-good")

    def test_no_recommendation_when_none_clears_measured_bar(self):
        results = {
            "baseline": {
                "positive_during_coverage": 0.22,
                "negative_coverage": 1.0,
                "artifact_bytes": 2_000_000,
                "detector_fps": 50.0,
            },
            "candidate": {
                "positive_during_coverage": 0.60,
                "negative_coverage": 0.50,
                "artifact_bytes": 3_000_000,
                "detector_fps": 60.0,
            },
        }
        self.assertIsNone(choose_material_candidate(results, baseline="baseline"))


if __name__ == "__main__":
    unittest.main()
