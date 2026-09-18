import unittest

from analytics_lab.detector_continuity import DetectionBox
from analytics_lab.detector_orientation_fallback import (
    select_orientation_fallback,
    unrotate_detection,
)


class DetectorOrientationFallbackTests(unittest.TestCase):
    def assertBoxAlmostEqual(self, actual, expected):
        self.assertAlmostEqual(actual.confidence, expected.confidence)
        self.assertAlmostEqual(actual.x_min, expected.x_min)
        self.assertAlmostEqual(actual.y_min, expected.y_min)
        self.assertAlmostEqual(actual.x_max, expected.x_max)
        self.assertAlmostEqual(actual.y_max, expected.y_max)

    def test_unrotate_ccw_maps_box_back_to_source_coordinates(self):
        source = DetectionBox(0.8, 0.2, 0.3, 0.4, 0.6)
        rotated_ccw = DetectionBox(0.8, 0.3, 0.6, 0.6, 0.8)
        self.assertBoxAlmostEqual(unrotate_detection(rotated_ccw, 1), source)

    def test_unrotate_cw_maps_box_back_to_source_coordinates(self):
        source = DetectionBox(0.8, 0.2, 0.3, 0.4, 0.6)
        rotated_cw = DetectionBox(0.8, 0.4, 0.2, 0.7, 0.4)
        self.assertBoxAlmostEqual(unrotate_detection(rotated_cw, -1), source)

    def test_fallback_admits_only_prior_linked_candidate(self):
        previous = DetectionBox(0.9, 0.2, 0.3, 0.4, 0.6)
        linked_ccw = DetectionBox(0.7, 0.3, 0.6, 0.6, 0.8)
        far_cw = DetectionBox(0.95, 0.0, 0.75, 0.2, 0.95)
        result = select_orientation_fallback(
            previous,
            {1: (linked_ccw,), -1: (far_cw,)},
            min_confidence=0.1,
            min_link_iou=0.05,
        )
        self.assertIsNotNone(result.selected)
        self.assertEqual(result.mapped_candidate_count, 2)
        self.assertEqual(result.linked_candidate_count, 1)
        self.assertBoxAlmostEqual(result.selected, DetectionBox(0.7, 0.2, 0.3, 0.4, 0.6))

    def test_fallback_fails_closed_when_no_rotated_box_links(self):
        previous = DetectionBox(0.9, 0.2, 0.3, 0.4, 0.6)
        far_ccw = DetectionBox(0.9, 0.0, 0.0, 0.1, 0.1)
        result = select_orientation_fallback(
            previous,
            {1: (far_ccw,), -1: ()},
            min_confidence=0.1,
            min_link_iou=0.05,
        )
        self.assertIsNone(result.selected)
        self.assertEqual(result.linked_candidate_count, 0)

    def test_fallback_rejects_unreviewed_rotation(self):
        previous = DetectionBox(0.9, 0.2, 0.3, 0.4, 0.6)
        with self.assertRaises(ValueError):
            select_orientation_fallback(
                previous,
                {2: ()},
                min_confidence=0.1,
                min_link_iou=0.05,
            )


if __name__ == "__main__":
    unittest.main()
