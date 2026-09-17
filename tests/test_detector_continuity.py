import unittest

from analytics_lab.detector_continuity import (
    ContinuityAccumulator,
    DetectionBox,
    box_iou,
    select_detection,
)


class DetectorContinuityTests(unittest.TestCase):
    def test_iou_and_linked_selection_prefer_spatial_continuity(self):
        previous = DetectionBox(0.9, 0.10, 0.10, 0.40, 0.80)
        nearby = DetectionBox(0.25, 0.12, 0.12, 0.42, 0.82)
        distractor = DetectionBox(0.95, 0.65, 0.10, 0.95, 0.80)
        self.assertGreater(box_iou(previous, nearby), 0.80)
        selected, linked = select_detection(previous, [distractor, nearby])
        self.assertEqual(selected, nearby)
        self.assertTrue(linked)

    def test_reacquires_highest_confidence_when_no_candidate_links(self):
        previous = DetectionBox(0.9, 0.10, 0.10, 0.30, 0.40)
        lower = DetectionBox(0.30, 0.60, 0.60, 0.80, 0.90)
        higher = DetectionBox(0.45, 0.65, 0.10, 0.95, 0.40)
        selected, linked = select_detection(previous, [lower, higher])
        self.assertEqual(selected, higher)
        self.assertFalse(linked)

    def test_accumulator_records_links_resets_and_gaps(self):
        acc = ContinuityAccumulator(min_link_iou=0.05)
        first = DetectionBox(0.8, 0.10, 0.10, 0.30, 0.50)
        linked = DetectionBox(0.7, 0.11, 0.10, 0.31, 0.50)
        reset = DetectionBox(0.6, 0.70, 0.10, 0.90, 0.50)
        self.assertEqual(acc.add([first]), first)
        self.assertEqual(acc.add([linked]), linked)
        self.assertEqual(acc.add([reset]), reset)
        self.assertIsNone(acc.add([]))
        totals = acc.freeze()
        self.assertEqual(totals.frames, 4)
        self.assertEqual(totals.selected_frames, 3)
        self.assertEqual(totals.transitions, 2)
        self.assertEqual(totals.linked_transitions, 1)
        self.assertEqual(totals.resets, 1)
        self.assertEqual(totals.coverage, 0.75)
        self.assertEqual(totals.link_rate, 0.5)
        self.assertEqual(totals.reset_rate, 0.5)
        self.assertAlmostEqual(totals.mean_confidence, 0.7)
        self.assertEqual(totals.min_confidence, 0.6)

    def test_invalid_boxes_and_thresholds_fail_closed(self):
        with self.assertRaises(ValueError):
            DetectionBox(1.1, 0.0, 0.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            DetectionBox(0.5, 1.0, 0.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            ContinuityAccumulator(min_link_iou=-0.1)


if __name__ == "__main__":
    unittest.main()
