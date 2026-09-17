import unittest

from analytics_lab.detector_thresholds import _Accumulator, choose_threshold


class DetectorThresholdTests(unittest.TestCase):
    def test_accumulator_tracks_duplicate_noise(self):
        acc = _Accumulator()
        for count in (0, 1, 2, 3):
            acc.add(count)
        out = acc.freeze()
        self.assertEqual(out.frames, 4)
        self.assertEqual(out.frames_with_detection, 3)
        self.assertEqual(out.detections, 6)
        self.assertEqual(out.duplicate_frames, 2)
        self.assertEqual(out.duplicate_excess, 3)
        self.assertEqual(out.coverage, 0.75)
        self.assertEqual(out.duplicate_frame_rate, 0.5)

    def test_selects_highest_threshold_that_clears_recall_and_noise_bars(self):
        rows = [
            {
                "threshold": 0.50,
                "positive_during_coverage": 0.30,
                "positive_duplicate_frame_rate": 0.0,
                "negative_coverage": 1.0,
                "negative_duplicate_frame_rate": 0.0,
            },
            {
                "threshold": 0.40,
                "positive_during_coverage": 0.58,
                "positive_duplicate_frame_rate": 0.02,
                "negative_coverage": 1.0,
                "negative_duplicate_frame_rate": 0.01,
            },
            {
                "threshold": 0.30,
                "positive_during_coverage": 0.80,
                "positive_duplicate_frame_rate": 0.20,
                "negative_coverage": 1.0,
                "negative_duplicate_frame_rate": 0.10,
            },
        ]
        self.assertEqual(choose_threshold(rows), 0.40)

    def test_no_recommendation_when_recall_or_noise_bar_fails(self):
        rows = [
            {
                "threshold": 0.40,
                "positive_during_coverage": 0.54,
                "positive_duplicate_frame_rate": 0.0,
                "negative_coverage": 1.0,
                "negative_duplicate_frame_rate": 0.0,
            },
            {
                "threshold": 0.30,
                "positive_during_coverage": 0.75,
                "positive_duplicate_frame_rate": 0.0,
                "negative_coverage": 1.0,
                "negative_duplicate_frame_rate": 0.06,
            },
        ]
        self.assertIsNone(choose_threshold(rows))

    def test_negative_detection_coverage_must_remain_high(self):
        rows = [{
            "threshold": 0.20,
            "positive_during_coverage": 0.90,
            "positive_duplicate_frame_rate": 0.0,
            "negative_coverage": 0.89,
            "negative_duplicate_frame_rate": 0.0,
        }]
        self.assertIsNone(choose_threshold(rows))


if __name__ == "__main__":
    unittest.main()
