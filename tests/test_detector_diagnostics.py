from pathlib import Path
import unittest

from analytics_lab.detector_diagnostics import _Accumulator, _window_name
from analytics_lab.evaluation import LabeledPersonDown
from analytics_lab.validation import ValidationSampleSpec


class Tests(unittest.TestCase):
    def test_stage_totals_separate_detector_pose_posture_and_tracks(self):
        acc = _Accumulator()
        acc.add(1, ("upright",), ("track-000001",))
        acc.add(0, (), ())
        acc.add(2, ("down", "unknown"), ("track-000001", "track-000002"))
        out = acc.freeze()
        self.assertEqual(out.frames, 3)
        self.assertEqual(out.frames_with_detection, 2)
        self.assertEqual(out.detections, 3)
        self.assertEqual(out.frames_with_pose, 2)
        self.assertEqual(out.pose_candidates, 3)
        self.assertEqual(out.posture_upright, 1)
        self.assertEqual(out.posture_down, 1)
        self.assertEqual(out.posture_unknown, 1)
        self.assertEqual(out.track_assignments, 3)
        self.assertEqual(out.distinct_tracks, 2)

    def test_single_label_is_split_before_during_after(self):
        sample = ValidationSampleSpec(
            sample_id="sample",
            site_id="site",
            camera_id="camera",
            authorization_ref="rights-ref",
            video_path=Path("authorized.mp4"),
            media_sha256="0" * 64,
            start_timestamp_ms=0,
            end_timestamp_ms=6000,
            labels=(LabeledPersonDown(1800, 5000, "fall-1"),),
        )
        self.assertEqual(_window_name(sample, 1799), "before")
        self.assertEqual(_window_name(sample, 1800), "during")
        self.assertEqual(_window_name(sample, 5000), "during")
        self.assertEqual(_window_name(sample, 5001), "after")


if __name__ == "__main__":
    unittest.main()
