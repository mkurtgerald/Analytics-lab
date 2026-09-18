import unittest

from analytics_lab.openpose_pose_geometry import AssociatedReferencePose
from analytics_lab.perception import BBox, Keypoint, PoseCandidate
from analytics_lab.person_down_e2e_diagnostics import _TemporalTrace, _corrected_posture
from analytics_lab.temporal import Config


class PersonDownE2EDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _associated(points, *, confidence=0.9, bbox=None):
        candidate = PoseCandidate(bbox or BBox(0, 0, 20, 10), tuple(points), confidence)
        return AssociatedReferencePose(
            candidate=candidate,
            decoder_score=confidence,
            valid_points=len(points),
            all_points_inside_selection=len(points),
            required_points=len(points),
            required_points_inside_selection=len(points),
            selection_iou=1.0,
        )

    def test_corrected_posture_preserves_four_point_result(self):
        item = self._associated((
            Keypoint("left_shoulder", 2, 4, 0.9),
            Keypoint("right_shoulder", 2, 6, 0.9),
            Keypoint("left_hip", 18, 4, 0.9),
            Keypoint("right_hip", 18, 6, 0.9),
        ))
        posture, confidence, basis = _corrected_posture(item)
        self.assertEqual(posture, "down")
        self.assertEqual(confidence, 0.9)
        self.assertEqual(basis, "horizontal_torso_and_wide_bbox")

    def test_corrected_posture_uses_only_accepted_three_point_fallback(self):
        item = self._associated((
            Keypoint("left_shoulder", 2, 4, 0.8),
            Keypoint("right_shoulder", 2, 6, 0.7),
            Keypoint("left_hip", 18, 4, 0.75),
        ), confidence=0.85)
        posture, confidence, basis = _corrected_posture(item)
        self.assertEqual(posture, "down")
        self.assertEqual(confidence, 0.7)
        self.assertEqual(basis, "three_point_horizontal_torso_and_wide_bbox")

    def test_missing_pose_fails_closed(self):
        self.assertEqual(_corrected_posture(None), ("unknown", 0.0, "no_associated_pose"))

    def test_temporal_trace_exposes_confidence_and_posture_resets(self):
        trace = _TemporalTrace(Config(down_duration_ms=1000, max_gap_ms=750, min_samples=2, min_confidence=0.7))
        trace.observe(0, "down", 0.8, "down")
        trace.observe(500, "down", 0.6, "down")
        trace.observe(1000, "down", 0.9, "down")
        trace.observe(1500, "unknown", 0.0, "missing")
        frozen = trace.freeze()
        self.assertEqual(frozen["qualified_down_frames"], 2)
        self.assertEqual(frozen["low_confidence_down_frames"], 1)
        self.assertEqual(frozen["reset_reason_counts"], {
            "down_confidence_below_threshold": 1,
            "posture_unknown": 1,
        })
        self.assertEqual(frozen["longest_qualified_down_run_ms"], 0)
        self.assertEqual(frozen["longest_qualified_down_run_samples"], 1)


if __name__ == "__main__":
    unittest.main()
