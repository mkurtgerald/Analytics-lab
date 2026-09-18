import unittest
from unittest.mock import patch

from analytics_lab.detector_continuity import DetectionBox
from analytics_lab.openpose_pose_geometry import AssociatedReferencePose
from analytics_lab.perception import BBox, Keypoint, PoseCandidate
from analytics_lab import openpose_association_diagnostics as association
from analytics_lab.posture_quality_diagnostics import (
    _PostureQualityAccumulator,
    _three_point_fallback,
    run_posture_quality_diagnostic,
)


class PostureQualityDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _associated(points, bbox=None):
        candidate = PoseCandidate(bbox or BBox(0, 0, 20, 10), tuple(points), 1.0)
        return AssociatedReferencePose(
            candidate=candidate,
            decoder_score=1.0,
            valid_points=len(points),
            all_points_inside_selection=len(points),
            required_points=len(points),
            required_points_inside_selection=len(points),
            selection_iou=1.0,
        )

    def test_unknown_reasons_and_geometry_are_aggregated(self):
        selected = DetectionBox(0.9, 0.0, 0.0, 1.0, 1.0)
        accumulator = _PostureQualityAccumulator()
        low_confidence = self._associated((
            Keypoint("left_shoulder", 2, 4, 0.9),
            Keypoint("right_shoulder", 2, 6, 0.9),
            Keypoint("left_hip", 18, 4, 0.05),
            Keypoint("right_hip", 18, 6, 0.9),
        ))
        missing_hips = self._associated((
            Keypoint("left_shoulder", 2, 4, 0.9),
            Keypoint("right_shoulder", 2, 6, 0.9),
        ))
        accumulator.add(selected=selected, decoded_count=1, associated=low_confidence)
        accumulator.add(selected=selected, decoded_count=1, associated=missing_hips)
        frozen = accumulator.freeze()["posture_quality"]
        self.assertEqual(frozen["unknown_frames"], 2)
        self.assertEqual(
            frozen["unknown_basis_counts"],
            {
                "required_keypoint_confidence_below_threshold": 1,
                "required_keypoints_missing": 1,
            },
        )
        self.assertEqual(frozen["unknown_required_points_histogram"], {"2": 1, "4": 1})
        metrics = frozen["metrics_by_posture"]["unknown"]
        self.assertEqual(metrics["min_required_confidence"]["count"], 2)
        self.assertEqual(metrics["torso_fraction"]["count"], 1)
        self.assertEqual(frozen["three_point_fallback"]["attempted_frames"], 0)

    def test_three_point_fallback_recovers_only_decisive_same_side_geometry(self):
        down = self._associated((
            Keypoint("left_shoulder", 2, 4, 0.8),
            Keypoint("right_shoulder", 2, 6, 0.8),
            Keypoint("left_hip", 18, 4, 0.8),
        ))
        upright = self._associated((
            Keypoint("left_shoulder", 4, 2, 0.8),
            Keypoint("right_shoulder", 6, 2, 0.8),
            Keypoint("left_hip", 4, 18, 0.8),
        ), bbox=BBox(0, 0, 10, 20))
        insufficient = self._associated((
            Keypoint("left_shoulder", 2, 4, 0.8),
            Keypoint("right_hip", 18, 6, 0.8),
        ))
        self.assertEqual(_three_point_fallback(down)[0], "down")
        self.assertEqual(_three_point_fallback(upright)[0], "upright")
        self.assertEqual(_three_point_fallback(insufficient)[0], "unknown")

    def test_accumulator_keeps_non_decisive_fallback_unknown(self):
        selected = DetectionBox(0.9, 0.0, 0.0, 1.0, 1.0)
        accumulator = _PostureQualityAccumulator()
        ambiguous = self._associated((
            Keypoint("left_shoulder", 3, 3, 0.8),
            Keypoint("right_shoulder", 7, 3, 0.8),
            Keypoint("left_hip", 8, 8, 0.8),
        ), bbox=BBox(0, 0, 10, 10))
        accumulator.add(selected=selected, decoded_count=1, associated=ambiguous)
        frozen = accumulator.freeze()["posture_quality"]
        self.assertEqual(frozen["three_point_fallback"]["attempted_frames"], 1)
        self.assertEqual(frozen["three_point_fallback"]["accepted_decisive_frames"], 0)
        self.assertEqual(frozen["corrected_posture_counts"], {"unknown": 1})

    def test_runner_restores_association_accumulator_after_failure(self):
        original = association._Accumulator
        with patch.object(association, "run_association_diagnostic", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                run_posture_quality_diagnostic("manifest.json", "candidates")
        self.assertIs(association._Accumulator, original)


if __name__ == "__main__":
    unittest.main()
