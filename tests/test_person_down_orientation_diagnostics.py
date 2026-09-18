import unittest

from analytics_lab.openpose_pose_geometry import AssociatedReferencePose
from analytics_lab.perception import BBox, Keypoint, PoseCandidate
from analytics_lab.person_down_orientation_diagnostics import (
    _DIAGONAL_AMBIGUITY_DELTA,
    _FragmentationAccumulator,
    _bounded_posture,
)


class PersonDownOrientationDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _associated(*, required_points=4, points=None, bbox=None):
        if points is None:
            points = (
                Keypoint("left_shoulder", 2, 4, 0.8),
                Keypoint("right_shoulder", 3, 6, 0.7),
                Keypoint("left_hip", 16, 4, 0.75),
                Keypoint("right_hip", 18, 6, 0.85),
            )[:required_points]
        return AssociatedReferencePose(
            candidate=PoseCandidate(bbox or BBox(0, 0, 20, 10), tuple(points), 1.0),
            decoder_score=0.9,
            valid_points=max(required_points, 4),
            all_points_inside_selection=required_points,
            required_points=required_points,
            required_points_inside_selection=required_points,
            selection_iou=0.5,
        )

    def test_fragmentation_accumulator_records_decisive_reset_geometry(self):
        item = self._associated()
        accumulator = _FragmentationAccumulator()
        accumulator.add(
            association_reason="baseline_overlap",
            posture="other",
            basis="geometry_not_decisive",
            associated=item,
            nearest=(item, {
                "selection_iou": 0.5,
                "center_distance_norm": 0.1,
                "edge_gap_norm": 0.0,
                "pose_width_ratio": 1.2,
                "pose_height_ratio": 0.8,
                "pose_area_ratio": 0.96,
            }),
            reset_reason="posture_other",
        )
        frozen = accumulator.freeze()
        self.assertEqual(frozen["frames"], 1)
        self.assertEqual(frozen["association_reason_counts"], {"baseline_overlap": 1})
        self.assertEqual(frozen["posture_counts"], {"other": 1})
        self.assertEqual(frozen["decisive_resets"]["count"], 1)
        self.assertEqual(frozen["decisive_resets"]["reason_counts"], {"posture_other": 1})
        self.assertEqual(frozen["decisive_resets"]["required_points_histogram"], {"4": 1})
        self.assertEqual(frozen["decisive_resets"]["pose_metrics"]["width_over_height"]["count"], 1)

    def test_fragmentation_accumulator_records_unmatched_nearest_pose_only_in_aggregate(self):
        item = self._associated(required_points=3)
        accumulator = _FragmentationAccumulator()
        accumulator.add(
            association_reason="no_candidate_within_bound",
            posture="unknown",
            basis="no_associated_pose",
            associated=None,
            nearest=(item, {
                "selection_iou": 0.0,
                "center_distance_norm": 0.2,
                "edge_gap_norm": 0.13,
                "pose_width_ratio": 1.5,
                "pose_height_ratio": 0.7,
                "pose_area_ratio": 1.05,
            }),
            reset_reason=None,
        )
        frozen = accumulator.freeze()
        self.assertEqual(
            frozen["unmatched_nearest_required_points_histogram"],
            {"3": 1},
        )
        self.assertEqual(
            frozen["unmatched_nearest_association_metrics"]["edge_gap_norm"],
            {"count": 1, "min": 0.13, "p50": 0.13, "p90": 0.13, "max": 0.13},
        )
        self.assertEqual(frozen["decisive_resets"]["count"], 0)

    def test_bounded_posture_turns_only_near_diagonal_other_into_unknown(self):
        diagonal = self._associated(
            points=(
                Keypoint("left_shoulder", 2, 2, 0.8),
                Keypoint("right_shoulder", 4, 2, 0.8),
                Keypoint("left_hip", 12, 12, 0.8),
                Keypoint("right_hip", 14, 12, 0.8),
            ),
            bbox=BBox(0, 0, 20, 18),
        )
        posture, confidence, basis = _bounded_posture(diagonal)
        self.assertEqual(posture, "unknown")
        self.assertEqual(confidence, 0.8)
        self.assertEqual(basis, "diagonal_torso_ambiguous")
        self.assertLessEqual(_DIAGONAL_AMBIGUITY_DELTA, 0.02)

        horizontal_but_not_wide = self._associated(
            points=(
                Keypoint("left_shoulder", 2, 3, 0.8),
                Keypoint("right_shoulder", 4, 3, 0.8),
                Keypoint("left_hip", 12, 9, 0.8),
                Keypoint("right_hip", 14, 9, 0.8),
            ),
            bbox=BBox(0, 0, 20, 18),
        )
        posture, _confidence, basis = _bounded_posture(horizontal_but_not_wide)
        self.assertEqual(posture, "other")
        self.assertEqual(basis, "geometry_not_decisive")

    def test_bounded_posture_preserves_decisive_down_and_upright(self):
        down = self._associated(
            points=(
                Keypoint("left_shoulder", 2, 4, 0.8),
                Keypoint("right_shoulder", 2, 6, 0.8),
                Keypoint("left_hip", 18, 4, 0.8),
                Keypoint("right_hip", 18, 6, 0.8),
            ),
            bbox=BBox(0, 0, 20, 10),
        )
        self.assertEqual(_bounded_posture(down)[0], "down")

        upright = self._associated(
            points=(
                Keypoint("left_shoulder", 4, 2, 0.8),
                Keypoint("right_shoulder", 6, 2, 0.8),
                Keypoint("left_hip", 4, 18, 0.8),
                Keypoint("right_hip", 6, 18, 0.8),
            ),
            bbox=BBox(0, 0, 10, 20),
        )
        self.assertEqual(_bounded_posture(upright)[0], "upright")


if __name__ == "__main__":
    unittest.main()
