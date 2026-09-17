import unittest

from analytics_lab.openpose_association_diagnostics import (
    _bounded_fallback,
    _candidate_metrics,
    _nearest_candidate,
    _summary,
)
from analytics_lab.openpose_pose_geometry import candidate_from_coco_pose
from analytics_lab.perception import BBox


def _pose(points):
    values = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, triple in points.items():
        values[index] = list(triple)
    return values


def _candidate(pose, selection, score=1.0):
    return candidate_from_coco_pose(
        pose, decoder_score=score, selection_bbox=selection,
        frame_width=200, frame_height=200, resized_width=200, resized_height=200,
    )


class OpenPoseAssociationDiagnosticTests(unittest.TestCase):
    def test_geometry_is_normalized_to_selection_box(self):
        selection = BBox(80.0, 80.0, 120.0, 160.0)
        pose = _pose({
            5: (16.0, 10.0, 0.9), 6: (18.0, 10.0, 0.9),
            11: (16.0, 12.0, 0.9), 12: (18.0, 12.0, 0.9),
        })
        item = _candidate(pose, selection, score=2.0)
        metrics = _candidate_metrics(item, selection)
        self.assertGreater(metrics["edge_gap_norm"], 0.0)
        self.assertGreater(metrics["center_distance_norm"], 0.0)
        self.assertEqual(metrics["selection_iou"], 0.0)
        self.assertGreater(metrics["pose_area_ratio"], 0.0)

    def test_nearest_candidate_prefers_edge_gap_before_decoder_score(self):
        selection = BBox(80.0, 80.0, 120.0, 160.0)
        near_pose = _pose({
            5: (14.0, 10.0, 0.8), 6: (16.0, 10.0, 0.8),
            11: (14.0, 12.0, 0.8), 12: (16.0, 12.0, 0.8),
        })
        far_pose = _pose({
            5: (2.0, 2.0, 0.9), 6: (4.0, 2.0, 0.9),
            11: (2.0, 4.0, 0.9), 12: (4.0, 4.0, 0.9),
        })
        near = _candidate(near_pose, selection, score=1.0)
        far = _candidate(far_pose, selection, score=100.0)
        chosen = _nearest_candidate((far, near), selection)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen[0].decoder_score, 1.0)

    def test_bounded_fallback_accepts_one_small_gap(self):
        selection = BBox(80.0, 80.0, 120.0, 160.0)
        near = _candidate(_pose({
            5: (16.0, 10.0, 0.8), 6: (18.0, 10.0, 0.8),
            11: (16.0, 12.0, 0.8), 12: (18.0, 12.0, 0.8),
        }), selection)
        chosen, reason, count = _bounded_fallback((near,), selection)
        self.assertIs(chosen, near)
        self.assertEqual(reason, "bounded_edge_gap")
        self.assertEqual(count, 1)

    def test_bounded_fallback_rejects_large_gap(self):
        selection = BBox(80.0, 80.0, 120.0, 160.0)
        far = _candidate(_pose({
            5: (20.0, 10.0, 0.8), 6: (22.0, 10.0, 0.8),
            11: (20.0, 12.0, 0.8), 12: (22.0, 12.0, 0.8),
        }), selection)
        chosen, reason, count = _bounded_fallback((far,), selection)
        self.assertIsNone(chosen)
        self.assertEqual(reason, "no_candidate_within_bound")
        self.assertEqual(count, 0)

    def test_bounded_fallback_fails_closed_on_two_nearby_poses(self):
        selection = BBox(80.0, 80.0, 120.0, 160.0)
        first = _candidate(_pose({
            5: (16.0, 10.0, 0.8), 6: (18.0, 10.0, 0.8),
            11: (16.0, 12.0, 0.8), 12: (18.0, 12.0, 0.8),
        }), selection, score=2.0)
        second = _candidate(_pose({
            5: (16.2, 14.0, 0.8), 6: (18.2, 14.0, 0.8),
            11: (16.2, 16.0, 0.8), 12: (18.2, 16.0, 0.8),
        }), selection, score=1.0)
        chosen, reason, count = _bounded_fallback((first, second), selection)
        self.assertIsNone(chosen)
        self.assertEqual(reason, "ambiguous_candidates_within_bound")
        self.assertEqual(count, 2)

    def test_summary_uses_bounded_nearest_rank_quantiles(self):
        summary = _summary([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["p50"], 3.0)
        self.assertEqual(summary["p90"], 5.0)
        self.assertEqual(summary["max"], 5.0)
        self.assertEqual(_summary([])["count"], 0)


if __name__ == "__main__":
    unittest.main()
