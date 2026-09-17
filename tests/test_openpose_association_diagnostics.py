import unittest

from analytics_lab.openpose_association_diagnostics import (
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


class OpenPoseAssociationDiagnosticTests(unittest.TestCase):
    def test_geometry_is_normalized_to_selection_box(self):
        selection = BBox(80.0, 80.0, 120.0, 160.0)
        pose = _pose({
            5: (16.0, 10.0, 0.9), 6: (18.0, 10.0, 0.9),
            11: (16.0, 12.0, 0.9), 12: (18.0, 12.0, 0.9),
        })
        item = candidate_from_coco_pose(
            pose, decoder_score=2.0, selection_bbox=selection,
            frame_width=200, frame_height=200, resized_width=200, resized_height=200,
        )
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
        near = candidate_from_coco_pose(
            near_pose, decoder_score=1.0, selection_bbox=selection,
            frame_width=200, frame_height=200, resized_width=200, resized_height=200,
        )
        far = candidate_from_coco_pose(
            far_pose, decoder_score=100.0, selection_bbox=selection,
            frame_width=200, frame_height=200, resized_width=200, resized_height=200,
        )
        chosen = _nearest_candidate((far, near), selection)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen[0].decoder_score, 1.0)

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
