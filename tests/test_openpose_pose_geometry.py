import unittest

from analytics_lab.openpose_pose_geometry import candidate_from_coco_pose, select_reference_pose
from analytics_lab.perception import BBox


def _pose(points):
    values = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, triple in points.items():
        values[index] = list(triple)
    return values


class OpenPoseGeometryTests(unittest.TestCase):
    def test_posture_bbox_comes_from_pose_not_detector(self):
        selected = BBox(20.0, 30.0, 180.0, 190.0)
        pose = _pose({
            0: (3.0, 3.0, 0.9), 5: (4.0, 5.0, 0.8), 6: (6.0, 5.0, 0.7),
            11: (4.0, 10.0, 0.9), 12: (6.0, 10.0, 0.6), 15: (7.0, 12.0, 0.7),
        })
        decoded = candidate_from_coco_pose(
            pose, decoder_score=3.5, selection_bbox=selected,
            frame_width=200, frame_height=200, resized_width=200, resized_height=200,
        )
        self.assertEqual(decoded.required_points, 4)
        self.assertEqual(decoded.valid_points, 6)
        self.assertNotEqual(decoded.candidate.bbox, selected)
        self.assertAlmostEqual(decoded.candidate.bbox.x1, 24.0)
        self.assertAlmostEqual(decoded.candidate.bbox.y2, 96.0)

    def test_selector_prefers_spatial_overlap_over_decoder_score(self):
        selected = BBox(40.0, 40.0, 160.0, 180.0)
        outside = _pose({0: (1.0, 1.0, 0.9), 5: (1.0, 2.0, 0.9), 6: (2.0, 2.0, 0.9), 11: (1.0, 3.0, 0.9)})
        inside = _pose({0: (8.0, 7.0, 0.6), 5: (8.0, 8.0, 0.6), 6: (10.0, 8.0, 0.6), 11: (8.0, 14.0, 0.6), 12: (10.0, 14.0, 0.6)})
        decoded = select_reference_pose(
            [outside, inside], [100.0, 1.0], selection_bbox=selected,
            frame_width=200, frame_height=200, resized_width=200, resized_height=200,
        )
        self.assertIsNotNone(decoded)
        self.assertGreater(decoded.selection_iou, 0.0)
        self.assertEqual(decoded.decoder_score, 1.0)

    def test_non_required_overlap_can_associate_prone_pose(self):
        selected = BBox(72.0, 72.0, 96.0, 96.0)
        pose = _pose({
            0: (10.0, 10.0, 0.9),
            5: (3.0, 5.0, 0.8), 6: (4.0, 5.0, 0.8),
            11: (13.0, 5.0, 0.8), 12: (14.0, 5.0, 0.8),
        })
        decoded = select_reference_pose(
            [pose], [2.0], selection_bbox=selected,
            frame_width=200, frame_height=200, resized_width=200, resized_height=200,
        )
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.required_points_inside_selection, 0)
        self.assertGreater(decoded.all_points_inside_selection, 0)

    def test_selector_rejects_truly_unassociated_pose(self):
        selected = BBox(120.0, 120.0, 180.0, 180.0)
        outside = _pose({0: (1.0, 1.0, 0.9), 5: (1.0, 2.0, 0.9), 6: (2.0, 2.0, 0.9)})
        decoded = select_reference_pose(
            [outside], [1.0], selection_bbox=selected,
            frame_width=200, frame_height=200, resized_width=200, resized_height=200,
        )
        self.assertIsNone(decoded)


if __name__ == "__main__":
    unittest.main()
