import unittest

from analytics_lab.openpose_reference import candidate_from_coco_pose, select_reference_pose
from analytics_lab.perception import BBox


def _pose(points):
    values = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, triple in points.items():
        values[index] = list(triple)
    return values


class OpenPoseReferenceTests(unittest.TestCase):
    def test_maps_required_coco_points_back_to_source_frame(self):
        selected = BBox(20.0, 30.0, 180.0, 190.0)
        pose = _pose({
            5: (4.0, 5.0, 0.8),
            6: (6.0, 5.0, 0.7),
            11: (4.0, 10.0, 0.9),
            12: (6.0, 10.0, 0.6),
        })
        decoded = candidate_from_coco_pose(
            pose,
            decoder_score=3.5,
            selection_bbox=selected,
            frame_width=200,
            frame_height=200,
            resized_width=200,
            resized_height=200,
        )
        self.assertEqual(decoded.required_points, 4)
        self.assertEqual(decoded.required_points_inside_selection, 4)
        self.assertEqual(decoded.candidate.confidence, 1.0)
        self.assertAlmostEqual(decoded.candidate.keypoint("left_shoulder").x, 32.0)
        self.assertAlmostEqual(decoded.candidate.keypoint("left_shoulder").y, 40.0)
        self.assertAlmostEqual(decoded.candidate.keypoint("right_hip").confidence, 0.6)

    def test_selector_prefers_pose_associated_with_continuity_box(self):
        selected = BBox(40.0, 40.0, 160.0, 180.0)
        outside = _pose({
            5: (1.0, 1.0, 0.9),
            6: (2.0, 1.0, 0.9),
            11: (1.0, 2.0, 0.9),
            12: (2.0, 2.0, 0.9),
        })
        inside = _pose({
            5: (8.0, 8.0, 0.6),
            6: (10.0, 8.0, 0.6),
            11: (8.0, 14.0, 0.6),
            12: (10.0, 14.0, 0.6),
        })
        decoded = select_reference_pose(
            [outside, inside],
            [100.0, 1.0],
            selection_bbox=selected,
            frame_width=200,
            frame_height=200,
            resized_width=200,
            resized_height=200,
        )
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.required_points_inside_selection, 4)
        self.assertEqual(decoded.decoder_score, 1.0)

    def test_selector_rejects_unassociated_pose(self):
        selected = BBox(100.0, 100.0, 180.0, 180.0)
        outside = _pose({5: (1.0, 1.0, 0.9), 6: (2.0, 1.0, 0.9)})
        decoded = select_reference_pose(
            [outside],
            [1.0],
            selection_bbox=selected,
            frame_width=200,
            frame_height=200,
            resized_width=200,
            resized_height=200,
        )
        self.assertIsNone(decoded)


if __name__ == "__main__":
    unittest.main()
