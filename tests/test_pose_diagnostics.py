import unittest

from analytics_lab.detector_continuity import DetectionBox
from analytics_lab.perception import BBox, Keypoint, PoseCandidate, classify_posture
from analytics_lab.pose_diagnostics import _PoseAccumulator, _pixel_bbox


class _Image:
    shape = (100, 200, 3)


class PoseDiagnosticTests(unittest.TestCase):
    def test_pixel_bbox_clamps_normalized_detector_output(self):
        box = DetectionBox(0.8, -0.1, 0.2, 1.2, 0.9)
        pixel = _pixel_bbox(box, _Image())
        self.assertEqual(pixel, BBox(0.0, 20.0, 200.0, 90.0))

    def test_accumulator_records_exact_unknown_reason_and_keypoint_confidence(self):
        pose = PoseCandidate(
            BBox(10, 10, 90, 90),
            (
                Keypoint("left_shoulder", 30, 30, 0.20),
                Keypoint("right_shoulder", 50, 30, 0.60),
                Keypoint("left_hip", 30, 60, 0.70),
                Keypoint("right_hip", 50, 60, 0.80),
            ),
            0.90,
        )
        result = classify_posture(pose)
        self.assertEqual(result.basis, "required_keypoint_confidence_below_threshold")
        acc = _PoseAccumulator()
        acc.add(
            image=_Image(),
            selected=DetectionBox(0.9, 0.05, 0.10, 0.45, 0.90),
            pose=pose,
            posture=result,
        )
        out = acc.freeze()
        self.assertEqual(out["posture_counts"], {"unknown": 1})
        self.assertEqual(
            out["classification_basis_counts"],
            {"required_keypoint_confidence_below_threshold": 1},
        )
        self.assertEqual(out["keypoints"]["left_shoulder"]["min"], 0.20)
        self.assertEqual(out["keypoints"]["left_shoulder"]["frames_at_or_above_0_35"], 0)
        self.assertEqual(out["keypoints"]["right_shoulder"]["frames_at_or_above_0_35"], 1)

    def test_accumulator_separates_unselected_frames(self):
        acc = _PoseAccumulator()
        acc.add(image=_Image(), selected=None, pose=None, posture=None)
        out = acc.freeze()
        self.assertEqual(out["frames"], 1)
        self.assertEqual(out["selected_frames"], 0)
        self.assertEqual(out["pose_frames"], 0)


if __name__ == "__main__":
    unittest.main()
