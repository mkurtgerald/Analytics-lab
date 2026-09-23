import math
import unittest

from analytics_lab.face_oid_ssd import (
    OID_V4_HUMAN_FACE_CLASS_ID,
    parse_oid_v4_detections,
)
from analytics_lab.tracking import NormalizedBox


class OIDV4FaceAdapterTests(unittest.TestCase):
    def test_filters_exact_human_face_class_and_maps_yxyx_box(self):
        faces = parse_oid_v4_detections(
            boxes=[[
                [0.10, 0.20, 0.40, 0.60],
                [0.15, 0.25, 0.45, 0.65],
                [0.20, 0.30, 0.50, 0.70],
            ]],
            scores=[[0.91, 0.99, 0.49]],
            classes=[[float(OID_V4_HUMAN_FACE_CLASS_ID), 1.0, 502.0]],
            num_detections=[3.0],
        )
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].confidence, 0.91)
        self.assertEqual(faces[0].box, NormalizedBox(0.20, 0.10, 0.60, 0.40))

    def test_clamps_coordinates_and_sorts_by_confidence(self):
        faces = parse_oid_v4_detections(
            boxes=[[
                [-0.10, 0.10, 0.40, 0.50],
                [0.20, 0.20, 1.20, 1.30],
            ]],
            scores=[[0.60, 0.80]],
            classes=[[502.0, 502.0]],
            num_detections=[2.0],
        )
        self.assertEqual([item.confidence for item in faces], [0.80, 0.60])
        self.assertEqual(faces[0].box, NormalizedBox(0.20, 0.20, 1.0, 1.0))
        self.assertEqual(faces[1].box, NormalizedBox(0.10, 0.0, 0.50, 0.40))

    def test_malformed_or_unbounded_outputs_fail_closed(self):
        with self.assertRaises(ValueError):
            parse_oid_v4_detections(
                boxes=[[]], scores=[[]], classes=[[]], num_detections=[101.0]
            )
        with self.assertRaises(ValueError):
            parse_oid_v4_detections(
                boxes=[[[0.1, 0.1, 0.2, 0.2]]],
                scores=[[math.nan]],
                classes=[[502.0]],
                num_detections=[1.0],
            )
        with self.assertRaises(ValueError):
            parse_oid_v4_detections(
                boxes=[[[0.1, 0.1, 0.2, 0.2]]],
                scores=[[0.9]],
                classes=[[502.5]],
                num_detections=[1.0],
            )


if __name__ == "__main__":
    unittest.main()
