import unittest

from analytics_lab.tracking import DetectionCandidate, NormalizedBox
from analytics_lab.weapons_oid_ssd import (
    OID_V4_WEAPON_CLASSES,
    parse_oid_v4_weapon_detections,
)


class WeaponOIDSSDTests(unittest.TestCase):
    def test_weapon_allowlist_is_exact_and_pinned(self):
        self.assertEqual(
            OID_V4_WEAPON_CLASSES,
            {
                285: "knife",
                325: "kitchen_knife",
                351: "rifle",
                361: "shotgun",
                365: "sword",
                408: "weapon",
                533: "handgun",
            },
        )

    def test_filters_non_weapon_and_sorts_deterministically(self):
        result = parse_oid_v4_weapon_detections(
            boxes=[[
                [0.10, 0.20, 0.30, 0.40],
                [0.20, 0.10, 0.60, 0.50],
                [0.05, 0.05, 0.15, 0.15],
                [0.30, 0.30, 0.70, 0.80],
            ]],
            scores=[[0.80, 0.95, 0.99, 0.70]],
            classes=[[533.0, 285.0, 502.0, 361.0]],
            num_detections=[4.0],
        )
        self.assertEqual([item.category for item in result], ["knife", "handgun", "shotgun"])
        self.assertEqual([item.model_class_id for item in result], [285, 533, 361])
        self.assertTrue(all(isinstance(item, DetectionCandidate) for item in result))

    def test_threshold_and_max_detection_bounds(self):
        result = parse_oid_v4_weapon_detections(
            boxes=[[[0.1, 0.1, 0.2, 0.2], [0.2, 0.2, 0.4, 0.4]]],
            scores=[[0.49, 0.90]],
            classes=[[533.0, 351.0]],
            num_detections=[2.0],
            confidence_threshold=0.50,
            max_detections=1,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].category, "rifle")

    def test_malformed_outputs_fail_closed(self):
        cases = (
            dict(boxes=[[[0, 0, 1, 1]]], scores=[[0.9]], classes=[[533.5]], num_detections=[1.0]),
            dict(boxes=[[[0, 0, 1]]], scores=[[0.9]], classes=[[533.0]], num_detections=[1.0]),
            dict(boxes=[[[0, 0, 1, 1]]], scores=[[1.1]], classes=[[533.0]], num_detections=[1.0]),
            dict(boxes=[[[0, 0, 1, 1]]], scores=[[0.9]], classes=[[533.0]], num_detections=[101.0]),
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                parse_oid_v4_weapon_detections(**case)

    def test_invalid_or_zero_area_boxes_are_not_emitted(self):
        result = parse_oid_v4_weapon_detections(
            boxes=[[
                [0.5, 0.5, 0.5, 0.7],
                [-1.0, -1.0, 2.0, 2.0],
            ]],
            scores=[[0.9, 0.8]],
            classes=[[533.0, 408.0]],
            num_detections=[2.0],
        )
        self.assertEqual(
            result,
            (
                DetectionCandidate(
                    category="weapon",
                    confidence=0.8,
                    box=NormalizedBox(0.0, 0.0, 1.0, 1.0),
                    model_class_id=408,
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
