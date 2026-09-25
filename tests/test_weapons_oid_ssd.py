import tempfile
from pathlib import Path
import unittest

from analytics_lab.tracking import DetectionCandidate, NormalizedBox
from analytics_lab.weapons_oid_ssd import (
    OID_V4_WEAPON_CLASSES,
    OpenVINOOIDSSDWeaponRuntime,
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


class WeaponRuntimeLifecycleTests(unittest.TestCase):
    def test_convert_compile_once_and_reuse_for_repeated_detections(self):
        calls = {"convert": 0, "compile": 0, "factory": 0, "detect": 0}
        candidate = DetectionCandidate(
            category="handgun",
            confidence=0.90,
            box=NormalizedBox(0.10, 0.20, 0.30, 0.40),
            model_class_id=533,
        )

        def convert_model(path):
            self.assertTrue(Path(path).is_file())
            calls["convert"] += 1
            return "converted"

        class Core:
            def compile_model(self, converted, device):
                self_outer.assertEqual(converted, "converted")
                self_outer.assertEqual(device, "CPU")
                calls["compile"] += 1
                return "compiled"

        class Detector:
            def detect(self, _frame):
                calls["detect"] += 1
                return (candidate,)

        def detector_factory(compiled, **kwargs):
            self.assertEqual(compiled, "compiled")
            self.assertEqual(kwargs["confidence_threshold"], 0.50)
            self.assertEqual(kwargs["max_detections"], 64)
            calls["factory"] += 1
            return Detector()

        self_outer = self
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "frozen_inference_graph.pb"
            path.write_bytes(b"synthetic-graph-fixture")
            runtime = OpenVINOOIDSSDWeaponRuntime.from_tensorflow_graph(
                path,
                core=Core(),
                convert_model=convert_model,
                detector_factory=detector_factory,
            )
            self.assertEqual(runtime.detect(object()), (candidate,))
            self.assertEqual(runtime.detect(object()), (candidate,))
            self.assertEqual(runtime.detect(object()), (candidate,))

        self.assertEqual(runtime.compile_count, 1)
        self.assertEqual(runtime.inference_count, 3)
        self.assertEqual(calls, {"convert": 1, "compile": 1, "factory": 1, "detect": 3})

    def test_runtime_rejects_non_detection_values_without_counting_inference(self):
        class Detector:
            def detect(self, _frame):
                return ("not-a-detection",)

        runtime = OpenVINOOIDSSDWeaponRuntime(Detector(), compile_count=1)
        with self.assertRaisesRegex(ValueError, "unsupported weapon value"):
            runtime.detect(object())
        self.assertEqual(runtime.inference_count, 0)


if __name__ == "__main__":
    unittest.main()
