import tempfile
from pathlib import Path
import unittest

from analytics_lab.tracking import DetectionCandidate, NormalizedBox
from analytics_lab.weapons_oid_ssd import (
    OID_V4_WEAPON_CLASSES,
    OpenVINOOIDSSDWeaponRuntime,
    parse_oid_v4_weapon_detections,
)


def _generated_weapon_outputs(count):
    """Distinct boxes in valid model tensors; the last proposal is a rifle."""
    return {
        "boxes": [[[(row + .1) / 10, (column + .1) / 10,
                    (row + .8) / 10, (column + .8) / 10]
                   for row, column in (divmod(index, 10) for index in range(count))]],
        "scores": [[.9] * count],
        "classes": [[285.0] * (count - 1) + [351.0]],
        "num_detections": [float(count)],
    }


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


class WeaponCompleteOutputTests(unittest.TestCase):
    def test_default_capacity_overflow_cannot_silently_drop_admitted_rifle(self):
        # All 65 boxes survive the class, confidence and geometry filters.
        outputs = _generated_weapon_outputs(65)
        with self.assertRaisesRegex(RuntimeError, "weapon count exceeds configured bound"):
            parse_oid_v4_weapon_detections(**outputs)

    def test_exact_default_capacity_preserves_every_admitted_weapon(self):
        detections = parse_oid_v4_weapon_detections(**_generated_weapon_outputs(64))
        self.assertEqual(len(detections), 64)
        self.assertEqual([item.category for item in detections], ["knife"] * 63 + ["rifle"])
        self.assertEqual(len({item.box for item in detections}), 64)

    def test_configured_capacity_one_rejects_two_admitted_weapons(self):
        with self.assertRaisesRegex(RuntimeError, "weapon count exceeds configured bound"):
            parse_oid_v4_weapon_detections(**_generated_weapon_outputs(2), max_detections=1)

    def test_filtered_proposals_do_not_consume_weapon_capacity(self):
        for capacity in (1, 64):
            with self.subTest(capacity=capacity):
                outputs = _generated_weapon_outputs(capacity)
                for index in range(100 - capacity):
                    kind = index % 3
                    outputs["boxes"][0].append(
                        [.2, .2, .2, .4] if kind == 2 else [.2, .2, .4, .4]
                    )
                    outputs["scores"][0].append(.49 if kind == 1 else .99)
                    outputs["classes"][0].append(502.0 if kind == 0 else 533.0)
                outputs["num_detections"] = [100.0]
                detections = parse_oid_v4_weapon_detections(**outputs, max_detections=capacity)
                self.assertEqual(len(detections), capacity)
                self.assertEqual(sum(item.category == "rifle" for item in detections), 1)

    def test_runtime_propagates_overflow_without_counting_success(self):
        outputs = _generated_weapon_outputs(65)

        class Detector:
            def detect(self, _frame):
                return parse_oid_v4_weapon_detections(**outputs)

        runtime = OpenVINOOIDSSDWeaponRuntime(Detector(), compile_count=1)
        with self.assertRaisesRegex(RuntimeError, "weapon count exceeds configured bound"):
            runtime.detect(object())
        self.assertEqual(runtime.inference_count, 0)
        self.assertEqual(runtime.compile_count, 1)

        outputs = _generated_weapon_outputs(64)
        detections = runtime.detect(object())
        self.assertEqual(len(detections), 64)
        self.assertEqual(sum(item.category == "rifle" for item in detections), 1)
        self.assertEqual(runtime.inference_count, 1)
        self.assertEqual(runtime.compile_count, 1)


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
