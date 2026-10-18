"""Generated row and fake-runtime coverage; no native libraries or assets."""
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analytics_lab.lpr_ocr import (
    LPROCRConfig, OpenVINOOMZPlateDetector, OpenVINOOMZVehicleDetector,
    PlateDetection, omz_vehicle_artifacts, parse_omz_vehicle_rows,
)
from analytics_lab.tracking import DetectionCandidate, NormalizedBox


@contextmanager
def _runtime(rows, *, shape=(1, 300, 300, 3), version="2026.3.1-generated"):
    calls = {"verify": [], "read": [], "compile": [], "resize": [], "blob": [], "infer": []}

    class Output:
        size = len(rows) * 7

        def reshape(self, requested):
            if requested != (-1, 7):
                raise AssertionError("unexpected detector reshape")
            return rows

    class Resized:
        def __getitem__(self, index):
            if index is not None:
                raise AssertionError("expected an NHWC batch dimension")
            return self

    class Compiled:
        inputs = (SimpleNamespace(shape=shape),)

        def output(self, index):
            if index != 0:
                raise AssertionError("unexpected detector output")
            return "output"

        def __call__(self, inputs):
            calls["infer"].append(inputs)
            return {"output": Output()}

    class Core:
        def read_model(self, **kwargs):
            if not calls["verify"]:
                raise AssertionError("model read before artifact verification")
            calls["read"].append(kwargs)
            return "verified-model"

        def compile_model(self, model, device):
            calls["compile"].append((model, device))
            return Compiled()

    def verify(root, specs):
        calls["verify"].append((root, specs))
        return tuple(SimpleNamespace(spec=spec, path=Path("generated") / spec.relative_path) for spec in specs)

    def resize(image, dimensions, *, interpolation):
        calls["resize"].append((image, dimensions, interpolation))
        return Resized()

    def contiguous(value, *, dtype):
        calls["blob"].append((value, dtype))
        return "bgr-float32-nhwc"

    modules = {
        "cv2": SimpleNamespace(resize=resize, INTER_LINEAR="linear"),
        "numpy": SimpleNamespace(ascontiguousarray=contiguous, asarray=lambda value: value, float32="float32"),
        "openvino": SimpleNamespace(Core=Core, __version__=version),
    }
    with patch.dict(sys.modules, modules), patch("analytics_lab.artifacts.verify_artifact_set", side_effect=verify):
        yield calls


class OMZVehicleParserTests(unittest.TestCase):
    def test_vehicle_class_only_with_inclusive_threshold_and_stable_order(self):
        result = parse_omz_vehicle_rows((
            (0, 2, .99, .1, .1, .2, .2),
            (0, 1, .5, .2, .1, .4, .3),
            (0, 1, .9, .3, .1, .5, .3),
            (0, 1, .9, .1, .1, .3, .3),
            (0, 1, .49, .1, .1, .2, .2),
            (0, 0, .99, .1, .1, .2, .2),
        ))
        self.assertEqual([item.confidence for item in result], [.9, .9, .5])
        self.assertEqual([item.box.x_min for item in result], [.1, .3, .2])
        self.assertTrue(all(type(item) is DetectionCandidate for item in result))
        self.assertTrue(all(item.category == "vehicle" and item.model_class_id == 1 for item in result))

    def test_clipping_degenerate_boxes_and_model_termination(self):
        result = parse_omz_vehicle_rows((
            (0, 1, .9, -.1, -.2, 1.2, 1.1),
            (0, 1, .9, .2, .3, .2, .4),
            (0, 1, .9, 1.1, .1, 1.2, .2),
            (-1, 0, 0, 0, 0, 0, 0),
            (0, 1, .99, .1, .1, .2, .2),
        ))
        self.assertEqual(result, (DetectionCandidate("vehicle", .9, NormalizedBox(0, 0, 1, 1), 1),))

    def test_exact_capacity_and_overflow(self):
        for capacity in (1, 64):
            rows = tuple((0, 1, .9, index / 100, .1, (index + .5) / 100, .2)
                         for index in range(capacity + 1))
            with self.subTest(capacity=capacity):
                self.assertEqual(len(parse_omz_vehicle_rows(rows[:capacity], max_vehicles=capacity)), capacity)
                with self.assertRaisesRegex(RuntimeError, "vehicle count exceeds"):
                    parse_omz_vehicle_rows(rows, max_vehicles=capacity)

    def test_filtered_extra_rows_do_not_consume_capacity(self):
        rows = ((0, 1, .5, .1, .1, .2, .2), (0, 2, .99, .1, .1, .2, .2),
                (0, 1, .49, .2, .2, .3, .3), (0, 1, .99, .2, .2, .2, .3))
        self.assertEqual(len(parse_omz_vehicle_rows(rows, max_vehicles=1)), 1)

    def test_row_consumption_is_bounded(self):
        consumed = []

        def endless_plates():
            while True:
                consumed.append(1)
                yield (0, 2, .99, .1, .1, .2, .2)

        with self.assertRaisesRegex(RuntimeError, "row limit"):
            parse_omz_vehicle_rows(endless_plates())
        self.assertEqual(len(consumed), 201)

    def test_invalid_configuration_rejects_before_native_construction(self):
        with patch("analytics_lab.lpr_ocr.OpenVINOOMZPlateDetector") as factory:
            for options in ({"confidence_threshold": -1}, {"confidence_threshold": 1.1},
                            {"confidence_threshold": True}, {"confidence_threshold": float("nan")},
                            {"max_vehicles": 0}, {"max_vehicles": 65}, {"max_vehicles": True}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    OpenVINOOMZVehicleDetector("generated", **options)
            factory.assert_not_called()


class OMZVehicleRuntimeTests(unittest.TestCase):
    def test_real_adapter_boundary_reuses_exact_artifacts_shape_bgr_and_cpu(self):
        rows = ((0, 1, .9, .1, .2, .8, .9), (0, 2, .99, .2, .3, .3, .4))
        image = SimpleNamespace(shape=(480, 640, 3))
        with _runtime(rows) as calls:
            detector = OpenVINOOMZVehicleDetector("generated-artifacts")
            first, second = detector.detect(image), detector.detect(image)
        self.assertEqual(first, second)
        self.assertEqual(first, (DetectionCandidate("vehicle", .9, NormalizedBox(.1, .2, .8, .9), 1),))
        self.assertEqual(detector.runtime_version, "2026.3.1-generated")
        self.assertEqual(len(calls["verify"]), 1)
        specs = calls["verify"][0][1]
        self.assertEqual(specs, omz_vehicle_artifacts())
        self.assertEqual([spec.size_bytes for spec in specs], [325452, 1286772])
        self.assertEqual([spec.sha384 for spec in specs], [
            "b0a70cfc4ccbe85dd5ba5549c2d142d7f922a0a1116657cdcc9f48cb6afc7cc40a596a655b9e6566396c48b72232ff65",
            "0406e296c9822f4a5e8ee300ad40e5c0a0eeeaf9206f829556cb836cb8f10a4f8a4dd94d1bb16a60562bf1568fd12618",
        ])
        self.assertEqual(calls["compile"], [("verified-model", "CPU")])
        self.assertEqual(calls["resize"], [(image, (300, 300), "linear")] * 2)
        self.assertEqual([dtype for _, dtype in calls["blob"]], ["float32", "float32"])
        self.assertEqual(calls["infer"], [["bgr-float32-nhwc"]] * 2)

    def test_plate_call_remains_class_two_only_with_one_inference(self):
        rows = ((0, 1, .99, .1, .1, .9, .9), (0, 2, .7, .2, .3, .4, .5))
        with _runtime(rows) as calls:
            detector = OpenVINOOMZPlateDetector("generated", config=LPROCRConfig(max_plates=1))
            self.assertEqual(detector(SimpleNamespace(shape=(480, 640, 3))),
                             (PlateDetection(.7, .2, .3, .4, .5),))
        self.assertEqual(len(calls["infer"]), 1)

    def test_existing_native_shape_and_version_guards_apply_to_vehicle(self):
        for options, message in (({"shape": (1, 3, 300, 300)}, "input shape"),
                                 ({"version": "2025.0"}, "runtime version")):
            with self.subTest(options=options), _runtime((), **options), self.assertRaisesRegex(RuntimeError, message):
                OpenVINOOMZVehicleDetector("generated")


if __name__ == "__main__":
    unittest.main()
