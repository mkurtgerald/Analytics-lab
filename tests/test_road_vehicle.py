import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analytics_lab.road_vehicle import OpenVINORoadVehicleDetector, parse_road_vehicle_rows, road_vehicle_artifacts
from analytics_lab.tracking import DetectionCandidate, NormalizedBox


class RoadVehicleParserTests(unittest.TestCase):
    def test_only_vehicle_class_two_inclusive_cutoff_and_sorted(self):
        result = parse_road_vehicle_rows((
            (0, 1, .99, .1, .1, .4, .4), (0, 3, .99, .1, .1, .4, .4),
            (0, 2, .5, .2, .2, .4, .4), (0, 2, .9, -.1, .1, 1.2, .8),
            (0, 2, .49, .2, .2, .4, .4),
        ))
        self.assertEqual(result, (DetectionCandidate("vehicle", .9, NormalizedBox(0, .1, 1, .8), 2),
                                  DetectionCandidate("vehicle", .5, NormalizedBox(.2, .2, .4, .4), 2)))

    def test_malformed_ignored_classes_fail_and_sentinel_stops(self):
        for label in (math.nan, math.inf, "2", True):
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_road_vehicle_rows(((0, label, .9, .1, .1, .4, .4),))
        with self.assertRaises(ValueError):
            parse_road_vehicle_rows(((0, 1, math.nan, .1, .1, .4, .4),))
        with self.assertRaises(ValueError):
            parse_road_vehicle_rows(((0, 2),))
        self.assertEqual(parse_road_vehicle_rows(((-1, 0, 0, 0, 0, 0, 0), (0, 2, 1, 0, 0, 1, 1))), ())

    def test_capacity_and_raw_row_overflow_fail_without_silent_slicing(self):
        row = (0, 2, .9, .1, .1, .4, .4)
        self.assertEqual(len(parse_road_vehicle_rows((row,) * 64)), 64)
        with self.assertRaisesRegex(RuntimeError, "vehicle count"):
            parse_road_vehicle_rows((row,) * 65)
        with self.assertRaisesRegex(RuntimeError, "row limit"):
            parse_road_vehicle_rows(((0, 1, .9, .1, .1, .4, .4),) * 201)

    def test_artifact_pins_match_primary_metadata(self):
        specs = road_vehicle_artifacts()
        self.assertEqual([item.relative_path for item in specs], [
            "FP16/person-vehicle-bike-detection-crossroad-0078.xml", "FP16/person-vehicle-bike-detection-crossroad-0078.bin"])
        self.assertEqual([item.size_bytes for item in specs], [526260, 2357042])
        self.assertEqual(sum(item.size_bytes for item in specs), 2883302)
        self.assertEqual([item.sha384 for item in specs], [
            "7997f14109662b3d422ee5b9c7a812690eb00227b17e366ab6098fb5924a9a8c5077b326e3dcc3d2e2cbce91797bab4d",
            "9fa3031c27af7e934a61ff02a01820d990b257a21ea0f9e195d79941fcbe97fac9c731bd404bd0c8dddfa4c1e1937cb4"])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required for injected native array tests")
class RoadVehicleRuntimeTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np, self.events, self.blobs = np, [], []
        self.input_shape, self.input_type = (1, 3, 1024, 1024), "f32"
        self.output_shape, self.output_type = (1, 1, 200, 7), "f32"
        self.output = np.zeros((1, 1, 200, 7), dtype=np.float32)
        self.output[0, 0, 0] = (0, 2, .9, .1, .1, .4, .4)
        self.output[0, 0, 1, 0] = -1
        self.version = "2026.3.1-fake"
        owner = self

        class Port:
            def __init__(self, is_input):
                self.is_input = is_input

            @property
            def shape(self):
                return owner.input_shape if self.is_input else owner.output_shape

            def get_names(self):
                return {"data"} if self.is_input else {"detection_out"}

            def get_element_type(self):
                return SimpleNamespace(get_type_name=lambda: owner.input_type if self.is_input else owner.output_type)

        class Compiled:
            inputs, outputs = [Port(True)], [Port(False)]

            def output(self, index):
                assert index == 0
                return "output"

            def __call__(self, values):
                owner.events.append("infer")
                owner.blobs.append(values[0])
                owner.assertEqual(values[0][0, :, 0, 0].tolist(), [7., 23., 241.])
                values[0][:] = 0  # Native mutation cannot reach source pixels.
                return {"output": owner.output}

        class Core:
            def __init__(self):
                owner.events.append("core")

            def read_model(self, **kwargs):
                owner.events.append(("read", kwargs))
                return SimpleNamespace(inputs=Compiled.inputs, outputs=Compiled.outputs)

            def compile_model(self, model, device):
                owner.events.append(("compile", device))
                return Compiled()

        def verify(root, specs):
            owner.events.append("verify")
            return tuple(SimpleNamespace(spec=item, path=Path("generated") / item.relative_path) for item in specs)

        def resize(image, size, *, interpolation):
            self.events.append(("resize", size, interpolation))
            self.assertEqual(image[0, 0].tolist(), [7, 23, 241])
            result = np.tile(image[0, 0], (1024, 1024, 1))
            return result

        self.patches = [patch.dict("sys.modules", {
            "cv2": SimpleNamespace(resize=resize, INTER_LINEAR=1),
            "openvino": SimpleNamespace(Core=Core, __version__=self.version)}),
            patch("analytics_lab.road_vehicle.verify_artifact_set", side_effect=verify)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_hash_before_load_cpu_compile_once_bgr_nchw_float32_and_warm_reuse(self):
        detector = OpenVINORoadVehicleDetector("generated")
        image = self.np.tile(self.np.array([7, 23, 241], dtype=self.np.uint8), (4, 8, 1))
        for _ in range(2):
            result = detector.detect(image)
            self.assertEqual(result[0].model_class_id, 2)
        self.assertEqual(self.events[:2], ["verify", "core"])
        self.assertEqual(self.events.count(("compile", "CPU")), 1)
        self.assertEqual((detector.compile_count, detector.inference_count), (1, 2))
        self.assertEqual(detector.runtime_version, self.version)
        self.assertEqual(image[0, 0].tolist(), [7, 23, 241])
        for blob in self.blobs:
            self.assertEqual((blob.shape, blob.dtype), ((1, 3, 1024, 1024), self.np.float32))
            self.assertTrue(blob.flags.owndata)

    def test_bad_hash_cpu_or_configuration_prevents_load(self):
        with patch("analytics_lab.road_vehicle.verify_artifact_set", side_effect=ValueError("checksum mismatch")):
            with self.assertRaises(ValueError):
                OpenVINORoadVehicleDetector("generated")
        self.assertEqual(self.events, [])
        for options in ({"device": "GPU"}, {"confidence_threshold": True}, {"confidence_threshold": 1.1},
                        {"max_vehicles": 0}, {"max_vehicles": 65}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                OpenVINORoadVehicleDetector("generated", **options)
        self.assertEqual(self.events, [])

    def test_layout_type_and_output_shape_guard_before_inference(self):
        self.input_shape = (1, 1024, 1024, 3)
        with self.assertRaisesRegex(RuntimeError, "input shape"):
            OpenVINORoadVehicleDetector("generated")
        self.input_shape, self.input_type = (1, 3, 1024, 1024), "u8"
        with self.assertRaisesRegex(RuntimeError, "float32 input"):
            OpenVINORoadVehicleDetector("generated")
        self.input_type, self.output_shape = "f32", (1, 1, 100, 7)
        with self.assertRaisesRegex(RuntimeError, "output shape"):
            OpenVINORoadVehicleDetector("generated")
        with patch.dict("sys.modules", {"openvino": SimpleNamespace(__version__="2025.0")}):
            with self.assertRaisesRegex(RuntimeError, "runtime version"):
                OpenVINORoadVehicleDetector("generated")
        self.assertNotIn("infer", self.events)

    def test_invalid_pixels_or_native_output_fail_closed(self):
        detector = OpenVINORoadVehicleDetector("generated")
        for image in (self.np.zeros((4, 8, 3), dtype=self.np.float32), self.np.zeros((4, 8), dtype=self.np.uint8)):
            with self.assertRaises(ValueError):
                detector.detect(image)
        self.assertEqual(detector.inference_count, 0)
        self.output = self.np.zeros((1, 1, 199, 7), dtype=self.np.float32)
        with self.assertRaises(RuntimeError):
            detector.detect(self.np.tile(self.np.array([7, 23, 241], dtype=self.np.uint8), (4, 8, 1)))


if __name__ == "__main__":
    unittest.main()
