import tempfile
from pathlib import Path
import unittest

from analytics_lab.face_oid_ssd import (
    OID_V4_HUMAN_FACE_CLASS_ID,
    parse_oid_v4_detections,
)
from analytics_lab.face_privacy import (
    FaceDetection,
    FacePrivacyConfig,
    _bounded_blur_plan,
    apply_face_privacy,
)
from analytics_lab.face_runtime import OpenVINOOIDSSDRuntime
from analytics_lab.tracking import NormalizedBox


class FaceRuntimeLifecycleTests(unittest.TestCase):
    def test_convert_compile_once_and_reuse_for_repeated_detections(self):
        calls = {"convert": 0, "compile": 0, "factory": 0, "detect": 0}

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
                return ()

        def detector_factory(compiled, **kwargs):
            self.assertEqual(compiled, "compiled")
            self.assertEqual(kwargs["confidence_threshold"], 0.50)
            self.assertEqual(kwargs["max_faces"], 64)
            calls["factory"] += 1
            return Detector()

        self_outer = self
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "frozen_inference_graph.pb"
            path.write_bytes(b"synthetic-graph-fixture")
            runtime = OpenVINOOIDSSDRuntime.from_tensorflow_graph(
                path,
                core=Core(),
                convert_model=convert_model,
                detector_factory=detector_factory,
            )
            self.assertEqual(runtime.detect(object()), ())
            self.assertEqual(runtime.detect(object()), ())
            self.assertEqual(runtime.detect(object()), ())

        self.assertEqual(runtime.compile_count, 1)
        self.assertEqual(runtime.inference_count, 3)
        self.assertEqual(calls, {"convert": 1, "compile": 1, "factory": 1, "detect": 3})


class FaceOutputHardeningTests(unittest.TestCase):
    def test_zero_faces_is_valid(self):
        self.assertEqual(
            parse_oid_v4_detections(
                boxes=[[]],
                scores=[[]],
                classes=[[]],
                num_detections=[0.0],
            ),
            (),
        )

    def test_multiple_faces_admit_only_class_502(self):
        faces = parse_oid_v4_detections(
            boxes=[[
                [0.10, 0.10, 0.20, 0.20],
                [0.20, 0.20, 0.35, 0.35],
                [0.30, 0.30, 0.45, 0.45],
            ]],
            scores=[[0.90, 0.95, 0.80]],
            classes=[[502.0, 1.0, 502.0]],
            num_detections=[3.0],
        )
        self.assertEqual(len(faces), 2)
        self.assertEqual([item.confidence for item in faces], [0.90, 0.80])

    def test_max_face_bound_truncates_parser_and_rejects_invalid_configuration(self):
        count = 65
        boxes = [[[0.10, 0.10, 0.20, 0.20] for _ in range(count)]]
        scores = [[0.90 for _ in range(count)]]
        classes = [[float(OID_V4_HUMAN_FACE_CLASS_ID) for _ in range(count)]]
        faces = parse_oid_v4_detections(
            boxes=boxes,
            scores=scores,
            classes=classes,
            num_detections=[float(count)],
            max_faces=64,
        )
        self.assertEqual(len(faces), 64)
        with self.assertRaises(ValueError):
            parse_oid_v4_detections(
                boxes=boxes,
                scores=scores,
                classes=classes,
                num_detections=[float(count)],
                max_faces=65,
            )

    def test_short_malformed_outputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "shorter than num_detections"):
            parse_oid_v4_detections(
                boxes=[[[0.10, 0.10, 0.20, 0.20]]],
                scores=[[0.90]],
                classes=[[502.0]],
                num_detections=[2.0],
            )


class FacePrivacyHardeningTests(unittest.TestCase):
    def test_privacy_enforces_max_face_bound_before_blur(self):
        class Frame:
            shape = (100, 100, 3)

            def copy(self):
                return self

        detection = FaceDetection(0.90, NormalizedBox(0.10, 0.10, 0.20, 0.20))

        class Detector:
            def detect(self, _frame):
                return (detection,) * 65

        with self.assertRaisesRegex(RuntimeError, "face count exceeds configured bound"):
            apply_face_privacy(Frame(), Detector(), config=FacePrivacyConfig(max_faces=64))

    def test_large_gaussian_region_uses_bounded_direct_sigma(self):
        target_width, target_height, sigma_x, sigma_y = _bounded_blur_plan(
            width=4096,
            height=2160,
            sigma=320.0,
        )
        self.assertLess(target_width, 4096)
        self.assertLess(target_height, 2160)
        self.assertGreaterEqual(target_width, 1)
        self.assertGreaterEqual(target_height, 1)
        self.assertGreater(sigma_x, 0.0)
        self.assertGreater(sigma_y, 0.0)
        self.assertLessEqual(sigma_x, 32.0)
        self.assertLessEqual(sigma_y, 32.0)


if __name__ == "__main__":
    unittest.main()
