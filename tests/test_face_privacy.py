import hashlib
from pathlib import Path
import tempfile
import unittest

from analytics_lab.face_privacy import (
    FaceDetection, FacePrivacyConfig, OpenCVYuNetDetector,
    apply_face_privacy, parse_yunet_rows, verify_face_model_artifact,
)
from analytics_lab.tracking import NormalizedBox


class FakeFrame:
    shape = (100, 200, 3)
    def copy(self): return self


class FakeYuNet:
    def setInputSize(self, size): self.size = size
    def detect(self, _frame):
        return 1, [[20.0, 10.0, 50.0, 40.0] + [0.0] * 10 + [0.95]]


class FacePrivacyContractTests(unittest.TestCase):
    def test_yunet_rows_become_normalized_faces(self):
        rows = [
            [20.0, 10.0, 50.0, 40.0] + [0.0] * 10 + [0.95],
            [5.0, 5.0, 10.0, 10.0] + [0.0] * 10 + [0.50],
        ]
        faces = parse_yunet_rows(rows, width=200, height=100, confidence_threshold=0.9)
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].box, NormalizedBox(0.1, 0.1, 0.35, 0.5))

    def test_model_artifact_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "face.onnx"
            path.write_bytes(b"face-model-fixture")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            identity = verify_face_model_artifact(path, expected_sha256=digest)
            self.assertEqual(identity.size_bytes, len(b"face-model-fixture"))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_face_model_artifact(path, expected_sha256="0" * 64)

    def test_yunet_adapter_uses_verified_external_artifact_and_factory(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "face.onnx"
            path.write_bytes(b"face-model-fixture")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            fake = FakeYuNet()
            calls = []
            def factory(model, config, size, score, nms, top_k):
                calls.append((model, config, size, score, nms, top_k))
                return fake
            detector = OpenCVYuNetDetector(path, expected_sha256=digest, factory=factory)
            faces = detector.detect(FakeFrame())
            self.assertEqual(len(faces), 1)
            self.assertEqual(calls[0][2], (200, 100))

    def test_invalid_values_fail_closed(self):
        with self.assertRaises(ValueError):
            FacePrivacyConfig(expand_ratio=0.9)
        with self.assertRaises(ValueError):
            FaceDetection(1.2, NormalizedBox(0.1, 0.1, 0.2, 0.2))


class FacePrivacyOpenCVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2  # noqa: F401
            import numpy as np  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("OpenCV/NumPy optional outside face evidence lane")

    def _frame(self):
        import numpy as np
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for y in range(120):
            for x in range(160):
                value = 255 if ((x // 3) + (y // 3)) % 2 else 0
                frame[y, x] = (value, 255 - value, value)
        return frame

    def _detector(self):
        class Detector:
            def detect(self, _frame):
                return (FaceDetection(0.99, NormalizedBox(0.25, 0.25, 0.75, 0.75)),)
        return Detector()

    def test_default_blur_changes_face_region_without_mutating_input(self):
        import numpy as np
        frame = self._frame()
        original = frame.copy()
        result = apply_face_privacy(frame, self._detector())
        self.assertEqual(result.audit.action, "blur")
        self.assertTrue(np.array_equal(frame, original))
        self.assertTrue(np.array_equal(result.frame_bgr[:10, :10], original[:10, :10]))
        self.assertFalse(np.array_equal(result.frame_bgr[40:80, 60:100], original[40:80, 60:100]))

    def test_unblur_is_permission_gated(self):
        import numpy as np
        frame = self._frame()
        denied = apply_face_privacy(frame, self._detector(), request_unblur=True, authorized_unblur=False)
        allowed = apply_face_privacy(frame, self._detector(), request_unblur=True, authorized_unblur=True)
        self.assertEqual(denied.audit.action, "unblur_denied")
        self.assertFalse(np.array_equal(denied.frame_bgr, frame))
        self.assertEqual(allowed.audit.action, "unblur")
        self.assertTrue(np.array_equal(allowed.frame_bgr, frame))


if __name__ == "__main__":
    unittest.main()
