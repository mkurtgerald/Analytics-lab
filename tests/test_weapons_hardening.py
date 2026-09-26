import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from analytics_lab import weapons_hardening_evidence as evidence
from analytics_lab.tracking import DetectionCandidate, NormalizedBox


class WeaponsHardeningEvidenceTests(unittest.TestCase):
    def test_synthetic_frame_is_deterministic_300_square_uint8(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy optional outside Weapons evidence lane")
        first = evidence._synthetic_frame()
        second = evidence._synthetic_frame()
        self.assertEqual(first.shape, (300, 300, 3))
        self.assertEqual(first.dtype, np.uint8)
        self.assertTrue(np.array_equal(first, second))

    def test_work_dir_must_be_below_runner_temp(self):
        with tempfile.TemporaryDirectory() as root:
            runner = Path(root) / "runner"
            runner.mkdir()
            outside = Path(root) / "outside"
            with patch.dict(os.environ, {"RUNNER_TEMP": str(runner)}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "below RUNNER_TEMP"):
                    evidence._bounded_work_dir(outside)

    def test_constants_are_frozen(self):
        self.assertEqual(evidence._FIXED_THRESHOLD, 0.50)
        self.assertEqual(evidence._INFERENCE_RUNS, 3)
        self.assertEqual(evidence._EXPECTED_OPENVINO_VERSION, "2026.3.1")
        self.assertEqual(evidence._EXPECTED_OPENCV_VERSION, "4.12.0")

    def test_detection_validator_rejects_non_allowlisted_value(self):
        from analytics_lab.weapons_oid_ssd_smoke import _validate_detections
        candidate = DetectionCandidate(
            category="not-a-weapon",
            confidence=0.9,
            box=NormalizedBox(0.1, 0.1, 0.2, 0.2),
            model_class_id=999,
        )
        with self.assertRaisesRegex(RuntimeError, "non-allowlisted"):
            _validate_detections((candidate,))


if __name__ == "__main__":
    unittest.main()
