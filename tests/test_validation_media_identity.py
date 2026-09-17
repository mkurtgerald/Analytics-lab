from pathlib import Path
import hashlib
import tempfile
import unittest

from analytics_lab.openvino_pipeline import OpenVINOOMZRunResult
from analytics_lab.validation import ValidationSampleSpec, run_validation_suite
from analytics_lab.video import VideoRunResult


class ValidationMediaIdentityTests(unittest.TestCase):
    def test_rejects_media_changed_while_sample_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "authorized.mp4"
            path.write_bytes(b"authorized-media-v1")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            sample = ValidationSampleSpec(
                "sample-1", "site-1", "camera-1", "rights:test",
                path, digest, 1000, 5000, (),
            )

            def runner(video_path, **kwargs):
                video_path.write_bytes(b"different-media")
                return OpenVINOOMZRunResult(
                    "2026.3.1-test",
                    "CPU",
                    VideoRunResult(2, 0, (), 1000, 2000),
                )

            with self.assertRaisesRegex(RuntimeError, "validation media changed during sample execution"):
                run_validation_suite(
                    [sample],
                    artifact_root="/models",
                    runner=runner,
                    clock_ns=iter([0, 1]).__next__,
                )

    def test_accepts_unchanged_media_after_sample_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "authorized.mp4"
            path.write_bytes(b"authorized-media-v1")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            sample = ValidationSampleSpec(
                "sample-1", "site-1", "camera-1", "rights:test",
                path, digest, 1000, 5000, (),
            )
            result = OpenVINOOMZRunResult(
                "2026.3.1-test",
                "CPU",
                VideoRunResult(2, 0, (), 1000, 2000),
            )
            out = run_validation_suite(
                [sample],
                artifact_root="/models",
                runner=lambda *args, **kwargs: result,
                clock_ns=iter([0, 1]).__next__,
            )
            self.assertEqual(out.sample_runs[0].media_sha256, digest)


if __name__ == "__main__":
    unittest.main()
