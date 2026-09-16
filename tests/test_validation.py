from pathlib import Path
import hashlib
import tempfile
import unittest

from analytics_lab.evaluation import LabeledPersonDown
from analytics_lab.openvino_pipeline import OpenVINOOMZRunResult
from analytics_lab.validation import ValidationSampleSpec, ValidationSuiteConfig, run_validation_suite
from analytics_lab.video import VideoRunResult


def event(start, end):
    return {
        "event_type": "person_down_candidate",
        "status": "candidate",
        "evidence": [{
            "type": "observation_window",
            "start_timestamp_ms": start,
            "end_timestamp_ms": end,
        }],
    }


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.a = self.root / "a.mp4"
        self.a.write_bytes(b"aa")
        self.b = self.root / "b.mp4"
        self.b.write_bytes(b"b")

    def spec(self, sample="s1", camera="c1", path=None, start=1000, end=5000, labels=()):
        media = path or self.a
        digest = hashlib.sha256(media.read_bytes()).hexdigest() if media.exists() and not media.is_symlink() else "0" * 64
        return ValidationSampleSpec(sample, "site", camera, "rights:test", media, digest, start, end, labels)

    def test_runs_and_binds_identity_metrics_and_runtime(self):
        calls = []
        results = [
            OpenVINOOMZRunResult("2026.3.1-test", "CPU", VideoRunResult(10, 8, (event(1000, 4000),))),
            OpenVINOOMZRunResult("2026.3.1-test", "CPU", VideoRunResult(20, 15, ())),
        ]

        def runner(path, **kwargs):
            calls.append((path, kwargs))
            return results[len(calls) - 1]

        ticks = iter([0, 2_000_000_000, 2_000_000_000, 6_000_000_000])
        one = self.spec(labels=(LabeledPersonDown(1000, 4500, "p1"),))
        two = self.spec("s2", "c2", self.b, 6000, 10000, ())
        out = run_validation_suite([one, two], artifact_root="/models", runner=runner, clock_ns=lambda: next(ticks))
        self.assertEqual(out.runtime_version, "2026.3.1-test")
        self.assertEqual(out.device, "CPU")
        self.assertEqual(out.total_frames_processed, 30)
        self.assertAlmostEqual(out.frames_per_second, 5.0)
        self.assertEqual(out.aggregate.sample_count, 2)
        self.assertEqual(len(out.model_artifacts), 4)
        self.assertEqual(out.sample_runs[0].authorization_ref, "rights:test")
        self.assertEqual(out.sample_runs[0].media_sha256, hashlib.sha256(self.a.read_bytes()).hexdigest())
        self.assertAlmostEqual(out.sample_runs[0].frames_per_second, 5.0)
        self.assertEqual(calls[0][1]["source_id"], "c1")
        self.assertEqual(calls[0][1]["session_id"], "s1")

    def test_preflight_rejects_missing_rights_urls_duplicates_and_overlap_before_runner(self):
        digest = hashlib.sha256(self.a.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            ValidationSampleSpec("s", "site", "cam", "", self.a, digest, 0, 1, ())
        with self.assertRaises(ValueError):
            ValidationSampleSpec("s", "site", "cam", "https://rights", self.a, digest, 0, 1, ())
        with self.assertRaises(ValueError):
            ValidationSampleSpec("s", "site", "cam", "r", Path("https://x"), "0" * 64, 0, 1, ())
        called = False

        def runner(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError

        duplicate = [self.spec("s", "c1"), self.spec("s", "c2", self.b, 6000, 7000)]
        with self.assertRaises(ValueError):
            run_validation_suite(duplicate, artifact_root="/m", runner=runner)
        overlap = [self.spec("s1", "c1"), self.spec("s2", "c1", self.b, 4000, 7000)]
        with self.assertRaises(ValueError):
            run_validation_suite(overlap, artifact_root="/m", runner=runner)
        self.assertFalse(called)

    def test_preflight_rejects_symlink_missing_checksum_and_byte_budget_before_runner(self):
        link = self.root / "link.mp4"
        link.symlink_to(self.a)
        with self.assertRaises(ValueError):
            run_validation_suite([self.spec(path=link)], artifact_root="/m", runner=lambda *a, **k: None)
        with self.assertRaises(ValueError):
            run_validation_suite([self.spec(path=self.root / "missing.mp4")], artifact_root="/m", runner=lambda *a, **k: None)
        bad = ValidationSampleSpec("bad", "site", "c1", "rights:test", self.a, "f" * 64, 1000, 5000, ())
        with self.assertRaises(ValueError):
            run_validation_suite([bad], artifact_root="/m", runner=lambda *a, **k: None)
        with self.assertRaises(RuntimeError):
            run_validation_suite(
                [self.spec()], artifact_root="/m",
                config=ValidationSuiteConfig(max_total_video_bytes=1),
                runner=lambda *a, **k: None,
            )

    def test_rejects_mixed_runtime_device_zero_frames_and_bad_clock(self):
        spec1 = self.spec()
        spec2 = self.spec("s2", "c2", self.b, 6000, 7000)
        mixed = [
            OpenVINOOMZRunResult("2026.3.1-a", "CPU", VideoRunResult(1, 0, ())),
            OpenVINOOMZRunResult("2026.3.1-b", "CPU", VideoRunResult(1, 0, ())),
        ]
        results = iter(mixed)
        ticks = iter([0, 1, 1, 2])
        with self.assertRaises(RuntimeError):
            run_validation_suite(
                [spec1, spec2], artifact_root="/m",
                runner=lambda *a, **k: next(results), clock_ns=lambda: next(ticks),
            )
        with self.assertRaises(RuntimeError):
            run_validation_suite(
                [spec1], artifact_root="/m",
                runner=lambda *a, **k: OpenVINOOMZRunResult("2026.3.1", "GPU", VideoRunResult(1, 0, ())),
                clock_ns=iter([0, 1]).__next__,
            )
        with self.assertRaises(RuntimeError):
            run_validation_suite(
                [spec1], artifact_root="/m",
                runner=lambda *a, **k: OpenVINOOMZRunResult("2026.3.1", "CPU", VideoRunResult(0, 0, ())),
                clock_ns=iter([0, 1]).__next__,
            )
        with self.assertRaises(RuntimeError):
            run_validation_suite(
                [spec1], artifact_root="/m",
                runner=lambda *a, **k: OpenVINOOMZRunResult("2026.3.1", "CPU", VideoRunResult(1, 0, ())),
                clock_ns=iter([2, 1]).__next__,
            )

    def test_label_bounds_and_sample_limit(self):
        with self.assertRaises(ValueError):
            self.spec(labels=(LabeledPersonDown(0, 2000, "x"),))
        with self.assertRaises(RuntimeError):
            run_validation_suite(
                [self.spec(), self.spec("s2", "c2", self.b, 6000, 7000)],
                artifact_root="/m",
                config=ValidationSuiteConfig(max_samples=1),
                runner=lambda *a, **k: None,
            )


if __name__ == "__main__":
    unittest.main()
