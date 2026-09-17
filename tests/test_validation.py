from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest import mock

from analytics_lab.evaluation import LabeledPersonDown
from analytics_lab.openvino_pipeline import OpenVINOOMZPerceptionAdapter, OpenVINOOMZRunResult
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


def video_result(frames, observations, events, start, end):
    return VideoRunResult(frames, observations, events, start, end)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.a = self.root / "a.mp4"
        self.a.write_bytes(b"aa")
        self.b = self.root / "b.mp4"
        self.b.write_bytes(b"b")

    def spec(self, sample="s1", camera="c1", path=None, start=1000, end=5000, labels=(), site="site"):
        media = path or self.a
        digest = hashlib.sha256(media.read_bytes()).hexdigest() if media.exists() and not media.is_symlink() else "0" * 64
        return ValidationSampleSpec(sample, site, camera, "rights:test", media, digest, start, end, labels)

    def test_runs_and_binds_identity_metrics_and_runtime(self):
        calls = []
        results = [
            OpenVINOOMZRunResult("2026.3.1-test", "CPU", video_result(10, 8, (event(1000, 4000),), 1000, 5000)),
            OpenVINOOMZRunResult("2026.3.1-test", "CPU", video_result(20, 15, (), 6000, 10000)),
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
        self.assertEqual(out.preparation_elapsed_ms, 0.0)
        self.assertEqual(out.aggregate.sample_count, 2)
        self.assertEqual(len(out.model_artifacts), 4)
        self.assertEqual(out.sample_runs[0].authorization_ref, "rights:test")
        self.assertEqual(out.sample_runs[0].media_sha256, hashlib.sha256(self.a.read_bytes()).hexdigest())
        self.assertAlmostEqual(out.sample_runs[0].frames_per_second, 5.0)
        self.assertEqual(calls[0][1]["source_id"], "c1")
        self.assertEqual(calls[0][1]["session_id"], "s1")

    @mock.patch("analytics_lab.openvino_pipeline.run_local_video")
    @mock.patch("analytics_lab.openvino_omz.verify_artifact_set")
    def test_default_suite_prepares_runtime_once_and_resets_per_sample_adapter(self, verify, run_video):
        verify.return_value = ()
        run_video.side_effect = [
            video_result(1, 0, (), 1000, 5000),
            video_result(1, 0, (), 6000, 10000),
        ]
        factory_calls = []

        class Runtime:
            runtime_version = "2026.3.1-test"

        def runtime_factory(artifacts, device):
            factory_calls.append((artifacts, device))
            return Runtime()

        ticks = iter([0, 2_000_000, 2_000_000, 3_000_000, 3_000_000, 5_000_000])
        out = run_validation_suite(
            [self.spec(), self.spec("s2", "c2", self.b, 6000, 10000)],
            artifact_root="/models",
            runtime_factory=runtime_factory,
            clock_ns=lambda: next(ticks),
        )
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(verify.call_count, 1)
        self.assertEqual(run_video.call_count, 2)
        first = run_video.call_args_list[0].kwargs["perception"]
        second = run_video.call_args_list[1].kwargs["perception"]
        self.assertIsInstance(first, OpenVINOOMZPerceptionAdapter)
        self.assertIsInstance(second, OpenVINOOMZPerceptionAdapter)
        self.assertIsNot(first, second)
        self.assertIs(first.backend, second.backend)
        self.assertEqual(out.preparation_elapsed_ms, 2.0)
        self.assertEqual(out.total_elapsed_ms, 5.0)
        self.assertAlmostEqual(out.frames_per_second, 400.0)

    def test_validation_allows_same_camera_id_at_different_sites(self):
        samples = [
            self.spec("east", "cam-01", self.a, 1000, 5000, (), site="site-east"),
            self.spec("west", "cam-01", self.b, 1000, 5000, (), site="site-west"),
        ]
        results = iter([
            OpenVINOOMZRunResult("2026.3.1-test", "CPU", video_result(1, 0, (), 1000, 5000)),
            OpenVINOOMZRunResult("2026.3.1-test", "CPU", video_result(1, 0, (), 1000, 5000)),
        ])
        ticks = iter([0, 1, 1, 2])
        out = run_validation_suite(
            samples,
            artifact_root="/models",
            runner=lambda *args, **kwargs: next(results),
            clock_ns=lambda: next(ticks),
        )
        self.assertEqual(out.aggregate.camera_count, 2)
        self.assertEqual(
            [(item.site_id, item.camera_id) for item in out.aggregate.by_camera],
            [("site-east", "cam-01"), ("site-west", "cam-01")],
        )

    def test_uses_decoded_timestamp_span_not_declared_manifest_duration(self):
        sample = self.spec(start=1000, end=3_601_000)
        result = OpenVINOOMZRunResult(
            "2026.3.1-test",
            "CPU",
            video_result(2, 0, (event(1000, 2000),), 1000, 2000),
        )
        out = run_validation_suite(
            [sample],
            artifact_root="/models",
            runner=lambda *args, **kwargs: result,
            clock_ns=iter([0, 1]).__next__,
        )
        self.assertEqual(out.aggregate.duration_ms, 1000)
        self.assertEqual(out.aggregate.false_alerts, 1)
        self.assertAlmostEqual(out.aggregate.false_alerts_per_camera_hour, 3600.0)

    def test_rejects_labels_that_exceed_decoded_coverage(self):
        late = self.spec(labels=(LabeledPersonDown(3000, 4000, "late"),))
        short = OpenVINOOMZRunResult(
            "2026.3.1-test", "CPU", video_result(2, 0, (), 1000, 2000)
        )
        with self.assertRaises(ValueError):
            run_validation_suite(
                [late], artifact_root="/models", runner=lambda *a, **k: short,
                clock_ns=iter([0, 1]).__next__,
            )

    def test_exact_decoded_coverage_can_exceed_nominal_manifest_end(self):
        nominal = self.spec(start=1000, end=5000)
        decoded = OpenVINOOMZRunResult(
            "2026.3.1-test", "CPU", video_result(2, 0, (), 1000, 6000)
        )
        out = run_validation_suite(
            [nominal], artifact_root="/models", runner=lambda *a, **k: decoded,
            clock_ns=iter([0, 1]).__next__,
        )
        self.assertEqual(out.aggregate.duration_ms, 5000)
        self.assertAlmostEqual(out.aggregate.camera_hours, 5000 / 3_600_000.0)

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

    def test_rejects_mixed_runtime_device_zero_frames_bad_coverage_and_bad_clock(self):
        spec1 = self.spec()
        spec2 = self.spec("s2", "c2", self.b, 6000, 7000)
        mixed = [
            OpenVINOOMZRunResult("2026.3.1-a", "CPU", video_result(1, 0, (), 1000, 5000)),
            OpenVINOOMZRunResult("2026.3.1-b", "CPU", video_result(1, 0, (), 6000, 7000)),
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
                runner=lambda *a, **k: OpenVINOOMZRunResult("2026.3.1", "GPU", video_result(1, 0, (), 1000, 5000)),
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
                clock_ns=iter([0, 1]).__next__,
            )
        with self.assertRaises(RuntimeError):
            run_validation_suite(
                [spec1], artifact_root="/m",
                runner=lambda *a, **k: OpenVINOOMZRunResult("2026.3.1", "CPU", video_result(1, 0, (), 1000, 5000)),
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
