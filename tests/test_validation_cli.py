import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analytics_lab.evaluation import EvaluationAggregate, EvaluationResult
from analytics_lab.validation import ModelArtifactIdentity, ValidationSampleRun, ValidationSuiteResult
from analytics_lab.validation_cli import load_manifest, main, result_document


class ValidationCliTests(unittest.TestCase):
    def _manifest(self, root: Path, *, extra_sample=None):
        video = root / "authorized.mp4"
        video.write_bytes(b"fixture")
        sample = {
            "sample_id": "sample-1",
            "site_id": "site-1",
            "camera_id": "camera-1",
            "authorization_ref": "rights-ticket-17",
            "video_path": str(video),
            "media_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "start_timestamp_ms": 1000,
            "end_timestamp_ms": 5000,
            "labels": [{"start_timestamp_ms": 1500, "end_timestamp_ms": 3500, "label_id": "down-1"}],
        }
        if extra_sample:
            sample.update(extra_sample)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "artifact_root": str(root / "models"),
            "config": {"required_device": "CPU", "max_samples": 4, "max_total_video_bytes": 1024},
            "samples": [sample],
        }), encoding="utf-8")
        return manifest, video

    @staticmethod
    def _result():
        evaluation = EvaluationResult(
            duration_ms=4000,
            positive_episodes=1,
            candidate_events=1,
            matched_events=1,
            missed_episodes=0,
            false_alerts=0,
            precision=1.0,
            recall=1.0,
            false_alerts_per_camera_hour=0.0,
            median_alert_delay_ms=500.0,
            matched_label_ids=("down-1",),
            alert_delays_ms=(500,),
        )
        aggregate = EvaluationAggregate(
            sample_count=1,
            site_count=1,
            camera_count=1,
            duration_ms=4000,
            camera_hours=4000 / 3_600_000.0,
            positive_episodes=1,
            candidate_events=1,
            matched_events=1,
            missed_episodes=0,
            false_alerts=0,
            precision=1.0,
            recall=1.0,
            false_alerts_per_camera_hour=0.0,
            median_alert_delay_ms=500.0,
            by_site=(),
            by_camera=(),
        )
        return ValidationSuiteResult(
            runtime_version="2026.3.1-test",
            device="CPU",
            model_artifacts=(ModelArtifactIdentity("detector", "detector.xml", 10, "a" * 96, "Apache-2.0"),),
            sample_runs=(ValidationSampleRun(
                sample_id="sample-1",
                site_id="site-1",
                camera_id="camera-1",
                authorization_ref="rights-ticket-17",
                media_sha256="b" * 64,
                runtime_version="2026.3.1-test",
                device="CPU",
                frames_processed=12,
                observations_processed=10,
                elapsed_ms=20.0,
                frames_per_second=600.0,
                evaluation=evaluation,
            ),),
            aggregate=aggregate,
            total_elapsed_ms=20.0,
            total_frames_processed=12,
            frames_per_second=600.0,
        )

    def test_load_manifest_builds_strict_rights_bound_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, video = self._manifest(root)
            artifact_root, samples, config = load_manifest(manifest)
            self.assertEqual(artifact_root, root / "models")
            self.assertEqual(config.required_device, "CPU")
            self.assertEqual(samples[0].video_path, video)
            self.assertEqual(samples[0].authorization_ref, "rights-ticket-17")
            self.assertEqual(samples[0].labels[0].label_id, "down-1")

    def test_manifest_rejects_unknown_fields_and_network_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _ = self._manifest(root, extra_sample={"unexpected": True})
            with self.assertRaises(ValueError):
                load_manifest(manifest)
            manifest, _ = self._manifest(root, extra_sample={"video_path": "https://example.invalid/video.mp4"})
            with self.assertRaises(ValueError):
                load_manifest(manifest)

    def test_result_document_contains_evidence_but_no_media_path(self):
        document = result_document(self._result())
        encoded = json.dumps(document, allow_nan=False, sort_keys=True)
        self.assertEqual(document["schema_version"], 1)
        self.assertIn("rights-ticket-17", encoded)
        self.assertIn("2026.3.1-test", encoded)
        self.assertIn("false_alerts_per_camera_hour", encoded)
        self.assertNotIn("video_path", encoded)
        self.assertNotIn("artifact_root", encoded)

    def test_main_runs_suite_and_emits_machine_readable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _ = self._manifest(root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("analytics_lab.validation_cli.run_validation_suite", return_value=self._result()) as runner:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = main(["--manifest", str(manifest)])
            self.assertEqual(code, 0)
            self.assertEqual(stderr.getvalue(), "")
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["validation"]["device"], "CPU")
            self.assertEqual(payload["validation"]["aggregate"]["missed_episodes"], 0)
            runner.assert_called_once()

    def test_main_error_does_not_echo_untrusted_paths(self):
        secret_path = "/tmp/secret-customer-video.mp4"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["--manifest", secret_path])
        self.assertEqual(code, 2)
        self.assertNotIn(secret_path, stderr.getvalue())
        self.assertIn("Validation rejected", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
