from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from analytics_lab.figshare_member_admission import AdmittedMember
from analytics_lab.figshare_person_down_diagnostic import (
    _EXPECTED_SHA256,
    run_figshare_person_down_diagnostic,
)


class FigsharePersonDownDiagnosticTests(unittest.TestCase):
    def _member(self, role: str, path: Path) -> AdmittedMember:
        return AdmittedMember(
            role=role,
            name=(
                "VideoDataset/ADL/SBJ_01_LOC3/ACT25_R_1/20240923130459.mp4"
                if role == "negative"
                else "VideoDataset/Fall/SBJ_10_LOC3/ACT10_R_2/20240915184434.mp4"
            ),
            compression_method=8,
            compressed_size=10,
            uncompressed_size=20,
            crc32=1,
            sha256=_EXPECTED_SHA256[role],
            local_path=path,
        )

    def _scan_result(self, sample_id: str, *, false_alerts: int) -> dict:
        return {
            "sample_id": sample_id,
            "authorization_ref": "figshare:28596332:v2:CC-BY-4.0:evaluation",
            "media_sha256": "0" * 64,
            "decoded_start_timestamp_ms": 0,
            "decoded_end_timestamp_ms": 4000,
            "decoded_duration_ms": 4000,
            "frames_processed": 120,
            "associated_frames": 100,
            "candidate_events": false_alerts,
            "evaluation": {
                "duration_ms": 4000,
                "positive_episodes": 0,
                "candidate_events": false_alerts,
                "matched_events": 0,
                "missed_episodes": 0,
                "false_alerts": false_alerts,
                "precision": None,
                "recall": None,
                "false_alerts_per_camera_hour": false_alerts * 900.0,
                "median_alert_delay_ms": None,
                "matched_label_ids": (),
                "alert_delays_ms": (),
            },
            "continuity": {"coverage": 1.0},
            "selection_windows": {"overall": {"frames": 120, "selected_frames": 120, "coverage": 1.0}},
            "orientation_fallback": {"accepted_frames": 0},
            "temporal_trace": {"longest_qualified_down_run_ms": 0},
            "fragmentation_windows": {"overall": {}},
            "detector_inference_ms": 10.0,
            "pose_inference_ms": 20.0,
            "pose_decode_ms": 5.0,
            "elapsed_ms": 100.0,
            "throughput_fps": 1200.0,
        }

    @patch("analytics_lab.figshare_person_down_diagnostic.ReferenceOpenPoseDecoder")
    @patch("analytics_lab.figshare_person_down_diagnostic._ReferenceRuntime")
    @patch("analytics_lab.figshare_person_down_diagnostic._OpenVINORuntime")
    @patch("analytics_lab.figshare_person_down_diagnostic.verify_artifact_set")
    @patch("analytics_lab.figshare_person_down_diagnostic._CompiledDetector")
    @patch("analytics_lab.figshare_person_down_diagnostic._provision")
    @patch("analytics_lab.figshare_person_down_diagnostic._target_candidate")
    @patch("analytics_lab.figshare_person_down_diagnostic._download_models")
    @patch("analytics_lab.figshare_person_down_diagnostic._scan_sample")
    @patch("analytics_lab.figshare_person_down_diagnostic.admit_pinned_pair")
    def test_negative_is_scored_and_positive_stays_unscored(
        self,
        admit,
        scan,
        download_models,
        target_candidate,
        provision,
        compiled_detector,
        verify_artifacts,
        openvino_runtime,
        reference_runtime,
        decoder,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            negative = self._member("negative", root / "negative.mp4")
            positive = self._member("positive", root / "positive.mp4")
            admit.return_value = (negative, positive)
            download_models.return_value = root / "artifacts"
            candidate = Mock(name="candidate")
            candidate.name = "person-detection-0200"
            target_candidate.return_value = candidate
            runtime = Mock()
            runtime.runtime_version = "2026.3.1-test"
            openvino_runtime.return_value = runtime
            scan.side_effect = [
                (self._scan_result("figshare-act25-negative", false_alerts=0), Mock()),
                (self._scan_result("figshare-act10-positive", false_alerts=1), Mock()),
            ]

            result = run_figshare_person_down_diagnostic(
                root / "output",
                root / "detector",
                opener=Mock(),
            )

        by_role = {item["role"]: item for item in result["samples"]}
        self.assertEqual(result["negative_false_alerts"], 0)
        self.assertAlmostEqual(result["negative_decoded_camera_hours"], 4000 / 3_600_000)
        self.assertEqual(by_role["negative"]["evaluation"]["false_alerts"], 0)
        self.assertTrue(by_role["negative"]["evaluation_policy"]["false_alerts_scored"])
        self.assertEqual(by_role["positive"]["evaluation"]["status"], "unscored-positive-clip")
        self.assertFalse(by_role["positive"]["evaluation"]["match_miss_scored"])
        self.assertFalse(result["positive_match_miss_scored"])
        self.assertFalse(result["positive_alert_delay_scored"])
        self.assertFalse(result["thresholds_or_temporal_rules_changed"])
        self.assertFalse(result["commercial_accuracy_claim"])
        specs = [call.args[0] for call in scan.call_args_list]
        self.assertEqual([spec.labels for spec in specs], [(), ()])
        self.assertEqual([spec.end_timestamp_ms for spec in specs], [1, 1])
        self.assertEqual([item["source_activity"]["activity_code"] for item in result["samples"]], ["ACT25", "ACT10"])

    @patch("analytics_lab.figshare_person_down_diagnostic._download_models")
    @patch("analytics_lab.figshare_person_down_diagnostic.admit_pinned_pair")
    def test_sha_identity_change_fails_before_model_download(self, admit, download_models) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = AdmittedMember(
                role="negative",
                name="VideoDataset/ADL/SBJ_01_LOC3/ACT25_R_1/20240923130459.mp4",
                compression_method=8,
                compressed_size=10,
                uncompressed_size=20,
                crc32=1,
                sha256="f" * 64,
                local_path=root / "bad.mp4",
            )
            positive = self._member("positive", root / "positive.mp4")
            admit.return_value = (bad, positive)
            with self.assertRaises(RuntimeError):
                run_figshare_person_down_diagnostic(root / "output", root / "detector", opener=Mock())
        download_models.assert_not_called()


if __name__ == "__main__":
    unittest.main()
