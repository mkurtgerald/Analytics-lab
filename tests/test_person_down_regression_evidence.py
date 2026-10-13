"""Offline metadata tests; no publisher media, models or native runtime runs."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analytics_lab import person_down_regression_evidence as evidence
from analytics_lab.artifacts import OPENVINO_OMZ_2023_FP16
from analytics_lab.evaluation import (
    EvaluationConfig, EvaluationSample, LabeledPersonDown,
    aggregate_person_down_evaluations, evaluate_person_down_candidates,
)
from analytics_lab.openvino_pipeline import OpenVINOOMZPipelineConfig
from analytics_lab.validation import ModelArtifactIdentity, ValidationSampleRun, ValidationSuiteResult
from analytics_lab.validation_seed import _manifest
from analytics_lab.video import VideoRunConfig


def _blob_sha1(payload):
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


class _Response(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.requests = []

    def read(self, count=-1):
        self.requests.append(count)
        return super().read(count)


class PersonDownRegressionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.responses = []
        self.requests = []
        self.prepared_directories = []
        self.media = {}
        self.specs = tuple(
            replace(spec, size_bytes=len(payload), git_blob_sha1=_blob_sha1(payload))
            for spec, payload in (
                (spec, f"generated metadata only:{spec.sample_id}".encode("ascii"))
                for spec in evidence.GMDCSA24_HELDOUT_S4
            )
        )
        for spec in self.specs:
            self.media[spec.sample_id] = f"generated metadata only:{spec.sample_id}".encode("ascii")
        self.payloads = {
            table["path"]: (evidence._HEADER + "\n" + table["row"] + "\n").encode("utf-8")
            for table in evidence._TABLES
        }
        self.tables = self._generated_tables(self.payloads)

    def _generated_tables(self, payloads):
        return tuple({
            **table, "size_bytes": len(payloads[table["path"]]),
            "git_blob_sha1": _blob_sha1(payloads[table["path"]]),
            "sha256": hashlib.sha256(payloads[table["path"]]).hexdigest(),
        } for table in evidence._TABLES)

    def _opener(self, request, timeout):
        self.assertEqual(timeout, 30)
        self.requests.append(request.full_url)
        path = next(path for path in self.payloads if request.full_url.endswith(path.replace(" ", "%20")))
        response = _Response(self.payloads[path])
        self.responses.append(response)
        return response

    def _prepare(self, directory, *, opener, subset):
        self.assertEqual(subset, "subject4-regression")
        self.assertEqual(opener, self._opener)
        directory = Path(directory)
        self.prepared_directories.append(directory)
        self.assertTrue(directory.is_dir())
        identities = {}
        for spec in self.specs:
            target = directory / "media" / "gmdcsa24" / spec.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.media[spec.sample_id])
            identities[spec.sample_id] = hashlib.sha256(self.media[spec.sample_id]).hexdigest()
        (directory / "artifacts").mkdir()
        # Exercises cleanup of all prepared artifacts without materializing weights.
        (directory / "artifacts" / "generated-metadata.txt").write_text("not a model", encoding="utf-8")
        document = _manifest(identities, self.specs)
        manifest = directory / "validation-manifest.json"
        manifest.write_text(json.dumps(document), encoding="utf-8")
        return manifest

    def _result(self, samples, *, negative_event=False, negative_duration=3333, **kwargs):
        runs = []
        evaluations = []
        for index, sample in enumerate(samples):
            duration = sample.end_timestamp_ms + 100 if index == 0 else negative_duration
            events = ()
            if index == 1 and negative_event:
                events = ({"event_type": "person_down_candidate", "status": "candidate", "evidence": [{
                    "type": "observation_window", "start_timestamp_ms": 0,
                    "end_timestamp_ms": 1000,
                }]},)
            evaluation = evaluate_person_down_candidates(
                events, sample.labels, video_start_timestamp_ms=0, video_end_timestamp_ms=duration,
            )
            evaluations.append(EvaluationSample(sample.sample_id, sample.site_id, sample.camera_id, 0, duration, evaluation))
            runs.append(ValidationSampleRun(
                sample.sample_id, sample.site_id, sample.camera_id, sample.authorization_ref,
                sample.media_sha256, "2026.3.1", "CPU", 2, 0, 1.0, 2000.0, evaluation,
            ))
        identities = tuple(ModelArtifactIdentity(
            spec.component, spec.relative_path, spec.size_bytes, spec.sha384, spec.license_id,
        ) for spec in OPENVINO_OMZ_2023_FP16)
        return ValidationSuiteResult(
            "2026.3.1", "CPU", identities, tuple(runs),
            aggregate_person_down_evaluations(evaluations), 2.0, 4, 2000.0,
        )

    def _run(self, *, prepare=None, suite=None, tables=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(evidence, "_TABLES", self.tables if tables is None else tables))
            stack.enter_context(patch.object(evidence, "GMDCSA24_HELDOUT_S4", self.specs))
            stack.enter_context(patch.object(evidence, "_runtime_versions", return_value={
                "openvino": "2026.3.1", "opencv": "4.12.0", "numpy": "2.2.6",
            }))
            self.prepare_call = stack.enter_context(patch.object(evidence, "prepare_seed", side_effect=prepare or self._prepare))
            self.suite_call = stack.enter_context(patch.object(evidence, "run_validation_suite", side_effect=suite or self._result))
            return evidence.run_regression(self.root, opener=self._opener)

    def _reject_manifest_change(self, change):
        def prepare(*args, **kwargs):
            manifest = self._prepare(*args, **kwargs)
            document = json.loads(manifest.read_text(encoding="utf-8"))
            change(document, manifest.parent)
            manifest.write_text(json.dumps(document), encoding="utf-8")
            return manifest
        with self.assertRaises((RuntimeError, ValueError)):
            self._run(prepare=prepare)
        self.suite_call.assert_not_called()
        self.assertTrue(self.prepared_directories)
        self.assertFalse(self.prepared_directories[-1].exists())

    def test_publisher_pins_and_rows_match_selected_subject4_media(self):
        self.assertEqual([table["path"] for table in evidence._TABLES], ["Subject 4/Fall.csv", "Subject 4/ADL.csv"])
        self.assertEqual([table["size_bytes"] for table in evidence._TABLES], [2399, 3092])
        self.assertEqual([table["git_blob_sha1"] for table in evidence._TABLES], [
            "31c1625bbf51eff2e5bd767455c400f50f5786e4", "4d97c1ba3b3a43290529ae5c7be236322706816d",
        ])
        for spec, table in zip(evidence.GMDCSA24_HELDOUT_S4, evidence._TABLES):
            self.assertEqual(spec.sample_id, table["sample_id"])
            self.assertEqual(Path(spec.relative_path).name, table["row"].split(",", 1)[0])
            interval = [spec.label_start_timestamp_ms, spec.label_end_timestamp_ms] if spec.label_id else None
            self.assertEqual(interval, table["fall_activity_interval_ms"])

    def test_complete_tables_are_bounded_and_immutable_identity_verified(self):
        with patch.object(evidence, "_TABLES", self.tables):
            evidence.verify_annotations(opener=self._opener)
        self.assertEqual(len(self.responses), 2)
        for response, table, url in zip(self.responses, self.tables, self.requests):
            self.assertEqual(response.requests, [table["size_bytes"] + 1])
            self.assertTrue(response.closed)
            self.assertIn("/" + evidence._COMMIT + "/", url)
            self.assertIn("Subject%204/", url)

    def test_annotation_overread_truncation_and_same_size_tamper_precede_preparation(self):
        original = self.payloads.copy()
        path = self.tables[0]["path"]
        for payload in (original[path] + b"x", original[path][:-1], original[path].replace(b"Walking", b"Wxlking", 1)):
            with self.subTest(payload_size=len(payload)):
                self.payloads = {**original, path: payload}
                with self.assertRaises(RuntimeError):
                    self._run()
                self.prepare_call.assert_not_called()
                self.suite_call.assert_not_called()
                self.assertEqual(list(self.root.iterdir()), [])

    def test_whole_table_identity_checks_unselected_rows_too(self):
        path = self.tables[0]["path"]
        self.payloads[path] += b"99.mp4,01,Generated,Generated,Generated, Standing[0 to 1]\n"
        tables = self._generated_tables(self.payloads)
        self.payloads[path] = self.payloads[path].replace(b"99.mp4", b"98.mp4")
        with self.assertRaises(RuntimeError):
            self._run(tables=tables)
        self.prepare_call.assert_not_called()
        self.suite_call.assert_not_called()

    def test_rehashed_changed_selected_row_duplicate_and_header_are_rejected(self):
        original = self.payloads.copy()
        path = self.tables[0]["path"]
        row = self.tables[0]["row"].encode("utf-8")
        variants = (original[path].replace(b"[1.7 to 6]", b"[1.8 to 6]"),
                    original[path] + row + b"\n", original[path].replace(b"File Name", b"File Xame"))
        for payload in variants:
            with self.subTest(payload=payload):
                self.payloads = {**original, path: payload}
                with self.assertRaises(RuntimeError):
                    self._run(tables=self._generated_tables(self.payloads))
                self.prepare_call.assert_not_called()
                self.suite_call.assert_not_called()

    def test_one_prepared_seed_and_one_frozen_suite_call_preserve_silent_positive_miss(self):
        report = self._run()
        self.prepare_call.assert_called_once()
        self.suite_call.assert_called_once()
        self.assertEqual(len(self.prepared_directories), 1)
        kwargs = self.suite_call.call_args.kwargs
        self.assertEqual(kwargs["pipeline_config"], OpenVINOOMZPipelineConfig(video=VideoRunConfig(max_frames=1000)))
        self.assertEqual(kwargs["evaluation_config"], EvaluationConfig())
        self.assertEqual(asdict(kwargs["config"]), {
            "max_samples": 2, "required_device": "CPU",
            "max_total_video_bytes": sum(spec.size_bytes for spec in self.specs),
        })
        self.assertEqual(report["fall_activity_candidate_matches"], 0)
        self.assertEqual(report["fall_activity_candidate_misses"], 1)
        self.assertEqual(report["activity_proxy_validation"]["aggregate"]["recall"], 0.0)
        self.assertIsNone(report["activity_proxy_validation"]["aggregate"]["precision"])
        self.assertNotIn("validation", report)
        self.assertFalse(report["false_alert_gate_eligible"])
        self.assertFalse(report["zero_candidates_is_success"])
        self.assertFalse(report["commercial_accuracy_claim"])
        self.assertFalse(report["production_promoted"])
        self.assertFalse(report["new_heldout_evidence"])
        self.assertIsNone(report["baseline_delta"])
        self.assertFalse(self.prepared_directories[0].exists())
        serialized = json.dumps(report, allow_nan=False, sort_keys=True)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("generated metadata only:", serialized)
        self.assertEqual(report["plan_sha256"], hashlib.sha256(evidence._canonical(report["plan"])).hexdigest())

    def test_negative_exposure_uses_actual_duration_and_preserves_false_alert(self):
        report = self._run(suite=lambda samples, **kwargs: self._result(samples, negative_event=True, **kwargs))
        self.assertEqual(report["normal_clip_decoded_exposure_ms"], 3333)
        self.assertAlmostEqual(report["normal_clip_decoded_camera_hours"], 3333 / 3600000)
        self.assertEqual(report["normal_clip_nominal_annotated_ms"], 4000)
        self.assertEqual(report["normal_clip_unannotated_tail_ms"], 0)
        self.assertEqual(report["normal_clip_observed_annotated_activity_ms"], 3333)
        self.assertEqual(report["normal_clip_scored_activity_false_alerts"], 1)
        self.assertAlmostEqual(report["normal_clip_scored_activity_camera_hours"], 3333 / 3600000)
        self.assertFalse(report["false_alert_gate_eligible"])
        self.assertEqual(report["normal_clip_candidate_events"], 1)
        self.assertEqual(report["activity_proxy_validation"]["aggregate"]["false_alerts"], 1)
        self.assertEqual(report["activity_proxy_validation"]["aggregate"]["missed_episodes"], 1)

    def test_unannotated_negative_tail_is_disclosed_without_extending_annotation(self):
        report = self._run(suite=lambda samples, **kwargs: self._result(
            samples, negative_duration=4500, negative_event=True, **kwargs,
        ))
        self.assertEqual(report["normal_clip_decoded_exposure_ms"], 4500)
        self.assertEqual(report["normal_clip_nominal_annotated_ms"], 4000)
        self.assertEqual(report["normal_clip_unannotated_tail_ms"], 500)
        self.assertEqual(report["normal_clip_observed_annotated_activity_ms"], 4000)
        self.assertEqual(report["normal_clip_candidate_events"], 1)
        self.assertAlmostEqual(report["normal_clip_decoded_camera_hours"], 4500 / 3600000)
        self.assertIsNone(report["normal_clip_scored_activity_false_alerts"])
        self.assertIsNone(report["normal_clip_scored_activity_camera_hours"])
        self.assertFalse(report["false_alert_gate_eligible"])

    def test_cleanup_after_seed_native_and_result_contract_failure(self):
        for phase in ("seed", "native", "result"):
            with self.subTest(phase=phase):
                def prepare(*args, **kwargs):
                    manifest = self._prepare(*args, **kwargs)
                    if phase == "seed":
                        raise RuntimeError("generated failure")
                    return manifest
                def suite(*args, **kwargs):
                    if phase == "native":
                        raise RuntimeError("generated failure")
                    return object()
                with self.assertRaises(RuntimeError):
                    self._run(prepare=prepare, suite=suite)
                self.assertFalse(self.prepared_directories[-1].exists())
                self.assertEqual(list(self.root.iterdir()), [])

    def test_sample_order_count_ids_and_labels_fail_before_native(self):
        changes = (
            lambda doc, _: doc["samples"].reverse(),
            lambda doc, _: doc["samples"].pop(),
            lambda doc, _: doc["samples"].append(doc["samples"][1].copy()),
            lambda doc, _: doc["samples"][0].update(sample_id="changed"),
            lambda doc, _: doc["samples"][0]["labels"][0].update(start_timestamp_ms=1701),
            lambda doc, _: doc["samples"][0]["labels"][0].update(label_id="changed"),
            lambda doc, _: doc["samples"][0].update(labels=[]),
            lambda doc, _: doc["samples"][1].update(labels=[{"start_timestamp_ms": 0, "end_timestamp_ms": 1, "label_id": "invented"}]),
            lambda doc, _: doc["samples"][1].update(start_timestamp_ms=1),
            lambda doc, _: doc["samples"][1].update(end_timestamp_ms=4001),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self._reject_manifest_change(change)

    def test_source_rights_camera_site_path_and_digest_fail_before_native(self):
        changes = (
            lambda doc, _: doc["samples"][0].update(authorization_ref="different-rights"),
            lambda doc, _: doc["samples"][0].update(site_id="different-site"),
            lambda doc, _: doc["samples"][0].update(camera_id="different-camera"),
            lambda doc, _: doc["samples"][0].update(video_path=doc["samples"][1]["video_path"]),
            lambda doc, _: doc["samples"][0].update(media_sha256="0" * 64),
            lambda doc, _: doc.update(artifact_root="changed-artifacts"),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self._reject_manifest_change(change)

    def test_rehashed_replacement_media_is_not_accepted_as_the_pinned_source(self):
        def change(document, directory):
            row = document["samples"][0]
            path = directory / row["video_path"]
            changed = path.read_bytes().replace(b"generated", b"invented ", 1)
            path.write_bytes(changed)
            row["media_sha256"] = hashlib.sha256(changed).hexdigest()
        self._reject_manifest_change(change)

    def test_suite_configuration_drift_fails_before_native(self):
        for field, value in (("max_samples", 3), ("required_device", "GPU"), ("max_total_video_bytes", 99999)):
            with self.subTest(field=field):
                self._reject_manifest_change(lambda doc, _, field=field, value=value: doc["config"].update({field: value}))

    def test_plan_annotation_lists_do_not_mutate_future_plans(self):
        plan = evidence.evidence_plan()
        plan["annotations"][0]["fall_activity_interval_ms"][0] = 0
        self.assertEqual(evidence.evidence_plan()["annotations"][0]["fall_activity_interval_ms"], [1700, 6000])

    def test_cli_rejects_nonhosted_selfhosted_and_outside_temp_before_runtime(self):
        for environment, path in (
            ({}, self.root / "child"),
            ({"GITHUB_ACTIONS": "false", "RUNNER_TEMP": str(self.root)}, self.root / "child"),
            ({"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "self-hosted", "RUNNER_TEMP": str(self.root)}, self.root / "child"),
            ({"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_TEMP": str(self.root)}, self.root.parent / "outside-regression"),
            ({"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_TEMP": str(self.root)}, self.root),
        ):
            with self.subTest(environment=environment, path=path), patch.dict(evidence.os.environ, environment, clear=True), \
                    patch.object(evidence, "_runtime_versions") as versions, patch.object(evidence, "run_regression") as run, \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(evidence.main(["--work-dir", str(path)]), 2)
                versions.assert_not_called()
                run.assert_not_called()

    def test_cli_hosted_success_outputs_metadata_only(self):
        environment = {"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_TEMP": str(self.root)}
        output = io.StringIO()
        report = {"schema_version": 1, "generated_test_metadata": True, "commercial_accuracy_claim": False}
        with patch.dict(evidence.os.environ, environment, clear=True), \
                patch.object(evidence, "_runtime_versions", return_value={}), \
                patch.object(evidence, "run_regression", return_value=report) as run, redirect_stdout(output):
            self.assertEqual(evidence.main(["--work-dir", str(self.root / "child")]), 0)
        run.assert_called_once_with(self.root / "child")
        self.assertEqual(json.loads(output.getvalue()), report)


if __name__ == "__main__":
    unittest.main()
