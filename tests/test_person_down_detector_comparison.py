"""Generated-byte acquisition and native-fake paired-comparison regressions."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analytics_lab import person_down_detector_comparison as comparison
from analytics_lab import person_down_regression_evidence as regression
from analytics_lab import validation_seed as seed
from analytics_lab.artifacts import OPENVINO_OMZ_2023_FP16, OPENVINO_OMZ_PERSON_0200_FP16
from analytics_lab.evaluation import EvaluationSample, aggregate_person_down_evaluations, evaluate_person_down_candidates
from analytics_lab.validation import ModelArtifactIdentity, ValidationSampleRun, ValidationSuiteResult


def _blob_sha1(payload):
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _event(start, end):
    return {"event_type": "person_down_candidate", "status": "candidate", "evidence": [{
        "type": "observation_window", "start_timestamp_ms": start, "end_timestamp_ms": end,
    }]}


class PersonDownDetectorComparisonTests(unittest.TestCase):
    def setUp(self):
        # Synthetic/generated fixture tests stub eligibility only; production
        # admission and GMDCSA24 media acquisition continue to fail closed.
        for target in (
            "analytics_lab.validation_seed.require_source",
            "analytics_lab.validation.require_source",
            "analytics_lab.person_down_regression_evidence.require_source",
        ):
            patcher = patch(target, new=lambda *_a, **_kw: seed.GMDCSA24)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runner_temp = Path(self.temporary.name)
        self.root = self.runner_temp / "comparison-work"
        self.environment = {"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_TEMP": str(self.runner_temp)}
        self.versions = {"openvino": "2026.3.1", "opencv": "4.12.0", "numpy": "2.2.6"}
        self.payloads = {}
        self.requests = []
        self.directories = []
        self.specs = tuple(replace(spec, size_bytes=len(payload), git_blob_sha1=_blob_sha1(payload))
                           for spec, payload in ((spec, f"generated media metadata:{spec.sample_id}".encode("ascii"))
                                                 for spec in seed.GMDCSA24_HELDOUT_S4))
        for spec in self.specs:
            self.payloads[spec.source_url] = f"generated media metadata:{spec.sample_id}".encode("ascii")
        generated_models = []
        for spec in (*OPENVINO_OMZ_2023_FP16, *OPENVINO_OMZ_PERSON_0200_FP16[:2]):
            payload = f"generated artifact metadata:{spec.relative_path}".encode("ascii")
            self.payloads[spec.source_url] = payload
            generated_models.append(replace(spec, size_bytes=len(payload), sha384=hashlib.sha384(payload).hexdigest()))
        self.baseline_models = tuple(generated_models[:4])
        self.candidate_models = (*generated_models[4:], *generated_models[2:4])
        self.tables = []
        for table in regression._TABLES:
            payload = (regression._HEADER + "\n" + table["row"] + "\n").encode("utf-8")
            url = f"https://raw.githubusercontent.com/{regression._REPOSITORY}/{regression._COMMIT}/" + table["path"].replace(" ", "%20")
            self.payloads[url] = payload
            self.tables.append({**table, "size_bytes": len(payload), "git_blob_sha1": _blob_sha1(payload),
                                "sha256": hashlib.sha256(payload).hexdigest()})
        self.tables = tuple(self.tables)

    def _opener(self, request, timeout):
        self.assertEqual(timeout, 30)
        self.requests.append(request.full_url)
        self.assertIn(request.full_url, self.payloads)
        return io.BytesIO(self.payloads[request.full_url])

    def _suite(self, samples, *, artifact_root, config, pipeline_config, evaluation_config,
               positive_events=None, negative_events=None, negative_duration=4970):
        candidate = pipeline_config.openvino.detector_model == comparison._CANDIDATE
        models = self.candidate_models if candidate else self.baseline_models
        positive_events = ((_event(1700, 4700),) if candidate else ()) if positive_events is None else positive_events
        negative_events = ((_event(0, 3000),) if candidate else ()) if negative_events is None else negative_events
        runs, evaluated = [], []
        for index, sample in enumerate(samples):
            duration = 6217 if index == 0 else negative_duration
            result = evaluate_person_down_candidates(
                positive_events if index == 0 else negative_events, sample.labels,
                video_start_timestamp_ms=0, video_end_timestamp_ms=duration, config=evaluation_config,
            )
            frames = 187 if index == 0 else 150
            observations = ((19, 29) if candidate else (11, 13))[index]
            runs.append(ValidationSampleRun(
                sample.sample_id, sample.site_id, sample.camera_id, sample.authorization_ref,
                sample.media_sha256, self.versions["openvino"], "CPU", frames, observations,
                10.0, frames * 100.0, result,
            ))
            evaluated.append(EvaluationSample(sample.sample_id, sample.site_id, sample.camera_id, 0, duration, result))
        identities = tuple(ModelArtifactIdentity(spec.component, spec.relative_path, spec.size_bytes, spec.sha384, spec.license_id)
                           for spec in models)
        return ValidationSuiteResult(self.versions["openvino"], "CPU", identities, tuple(runs),
                                     aggregate_person_down_evaluations(evaluated), 20.0, 337, 16850.0)

    def _context(self, *, suite=None, change=None, opener=None):
        stack = ExitStack()
        stack.enter_context(patch.dict(comparison.os.environ, self.environment, clear=True))
        stack.enter_context(patch.object(comparison.platform, "system", return_value="Linux"))
        stack.enter_context(patch.object(comparison, "_runtime_versions", return_value=self.versions))
        stack.enter_context(patch.object(comparison, "_cpu_metadata", return_value={"linux_cpu_model_name": "generated metadata", "performance_claim": False}))
        for module in (comparison, regression, seed):
            stack.enter_context(patch.object(module, "GMDCSA24_HELDOUT_S4", self.specs))
            stack.enter_context(patch.object(module, "OPENVINO_OMZ_2023_FP16", self.baseline_models))
        for module in (comparison, regression):
            stack.enter_context(patch.object(module, "_TABLES", self.tables))
        stack.enter_context(patch.object(comparison, "OPENVINO_OMZ_PERSON_0200_FP16", self.candidate_models))
        original_prepare = seed.prepare_seed

        def prepare(directory, **kwargs):
            self.directories.append(Path(directory))
            manifest = original_prepare(directory, **kwargs)
            if change is not None:
                document = json.loads(manifest.read_text(encoding="utf-8"))
                change(document, manifest.parent)
                manifest.write_text(json.dumps(document), encoding="utf-8")
            return manifest

        self.prepare_call = stack.enter_context(patch.object(comparison, "prepare_seed", side_effect=prepare))
        self.suite_call = stack.enter_context(patch.object(comparison, "run_validation_suite", side_effect=suite or self._suite))
        stack.enter_context(redirect_stderr(io.StringIO()))
        return stack

    def _run(self, *, suite=None, change=None, opener=None):
        with self._context(suite=suite, change=change):
            return comparison.run_comparison(self.root, opener=opener or self._opener)

    def _assert_clean(self):
        self.assertTrue(self.root.is_dir())
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertTrue(all(not directory.exists() for directory in self.directories))

    def test_real_plan_binds_exact_six_artifacts_and_frozen_pair(self):
        plan = comparison.comparison_plan()
        self.assertEqual(plan["max_media_bytes"], 7_996_782)
        self.assertEqual(plan["max_unique_model_bytes"], 14_321_809)
        self.assertEqual(len(plan["unique_models"]), 6)
        self.assertEqual(sum(spec.size_bytes for spec in OPENVINO_OMZ_PERSON_0200_FP16[:2]), 3_889_273)
        self.assertEqual(plan["suite_order"], [comparison._BASELINE, comparison._CANDIDATE])
        self.assertEqual(plan["evaluation_config"], {"match_tolerance_ms": 1000})
        self.assertFalse(plan["false_alert_gate_eligible"])
        self.assertFalse(plan["new_heldout_evidence"])
        plan["annotations"][0]["fall_activity_interval_ms"][0] = 0
        self.assertEqual(comparison.comparison_plan()["annotations"][0]["fall_activity_interval_ms"], [1700, 6000])

    def test_prepare_once_six_unique_artifacts_and_two_identical_settings_calls(self):
        report = self._run()
        self.prepare_call.assert_called_once()
        self.assertEqual(self.prepare_call.call_args.kwargs["subset"], "subject4-regression")
        self.assertEqual(len(self.requests), 10)  # Two tables, two metadata clips, six metadata artifacts.
        self.assertEqual(set(self.requests), set(self.payloads))
        self.assertEqual(len(set(self.requests)), len(self.requests))
        self.assertEqual(self.suite_call.call_count, 2)
        left, right = self.suite_call.call_args_list
        self.assertEqual(left.args, right.args)
        for key in ("artifact_root", "config", "evaluation_config"):
            self.assertEqual(left.kwargs[key], right.kwargs[key])
        settings = [asdict(call.kwargs["pipeline_config"]) for call in (left, right)]
        self.assertEqual([item["openvino"].pop("detector_model") for item in settings],
                         [comparison._BASELINE, comparison._CANDIDATE])
        self.assertEqual(settings[0], settings[1])
        self.assertEqual(settings[0]["openvino"]["detection_threshold"], .50)
        self.assertEqual(settings[0]["video"]["max_frames"], 1000)
        self.assertEqual(report["plan_sha256"], hashlib.sha256(comparison._canonical(report["plan"])).hexdigest())
        self.assertEqual(report["delta_direction"], "candidate_minus_baseline")
        self.assertEqual(report["aggregate_deltas"], {
            "observations_processed": 24, "candidate_events": 2, "matched_events": 1, "missed_episodes": -1,
        })
        self.assertEqual(report["sample_deltas"], [
            {"sample_id": self.specs[0].sample_id, "observations_processed": 8,
             "candidate_events": 1, "matched_events": 1, "missed_episodes": -1},
            {"sample_id": self.specs[1].sample_id, "observations_processed": 16,
             "candidate_events": 1, "matched_events": 0, "missed_episodes": 0},
        ])
        for key, models in (("baseline", self.baseline_models), ("candidate", self.candidate_models)):
            actual = report[key]["activity_proxy_validation"]["model_artifacts"]
            self.assertEqual([item["relative_path"] for item in actual], [spec.relative_path for spec in models])
        self.assertTrue(report["temporary_assets_removed"])
        self._assert_clean()

    def test_tail_preserves_proxy_counts_but_cannot_supply_annotated_or_clinical_gate(self):
        for duration in (4000, 4001, 4970):
            with self.subTest(duration=duration):
                report = self._run(suite=lambda *args, **kwargs: self._suite(*args, negative_duration=duration, **kwargs))
                candidate = report["candidate"]
                self.assertEqual(candidate["normal_clip_candidate_events"], 1)
                self.assertEqual(candidate["normal_clip_unannotated_tail_ms"], max(0, duration - 4000))
                self.assertEqual(candidate["normal_clip_observed_annotated_activity_ms"], min(duration, 4000))
                self.assertIsNone(candidate["clinical_false_positives"])
                self.assertIsNone(candidate["clinical_false_positives_per_camera_hour"])
                self.assertFalse(candidate["false_alert_gate_eligible"])
                if duration > 4000:
                    for field in ("normal_clip_scored_activity_false_alerts", "normal_clip_scored_activity_camera_hours",
                                  "normal_clip_scored_activity_false_alerts_per_camera_hour"):
                        self.assertIsNone(candidate[field])
                    self.assertIsNone(report["normal_clip_scored_activity_false_alerts_delta"])
                else:
                    self.assertEqual(candidate["normal_clip_scored_activity_false_alerts"], 1)
                    self.assertEqual(report["normal_clip_scored_activity_false_alerts_delta"], 1)
                self.assertFalse(report["false_alert_gate_eligible"])
                self._assert_clean()

    def test_silent_detectors_preserve_both_positive_misses_without_promotion(self):
        report = self._run(suite=lambda *args, **kwargs: self._suite(*args, positive_events=(), negative_events=(), **kwargs))
        for key in ("baseline", "candidate"):
            self.assertEqual(report[key]["fall_activity_candidate_misses"], 1)
            self.assertEqual(report[key]["normal_clip_candidate_events"], 0)
        self.assertFalse(report["zero_candidates_is_success"])
        self.assertFalse(report["production_promoted"])

    def test_negative_reduction_cannot_hide_lost_positive(self):
        def suite(*args, **kwargs):
            candidate = kwargs["pipeline_config"].openvino.detector_model == comparison._CANDIDATE
            return self._suite(*args, positive_events=() if candidate else (_event(1700, 4700),),
                               negative_events=() if candidate else (_event(0, 3000),), **kwargs)
        report = self._run(suite=suite)
        self.assertEqual(report["sample_deltas"][0]["matched_events"], -1)
        self.assertEqual(report["sample_deltas"][0]["missed_episodes"], 1)
        self.assertEqual(report["sample_deltas"][1]["candidate_events"], -1)
        self.assertFalse(report["production_promoted"])

    def test_manifest_scope_rights_original_labels_and_budget_reject_before_either_suite(self):
        changes = (
            lambda doc, _: doc["samples"].reverse(),
            lambda doc, _: doc["samples"].pop(),
            lambda doc, _: doc["samples"][0].update(sample_id="different"),
            lambda doc, _: doc["samples"][0].update(authorization_ref="different-rights"),
            lambda doc, _: doc["samples"][0].update(site_id="different-site"),
            lambda doc, _: doc["samples"][0].update(camera_id="different-camera"),
            lambda doc, _: doc["samples"][0].update(video_path=doc["samples"][1]["video_path"]),
            lambda doc, _: doc["samples"][0].update(media_sha256="0" * 64),
            lambda doc, _: doc["samples"][0].update(start_timestamp_ms=1),
            lambda doc, _: doc["samples"][0].update(end_timestamp_ms=6001),
            lambda doc, _: doc["samples"][0]["labels"][0].update(end_timestamp_ms=5999),
            lambda doc, _: doc["samples"][0].update(labels=[]),
            lambda doc, _: doc["samples"][1].update(labels=[{"label_id": "invented", "start_timestamp_ms": 0, "end_timestamp_ms": 1000}]),
            lambda doc, _: doc.update(artifact_root="other-artifacts"),
            lambda doc, _: doc["config"].update(max_samples=3),
            lambda doc, _: doc["config"].update(required_device="GPU"),
            lambda doc, _: doc["config"].update(max_total_video_bytes=9999),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index), self.assertRaises((RuntimeError, ValueError)):
                self._run(change=change)
            self.suite_call.assert_not_called()
            self._assert_clean()

    def test_rehashed_replacement_bytes_do_not_replace_pinned_git_source(self):
        def change(document, directory):
            item = document["samples"][0]
            path = directory / item["video_path"]
            payload = path.read_bytes().replace(b"generated", b"invented ", 1)
            path.write_bytes(payload)
            item["media_sha256"] = hashlib.sha256(payload).hexdigest()
        with self.assertRaises(ValueError):
            self._run(change=change)
        self.suite_call.assert_not_called()
        self._assert_clean()

    def test_failed_annotation_identity_stops_before_preparation(self):
        def opener(request, timeout):
            if request.full_url.endswith("Fall.csv"):
                return io.BytesIO(b"malformed generated annotation metadata")
            return self._opener(request, timeout)
        with self.assertRaises(RuntimeError):
            self._run(opener=opener)
        self.prepare_call.assert_not_called()
        self.suite_call.assert_not_called()
        self._assert_clean()

    def test_model_scope_rejects_foreign_pose_pair_and_duplicate_or_missing_artifacts(self):
        with self._context():
            changes = (
                (*self.candidate_models[:2], replace(self.candidate_models[2], sha384="0" * 96), self.candidate_models[3]),
                self.candidate_models[:3],
                (*self.candidate_models[:1], self.candidate_models[0], *self.candidate_models[2:]),
            )
            for specs in changes:
                with self.subTest(specs=specs), patch.object(comparison, "OPENVINO_OMZ_PERSON_0200_FP16", specs), \
                        self.assertRaises(RuntimeError):
                    comparison.run_comparison(self.root, opener=self._opener)
                self.suite_call.assert_not_called()
                self.prepare_call.assert_not_called()
        self._assert_clean()

    def test_result_identity_drift_rejects_each_suite_before_emitting_deltas(self):
        changes = (
            lambda result: replace(result, device="GPU"),
            lambda result: replace(result, runtime_version="2026.3.1-different"),
            lambda result: replace(result, sample_runs=tuple(reversed(result.sample_runs))),
            lambda result: replace(result, model_artifacts=(*result.model_artifacts[:-1], replace(result.model_artifacts[-1], sha384="0" * 96))),
        )
        for field, value in (("site_id", "other"), ("camera_id", "other"), ("authorization_ref", "other"),
                             ("media_sha256", "0" * 64), ("runtime_version", "other"), ("device", "GPU"),
                             ("frames_processed", 1), ("frames_processed", 1001)):
            changes += (lambda result, field=field, value=value: replace(result, sample_runs=(
                replace(result.sample_runs[0], **{field: value}), result.sample_runs[1])),)
        for bad_selector in (comparison._BASELINE, comparison._CANDIDATE):
            for index, change in enumerate(changes):
                def suite(*args, **kwargs):
                    result = self._suite(*args, **kwargs)
                    return change(result) if kwargs["pipeline_config"].openvino.detector_model == bad_selector else result
                with self.subTest(selector=bad_selector, change=index), self.assertRaises(RuntimeError):
                    self._run(suite=suite)
                self.assertEqual(self.suite_call.call_count, 1 if bad_selector == comparison._BASELINE else 2)
                self._assert_clean()

    def test_different_frame_count_or_decoded_duration_cannot_form_paired_delta(self):
        for mismatch in ("frames", "duration"):
            def suite(*args, **kwargs):
                if kwargs["pipeline_config"].openvino.detector_model == comparison._CANDIDATE:
                    if mismatch == "duration":
                        return self._suite(*args, negative_duration=4971, **kwargs)
                    result = self._suite(*args, **kwargs)
                    return replace(result, total_frames_processed=338,
                                   sample_runs=(replace(result.sample_runs[0], frames_processed=188), result.sample_runs[1]))
                return self._suite(*args, **kwargs)
            with self.subTest(mismatch=mismatch), self.assertRaisesRegex(RuntimeError, "decoded exposure differs"):
                self._run(suite=suite)
            self.assertEqual(self.suite_call.call_count, 2)
            self._assert_clean()

    def test_positive_endpoint_is_never_clipped_to_make_a_suite_pass(self):
        def suite(*args, **kwargs):
            result = self._suite(*args, **kwargs)
            first = result.sample_runs[0]
            return replace(result, sample_runs=(replace(first, evaluation=replace(first.evaluation, duration_ms=5999)), result.sample_runs[1]))
        with self.assertRaisesRegex(RuntimeError, "positive label exceeds decoded coverage"):
            self._run(suite=suite)
        self.assertEqual(self.suite_call.call_count, 1)
        self._assert_clean()

    def test_impossible_typed_counts_metrics_and_aggregate_do_not_become_comparison_evidence(self):
        changes = (
            lambda result: replace(result, aggregate=replace(result.aggregate, matched_events=99)),
            lambda result: replace(result, total_frames_processed=999),
            lambda result: replace(result, sample_runs=(replace(result.sample_runs[0], observations_processed=-1), result.sample_runs[1])),
        )
        for field, value in (("candidate_events", -1), ("matched_events", 2), ("missed_episodes", 2),
                             ("false_alerts", 2), ("precision", .5), ("recall", .5),
                             ("false_alerts_per_camera_hour", -1.0), ("duration_ms", float("nan"))):
            changes += (lambda result, field=field, value=value: replace(result, sample_runs=(
                replace(result.sample_runs[0], evaluation=replace(result.sample_runs[0].evaluation, **{field: value})),
                result.sample_runs[1])),)
        for index, change in enumerate(changes):
            with self.subTest(change=index), self.assertRaises(RuntimeError):
                self._run(suite=lambda *args, **kwargs: change(self._suite(*args, **kwargs)))
            self.assertEqual(self.suite_call.call_count, 1)
            self._assert_clean()

    def test_corrupt_extra_artifact_fails_hash_verification_before_native(self):
        def opener(request, timeout):
            if "/person-detection-0200/" in request.full_url:
                self.requests.append(request.full_url)
                return io.BytesIO(self.payloads[request.full_url].replace(b"generated", b"invented ", 1))
            return self._opener(request, timeout)
        with self.assertRaisesRegex(ValueError, "artifact checksum mismatch"):
            self._run(opener=opener)
        self.suite_call.assert_not_called()
        self._assert_clean()

    def test_cleanup_on_extra_artifact_and_each_suite_failure_without_retry(self):
        for phase in ("extra", comparison._BASELINE, comparison._CANDIDATE):
            def opener(request, timeout):
                if phase == "extra" and "/person-detection-0200/" in request.full_url:
                    raise RuntimeError("generated acquisition failure")
                return self._opener(request, timeout)
            def suite(*args, **kwargs):
                if kwargs["pipeline_config"].openvino.detector_model == phase:
                    raise RuntimeError("generated suite failure")
                return self._suite(*args, **kwargs)
            with self.subTest(phase=phase), self.assertRaises(RuntimeError):
                self._run(suite=suite, opener=opener)
            self.prepare_call.assert_called_once()
            self.assertEqual(self.suite_call.call_count, 0 if phase == "extra" else (1 if phase == comparison._BASELINE else 2))
            self._assert_clean()

    def test_hosted_linux_and_dedicated_temp_child_required_before_runtime_or_acquisition(self):
        for environment, system, path in (
            ({}, "Linux", self.root),
            ({**self.environment, "RUNNER_ENVIRONMENT": "self-hosted"}, "Linux", self.root),
            (self.environment, "Windows", self.root),
            (self.environment, "Linux", self.runner_temp),
            (self.environment, "Linux", self.runner_temp.parent / "outside-comparison"),
        ):
            with self.subTest(environment=environment, system=system, path=path), \
                    patch.dict(comparison.os.environ, environment, clear=True), \
                    patch.object(comparison.platform, "system", return_value=system), \
                    patch.object(comparison, "_runtime_versions") as versions, \
                    patch.object(comparison, "verify_annotations") as annotations, \
                    self.assertRaises((RuntimeError, ValueError)):
                comparison.run_comparison(path, opener=self._opener)
            versions.assert_not_called()
            annotations.assert_not_called()

    def test_cpu_identity_unknown_if_unavailable_and_exact_if_single_linux_model(self):
        with patch.object(comparison.platform, "system", return_value="Linux"), \
                patch.object(Path, "open", side_effect=OSError("generated unavailable metadata")):
            self.assertEqual(comparison._cpu_metadata()["linux_cpu_model_name"], "unknown")
        with patch.object(comparison.platform, "system", return_value="Linux"), \
                patch.object(Path, "open", return_value=io.StringIO("processor : 0\nmodel name : Generated CPU 123 @ 4.0GHz\n")):
            self.assertEqual(comparison._cpu_metadata()["linux_cpu_model_name"], "Generated CPU 123 @ 4.0GHz")

    def test_cli_emits_one_metadata_document_only_after_complete_cleanup(self):
        output = io.StringIO()
        with self._context(), redirect_stdout(output), patch.object(comparison.urllib.request, "urlopen", side_effect=AssertionError("no network")):
            # The explicit fake opener supplies only generated metadata bytes.
            original = comparison.run_comparison
            with patch.object(comparison, "run_comparison", side_effect=lambda path: original(path, opener=self._opener)):
                self.assertEqual(comparison.main(["--work-dir", str(self.root)]), 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report["temporary_assets_removed"])
        self.assertFalse(report["performance_claim"])
        self._assert_clean()


if __name__ == "__main__":
    unittest.main()
