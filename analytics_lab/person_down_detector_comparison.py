"""One hosted paired replay of two reviewed detectors on the historical S4 pair.

Only the person-detector selector changes. Publisher activity labels remain a
retrospective proxy, not clinical person-down truth or a false-alert gate. All
six unique model files and both clips are ephemeral; only metadata is emitted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Callable
import urllib.request

from .artifacts import (
    OPENVINO_OMZ_2023_FP16, OPENVINO_OMZ_PERSON_0200_FP16, verify_artifact_set,
)
from .evaluation import EvaluationConfig, EvaluationSample, aggregate_person_down_evaluations
from .openvino_omz import OpenVINOOMZConfig
from .openvino_pipeline import OpenVINOOMZPipelineConfig
from .person_down_regression_evidence import (
    _TABLES, _canonical, _runtime_versions, evidence_plan, verify_annotations,
)
from .validation import ModelArtifactIdentity, ValidationSuiteResult, run_validation_suite
from .validation_cli import _safe_error_code, load_manifest
from .validation_seed import (
    GMDCSA24_HELDOUT_S4, _AUTHORIZATION_REF, _download_exact, _target,
    prepare_seed, verify_seed_media,
)
from .video import VideoRunConfig

_BASELINE = "person-detection-retail-0013"
_CANDIDATE = "person-detection-0200"


def _pipeline_configs() -> tuple[OpenVINOOMZPipelineConfig, OpenVINOOMZPipelineConfig]:
    baseline = OpenVINOOMZPipelineConfig(
        openvino=OpenVINOOMZConfig(detector_model=_BASELINE, detection_threshold=0.50),
        video=VideoRunConfig(max_frames=1000),
    )
    return baseline, replace(baseline, openvino=replace(baseline.openvino, detector_model=_CANDIDATE))


def _model_scope() -> tuple:
    baseline, candidate = OPENVINO_OMZ_2023_FP16, OPENVINO_OMZ_PERSON_0200_FP16
    expected = lambda name: tuple(f"{name}/FP16/{name}.{extension}" for extension in ("xml", "bin"))
    if (len(baseline) != 4 or len(candidate) != 4
            or tuple(item.relative_path for item in baseline[:2]) != expected(_BASELINE)
            or tuple(item.relative_path for item in candidate[:2]) != expected(_CANDIDATE)
            or tuple(item.relative_path for item in baseline[2:]) != expected("human-pose-estimation-0001")
            or candidate[2:] != baseline[2:]):
        raise RuntimeError("comparison model scope changed")
    unique = (*baseline, *candidate[:2])
    if len({item.relative_path for item in unique}) != 6:
        raise RuntimeError("comparison model scope is not six unique artifacts")
    return unique


def comparison_plan() -> dict[str, Any]:
    """Bind exact source/annotation identities and the paired frozen settings."""
    original = evidence_plan()
    configs = _pipeline_configs()
    return {
        **{key: original[key] for key in (
            "source_repository", "source_commit", "source_rights", "annotations", "media", "decoder",
        )},
        "mode": "retrospective_paired_person_detector_activity_proxy",
        "subset": "subject4-regression",
        "suite_order": [_BASELINE, _CANDIDATE],
        "pipeline_configs": [asdict(config) for config in configs],
        "evaluation_config": asdict(EvaluationConfig(match_tolerance_ms=1000)),
        "model_sets": [
            {"detector_model": selector, "artifacts": [asdict(item) for item in specs]}
            for selector, specs in ((_BASELINE, OPENVINO_OMZ_2023_FP16), (_CANDIDATE, OPENVINO_OMZ_PERSON_0200_FP16))
        ],
        "unique_models": [asdict(item) for item in _model_scope()],
        "max_media_bytes": sum(item.size_bytes for item in GMDCSA24_HELDOUT_S4),
        "max_unique_model_bytes": sum(item.size_bytes for item in _model_scope()),
        "suite_calls": 2,
        "crop_coordinate_policy": "shared integer pixel crop bounds and frame-relative pose remapping",
        "score_calibration": "detector scores are model scores, not calibrated probabilities",
        "scoring_scope": "whole-clip activity proxy; unannotated tails cannot qualify annotated-negative counts/rates",
        "false_alert_gate_eligible": False,
        "clinical_person_down_ground_truth": False,
        "new_heldout_evidence": False,
        "performance_claim": False,
    }


def _hosted_root(work_dir: str | Path) -> Path:
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
            or not os.environ.get("RUNNER_TEMP") or platform.system() != "Linux"):
        raise RuntimeError("detector comparison requires the hosted Linux ephemeral lane")
    runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    raw = Path(work_dir)
    root = raw.resolve()
    root.relative_to(runner_temp)
    if root == runner_temp or raw.is_symlink():
        raise ValueError("work directory must be a dedicated ephemeral child")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("work directory must be a directory")
    return root


def _cpu_metadata() -> dict[str, Any]:
    model_name = "unknown"
    if platform.system() == "Linux":
        try:
            with Path("/proc/cpuinfo").open("r", encoding="utf-8") as handle:
                text = handle.read(2 * 1024 * 1024 + 1)
            if len(text) <= 2 * 1024 * 1024:
                names = {line.split(":", 1)[1].strip() for line in text.splitlines()
                         if line.split(":", 1)[0].strip() == "model name" and ":" in line}
                if len(names) == 1 and next(iter(names)):
                    model_name = next(iter(names))
        except (OSError, UnicodeError):
            pass
    return {
        "system": platform.system(), "release": platform.release(),
        "machine": platform.machine(), "python": platform.python_version(),
        "logical_cpus": os.cpu_count(), "linux_cpu_model_name": model_name,
        "performance_claim": False,
    }


def _verify_manifest(manifest: Path, directory: Path, plan: dict[str, Any]) -> tuple:
    artifact_root, samples, config = load_manifest(manifest)
    if (artifact_root != directory / "artifacts" or asdict(config) != {
            "max_samples": 2, "required_device": "CPU", "max_total_video_bytes": plan["max_media_bytes"]}):
        raise RuntimeError("comparison manifest budget or artifact root changed")
    if (len(GMDCSA24_HELDOUT_S4) != 2 or len(_TABLES) != 2
            or [sample.sample_id for sample in samples] != [table["sample_id"] for table in _TABLES]
            or [spec.sample_id for spec in GMDCSA24_HELDOUT_S4] != [table["sample_id"] for table in _TABLES]):
        raise RuntimeError("comparison source selection changed")
    for sample, spec, table in zip(samples, GMDCSA24_HELDOUT_S4, _TABLES):
        labels = [(item.start_timestamp_ms, item.end_timestamp_ms, item.label_id) for item in sample.labels]
        expected = [(spec.label_start_timestamp_ms, spec.label_end_timestamp_ms, spec.label_id)] if spec.label_id else []
        if (sample.start_timestamp_ms != 0 or sample.end_timestamp_ms != spec.end_timestamp_ms
                or labels != expected
                or (list(expected[0][:2]) if expected else None) != table["fall_activity_interval_ms"]):
            raise RuntimeError("comparison activity-label binding changed")
        if (sample.site_id != "gmdcsa24-home-setup-1"
                or sample.camera_id != f"gmdcsa24-clip-{spec.sample_id}"
                or sample.authorization_ref != _AUTHORIZATION_REF
                or sample.video_path != directory / "media" / "gmdcsa24" / spec.relative_path
                or verify_seed_media(sample.video_path, spec) != sample.media_sha256):
            raise RuntimeError("comparison immutable source binding changed")
    return artifact_root, samples, config


def _validate_result(result: Any, samples: tuple, specs: tuple, versions: dict[str, str]) -> None:
    expected_models = tuple(ModelArtifactIdentity(
        spec.component, spec.relative_path, spec.size_bytes, spec.sha384, spec.license_id,
    ) for spec in specs)
    if (not isinstance(result, ValidationSuiteResult) or result.device != "CPU"
            or result.runtime_version != versions["openvino"] or result.model_artifacts != expected_models
            or [item.sample_id for item in result.sample_runs] != [item.sample_id for item in samples]):
        raise RuntimeError("comparison suite result identity changed")
    for run, sample in zip(result.sample_runs, samples):
        if ((run.site_id, run.camera_id, run.authorization_ref, run.media_sha256)
                != (sample.site_id, sample.camera_id, sample.authorization_ref, sample.media_sha256)
                or run.device != "CPU" or run.runtime_version != result.runtime_version
                or type(run.frames_processed) is not int or not 2 <= run.frames_processed <= 1000
                or type(run.observations_processed) is not int or run.observations_processed < 0
                or type(run.evaluation.duration_ms) is not int or run.evaluation.duration_ms <= 0
                or run.evaluation.positive_episodes != len(sample.labels)):
            raise RuntimeError("comparison sample result binding changed")
        if any(label.end_timestamp_ms > run.evaluation.duration_ms for label in sample.labels):
            raise RuntimeError("comparison positive label exceeds decoded coverage")
        evaluation = run.evaluation
        counts = [getattr(evaluation, field) for field in (
            "positive_episodes", "candidate_events", "matched_events", "missed_episodes", "false_alerts",
        )]
        if (any(type(value) is not int or value < 0 for value in counts)
                or evaluation.matched_events + evaluation.missed_episodes != evaluation.positive_episodes
                or evaluation.matched_events + evaluation.false_alerts != evaluation.candidate_events
                or evaluation.candidate_events > run.observations_processed
                or len(evaluation.matched_label_ids) != evaluation.matched_events
                or len(set(evaluation.matched_label_ids)) != evaluation.matched_events
                or not set(evaluation.matched_label_ids).issubset({label.label_id for label in sample.labels})
                or len(evaluation.alert_delays_ms) != evaluation.matched_events
                or evaluation.precision != (evaluation.matched_events / evaluation.candidate_events if evaluation.candidate_events else None)
                or evaluation.recall != (evaluation.matched_events / evaluation.positive_episodes if evaluation.positive_episodes else None)
                or evaluation.false_alerts_per_camera_hour != evaluation.false_alerts / (evaluation.duration_ms / 3600000)):
            raise RuntimeError("comparison evaluation counts or metrics are inconsistent")
    evaluated = [EvaluationSample(
        sample.sample_id, sample.site_id, sample.camera_id, 0, run.evaluation.duration_ms, run.evaluation,
    ) for sample, run in zip(samples, result.sample_runs)]
    if (result.aggregate != aggregate_person_down_evaluations(evaluated)
            or result.total_frames_processed != sum(run.frames_processed for run in result.sample_runs)):
        raise RuntimeError("comparison aggregate does not match the sample results")


def _proxy_report(result: ValidationSuiteResult) -> dict[str, Any]:
    positive, negative = (run.evaluation for run in result.sample_runs)
    nominal_normal_ms = _TABLES[1]["normal_activity_interval_ms"][1]
    scored = negative.duration_ms <= nominal_normal_ms
    return {
        "activity_proxy_validation": asdict(result),
        "fall_activity_candidate_matches": positive.matched_events,
        "fall_activity_candidate_misses": positive.missed_episodes,
        "normal_clip_candidate_events": negative.candidate_events,
        "normal_clip_decoded_exposure_ms": negative.duration_ms,
        "normal_clip_decoded_camera_hours": negative.duration_ms / 3600000,
        "normal_clip_nominal_annotated_ms": nominal_normal_ms,
        "normal_clip_observed_annotated_activity_ms": min(negative.duration_ms, nominal_normal_ms),
        "normal_clip_unannotated_tail_ms": max(0, negative.duration_ms - nominal_normal_ms),
        "normal_clip_scored_activity_false_alerts": negative.false_alerts if scored else None,
        "normal_clip_scored_activity_camera_hours": negative.duration_ms / 3600000 if scored else None,
        "normal_clip_scored_activity_false_alerts_per_camera_hour": negative.false_alerts_per_camera_hour if scored else None,
        "clinical_false_positives": None, "clinical_false_positives_per_camera_hour": None,
        "false_alert_gate_eligible": False,
    }


def run_comparison(work_dir: str | Path, *, opener: Callable[..., Any] = urllib.request.urlopen) -> dict[str, Any]:
    root = _hosted_root(work_dir)
    versions = _runtime_versions()
    if versions["numpy"] != "2.2.6":
        raise RuntimeError("comparison NumPy version changed")
    plan = comparison_plan()
    print("detector_comparison_phase=verify_annotations", file=sys.stderr, flush=True)
    verify_annotations(opener=opener)
    configs = _pipeline_configs()
    cpu_metadata = _cpu_metadata()
    with tempfile.TemporaryDirectory(prefix="person-down-detector-comparison-", dir=root) as temporary:
        directory = Path(temporary)
        print("detector_comparison_phase=prepare_sources", file=sys.stderr, flush=True)
        manifest = prepare_seed(directory, opener=opener, subset="subject4-regression")
        artifact_root, samples, config = _verify_manifest(manifest, directory, plan)
        for spec in OPENVINO_OMZ_PERSON_0200_FP16[:2]:
            target = _target(artifact_root, spec.relative_path)
            _download_exact(spec.source_url, target, expected_size=spec.size_bytes, opener=opener)
        verify_artifact_set(artifact_root, _model_scope())
        results = []
        for selector, pipeline, specs in zip(
                (_BASELINE, _CANDIDATE), configs, (OPENVINO_OMZ_2023_FP16, OPENVINO_OMZ_PERSON_0200_FP16)):
            print(f"detector_comparison_phase=suite_{selector}", file=sys.stderr, flush=True)
            result = run_validation_suite(
                samples, artifact_root=artifact_root, config=config,
                pipeline_config=pipeline, evaluation_config=EvaluationConfig(match_tolerance_ms=1000),
            )
            _validate_result(result, samples, specs, versions)
            results.append(result)
        baseline, candidate = results
        if any((left.frames_processed, left.evaluation.duration_ms) != (right.frames_processed, right.evaluation.duration_ms)
               for left, right in zip(baseline.sample_runs, candidate.sample_runs)):
            raise RuntimeError("comparison decoded exposure differs between detectors")
        reports = [_proxy_report(result) for result in results]
        scored_delta = None
        if all(report["normal_clip_scored_activity_false_alerts"] is not None for report in reports):
            scored_delta = reports[1]["normal_clip_scored_activity_false_alerts"] - reports[0]["normal_clip_scored_activity_false_alerts"]
        report = {
            "schema_version": 1, "plan": plan,
            "plan_sha256": hashlib.sha256(_canonical(plan)).hexdigest(),
            "runtime_versions": versions, "cpu_metadata": cpu_metadata,
            "materialized_media": [{"sample_id": sample.sample_id, "sha256": sample.media_sha256} for sample in samples],
            "baseline": reports[0], "candidate": reports[1],
            "delta_direction": "candidate_minus_baseline",
            "aggregate_deltas": {
                "observations_processed": sum(run.observations_processed for run in candidate.sample_runs)
                - sum(run.observations_processed for run in baseline.sample_runs),
                **{field: getattr(candidate.aggregate, field) - getattr(baseline.aggregate, field)
                   for field in ("candidate_events", "matched_events", "missed_episodes")},
            },
            "sample_deltas": [{
                "sample_id": left.sample_id,
                "observations_processed": right.observations_processed - left.observations_processed,
                **{field: getattr(right.evaluation, field) - getattr(left.evaluation, field)
                   for field in ("candidate_events", "matched_events", "missed_episodes")},
            } for left, right in zip(baseline.sample_runs, candidate.sample_runs)],
            "normal_clip_scored_activity_false_alerts_delta": scored_delta,
            "clinical_false_positive_delta": None,
            "zero_candidates_is_success": False, "false_alert_gate_eligible": False,
            "clinical_person_down_ground_truth": False, "new_heldout_evidence": False,
            "commercial_accuracy_claim": False, "performance_claim": False, "production_promoted": False,
            "coverage_identity_scope": "same immutable clips, decoder, frame counts and decoded duration; no decoded-frame digest",
        }
    if directory.exists():
        raise RuntimeError("comparison temporary assets were not removed")
    report["temporary_assets_removed"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_comparison(args.work_dir)
        print(json.dumps(report, allow_nan=False, sort_keys=True, separators=(",", ":")))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as error:
        print(f"Person-down comparison rejected ({type(error).__name__}; code={_safe_error_code(error)}).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
