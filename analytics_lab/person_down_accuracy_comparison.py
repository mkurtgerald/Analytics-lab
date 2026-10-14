"""One hosted OFF/ON replay of guarded 0200 orientation recovery.

Use exactly the reviewed S4 activity-proxy pair and four verified model files.
Missing-frame evidence is explicit in both arms. No threshold tuning, new data,
clinical accuracy gate, or default promotion is performed by this comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable
import urllib.request

from .artifacts import OPENVINO_OMZ_PERSON_0200_FP16, verify_artifact_set
from .evaluation import EvaluationConfig
from .openvino_omz import OpenVINOOMZConfig
from .openvino_pipeline import OpenVINOOMZPipelineConfig
from .person_down_detector_comparison import (
    _cpu_metadata, _hosted_root, _paired_report, _validate_result, _verify_manifest,
)
from .person_down_regression_evidence import _runtime_versions, evidence_plan, verify_annotations
from .validation import run_validation_suite
from .validation_cli import _safe_error_code
from .validation_seed import prepare_seed
from .video import VideoRunConfig


def _pipeline_configs() -> tuple[OpenVINOOMZPipelineConfig, OpenVINOOMZPipelineConfig]:
    baseline = OpenVINOOMZPipelineConfig(
        openvino=OpenVINOOMZConfig(detector_model="person-detection-0200", detection_threshold=.50),
        video=VideoRunConfig(max_frames=1000),
    )
    return baseline, replace(baseline, orientation_recovery=True)


def comparison_plan() -> dict[str, Any]:
    original = evidence_plan()
    models = OPENVINO_OMZ_PERSON_0200_FP16
    expected = tuple(f"{name}/FP16/{name}.{extension}"
                     for name in ("person-detection-0200", "human-pose-estimation-0001")
                     for extension in ("xml", "bin"))
    if len(models) != 4 or tuple(item.relative_path for item in models) != expected:
        raise RuntimeError("accuracy comparison requires exactly the four reviewed 0200/pose artifacts")
    return {
        **{key: original[key] for key in (
            "source_repository", "source_commit", "source_rights", "annotations", "media", "decoder",
        )},
        "mode": "retrospective_paired_guarded_orientation_activity_proxy",
        "subset": "subject4-regression",
        "suite_order": ["orientation_off", "orientation_on"],
        "pipeline_configs": [asdict(config) for config in _pipeline_configs()],
        "evaluation_config": asdict(EvaluationConfig(match_tolerance_ms=1000)),
        "unique_models": [asdict(item) for item in models],
        "max_unique_model_bytes": sum(item.size_bytes for item in models),
        "max_media_bytes": original["max_media_bytes"],
        "suite_calls": 2,
        "missing_frame_policy": "known missing detections obey the unchanged configured unknown-evidence gap in both arms",
        "recovery_policy": "primary detections first; only an immediate single-person history may link one +/-90-degree detection at unchanged confidence/IoU thresholds",
        "max_detector_passes_per_frame": 3,
        "pose_coordinate_policy": "pose inferred on original-frame crop through exact integer pixel bounds",
        "state_scope": "one fresh recovery/tracker/temporal state per clip; one compiled backend per suite",
        "score_calibration": "model scores, not calibrated probabilities",
        "scoring_scope": "whole-clip activity proxy; unannotated tails cannot qualify annotated-negative counts/rates",
        "false_alert_gate_eligible": False,
        "clinical_person_down_ground_truth": False,
        "new_heldout_evidence": False,
        "performance_claim": False,
    }


def run_comparison(work_dir: str | Path, *, opener: Callable[..., Any] = urllib.request.urlopen) -> dict[str, Any]:
    root = _hosted_root(work_dir)
    versions = _runtime_versions()
    if versions["numpy"] != "2.2.6":
        raise RuntimeError("accuracy comparison NumPy version changed")
    plan = comparison_plan()
    print("accuracy_comparison_phase=verify_annotations", file=sys.stderr, flush=True)
    verify_annotations(opener=opener)
    cpu_metadata = _cpu_metadata()
    with tempfile.TemporaryDirectory(prefix="person-down-accuracy-", dir=root) as temporary:
        directory = Path(temporary)
        print("accuracy_comparison_phase=prepare_sources", file=sys.stderr, flush=True)
        manifest = prepare_seed(directory, opener=opener, subset="subject4-regression",
                                detector_model="person-detection-0200")
        artifact_root, samples, config = _verify_manifest(manifest, directory, plan)
        verify_artifact_set(artifact_root, OPENVINO_OMZ_PERSON_0200_FP16)
        results = []
        for mode, pipeline in zip(plan["suite_order"], _pipeline_configs()):
            print(f"accuracy_comparison_phase=suite_{mode}", file=sys.stderr, flush=True)
            result = run_validation_suite(
                samples, artifact_root=artifact_root, config=config,
                pipeline_config=pipeline, evaluation_config=EvaluationConfig(match_tolerance_ms=1000),
            )
            _validate_result(result, samples, OPENVINO_OMZ_PERSON_0200_FP16, versions)
            results.append(result)
        report = _paired_report(plan, versions, cpu_metadata, samples, results)
    if directory.exists():
        raise RuntimeError("accuracy comparison temporary assets were not removed")
    report["temporary_assets_removed"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_comparison(args.work_dir), allow_nan=False, sort_keys=True, separators=(",", ":")))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as error:
        print(f"Person-down accuracy comparison rejected ({type(error).__name__}; code={_safe_error_code(error)}).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
