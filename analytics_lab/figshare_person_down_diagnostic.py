"""Run the retained person-down evidence path on the pinned Figshare pair.

The Figshare source supplies clip-level Fall/ADL activity classes, not a
published frame-level event interval for the two admitted clips. This diagnostic
therefore scores the ADL clip normally as a negative while treating the Fall
clip as diagnostic-only: candidate count, detector/pose/posture continuity,
decoded timing and resource measurements are retained, but match/miss and alert
latency are deliberately not scored. No threshold, model, association, posture
or temporal rule is changed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable
import urllib.request

from .artifacts import OPENVINO_OMZ_2023_FP16, verify_artifact_set
from .detector_thresholds import (
    _CONTINUITY_MIN_LINK_IOU,
    _CONTINUITY_THRESHOLD,
    _provision,
    _target_candidate,
)
from .detector_shootout import _CompiledDetector
from .figshare_member_admission import AdmittedMember, admit_pinned_pair
from .openpose_diagnostics import _ReferenceRuntime
from .openpose_reference import ReferenceOpenPoseDecoder
from .openvino_omz import _OpenVINORuntime
from .person_down_e2e_diagnostics import _measured_temporal_config
from .person_down_orientation_diagnostics import _scan_sample
from .source_rights import FIGSHARE_FALL_2017, require_source
from .validation import ValidationSampleSpec
from .validation_seed import _download_models

_RUNTIME_PREFIX = "2026.3.1"
_AUTHORIZATION_REF = "figshare:28596332:v2:CC-BY-4.0:evaluation"
_ACTIVITY_REFERENCE = {
    "title": "Vision Transformer Based Fall Detection: A Spatial Temporal Attention Mechanism for Robust Video Analysis",
    "doi": "10.30970/eli.33.12",
    "license_id": "CC-BY-4.0",
}
_EXPECTED_SHA256 = {
    "negative": "7e6f026e68c280234ac34764a26b7f073e6f1367a259ed756a30a663542d3c92",
    "positive": "8c7c13e1a9a5321e25b4203d35e65e072e2e6fcfd21ed806d7fd17b58b4438fd",
}
_ACTIVITY = {
    "negative": {
        "activity_code": "ACT25",
        "activity_name": "Descend",
        "source_clip_class": "ADL",
        "frame_level_interval_available": False,
    },
    "positive": {
        "activity_code": "ACT10",
        "activity_name": "Sit on chair, fall",
        "source_clip_class": "Fall",
        "frame_level_interval_available": False,
    },
}


def _sample_spec(item: AdmittedMember) -> ValidationSampleSpec:
    if not isinstance(item, AdmittedMember) or item.role not in _ACTIVITY:
        raise TypeError("admitted Figshare member role is unsupported")
    expected = _EXPECTED_SHA256[item.role]
    if item.sha256 != expected:
        raise RuntimeError("admitted Figshare SHA-256 identity changed")
    activity = _ACTIVITY[item.role]
    return ValidationSampleSpec(
        sample_id=f"figshare-{activity['activity_code'].lower()}-{item.role}",
        site_id="figshare-28596332-location-3",
        camera_id=f"figshare-{activity['activity_code'].lower()}-clip",
        authorization_ref=_AUTHORIZATION_REF,
        video_path=item.local_path,
        media_sha256=item.sha256,
        start_timestamp_ms=0,
        # The scanner expands this declared lower bound to the decoded final
        # timestamp. It is intentionally not a fabricated source annotation.
        end_timestamp_ms=1,
        labels=(),
    )


def _source_document() -> dict[str, Any]:
    rights = require_source(
        FIGSHARE_FALL_2017.source_id,
        purpose="evaluation",
        require_real_world=True,
    )
    if rights is not FIGSHARE_FALL_2017:
        raise RuntimeError("reviewed Figshare rights identity changed")
    return {
        "source_id": rights.source_id,
        "title": rights.title,
        "version": rights.version,
        "license_id": rights.license_id,
        "provenance_ref": rights.provenance_ref,
        "attribution_required": rights.attribution_required,
        "activity_reference": dict(_ACTIVITY_REFERENCE),
        "annotation_scope": "clip-level activity class only",
        "frame_level_positive_interval_available": False,
    }


def run_figshare_person_down_diagnostic(
    output_dir: str | Path,
    candidate_dir: str | Path,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists() and root.is_symlink():
        raise RuntimeError("output directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise RuntimeError("output path must be a directory")

    preparation_started = time.perf_counter()
    admitted = admit_pinned_pair(root / "media", opener=opener)
    if len(admitted) != 2 or {item.role for item in admitted} != {"negative", "positive"}:
        raise RuntimeError("Figshare diagnostic requires the exact admitted pair")

    artifact_root = _download_models(root / "models", opener=opener)
    candidate = _target_candidate()
    detector_root = Path(candidate_dir)
    _provision(detector_root, candidate)
    detector = _CompiledDetector(candidate, detector_root)

    verified = verify_artifact_set(artifact_root, OPENVINO_OMZ_2023_FP16)
    base_runtime = _OpenVINORuntime(verified, "CPU")
    if not base_runtime.runtime_version.startswith(_RUNTIME_PREFIX):
        raise RuntimeError("OpenVINO runtime version does not match reviewed release")
    runtime = _ReferenceRuntime(base_runtime)
    decoder = ReferenceOpenPoseDecoder()
    preparation_elapsed_ms = (time.perf_counter() - preparation_started) * 1000.0

    samples: list[dict[str, Any]] = []
    total_frames = 0
    total_elapsed_ms = 0.0
    for item in admitted:
        spec = _sample_spec(item)
        measured, _ = _scan_sample(spec, detector, runtime, decoder)
        total_frames += int(measured["frames_processed"])
        total_elapsed_ms += float(measured["elapsed_ms"])
        measured["source_activity"] = dict(_ACTIVITY[item.role])
        measured["member_name"] = item.name
        measured["role"] = item.role
        if item.role == "positive":
            # With no independent frame-level event window, evaluating this
            # clip as a false alert or matched event would manufacture ground
            # truth. Preserve the perception/temporal trace but not that score.
            measured["evaluation"] = {
                "status": "unscored-positive-clip",
                "match_miss_scored": False,
                "alert_delay_scored": False,
                "reason": "source provides clip-level Fall activity only; no frame-level positive interval is published",
            }
        else:
            measured["evaluation_policy"] = {
                "status": "scored-negative-clip",
                "false_alerts_scored": True,
                "reason": "source class is ADL and contains no positive person-down label",
            }
        samples.append(measured)

    by_role = {item["role"]: item for item in samples}
    negative_evaluation = by_role["negative"]["evaluation"]
    return {
        "schema_version": 1,
        "diagnostic": "figshare_person_down_generalization",
        "source": _source_document(),
        "detector": candidate.name,
        "detector_threshold": _CONTINUITY_THRESHOLD,
        "continuity_min_link_iou": _CONTINUITY_MIN_LINK_IOU,
        "pose_model": "human-pose-estimation-0001",
        "runtime_version": base_runtime.runtime_version,
        "device": "CPU",
        "temporal_config": asdict(_measured_temporal_config()),
        "preparation_elapsed_ms": preparation_elapsed_ms,
        "samples": samples,
        "negative_false_alerts": int(negative_evaluation["false_alerts"]),
        "negative_decoded_camera_hours": float(negative_evaluation["duration_ms"]) / 3_600_000.0,
        "total_frames_processed": total_frames,
        "total_elapsed_ms": total_elapsed_ms,
        "throughput_fps": total_frames * 1000.0 / total_elapsed_ms if total_elapsed_ms > 0 else 0.0,
        "positive_match_miss_scored": False,
        "positive_alert_delay_scored": False,
        "media_retained_in_repository": False,
        "thresholds_or_temporal_rules_changed": False,
        "evidence_only": True,
        "commercial_accuracy_claim": False,
        "inferences_not_supported": ["injury", "cause", "fault", "intent", "medical condition"],
    }


def _github_escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = run_figshare_person_down_diagnostic(args.output_dir, args.candidate_dir)
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        print(encoded)
        import os
        if os.getenv("GITHUB_ACTIONS") == "true":
            print("::notice title=Figshare person-down diagnostic::" + _github_escape(encoded))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError):
        print("Figshare person-down diagnostic rejected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
