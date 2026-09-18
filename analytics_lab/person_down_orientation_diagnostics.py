"""Evidence-only person-down diagnostic with bounded orientation recovery.

Subject-2 held-out evidence showed that the reviewed person-detection-0200 path
lost the person on 44/138 labelled fall frames while the hard negative remained
safe. This diagnostic keeps the exact same detector, weights, confidence floor,
OpenPose path, posture rules and temporal rules. Only when the primary detector
returns no admitted person box and a prior continuity box exists, it evaluates
+/-90-degree views with the same detector and admits at most one mapped box that
still clears the existing spatial-continuity IoU floor.

The experiment is deliberately evidence-only. It does not train, add another
model family, alter production defaults, retain media, or infer injury/cause.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

from .artifacts import OPENVINO_OMZ_2023_FP16, verify_artifact_set
from .detector_continuity import ContinuityAccumulator
from .detector_orientation_fallback import select_orientation_fallback
from .detector_shootout import _CompiledDetector, _sha256
from .detector_thresholds import (
    _CONTINUITY_MIN_LINK_IOU,
    _CONTINUITY_THRESHOLD,
    _person_detections,
    _provision,
    _target_candidate,
)
from .evaluation import EvaluationSample, aggregate_person_down_evaluations, evaluate_person_down_candidates
from .openpose_association_diagnostics import _bounded_fallback, _decoded_candidates
from .openpose_diagnostics import _ReferenceRuntime, _pixel_bbox
from .openpose_pose_geometry import select_reference_pose
from .openpose_reference import ReferenceOpenPoseDecoder
from .openvino_omz import _OpenVINORuntime
from .person_down_e2e_diagnostics import _TemporalTrace, _corrected_posture, _measured_temporal_config
from .temporal import Observation, PersonDownEngine
from .validation import ValidationSampleSpec
from .validation_cli import load_manifest
from .video import OpenCVVideoFileSource

_RUNTIME_PREFIX = "2026.3.1"
_TRACK_ID = "continuity-track"
_ROTATIONS = (-1, 1)


def _window_name(sample: ValidationSampleSpec, timestamp_ms: int) -> str:
    if len(sample.labels) != 1:
        return "overall"
    label = sample.labels[0]
    if timestamp_ms < label.start_timestamp_ms:
        return "before"
    if timestamp_ms <= label.end_timestamp_ms:
        return "during"
    return "after"


def _freeze_windows(windows: dict[str, dict[str, int]]) -> dict[str, dict[str, int | float]]:
    frozen: dict[str, dict[str, int | float]] = {}
    for name, row in windows.items():
        frames = int(row["frames"])
        selected = int(row["selected_frames"])
        frozen[name] = {
            "frames": frames,
            "selected_frames": selected,
            "fallback_selected_frames": int(row["fallback_selected_frames"]),
            "coverage": selected / frames if frames else 0.0,
        }
    return frozen


def _scan_sample(
    sample: ValidationSampleSpec,
    detector: _CompiledDetector,
    runtime: _ReferenceRuntime,
    decoder: ReferenceOpenPoseDecoder,
) -> tuple[dict[str, Any], EvaluationSample]:
    path = sample.video_path
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("orientation diagnostic media identity mismatch")

    temporal_config = _measured_temporal_config()
    continuity = ContinuityAccumulator(min_link_iou=_CONTINUITY_MIN_LINK_IOU)
    engine = PersonDownEngine(sample.sample_id, "staged-real-orientation", temporal_config)
    trace = _TemporalTrace(temporal_config)
    events: list[dict] = []
    detector_ms = 0.0
    fallback_detector_ms = 0.0
    pose_inference_ms = 0.0
    pose_decode_ms = 0.0
    associated_frames = 0
    frames_processed = 0
    fallback_attempted_frames = 0
    fallback_accepted_frames = 0
    fallback_mapped_candidates = 0
    fallback_linked_candidates = 0
    first_timestamp_ms: int | None = None
    last_timestamp_ms: int | None = None
    windows: dict[str, dict[str, int]] = {}
    started_sample = time.perf_counter()

    with OpenCVVideoFileSource(path, start_timestamp_ms=sample.start_timestamp_ms) as frames:
        for frame in frames:
            frames_processed += 1
            if first_timestamp_ms is None:
                first_timestamp_ms = frame.timestamp_ms
            last_timestamp_ms = frame.timestamp_ms
            window = _window_name(sample, frame.timestamp_ms)
            row = windows.setdefault(window, {"frames": 0, "selected_frames": 0, "fallback_selected_frames": 0})
            row["frames"] += 1

            started = time.perf_counter()
            detections = _person_detections(detector, frame.image)
            detector_ms += (time.perf_counter() - started) * 1000.0
            primary = tuple(
                item for item in detections if float(item.confidence) >= _CONTINUITY_THRESHOLD
            )

            fallback_selected = None
            if not primary and continuity.previous is not None:
                fallback_attempted_frames += 1
                rotated: dict[int, tuple] = {}
                started = time.perf_counter()
                for turn in _ROTATIONS:
                    rotated_image = detector.np.ascontiguousarray(detector.np.rot90(frame.image, k=turn))
                    rotated[turn] = _person_detections(detector, rotated_image)
                elapsed = (time.perf_counter() - started) * 1000.0
                detector_ms += elapsed
                fallback_detector_ms += elapsed
                selection = select_orientation_fallback(
                    continuity.previous,
                    rotated,
                    min_confidence=_CONTINUITY_THRESHOLD,
                    min_link_iou=_CONTINUITY_MIN_LINK_IOU,
                )
                fallback_mapped_candidates += selection.mapped_candidate_count
                fallback_linked_candidates += selection.linked_candidate_count
                fallback_selected = selection.selected

            candidates = primary if primary else (() if fallback_selected is None else (fallback_selected,))
            selected = continuity.add(candidates)
            used_fallback = fallback_selected is not None and selected is not None and not primary
            if selected is not None:
                row["selected_frames"] += 1
                if used_fallback:
                    fallback_accepted_frames += 1
                    row["fallback_selected_frames"] += 1

            corrected_associated = None
            if selected is not None:
                bbox = _pixel_bbox(selected, frame.image)
                if bbox is not None:
                    started = time.perf_counter()
                    heatmaps, pafs, resized_width, resized_height = runtime.infer(frame.image)
                    pose_inference_ms += (time.perf_counter() - started) * 1000.0
                    started = time.perf_counter()
                    poses, scores = decoder(heatmaps, pafs)
                    shape = frame.image.shape
                    kwargs = dict(
                        selection_bbox=bbox,
                        frame_width=int(shape[1]),
                        frame_height=int(shape[0]),
                        resized_width=resized_width,
                        resized_height=resized_height,
                    )
                    corrected_associated = select_reference_pose(poses, scores, **kwargs)
                    if corrected_associated is None:
                        decoded = _decoded_candidates(poses, scores, **kwargs)
                        corrected_associated, _reason, _count = _bounded_fallback(decoded, bbox)
                    pose_decode_ms += (time.perf_counter() - started) * 1000.0

            if corrected_associated is not None:
                associated_frames += 1
            posture, confidence, basis = _corrected_posture(corrected_associated)
            trace.observe(frame.timestamp_ms, posture, confidence, basis)
            events.extend(engine.observe(Observation(
                timestamp_ms=frame.timestamp_ms,
                track_id=_TRACK_ID,
                posture=posture,
                confidence=confidence,
            )))

    elapsed_ms = (time.perf_counter() - started_sample) * 1000.0
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("orientation diagnostic media changed during execution")
    if frames_processed < 1 or first_timestamp_ms is None or last_timestamp_ms is None:
        raise RuntimeError("orientation diagnostic decoded no frames")
    decoded_start = min(sample.start_timestamp_ms, first_timestamp_ms)
    decoded_end = max(sample.end_timestamp_ms, last_timestamp_ms)
    if decoded_end <= decoded_start:
        raise RuntimeError("decoded interval must have positive duration")

    evaluation = evaluate_person_down_candidates(
        events,
        sample.labels,
        video_start_timestamp_ms=decoded_start,
        video_end_timestamp_ms=decoded_end,
    )
    continuity_totals = continuity.freeze()
    result = {
        "sample_id": sample.sample_id,
        "authorization_ref": sample.authorization_ref,
        "media_sha256": sample.media_sha256,
        "decoded_start_timestamp_ms": decoded_start,
        "decoded_end_timestamp_ms": decoded_end,
        "decoded_duration_ms": decoded_end - decoded_start,
        "frames_processed": frames_processed,
        "associated_frames": associated_frames,
        "candidate_events": len(events),
        "evaluation": asdict(evaluation),
        "continuity": asdict(continuity_totals) | {
            "coverage": continuity_totals.coverage,
            "link_rate": continuity_totals.link_rate,
            "reset_rate": continuity_totals.reset_rate,
        },
        "selection_windows": _freeze_windows(windows),
        "orientation_fallback": {
            "attempted_frames": fallback_attempted_frames,
            "accepted_frames": fallback_accepted_frames,
            "mapped_candidates": fallback_mapped_candidates,
            "linked_candidates": fallback_linked_candidates,
            "detector_inference_ms": fallback_detector_ms,
        },
        "temporal_trace": trace.freeze(),
        "detector_inference_ms": detector_ms,
        "pose_inference_ms": pose_inference_ms,
        "pose_decode_ms": pose_decode_ms,
        "elapsed_ms": elapsed_ms,
        "throughput_fps": frames_processed * 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0,
    }
    evaluated = EvaluationSample(
        sample_id=sample.sample_id,
        site_id=sample.site_id,
        camera_id=sample.camera_id,
        start_timestamp_ms=decoded_start,
        end_timestamp_ms=decoded_end,
        result=evaluation,
    )
    return result, evaluated


def run_orientation_diagnostic(manifest: str | Path, candidate_root: str | Path) -> dict[str, Any]:
    artifact_root, samples, config = load_manifest(manifest)
    if config.required_device != "CPU" or len(samples) != 2:
        raise RuntimeError("orientation diagnostic requires the bounded two-clip CPU subset")

    preparation_started = time.perf_counter()
    candidate = _target_candidate()
    candidate_dir = Path(candidate_root)
    _provision(candidate_dir, candidate)
    detector = _CompiledDetector(candidate, candidate_dir)
    verified = verify_artifact_set(artifact_root, OPENVINO_OMZ_2023_FP16)
    base_runtime = _OpenVINORuntime(verified, "CPU")
    if not base_runtime.runtime_version.startswith(_RUNTIME_PREFIX):
        raise RuntimeError("OpenVINO runtime version does not match reviewed release")
    runtime = _ReferenceRuntime(base_runtime)
    decoder = ReferenceOpenPoseDecoder()
    preparation_elapsed_ms = (time.perf_counter() - preparation_started) * 1000.0

    results: list[dict[str, Any]] = []
    evaluated: list[EvaluationSample] = []
    for sample in samples:
        item, sample_evaluation = _scan_sample(sample, detector, runtime, decoder)
        results.append(item)
        evaluated.append(sample_evaluation)
    aggregate = aggregate_person_down_evaluations(evaluated)
    total_frames = sum(int(item["frames_processed"]) for item in results)
    total_elapsed_ms = sum(float(item["elapsed_ms"]) for item in results)
    return {
        "schema_version": 1,
        "diagnostic": "person_down_orientation_fallback",
        "detector": candidate.name,
        "detector_threshold": _CONTINUITY_THRESHOLD,
        "continuity_min_link_iou": _CONTINUITY_MIN_LINK_IOU,
        "orientation_fallback": {
            "rot90_quarter_turns": list(_ROTATIONS),
            "same_detector_and_weights": True,
            "primary_miss_only": True,
            "requires_prior_spatial_link": True,
            "production_promoted": False,
        },
        "pose_model": "human-pose-estimation-0001",
        "runtime_version": base_runtime.runtime_version,
        "device": "CPU",
        "temporal_config": asdict(_measured_temporal_config()),
        "preparation_elapsed_ms": preparation_elapsed_ms,
        "samples": results,
        "aggregate": asdict(aggregate),
        "total_frames_processed": total_frames,
        "total_elapsed_ms": total_elapsed_ms,
        "throughput_fps": total_frames * 1000.0 / total_elapsed_ms if total_elapsed_ms > 0 else 0.0,
        "evidence_only": True,
        "commercial_accuracy_claim": False,
        "inferences_not_supported": ["fall", "injury", "cause", "fault", "intent"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(
            run_orientation_diagnostic(args.manifest, args.candidate_dir),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError):
        print("Person-down orientation diagnostic rejected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
