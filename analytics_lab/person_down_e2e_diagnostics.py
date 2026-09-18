"""Evidence-only end-to-end staged-real person-down temporal diagnostic.

This module reconnects the measured low-confidence detector continuity, bounded
OpenPose association, bounded three-keypoint posture fallback and the existing
PersonDownEngine on the exact rights-reviewed two-clip seed. It emits aggregate
measurements only; no media/model artifact or identity data is retained, and no
fall, injury, cause, intent or fault is inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

from .artifacts import OPENVINO_OMZ_2023_FP16, verify_artifact_set
from .detector_continuity import ContinuityAccumulator
from .detector_shootout import _CompiledDetector, _sha256
from .detector_thresholds import (
    _CONTINUITY_MIN_LINK_IOU,
    _CONTINUITY_THRESHOLD,
    _person_detections,
    _provision,
    _target_candidate,
)
from .evaluation import (
    EvaluationSample,
    aggregate_person_down_evaluations,
    evaluate_person_down_candidates,
)
from .openpose_association_diagnostics import _bounded_fallback, _decoded_candidates, _summary
from .openpose_diagnostics import _ReferenceRuntime, _pixel_bbox
from .openpose_pose_geometry import AssociatedReferencePose, select_reference_pose
from .openpose_reference import ReferenceOpenPoseDecoder
from .openvino_omz import _OpenVINORuntime
from .perception import classify_posture
from .posture_quality_diagnostics import _POSTURE_CONFIG, _REQUIRED, _three_point_fallback
from .temporal import Config as TemporalConfig, Observation, PersonDownEngine
from .validation import ValidationSampleSpec
from .validation_cli import load_manifest
from .video import OpenCVVideoFileSource

_RUNTIME_PREFIX = "2026.3.1"
_TRACK_ID = "continuity-track"
_BASELINE_TEMPORAL_MIN_CONFIDENCE = TemporalConfig().min_confidence
_PRIOR_MEASURED_TEMPORAL_CONFIG = TemporalConfig(
    min_confidence=_POSTURE_CONFIG.min_keypoint_confidence,
)


def _measured_temporal_config() -> TemporalConfig:
    """Apply one bounded persistence correction to the prior measured config.

    The accepted confidence correction produced 85 qualifying positive `down`
    frames, but the longest uninterrupted run was only 464 ms / 15 frames.
    Measured reset causes were dominated by `unknown` posture interruptions,
    while `upright` and `other` remain conflicting evidence. Bridge only an
    `unknown` span bounded by the existing 750 ms inter-observation gap budget;
    keep duration, sample count, confidence, TTL and capacity unchanged.
    """
    return TemporalConfig(
        min_confidence=_PRIOR_MEASURED_TEMPORAL_CONFIG.min_confidence,
        max_unknown_gap_ms=_PRIOR_MEASURED_TEMPORAL_CONFIG.max_gap_ms,
    )


def _corrected_posture(
    associated: AssociatedReferencePose | None,
) -> tuple[str, float, str]:
    """Return the measured posture path without broadening the accepted fallback."""
    if associated is None:
        return "unknown", 0.0, "no_associated_pose"
    result = classify_posture(associated.candidate, _POSTURE_CONFIG)
    if result.posture != "unknown" or associated.required_points != 3:
        return result.posture, result.confidence, result.basis
    fallback, basis = _three_point_fallback(associated)
    if fallback not in {"upright", "down"}:
        return "unknown", result.confidence, basis
    points = [associated.candidate.keypoint(name) for name in _REQUIRED]
    present = [point for point in points if point is not None]
    if len(present) != 3:
        raise RuntimeError("three-point fallback accepted an invalid required-point count")
    confidence = min(
        float(associated.candidate.confidence),
        *(float(point.confidence) for point in present),
    )
    return fallback, confidence, basis


class _TemporalTrace:
    def __init__(self, config: TemporalConfig) -> None:
        self.config = config
        self.postures: Counter[str] = Counter()
        self.bases: Counter[str] = Counter()
        self.down_confidences: list[float] = []
        self.qualified_down_frames = 0
        self.low_confidence_down_frames = 0
        self.bridged_unknown_frames = 0
        self.longest_bridged_unknown_gap_ms = 0
        self.reset_reasons: Counter[str] = Counter()
        self._run_start: int | None = None
        self._run_count = 0
        self._last_qualified_down_ms: int | None = None
        self._unknown_since_down = False
        self.longest_down_run_ms = 0
        self.longest_down_run_samples = 0
        self._raw_run_start: int | None = None
        self._raw_run_count = 0
        self.longest_raw_down_run_ms = 0
        self.longest_raw_down_run_samples = 0

    def _reset_qualified_run(self) -> None:
        self._run_start = None
        self._run_count = 0
        self._last_qualified_down_ms = None
        self._unknown_since_down = False

    def observe(self, timestamp_ms: int, posture: str, confidence: float, basis: str) -> None:
        self.postures[posture] += 1
        self.bases[basis] += 1
        if posture == "down":
            self.down_confidences.append(confidence)
            if self._raw_run_start is None:
                self._raw_run_start = timestamp_ms
                self._raw_run_count = 0
            self._raw_run_count += 1
            raw_duration = timestamp_ms - self._raw_run_start
            if (raw_duration, self._raw_run_count) > (
                self.longest_raw_down_run_ms,
                self.longest_raw_down_run_samples,
            ):
                self.longest_raw_down_run_ms = raw_duration
                self.longest_raw_down_run_samples = self._raw_run_count
        else:
            self._raw_run_start = None
            self._raw_run_count = 0

        qualified = posture == "down" and confidence >= self.config.min_confidence
        if posture == "down":
            if qualified:
                self.qualified_down_frames += 1
            else:
                self.low_confidence_down_frames += 1

        if qualified:
            if (
                self._unknown_since_down
                and self._last_qualified_down_ms is not None
                and timestamp_ms - self._last_qualified_down_ms > self.config.max_unknown_gap_ms
            ):
                self.reset_reasons["unknown_gap_exceeded"] += 1
                self._reset_qualified_run()
            if self._run_start is None:
                self._run_start = timestamp_ms
                self._run_count = 0
            self._run_count += 1
            self._last_qualified_down_ms = timestamp_ms
            self._unknown_since_down = False
            duration = timestamp_ms - self._run_start
            if (duration, self._run_count) > (self.longest_down_run_ms, self.longest_down_run_samples):
                self.longest_down_run_ms = duration
                self.longest_down_run_samples = self._run_count
            return

        if (
            posture == "unknown"
            and self.config.max_unknown_gap_ms > 0
            and self._run_count
            and self._last_qualified_down_ms is not None
            and timestamp_ms - self._last_qualified_down_ms <= self.config.max_unknown_gap_ms
        ):
            self.bridged_unknown_frames += 1
            self.longest_bridged_unknown_gap_ms = max(
                self.longest_bridged_unknown_gap_ms,
                timestamp_ms - self._last_qualified_down_ms,
            )
            self._unknown_since_down = True
            return

        if self._run_count:
            reason = (
                "down_confidence_below_threshold"
                if posture == "down"
                else f"posture_{posture}"
            )
            self.reset_reasons[reason] += 1
        self._reset_qualified_run()

    def freeze(self) -> dict[str, Any]:
        return {
            "posture_counts": dict(sorted(self.postures.items())),
            "basis_counts": dict(sorted(self.bases.items())),
            "qualified_down_frames": self.qualified_down_frames,
            "low_confidence_down_frames": self.low_confidence_down_frames,
            "bridged_unknown_frames": self.bridged_unknown_frames,
            "longest_bridged_unknown_gap_ms": self.longest_bridged_unknown_gap_ms,
            "down_confidence": _summary(self.down_confidences),
            "longest_raw_down_run_ms": self.longest_raw_down_run_ms,
            "longest_raw_down_run_samples": self.longest_raw_down_run_samples,
            "longest_qualified_down_run_ms": self.longest_down_run_ms,
            "longest_qualified_down_run_samples": self.longest_down_run_samples,
            "reset_reason_counts": dict(sorted(self.reset_reasons.items())),
        }


def _scan_sample(
    sample: ValidationSampleSpec,
    detector: _CompiledDetector,
    runtime: _ReferenceRuntime,
    decoder: ReferenceOpenPoseDecoder,
    temporal_config: TemporalConfig,
) -> tuple[dict[str, Any], EvaluationSample]:
    path = sample.video_path
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("end-to-end diagnostic media identity mismatch")
    continuity = ContinuityAccumulator(min_link_iou=_CONTINUITY_MIN_LINK_IOU)
    engine = PersonDownEngine(sample.sample_id, "staged-real-e2e", temporal_config)
    trace = _TemporalTrace(temporal_config)
    events: list[dict] = []
    detector_ms = 0.0
    pose_inference_ms = 0.0
    pose_decode_ms = 0.0
    associated_frames = 0
    frames_processed = 0
    first_timestamp_ms: int | None = None
    last_timestamp_ms: int | None = None
    started_sample = time.perf_counter()

    with OpenCVVideoFileSource(path, start_timestamp_ms=sample.start_timestamp_ms) as frames:
        for frame in frames:
            frames_processed += 1
            if first_timestamp_ms is None:
                first_timestamp_ms = frame.timestamp_ms
            last_timestamp_ms = frame.timestamp_ms
            started = time.perf_counter()
            detections = _person_detections(detector, frame.image)
            detector_ms += (time.perf_counter() - started) * 1000.0
            selected = continuity.add(
                item for item in detections if float(item.confidence) >= _CONTINUITY_THRESHOLD
            )
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
                        candidates = _decoded_candidates(poses, scores, **kwargs)
                        corrected_associated, _reason, _count = _bounded_fallback(candidates, bbox)
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
        raise RuntimeError("end-to-end diagnostic media changed during execution")
    if frames_processed < 1 or first_timestamp_ms is None or last_timestamp_ms is None:
        raise RuntimeError("end-to-end diagnostic decoded no frames")
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
    sample_result = {
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
        "continuity": asdict(continuity_totals),
        "temporal_trace": trace.freeze(),
        "detector_inference_ms": detector_ms,
        "pose_inference_ms": pose_inference_ms,
        "pose_decode_ms": pose_decode_ms,
        "elapsed_ms": elapsed_ms,
        "throughput_fps": frames_processed * 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0,
    }
    evaluated_sample = EvaluationSample(
        sample_id=sample.sample_id,
        site_id=sample.site_id,
        camera_id=sample.camera_id,
        start_timestamp_ms=decoded_start,
        end_timestamp_ms=decoded_end,
        result=evaluation,
    )
    return sample_result, evaluated_sample


def run_person_down_e2e_diagnostic(
    manifest: str | Path,
    candidate_root: str | Path,
    *,
    temporal_config: TemporalConfig | None = None,
) -> dict[str, Any]:
    artifact_root, samples, config = load_manifest(manifest)
    if config.required_device != "CPU" or len(samples) != 2:
        raise RuntimeError("end-to-end diagnostics require the bounded two-clip CPU seed")
    cfg = temporal_config or _measured_temporal_config()
    if not isinstance(cfg, TemporalConfig):
        raise ValueError("temporal_config must be a TemporalConfig")

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
        item, evaluation_sample = _scan_sample(sample, detector, runtime, decoder, cfg)
        results.append(item)
        evaluated.append(evaluation_sample)
    aggregate = aggregate_person_down_evaluations(evaluated)
    total_frames = sum(int(item["frames_processed"]) for item in results)
    total_elapsed_ms = sum(float(item["elapsed_ms"]) for item in results)
    return {
        "schema_version": 3,
        "diagnostic": "person_down_staged_real_e2e",
        "detector": candidate.name,
        "detector_threshold": _CONTINUITY_THRESHOLD,
        "continuity_min_link_iou": _CONTINUITY_MIN_LINK_IOU,
        "pose_model": "human-pose-estimation-0001",
        "runtime_version": base_runtime.runtime_version,
        "device": "CPU",
        "temporal_config": asdict(cfg),
        "temporal_correction": {
            "baseline_min_confidence": _BASELINE_TEMPORAL_MIN_CONFIDENCE,
            "prior_measured_min_confidence": _PRIOR_MEASURED_TEMPORAL_CONFIG.min_confidence,
            "measured_min_confidence": cfg.min_confidence,
            "prior_max_unknown_gap_ms": _PRIOR_MEASURED_TEMPORAL_CONFIG.max_unknown_gap_ms,
            "measured_max_unknown_gap_ms": cfg.max_unknown_gap_ms,
            "derived_from": "qualified_down_persistence_fragmented_by_measured_unknown_posture_breaks",
            "upright_or_other_bridge_allowed": False,
            "low_confidence_down_bridge_allowed": False,
            "production_promoted": False,
        },
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
            run_person_down_e2e_diagnostic(args.manifest, args.candidate_dir),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError):
        print("Person-down staged-real end-to-end diagnostic rejected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
