"""Evidence-only person-down diagnostic with bounded orientation and pose recovery.

Held-out measurements showed two distinct failure modes in the reviewed path:
fallen people can be lost by the upright-oriented detector, and full-frame
OpenPose can decode a pose too far from the continuity-selected person box to
associate safely. This diagnostic keeps the exact same detector, weights,
confidence floor, pose model/decoder, posture rules and temporal rules.

Detector recovery remains limited to +/-90-degree views of the same detector
after a primary miss with prior spatial continuity. Pose recovery is attempted
only after full-frame safe association fails: the already-selected person box
is expanded by a fixed bounded margin, the same pinned OpenPose model is run on
that crop, and the same association rules are applied inside the crop.

The experiment is deliberately evidence-only. It does not train, add another
model family, alter temporal persistence, retain media, or infer injury/cause.
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
from .openpose_association_diagnostics import (
    _bounded_fallback,
    _candidate_metrics,
    _decoded_candidates,
    _nearest_candidate,
    _summary,
)
from .openpose_diagnostics import _ReferenceRuntime, _pixel_bbox
from .openpose_pose_geometry import AssociatedReferencePose, select_reference_pose
from .openpose_reference import ReferenceOpenPoseDecoder
from .openvino_omz import _OpenVINORuntime
from .perception import BBox
from .person_down_e2e_diagnostics import _TemporalTrace, _corrected_posture, _measured_temporal_config
from .pose_crop import (
    DEFAULT_PADDING_FRACTION,
    OPENPOSE_MAX_ASPECT_RATIO,
    extract_person_crop,
    plan_person_crop,
)
from .posture_quality_diagnostics import _pose_metrics
from .temporal import Observation, PersonDownEngine
from .validation import ValidationSampleSpec
from .validation_cli import load_manifest
from .video import OpenCVVideoFileSource

_RUNTIME_PREFIX = "2026.3.1"
_TRACK_ID = "continuity-track"
_ROTATIONS = (-1, 1)
# The first exact-head fragmentation measurement isolated one positive reset as
# truly diagonal/non-decisive: |horizontal - vertical| ~= 0.00017. In the same
# evidence run the prone-normal `other` population stayed at least ~0.056 away
# from the diagonal using its measured horizontal-min / vertical-max envelope.
# Keep a wide safety margin and convert only this narrow ambiguous geometry to
# `unknown`, where the existing 750 ms unknown-gap budget still fails closed.
_DIAGONAL_AMBIGUITY_DELTA = 0.02
_ASSOCIATION_METRICS = (
    "selection_iou",
    "center_distance_norm",
    "edge_gap_norm",
    "pose_width_ratio",
    "pose_height_ratio",
    "pose_area_ratio",
)
_POSTURE_METRICS = (
    "min_required_confidence",
    "torso_fraction",
    "vertical_fraction",
    "horizontal_fraction",
    "width_over_height",
    "height_over_width",
)


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


def _bounded_posture(item: AssociatedReferencePose | None) -> tuple[str, float, str]:
    """Turn only measured near-diagonal non-decisive geometry into unknown.

    This does not promote a pose to `down`, does not bridge `upright`, and does
    not alter the temporal duration/gap rules. It prevents a pose whose torso is
    essentially 45 degrees from being treated as contradictory evidence when
    the geometric classifier itself says the orientation is non-decisive.
    """
    posture, confidence, basis = _corrected_posture(item)
    if item is None or posture != "other" or basis != "geometry_not_decisive":
        return posture, confidence, basis
    metrics = _pose_metrics(item)
    horizontal = metrics.get("horizontal_fraction")
    vertical = metrics.get("vertical_fraction")
    if horizontal is None or vertical is None:
        return posture, confidence, basis
    if abs(float(horizontal) - float(vertical)) <= _DIAGONAL_AMBIGUITY_DELTA:
        return "unknown", confidence, "diagonal_torso_ambiguous"
    return posture, confidence, basis


class _FragmentationAccumulator:
    """Aggregate association/posture causes without retaining frame-level data."""

    def __init__(self) -> None:
        self.frames = 0
        self.association_reasons: Counter[str] = Counter()
        self.postures: Counter[str] = Counter()
        self.posture_bases: Counter[str] = Counter()
        self.unmatched_required_points: Counter[int] = Counter()
        self.unmatched_association_metrics: dict[str, list[float]] = {
            name: [] for name in _ASSOCIATION_METRICS
        }
        self.reset_reasons: Counter[str] = Counter()
        self.reset_bases: Counter[str] = Counter()
        self.reset_association_reasons: Counter[str] = Counter()
        self.reset_required_points: Counter[int] = Counter()
        self.reset_posture_metrics: dict[str, list[float]] = {
            name: [] for name in _POSTURE_METRICS
        }

    def add(
        self,
        *,
        association_reason: str,
        posture: str,
        basis: str,
        associated: AssociatedReferencePose | None,
        nearest: tuple[AssociatedReferencePose, dict[str, float | int]] | None,
        reset_reason: str | None,
    ) -> None:
        self.frames += 1
        self.association_reasons[association_reason] += 1
        self.postures[posture] += 1
        self.posture_bases[basis] += 1
        if associated is None and nearest is not None:
            item, metrics = nearest
            self.unmatched_required_points[item.required_points] += 1
            for name in _ASSOCIATION_METRICS:
                value = metrics.get(name)
                if value is not None:
                    self.unmatched_association_metrics[name].append(float(value))
        if reset_reason is None:
            return
        self.reset_reasons[reset_reason] += 1
        self.reset_bases[basis] += 1
        self.reset_association_reasons[association_reason] += 1
        if associated is None:
            return
        self.reset_required_points[associated.required_points] += 1
        metrics = _pose_metrics(associated)
        for name in _POSTURE_METRICS:
            value = metrics.get(name)
            if value is not None:
                self.reset_posture_metrics[name].append(float(value))

    def freeze(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "association_reason_counts": dict(sorted(self.association_reasons.items())),
            "posture_counts": dict(sorted(self.postures.items())),
            "posture_basis_counts": dict(sorted(self.posture_bases.items())),
            "unmatched_nearest_required_points_histogram": {
                str(key): value for key, value in sorted(self.unmatched_required_points.items())
            },
            "unmatched_nearest_association_metrics": {
                name: _summary(values)
                for name, values in self.unmatched_association_metrics.items()
            },
            "decisive_resets": {
                "count": sum(self.reset_reasons.values()),
                "reason_counts": dict(sorted(self.reset_reasons.items())),
                "posture_basis_counts": dict(sorted(self.reset_bases.items())),
                "association_reason_counts": dict(sorted(self.reset_association_reasons.items())),
                "required_points_histogram": {
                    str(key): value for key, value in sorted(self.reset_required_points.items())
                },
                "pose_metrics": {
                    name: _summary(values)
                    for name, values in self.reset_posture_metrics.items()
                },
            },
        }


def _associate_pose(
    image: Any,
    selection_bbox: BBox,
    runtime: _ReferenceRuntime,
    decoder: ReferenceOpenPoseDecoder,
) -> tuple[
    AssociatedReferencePose | None,
    tuple[AssociatedReferencePose, dict[str, float | int]] | None,
    str,
    float,
    float,
]:
    started = time.perf_counter()
    heatmaps, pafs, resized_width, resized_height = runtime.infer(image)
    inference_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    poses, scores = decoder(heatmaps, pafs)
    shape = image.shape
    kwargs = dict(
        selection_bbox=selection_bbox,
        frame_width=int(shape[1]),
        frame_height=int(shape[0]),
        resized_width=resized_width,
        resized_height=resized_height,
    )
    associated = select_reference_pose(poses, scores, **kwargs)
    if associated is not None:
        nearest = (associated, _candidate_metrics(associated, selection_bbox))
        reason = "baseline_overlap"
    else:
        decoded = _decoded_candidates(poses, scores, **kwargs)
        nearest = _nearest_candidate(decoded, selection_bbox)
        associated, reason, _count = _bounded_fallback(decoded, selection_bbox)
    decode_ms = (time.perf_counter() - started) * 1000.0
    return associated, nearest, reason, inference_ms, decode_ms


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
    crop_pose_inference_ms = 0.0
    crop_pose_decode_ms = 0.0
    associated_frames = 0
    frames_processed = 0
    fallback_attempted_frames = 0
    fallback_accepted_frames = 0
    fallback_mapped_candidates = 0
    fallback_linked_candidates = 0
    crop_attempted_frames = 0
    crop_associated_frames = 0
    first_timestamp_ms: int | None = None
    last_timestamp_ms: int | None = None
    windows: dict[str, dict[str, int]] = {}
    fragmentation: dict[str, _FragmentationAccumulator] = {}
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
            nearest = None
            association_reason = "no_selected_detection"
            if selected is not None:
                bbox = _pixel_bbox(selected, frame.image)
                if bbox is None:
                    association_reason = "invalid_selection_bbox"
                else:
                    (
                        corrected_associated,
                        nearest,
                        association_reason,
                        full_inference_ms,
                        full_decode_ms,
                    ) = _associate_pose(frame.image, bbox, runtime, decoder)
                    pose_inference_ms += full_inference_ms
                    pose_decode_ms += full_decode_ms

                    if corrected_associated is None:
                        crop_attempted_frames += 1
                        shape = frame.image.shape
                        plan = plan_person_crop(
                            bbox,
                            frame_width=int(shape[1]),
                            frame_height=int(shape[0]),
                        )
                        crop = extract_person_crop(frame.image, plan, detector.np)
                        (
                            crop_associated,
                            crop_nearest,
                            crop_reason,
                            crop_inference_ms,
                            crop_decode_ms,
                        ) = _associate_pose(crop, plan.selection_bbox, runtime, decoder)
                        pose_inference_ms += crop_inference_ms
                        pose_decode_ms += crop_decode_ms
                        crop_pose_inference_ms += crop_inference_ms
                        crop_pose_decode_ms += crop_decode_ms
                        corrected_associated = crop_associated
                        nearest = crop_nearest
                        association_reason = f"crop_{crop_reason}"
                        if corrected_associated is not None:
                            crop_associated_frames += 1

            if corrected_associated is not None:
                associated_frames += 1
            posture, confidence, basis = _bounded_posture(corrected_associated)
            before_resets = trace.reset_reasons.copy()
            trace.observe(frame.timestamp_ms, posture, confidence, basis)
            reset_delta = trace.reset_reasons - before_resets
            if sum(reset_delta.values()) > 1:
                raise RuntimeError("one observation produced multiple temporal reset reasons")
            reset_reason = next(iter(reset_delta), None)
            targets = ("overall",) if window == "overall" else ("overall", window)
            for name in targets:
                fragmentation.setdefault(name, _FragmentationAccumulator()).add(
                    association_reason=association_reason,
                    posture=posture,
                    basis=basis,
                    associated=corrected_associated,
                    nearest=nearest,
                    reset_reason=reset_reason,
                )
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
        "pose_crop_fallback": {
            "attempted_frames": crop_attempted_frames,
            "associated_frames": crop_associated_frames,
            "pose_inference_ms": crop_pose_inference_ms,
            "pose_decode_ms": crop_pose_decode_ms,
        },
        "temporal_trace": trace.freeze(),
        "fragmentation_windows": {
            name: accumulator.freeze()
            for name, accumulator in sorted(fragmentation.items())
        },
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
        "schema_version": 4,
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
        "pose_crop_fallback": {
            "trigger": "safe_full_frame_association_failure_only",
            "same_pose_model_and_decoder": True,
            "padding_fraction": DEFAULT_PADDING_FRACTION,
            "max_input_aspect_ratio": OPENPOSE_MAX_ASPECT_RATIO,
            "geometric_distortion": False,
            "association_rules_changed": False,
            "posture_rules_changed": False,
            "temporal_rules_changed": False,
            "production_promoted": False,
        },
        "fragmentation_diagnostic": {
            "aggregate_only": True,
            "frame_timestamps_retained": False,
            "measures": [
                "association_reason_counts",
                "unmatched_nearest_association_geometry",
                "posture_basis_counts",
                "temporal_reset_reason_counts",
                "reset_pose_geometry",
            ],
            "changes_inference_or_temporal_behavior": False,
        },
        "bounded_posture_correction": {
            "scope": "geometry_not_decisive_only",
            "diagonal_orientation_delta": _DIAGONAL_AMBIGUITY_DELTA,
            "output": "unknown",
            "down_promotion": False,
            "upright_bridge": False,
            "temporal_rules_changed": False,
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
