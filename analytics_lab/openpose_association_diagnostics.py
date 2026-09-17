"""Evidence-only diagnostics for decoded-pose to detector-track association.

This keeps the already measured detector continuity and pinned OpenPose decoding
path unchanged, then records aggregate geometry for the nearest decoded pose on
frames where the current overlap-based association succeeds or fails. No media
or model artifact is retained or emitted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

from .artifacts import OPENVINO_OMZ_2023_FP16, verify_artifact_set
from .detector_continuity import ContinuityAccumulator, DetectionBox
from .detector_shootout import _CompiledDetector, _sha256, _window
from .detector_thresholds import (
    _CONTINUITY_MIN_LINK_IOU,
    _CONTINUITY_THRESHOLD,
    _person_detections,
    _provision,
    _target_candidate,
)
from .openpose_diagnostics import _Accumulator, _ReferenceRuntime, _pixel_bbox
from .openpose_pose_geometry import (
    AssociatedReferencePose,
    candidate_from_coco_pose,
    select_reference_pose,
)
from .openpose_reference import ReferenceOpenPoseDecoder
from .openvino_omz import _OpenVINORuntime
from .perception import BBox
from .validation import ValidationSampleSpec
from .validation_cli import load_manifest
from .video import OpenCVVideoFileSource

_RUNTIME_PREFIX = "2026.3.1"


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "max": max(values),
    }


def _candidate_metrics(item: AssociatedReferencePose, selection: BBox) -> dict[str, float | int]:
    pose = item.candidate.bbox
    selection_diagonal = math.hypot(selection.width, selection.height)
    if selection_diagonal <= 0:
        raise ValueError("selection diagonal must be positive")
    pose_center_x = (pose.x1 + pose.x2) / 2.0
    pose_center_y = (pose.y1 + pose.y2) / 2.0
    selection_center_x = (selection.x1 + selection.x2) / 2.0
    selection_center_y = (selection.y1 + selection.y2) / 2.0
    center_distance = math.hypot(
        pose_center_x - selection_center_x,
        pose_center_y - selection_center_y,
    ) / selection_diagonal
    gap_x = max(selection.x1 - pose.x2, pose.x1 - selection.x2, 0.0)
    gap_y = max(selection.y1 - pose.y2, pose.y1 - selection.y2, 0.0)
    edge_gap = math.hypot(gap_x, gap_y) / selection_diagonal
    return {
        "selection_iou": item.selection_iou,
        "all_points_inside_selection": item.all_points_inside_selection,
        "required_points_inside_selection": item.required_points_inside_selection,
        "center_distance_norm": center_distance,
        "edge_gap_norm": edge_gap,
        "pose_width_ratio": pose.width / selection.width,
        "pose_height_ratio": pose.height / selection.height,
        "pose_area_ratio": pose.area / selection.area,
    }


def _decoded_candidates(
    poses: Iterable[Any],
    scores: Iterable[float],
    *,
    selection_bbox: BBox,
    frame_width: int,
    frame_height: int,
    resized_width: int,
    resized_height: int,
) -> tuple[AssociatedReferencePose, ...]:
    result: list[AssociatedReferencePose] = []
    for pose, score in zip(poses, scores):
        try:
            item = candidate_from_coco_pose(
                pose,
                decoder_score=float(score),
                selection_bbox=selection_bbox,
                frame_width=frame_width,
                frame_height=frame_height,
                resized_width=resized_width,
                resized_height=resized_height,
            )
        except ValueError:
            continue
        if item.required_points > 0:
            result.append(item)
    return tuple(result)


def _nearest_candidate(
    candidates: tuple[AssociatedReferencePose, ...], selection: BBox
) -> tuple[AssociatedReferencePose, dict[str, float | int]] | None:
    ranked: list[tuple[tuple[float, float, int, float], AssociatedReferencePose, dict[str, float | int]]] = []
    for item in candidates:
        metrics = _candidate_metrics(item, selection)
        ranked.append((
            (
                float(metrics["edge_gap_norm"]),
                float(metrics["center_distance_norm"]),
                -int(item.valid_points),
                -float(item.decoder_score),
            ),
            item,
            metrics,
        ))
    if not ranked:
        return None
    ranked.sort(key=lambda value: value[0])
    return ranked[0][1], ranked[0][2]


@dataclass
class _GeometryAccumulator:
    matched_frames: int = 0
    unmatched_frames_with_pose: int = 0
    matched: dict[str, list[float]] = field(default_factory=lambda: {
        "selection_iou": [], "center_distance_norm": [], "edge_gap_norm": [],
        "pose_width_ratio": [], "pose_height_ratio": [], "pose_area_ratio": [],
    })
    unmatched: dict[str, list[float]] = field(default_factory=lambda: {
        "selection_iou": [], "center_distance_norm": [], "edge_gap_norm": [],
        "pose_width_ratio": [], "pose_height_ratio": [], "pose_area_ratio": [],
    })
    unmatched_inside_points: dict[int, int] = field(default_factory=dict)
    unmatched_required_inside: dict[int, int] = field(default_factory=dict)

    def add(self, *, associated: AssociatedReferencePose | None,
            nearest: tuple[AssociatedReferencePose, dict[str, float | int]] | None) -> None:
        if associated is not None:
            self.matched_frames += 1
            if nearest is None:
                raise RuntimeError("associated pose missing diagnostic candidate")
            metrics = _candidate_metrics(associated, associated.candidate.bbox)
            # The caller replaces this self-relative placeholder below with the
            # actual selection-relative metrics by passing the associated pose as
            # nearest when available. Keep the branch explicit to catch misuse.
            metrics = nearest[1]
            for name in self.matched:
                self.matched[name].append(float(metrics[name]))
            return
        if nearest is None:
            return
        self.unmatched_frames_with_pose += 1
        item, metrics = nearest
        for name in self.unmatched:
            self.unmatched[name].append(float(metrics[name]))
        inside = int(metrics["all_points_inside_selection"])
        required = int(metrics["required_points_inside_selection"])
        self.unmatched_inside_points[inside] = self.unmatched_inside_points.get(inside, 0) + 1
        self.unmatched_required_inside[required] = self.unmatched_required_inside.get(required, 0) + 1

    def freeze(self) -> dict[str, Any]:
        return {
            "matched_frames": self.matched_frames,
            "unmatched_frames_with_decoded_pose": self.unmatched_frames_with_pose,
            "matched_nearest_geometry": {name: _summary(values) for name, values in self.matched.items()},
            "unmatched_nearest_geometry": {name: _summary(values) for name, values in self.unmatched.items()},
            "unmatched_all_points_inside_histogram": {
                str(key): value for key, value in sorted(self.unmatched_inside_points.items())
            },
            "unmatched_required_points_inside_histogram": {
                str(key): value for key, value in sorted(self.unmatched_required_inside.items())
            },
        }


def _scan_sample(
    sample: ValidationSampleSpec,
    detector: _CompiledDetector,
    runtime: _ReferenceRuntime,
    decoder: ReferenceOpenPoseDecoder,
) -> tuple[dict[str, Any], float, float]:
    path = sample.video_path
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("association diagnostic media identity mismatch")
    continuity = ContinuityAccumulator(min_link_iou=_CONTINUITY_MIN_LINK_IOU)
    overall = _Accumulator()
    windows = {name: _Accumulator() for name in ("before", "during", "after")}
    geometry = _GeometryAccumulator()
    geometry_windows = {name: _GeometryAccumulator() for name in ("before", "during", "after")}
    inference_ms = 0.0
    decode_ms = 0.0
    with OpenCVVideoFileSource(path, start_timestamp_ms=sample.start_timestamp_ms) as frames:
        for frame in frames:
            detections = _person_detections(detector, frame.image)
            selected = continuity.add(
                item for item in detections if float(item.confidence) >= _CONTINUITY_THRESHOLD
            )
            decoded_count = None
            associated = None
            nearest = None
            if selected is not None:
                bbox = _pixel_bbox(selected, frame.image)
                if bbox is not None:
                    started = time.perf_counter()
                    heatmaps, pafs, resized_width, resized_height = runtime.infer(frame.image)
                    inference_ms += (time.perf_counter() - started) * 1000.0
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
                    associated = select_reference_pose(poses, scores, **kwargs)
                    candidates = _decoded_candidates(poses, scores, **kwargs)
                    nearest = _nearest_candidate(candidates, bbox)
                    if associated is not None:
                        associated_metrics = _candidate_metrics(associated, bbox)
                        nearest = (associated, associated_metrics)
                    decode_ms += (time.perf_counter() - started) * 1000.0
                    decoded_count = len(poses)
            overall.add(selected=selected, decoded_count=decoded_count, associated=associated)
            geometry.add(associated=associated, nearest=nearest)
            window_name = _window(sample, frame.timestamp_ms)
            if window_name is not None:
                windows[window_name].add(
                    selected=selected, decoded_count=decoded_count, associated=associated
                )
                geometry_windows[window_name].add(associated=associated, nearest=nearest)
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("association diagnostic media changed during execution")
    result: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "overall": overall.freeze(),
        "association_geometry": geometry.freeze(),
    }
    if len(sample.labels) == 1:
        result["windows"] = {name: accumulator.freeze() for name, accumulator in windows.items()}
        result["association_geometry_windows"] = {
            name: accumulator.freeze() for name, accumulator in geometry_windows.items()
        }
    return result, inference_ms, decode_ms


def run_association_diagnostic(manifest: str | Path, candidate_root: str | Path) -> dict[str, Any]:
    artifact_root, samples, config = load_manifest(manifest)
    if config.required_device != "CPU" or len(samples) != 2:
        raise RuntimeError("association diagnostics require the bounded two-clip CPU seed")
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
    results = []
    inference_ms = 0.0
    decode_ms = 0.0
    for sample in samples:
        item, item_inference_ms, item_decode_ms = _scan_sample(sample, detector, runtime, decoder)
        results.append(item)
        inference_ms += item_inference_ms
        decode_ms += item_decode_ms
    inferred_frames = sum(int(item["overall"]["inferred_frames"]) for item in results)
    return {
        "schema_version": 1,
        "diagnostic": "openpose_track_association_geometry",
        "detector": candidate.name,
        "detector_threshold": _CONTINUITY_THRESHOLD,
        "continuity_min_link_iou": _CONTINUITY_MIN_LINK_IOU,
        "pose_model": "human-pose-estimation-0001",
        "association_rule": "pose_bbox_iou_gt_0_or_any_decoded_keypoint_inside_selection",
        "samples": results,
        "inference_ms": inference_ms,
        "decode_ms": decode_ms,
        "inferred_frames": inferred_frames,
        "pose_model_fps": inferred_frames * 1000.0 / inference_ms if inference_ms > 0 else 0.0,
        "evidence_only": True,
        "commercial_accuracy_claim": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(
            run_association_diagnostic(args.manifest, args.candidate_dir),
            allow_nan=False, sort_keys=True, separators=(",", ":"),
        ))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError):
        print("OpenPose association diagnostic rejected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
