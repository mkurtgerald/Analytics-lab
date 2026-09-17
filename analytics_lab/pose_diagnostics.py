"""Evidence-only pose/posture diagnostics on the continuity-selected real track.

This command reuses the rights-bound two-clip GMDCSA-24 seed, the measured
person-detection-0200 low-threshold continuity selector, and the reviewed
human-pose-estimation-0001 model. It emits aggregate geometry/confidence and
classification-reason evidence only; it does not alter the production path or
retain decoded media.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

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
from .openvino_omz import PersonDetection, _OpenVINORuntime, pose_candidate_from_heatmaps
from .perception import BBox, PoseCandidate, PostureResult, classify_posture
from .validation import ValidationSampleSpec
from .validation_cli import load_manifest
from .video import OpenCVVideoFileSource

_REQUIRED = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
_KEYPOINT_THRESHOLD = 0.35
_RUNTIME_PREFIX = "2026.3.1"


def _pixel_bbox(box: DetectionBox, image: Any) -> BBox | None:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("image shape is required")
    height, width = int(shape[0]), int(shape[1])
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    left = min(1.0, max(0.0, float(box.x_min)))
    top = min(1.0, max(0.0, float(box.y_min)))
    right = min(1.0, max(0.0, float(box.x_max)))
    bottom = min(1.0, max(0.0, float(box.y_max)))
    if right <= left or bottom <= top:
        return None
    return BBox(left * width, top * height, right * width, bottom * height)


def _pose_geometry(pose: PoseCandidate) -> dict[str, float] | None:
    points = [pose.keypoint(name) for name in _REQUIRED]
    if any(point is None for point in points):
        return None
    ls, rs, lh, rh = [point for point in points if point is not None]
    shoulder_x = (ls.x + rs.x) / 2.0
    shoulder_y = (ls.y + rs.y) / 2.0
    hip_x = (lh.x + rh.x) / 2.0
    hip_y = (lh.y + rh.y) / 2.0
    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y
    torso = math.hypot(dx, dy)
    diagonal = math.hypot(pose.bbox.width, pose.bbox.height)
    if torso <= 0.0 or diagonal <= 0.0:
        return {"torso_fraction": 0.0, "verticality": 0.0, "horizontality": 0.0}
    return {
        "torso_fraction": torso / diagonal,
        "verticality": abs(dy) / torso,
        "horizontality": abs(dx) / torso,
    }


class _PoseAccumulator:
    def __init__(self) -> None:
        self.frames = 0
        self.selected_frames = 0
        self.pose_frames = 0
        self.postures: Counter[str] = Counter()
        self.bases: Counter[str] = Counter()
        self.selected_confidences: list[float] = []
        self.pose_confidences: list[float] = []
        self.keypoint_confidences: dict[str, list[float]] = {name: [] for name in _REQUIRED}
        self.keypoint_above_threshold: Counter[str] = Counter()
        self.width_over_height: list[float] = []
        self.area_fractions: list[float] = []
        self.torso_fractions: list[float] = []
        self.verticality: list[float] = []
        self.horizontality: list[float] = []

    def add(
        self,
        *,
        image: Any,
        selected: DetectionBox | None,
        pose: PoseCandidate | None,
        posture: PostureResult | None,
    ) -> None:
        self.frames += 1
        if selected is None:
            return
        self.selected_frames += 1
        self.selected_confidences.append(float(selected.confidence))
        if pose is None or posture is None:
            return
        self.pose_frames += 1
        self.postures[posture.posture] += 1
        self.bases[posture.basis] += 1
        self.pose_confidences.append(float(pose.confidence))
        shape = getattr(image, "shape", None)
        height, width = int(shape[0]), int(shape[1])
        self.width_over_height.append(pose.bbox.width / pose.bbox.height)
        self.area_fractions.append(pose.bbox.area / float(width * height))
        for name in _REQUIRED:
            point = pose.keypoint(name)
            if point is None:
                continue
            confidence = float(point.confidence)
            self.keypoint_confidences[name].append(confidence)
            if confidence >= _KEYPOINT_THRESHOLD:
                self.keypoint_above_threshold[name] += 1
        geometry = _pose_geometry(pose)
        if geometry is not None:
            self.torso_fractions.append(geometry["torso_fraction"])
            self.verticality.append(geometry["verticality"])
            self.horizontality.append(geometry["horizontality"])

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def freeze(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "selected_frames": self.selected_frames,
            "selection_coverage": self.selected_frames / self.frames if self.frames else 0.0,
            "pose_frames": self.pose_frames,
            "posture_counts": dict(sorted(self.postures.items())),
            "classification_basis_counts": dict(sorted(self.bases.items())),
            "selected_confidence": self._stats(self.selected_confidences),
            "pose_confidence": self._stats(self.pose_confidences),
            "keypoints": {
                name: self._stats(values) | {
                    "frames_at_or_above_0_35": self.keypoint_above_threshold[name],
                    "fraction_at_or_above_0_35": (
                        self.keypoint_above_threshold[name] / len(values) if values else 0.0
                    ),
                }
                for name, values in self.keypoint_confidences.items()
            },
            "bbox_width_over_height": self._stats(self.width_over_height),
            "bbox_area_fraction": self._stats(self.area_fractions),
            "torso_fraction_of_bbox_diagonal": self._stats(self.torso_fractions),
            "torso_verticality": self._stats(self.verticality),
            "torso_horizontality": self._stats(self.horizontality),
        }


def _sample_scan(
    sample: ValidationSampleSpec,
    detector: _CompiledDetector,
    pose_runtime: _OpenVINORuntime,
) -> tuple[dict[str, Any], float]:
    path = sample.video_path
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("pose diagnostic media identity mismatch")
    continuity = ContinuityAccumulator(min_link_iou=_CONTINUITY_MIN_LINK_IOU)
    overall = _PoseAccumulator()
    windows = {name: _PoseAccumulator() for name in ("before", "during", "after")}
    pose_inference_ms = 0.0
    with OpenCVVideoFileSource(path, start_timestamp_ms=sample.start_timestamp_ms) as frames:
        for frame in frames:
            detections = _person_detections(detector, frame.image)
            selected = continuity.add(
                item for item in detections if float(item.confidence) >= _CONTINUITY_THRESHOLD
            )
            pose = None
            posture = None
            if selected is not None:
                bbox = _pixel_bbox(selected, frame.image)
                if bbox is not None:
                    started = time.perf_counter()
                    heatmaps = pose_runtime.pose_heatmaps(frame.image, bbox)
                    pose_inference_ms += (time.perf_counter() - started) * 1000.0
                    pose = pose_candidate_from_heatmaps(
                        heatmaps, PersonDetection(bbox, float(selected.confidence))
                    )
                    posture = classify_posture(pose)
            overall.add(image=frame.image, selected=selected, pose=pose, posture=posture)
            window_name = _window(sample, frame.timestamp_ms)
            if window_name is not None:
                windows[window_name].add(
                    image=frame.image, selected=selected, pose=pose, posture=posture
                )
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("pose diagnostic media changed during execution")
    result: dict[str, Any] = {"sample_id": sample.sample_id, "overall": overall.freeze()}
    if len(sample.labels) == 1:
        result["windows"] = {name: acc.freeze() for name, acc in windows.items()}
    return result, pose_inference_ms


def run_pose_diagnostics(manifest: str | Path, candidate_root: str | Path) -> dict[str, Any]:
    artifact_root, samples, config = load_manifest(manifest)
    if config.required_device != "CPU" or len(samples) != 2:
        raise RuntimeError("pose diagnostics require the bounded two-clip CPU seed")
    candidate = _target_candidate()
    candidate_dir = Path(candidate_root)
    _provision(candidate_dir, candidate)
    detector = _CompiledDetector(candidate, candidate_dir)
    verified = verify_artifact_set(artifact_root, OPENVINO_OMZ_2023_FP16)
    pose_runtime = _OpenVINORuntime(verified, "CPU")
    if not pose_runtime.runtime_version.startswith(_RUNTIME_PREFIX):
        raise RuntimeError("OpenVINO runtime version does not match reviewed release")
    sample_results: list[dict[str, Any]] = []
    pose_inference_ms = 0.0
    for sample in samples:
        item, elapsed = _sample_scan(sample, detector, pose_runtime)
        sample_results.append(item)
        pose_inference_ms += elapsed
    pose_frames = sum(int(item["overall"]["pose_frames"]) for item in sample_results)
    return {
        "schema_version": 1,
        "detector": candidate.name,
        "detector_threshold": _CONTINUITY_THRESHOLD,
        "continuity_min_link_iou": _CONTINUITY_MIN_LINK_IOU,
        "pose_model": "human-pose-estimation-0001",
        "posture_thresholds": {
            "min_pose_confidence": 0.35,
            "min_required_keypoint_confidence": _KEYPOINT_THRESHOLD,
        },
        "samples": sample_results,
        "pose_inference_ms": pose_inference_ms,
        "pose_fps": pose_frames * 1000.0 / pose_inference_ms if pose_inference_ms > 0 else 0.0,
        "evidence_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(
            run_pose_diagnostics(args.manifest, args.candidate_dir),
            allow_nan=False, sort_keys=True, separators=(",", ":"),
        ))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError):
        print("Pose diagnostic rejected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
