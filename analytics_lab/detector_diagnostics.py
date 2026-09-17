"""Bounded detector/pose/posture attribution for reviewed real-video evidence.

This diagnostic does not retain frames, download media/models, or make an
accuracy claim. It reuses the exact rights-bound validation manifest and the
reviewed OpenVINO/OMZ artifacts to quantify where perception becomes sparse.
Output is aggregate-only and contains no media paths or image data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .openvino_omz import (
    OpenVINOOMZPoseBackend,
    _image_size,
    parse_person_detections,
    pose_candidate_from_heatmaps,
)
from .perception import IoUTracker, classify_posture
from .validation import ValidationSampleSpec
from .validation_cli import load_manifest
from .video import OpenCVVideoFileSource


@dataclass(frozen=True)
class StageTotals:
    frames: int
    frames_with_detection: int
    detections: int
    frames_with_pose: int
    pose_candidates: int
    posture_upright: int
    posture_down: int
    posture_other: int
    posture_unknown: int
    track_assignments: int
    distinct_tracks: int


class _Accumulator:
    def __init__(self) -> None:
        self.frames = 0
        self.frames_with_detection = 0
        self.detections = 0
        self.frames_with_pose = 0
        self.pose_candidates = 0
        self.posture_upright = 0
        self.posture_down = 0
        self.posture_other = 0
        self.posture_unknown = 0
        self.track_assignments = 0
        self.track_ids: set[str] = set()

    def add(self, detections: int, postures: tuple[str, ...], track_ids: tuple[str, ...]) -> None:
        self.frames += 1
        if detections:
            self.frames_with_detection += 1
        self.detections += detections
        if postures:
            self.frames_with_pose += 1
        self.pose_candidates += len(postures)
        for posture in postures:
            if posture == "upright":
                self.posture_upright += 1
            elif posture == "down":
                self.posture_down += 1
            elif posture == "other":
                self.posture_other += 1
            elif posture == "unknown":
                self.posture_unknown += 1
            else:
                raise RuntimeError("unsupported posture diagnostic")
        self.track_assignments += len(track_ids)
        self.track_ids.update(track_ids)

    def freeze(self) -> StageTotals:
        return StageTotals(
            self.frames,
            self.frames_with_detection,
            self.detections,
            self.frames_with_pose,
            self.pose_candidates,
            self.posture_upright,
            self.posture_down,
            self.posture_other,
            self.posture_unknown,
            self.track_assignments,
            len(self.track_ids),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _window_name(sample: ValidationSampleSpec, timestamp_ms: int) -> str | None:
    if len(sample.labels) != 1:
        return None
    label = sample.labels[0]
    if timestamp_ms < label.start_timestamp_ms:
        return "before"
    if timestamp_ms <= label.end_timestamp_ms:
        return "during"
    return "after"


def _diagnose_sample(sample: ValidationSampleSpec, backend: OpenVINOOMZPoseBackend) -> dict[str, Any]:
    path = sample.video_path
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("diagnostic media identity mismatch")

    tracker = IoUTracker()
    totals = _Accumulator()
    windows = {name: _Accumulator() for name in ("before", "during", "after")}
    runtime = getattr(backend, "_runtime", None)
    if runtime is None:
        raise RuntimeError("diagnostic backend runtime unavailable")

    with OpenCVVideoFileSource(path, start_timestamp_ms=sample.start_timestamp_ms) as frames:
        for frame in frames:
            width, height = _image_size(frame.image)
            detections = parse_person_detections(
                runtime.detect_rows(frame.image),
                image_width=width,
                image_height=height,
                threshold=backend.config.detection_threshold,
                max_people=backend.config.max_people,
            )
            poses = tuple(
                pose_candidate_from_heatmaps(runtime.pose_heatmaps(frame.image, detection.bbox), detection)
                for detection in detections
            )
            track_ids = tracker.update(frame.index, tuple(pose.bbox for pose in poses))
            postures = tuple(classify_posture(pose).posture for pose in poses)
            totals.add(len(detections), postures, track_ids)
            window = _window_name(sample, frame.timestamp_ms)
            if window is not None:
                windows[window].add(len(detections), postures, track_ids)

    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("diagnostic media changed during execution")

    result: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "media_sha256": sample.media_sha256,
        "overall": asdict(totals.freeze()),
    }
    if len(sample.labels) == 1:
        result["windows"] = {name: asdict(acc.freeze()) for name, acc in windows.items()}
    return result


def run_diagnostics(manifest: str | Path) -> dict[str, Any]:
    artifact_root, samples, config = load_manifest(manifest)
    if config.required_device != "CPU":
        raise RuntimeError("diagnostic device must be CPU")
    backend = OpenVINOOMZPoseBackend(artifact_root)
    return {
        "schema_version": 1,
        "runtime_version": backend.runtime_version,
        "device": backend.config.device,
        "samples": [_diagnose_sample(sample, backend) for sample in samples],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_diagnostics(args.manifest), allow_nan=False, sort_keys=True, separators=(",", ":")))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as error:
        print(f"Detector diagnostic rejected ({type(error).__name__}).")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
