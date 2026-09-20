"""First-attempt frozen tracker head-to-head runner.

This module compares the already-frozen simple-IoU control with the portable
ByteTrack slice on identical detector observations and an already validated,
independently authored ground-truth package. It does not download media, author
labels, run a detector, tune tracker parameters, or establish commercial
accuracy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import tracemalloc
from typing import Any

from .bytetrack import ByteTrackAssociationBackend
from .iou_tracker import SimpleIoUAssociationBackend
from .tracking import DetectionCandidate, TrackingSession
from .tracking_evaluation import (
    TrackingEvaluationFrame,
    TrackingMetrics,
    evaluate_tracking,
)
from .tracking_ground_truth_package import TrackingGroundTruthPackage


@dataclass(frozen=True)
class TrackingDetectionFrame:
    """One detector-observation frame shared unchanged by both trackers."""

    frame_index: int
    timestamp_ms: int
    detections: tuple[DetectionCandidate, ...]

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if type(self.timestamp_ms) is not int or self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be a nonnegative integer")
        if not isinstance(self.detections, tuple) or any(
            not isinstance(item, DetectionCandidate) for item in self.detections
        ):
            raise ValueError("detections must be a tuple of DetectionCandidate values")


@dataclass(frozen=True)
class TrackerRuntimeMetrics:
    """Bounded association-only runtime/resource observations for one tracker."""

    frames: int
    detection_observations: int
    track_observations: int
    elapsed_seconds: float
    cpu_seconds: float
    throughput_fps: float
    mean_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    peak_python_bytes: int
    max_retained_tracks: int

    def __post_init__(self) -> None:
        for name in (
            "frames",
            "detection_observations",
            "track_observations",
            "peak_python_bytes",
            "max_retained_tracks",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "elapsed_seconds",
            "cpu_seconds",
            "throughput_fps",
            "mean_latency_ms",
            "p95_latency_ms",
            "max_latency_ms",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class TrackerRunResult:
    association: TrackingMetrics
    runtime: TrackerRuntimeMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.association, TrackingMetrics):
            raise ValueError("association must be TrackingMetrics")
        if not isinstance(self.runtime, TrackerRuntimeMetrics):
            raise ValueError("runtime must be TrackerRuntimeMetrics")


@dataclass(frozen=True)
class TrackerHeadToHeadResult:
    """Immutable first-attempt result ready for external evidence persistence."""

    annotation_sha256: str
    frame_manifest_sha256: str
    ground_truth_observations: int
    detection_observations: int
    simple_iou: TrackerRunResult
    portable_bytetrack: TrackerRunResult
    evidence_scope: str = "engineering_smoke_not_commercial_accuracy"
    first_attempt: bool = True

    def __post_init__(self) -> None:
        for name in ("annotation_sha256", "frame_manifest_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in ("ground_truth_observations", "detection_observations"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.simple_iou, TrackerRunResult) or not isinstance(
            self.portable_bytetrack, TrackerRunResult
        ):
            raise ValueError("tracker results must be TrackerRunResult values")
        if self.evidence_scope != "engineering_smoke_not_commercial_accuracy":
            raise ValueError("unsupported evidence_scope")
        if self.first_attempt is not True:
            raise ValueError("first_attempt must remain true for this runner")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result without changing metric names."""

        return asdict(self)


def _p95(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def _run_backend(
    backend: object,
    ground_truth: tuple[object, ...],
    detections: tuple[TrackingDetectionFrame, ...],
) -> TrackerRunResult:
    session = TrackingSession(backend)  # type: ignore[arg-type]
    evaluation_frames: list[TrackingEvaluationFrame] = []
    latencies_ms: list[float] = []
    detection_observations = 0
    track_observations = 0
    max_retained_tracks = 0

    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
        peak_before = 0
    else:
        peak_before = tracemalloc.get_traced_memory()[1]

    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    try:
        for truth_frame, detection_frame in zip(ground_truth, detections, strict=True):
            frame_start = time.perf_counter_ns()
            tracks = session.update(
                detection_frame.frame_index,
                detection_frame.timestamp_ms,
                detection_frame.detections,
            )
            latencies_ms.append((time.perf_counter_ns() - frame_start) / 1_000_000.0)
            detection_observations += len(detection_frame.detections)
            track_observations += len(tracks)
            retained = getattr(backend, "retained_tracks", len(tracks))
            if type(retained) is int and retained >= 0:
                max_retained_tracks = max(max_retained_tracks, retained)
            evaluation_frames.append(
                TrackingEvaluationFrame(
                    detection_frame.frame_index,
                    truth_frame.objects,  # type: ignore[attr-defined]
                    tracks,
                )
            )
    finally:
        elapsed_seconds = (time.perf_counter_ns() - wall_start) / 1_000_000_000.0
        cpu_seconds = (time.process_time_ns() - cpu_start) / 1_000_000_000.0
        _, peak_after = tracemalloc.get_traced_memory()
        peak_python_bytes = peak_after if owns_tracing else max(0, peak_after - peak_before)
        if owns_tracing:
            tracemalloc.stop()

    latency_values = tuple(latencies_ms)
    frames = len(detections)
    elapsed_floor = max(elapsed_seconds, 1e-12)
    runtime = TrackerRuntimeMetrics(
        frames=frames,
        detection_observations=detection_observations,
        track_observations=track_observations,
        elapsed_seconds=elapsed_seconds,
        cpu_seconds=cpu_seconds,
        throughput_fps=frames / elapsed_floor if frames else 0.0,
        mean_latency_ms=sum(latency_values) / frames if frames else 0.0,
        p95_latency_ms=_p95(latency_values),
        max_latency_ms=max(latency_values, default=0.0),
        peak_python_bytes=peak_python_bytes,
        max_retained_tracks=max_retained_tracks,
    )
    return TrackerRunResult(evaluate_tracking(evaluation_frames), runtime)


def compare_frozen_trackers(
    package: TrackingGroundTruthPackage,
    detection_frames: tuple[TrackingDetectionFrame, ...],
) -> TrackerHeadToHeadResult:
    """Run the frozen simple-IoU and portable ByteTrack slices once.

    ``package`` must already have been produced by
    :func:`parse_tracking_ground_truth_package`, which binds independent labels
    to canonical RGB24 frame hashes. The two trackers receive the exact same
    immutable detection tuples. Defaults are intentionally not configurable in
    this first-attempt runner so the pre-measurement control cannot be tuned
    after inspecting the comparison.
    """

    if not isinstance(package, TrackingGroundTruthPackage):
        raise ValueError("package must be a validated TrackingGroundTruthPackage")
    if not isinstance(detection_frames, tuple) or any(
        not isinstance(item, TrackingDetectionFrame) for item in detection_frames
    ):
        raise ValueError("detection_frames must be a tuple of TrackingDetectionFrame values")
    if len(detection_frames) != len(package.frames):
        raise ValueError("detection frames must exactly cover the ground-truth package")

    expected_indices = tuple(frame.frame_index for frame in package.frames)
    actual_indices = tuple(frame.frame_index for frame in detection_frames)
    if actual_indices != expected_indices:
        raise ValueError("detection frames must exactly match ground-truth frame order")
    timestamps = tuple(frame.timestamp_ms for frame in detection_frames)
    if any(current <= prior for prior, current in zip(timestamps, timestamps[1:])):
        raise ValueError("detection timestamps must be strictly increasing")

    detection_observations = sum(len(frame.detections) for frame in detection_frames)
    simple = _run_backend(
        SimpleIoUAssociationBackend(), package.frames, detection_frames
    )
    byte = _run_backend(
        ByteTrackAssociationBackend(), package.frames, detection_frames
    )
    return TrackerHeadToHeadResult(
        annotation_sha256=package.annotation_sha256,
        frame_manifest_sha256=package.frame_manifest_sha256,
        ground_truth_observations=package.object_observations,
        detection_observations=detection_observations,
        simple_iou=simple,
        portable_bytetrack=byte,
    )
