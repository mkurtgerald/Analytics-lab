"""Bounded, platform-neutral multi-object tracking evaluation.

This module measures tracker association outputs against labeled ground-truth
boxes. It deliberately avoids identity claims and does not implement MOTA,
HOTA, or any benchmark-specific protocol. The metrics here are engineering
stage diagnostics: matched/missed observations, false track observations,
session-local track-ID switches, track fragmentations, and matched IoU.

Real-world accuracy claims require a rights-cleared labeled dataset, exact data
identity/provenance, detector/runtime identity, and a documented evaluation
protocol. Synthetic fixtures exercise this code only; they are not accuracy
evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .tracking import NormalizedBox, TrackedDetection


def _iou(a: NormalizedBox, b: NormalizedBox) -> float:
    left = max(a.x_min, b.x_min)
    top = max(a.y_min, b.y_min)
    right = min(a.x_max, b.x_max)
    bottom = min(a.y_max, b.y_max)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    return intersection / (area_a + area_b - intersection)


@dataclass(frozen=True)
class GroundTruthObject:
    """One labeled object observation in a frame.

    ``object_id`` is a dataset-local annotation identifier, not a real-world
    identity assertion.
    """

    object_id: str
    category: str
    box: NormalizedBox

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id or len(self.object_id) > 128:
            raise ValueError("object_id must contain 1-128 characters")
        if not isinstance(self.category, str) or not self.category.strip() or len(self.category.strip()) > 64:
            raise ValueError("category must contain 1-64 non-whitespace characters")
        object.__setattr__(self, "category", self.category.strip())
        if not isinstance(self.box, NormalizedBox):
            raise ValueError("box must be a NormalizedBox")


@dataclass(frozen=True)
class TrackingEvaluationFrame:
    """Labeled objects and tracker outputs for one monotonically ordered frame."""

    frame_index: int
    ground_truth: tuple[GroundTruthObject, ...]
    tracks: tuple[TrackedDetection, ...]

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if not isinstance(self.ground_truth, tuple) or any(
            not isinstance(item, GroundTruthObject) for item in self.ground_truth
        ):
            raise ValueError("ground_truth must be a tuple of GroundTruthObject values")
        if not isinstance(self.tracks, tuple) or any(
            not isinstance(item, TrackedDetection) for item in self.tracks
        ):
            raise ValueError("tracks must be a tuple of TrackedDetection values")
        gt_ids = [item.object_id for item in self.ground_truth]
        track_ids = [item.track_id for item in self.tracks]
        if len(gt_ids) != len(set(gt_ids)):
            raise ValueError("duplicate ground-truth object_id in one frame")
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("duplicate track_id in one frame")


@dataclass(frozen=True)
class TrackingEvaluationConfig:
    min_iou: float = 0.5
    max_frames: int = 100_000
    max_objects_per_frame: int = 512

    def __post_init__(self) -> None:
        if type(self.min_iou) not in (int, float) or not math.isfinite(self.min_iou):
            raise ValueError("min_iou must be finite")
        object.__setattr__(self, "min_iou", float(self.min_iou))
        if not 0.0 < self.min_iou <= 1.0:
            raise ValueError("min_iou must be within (0, 1]")
        for name in ("max_frames", "max_objects_per_frame"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")


@dataclass(frozen=True)
class TrackingMetrics:
    frames: int
    ground_truth_observations: int
    matched_observations: int
    misses: int
    false_track_observations: int
    id_switches: int
    fragmentations: int
    mean_matched_iou: float
    continuity: float

    def __post_init__(self) -> None:
        for name in (
            "frames",
            "ground_truth_observations",
            "matched_observations",
            "misses",
            "false_track_observations",
            "id_switches",
            "fragmentations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("mean_matched_iou", "continuity"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite within [0, 1]")


def _match_frame(
    ground_truth: tuple[GroundTruthObject, ...],
    tracks: tuple[TrackedDetection, ...],
    min_iou: float,
) -> tuple[tuple[tuple[int, int, float], ...], tuple[int, ...], tuple[int, ...]]:
    """Return deterministic maximum-cardinality IoU-gated frame matches."""

    adjacency: dict[int, tuple[tuple[int, float], ...]] = {}
    for gt_index, gt in enumerate(ground_truth):
        edges: list[tuple[int, float]] = []
        for track_index, track in enumerate(tracks):
            if gt.category != track.category:
                continue
            value = _iou(gt.box, track.box)
            if value >= min_iou:
                edges.append((track_index, value))
        edges.sort(key=lambda item: (-item[1], tracks[item[0]].track_id, item[0]))
        adjacency[gt_index] = tuple(edges)

    by_track: dict[int, int] = {}

    def augment(gt_index: int, seen_tracks: set[int]) -> bool:
        for track_index, _ in adjacency.get(gt_index, ()):
            if track_index in seen_tracks:
                continue
            seen_tracks.add(track_index)
            incumbent = by_track.get(track_index)
            if incumbent is None or augment(incumbent, seen_tracks):
                by_track[track_index] = gt_index
                return True
        return False

    for gt_index in sorted(
        adjacency,
        key=lambda index: (len(adjacency[index]), ground_truth[index].object_id, index),
    ):
        augment(gt_index, set())

    matches: list[tuple[int, int, float]] = []
    for track_index, gt_index in by_track.items():
        matches.append((gt_index, track_index, _iou(ground_truth[gt_index].box, tracks[track_index].box)))
    matches.sort()
    matched_gt = {item[0] for item in matches}
    matched_tracks = {item[1] for item in matches}
    return (
        tuple(matches),
        tuple(index for index in range(len(ground_truth)) if index not in matched_gt),
        tuple(index for index in range(len(tracks)) if index not in matched_tracks),
    )


def evaluate_tracking(
    frames: Iterable[TrackingEvaluationFrame],
    config: TrackingEvaluationConfig | None = None,
) -> TrackingMetrics:
    """Evaluate one ordered labeled sequence.

    An ``id_switch`` is counted when a dataset-local ground-truth object that has
    previously been matched becomes matched to a different session-local tracker
    ID. A ``fragmentation`` is counted when a previously matched ground-truth
    object has at least one unmatched labeled observation and is later matched
    again. A gap followed by a new track ID can therefore count as both one
    fragmentation and one ID switch; those diagnose different failure modes.
    """

    cfg = config or TrackingEvaluationConfig()
    if not isinstance(cfg, TrackingEvaluationConfig):
        raise ValueError("config must be TrackingEvaluationConfig")

    try:
        values = tuple(frames)
    except TypeError as exc:
        raise ValueError("frames must be iterable") from exc
    if len(values) > cfg.max_frames:
        raise RuntimeError("frame count exceeds configured evaluation bound")
    if any(not isinstance(frame, TrackingEvaluationFrame) for frame in values):
        raise ValueError("frames must contain TrackingEvaluationFrame values")

    last_frame = -1
    gt_total = matched_total = misses = false_tracks = switches = fragments = 0
    iou_sum = 0.0
    last_track_by_object: dict[str, str] = {}
    was_matched_by_object: dict[str, bool] = {}
    gap_since_match: set[str] = set()

    for frame in values:
        if frame.frame_index <= last_frame:
            raise ValueError("frame_index values must be strictly increasing")
        last_frame = frame.frame_index
        if len(frame.ground_truth) > cfg.max_objects_per_frame or len(frame.tracks) > cfg.max_objects_per_frame:
            raise RuntimeError("object count exceeds configured evaluation bound")

        matches, unmatched_gt, unmatched_tracks = _match_frame(
            frame.ground_truth, frame.tracks, cfg.min_iou
        )
        gt_total += len(frame.ground_truth)
        matched_total += len(matches)
        misses += len(unmatched_gt)
        false_tracks += len(unmatched_tracks)

        for gt_index in unmatched_gt:
            object_id = frame.ground_truth[gt_index].object_id
            if was_matched_by_object.get(object_id, False):
                gap_since_match.add(object_id)

        for gt_index, track_index, value in matches:
            gt = frame.ground_truth[gt_index]
            track = frame.tracks[track_index]
            object_id = gt.object_id
            previous_track = last_track_by_object.get(object_id)
            if previous_track is not None and previous_track != track.track_id:
                switches += 1
            if object_id in gap_since_match:
                fragments += 1
                gap_since_match.discard(object_id)
            was_matched_by_object[object_id] = True
            last_track_by_object[object_id] = track.track_id
            iou_sum += value

    continuity = matched_total / gt_total if gt_total else 0.0
    mean_iou = iou_sum / matched_total if matched_total else 0.0
    return TrackingMetrics(
        frames=len(values),
        ground_truth_observations=gt_total,
        matched_observations=matched_total,
        misses=misses,
        false_track_observations=false_tracks,
        id_switches=switches,
        fragmentations=fragments,
        mean_matched_iou=mean_iou,
        continuity=continuity,
    )
