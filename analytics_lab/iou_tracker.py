"""Frozen simple-IoU control backend for tracker head-to-head measurement.

This control intentionally has no motion model, low-confidence recovery, lost
buffer, appearance/ReID features, or native/runtime dependency. It receives the
same detector observations as the ByteTrack slice. Pre-measurement defaults are
frozen here so real-video results cannot be used to tune the control after the
fact: existing tracks use detections >=0.5, new tracks require >=0.6, and raw
same-category IoU must be >=0.3. Track IDs are session-local only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .tracking import DetectionCandidate, NormalizedBox, TrackedDetection


def _score(value: float, name: str) -> float:
    if type(value) not in (int, float) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return float(value)


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
class SimpleIoUConfig:
    """Pre-measurement control parameters; do not tune after inspecting results."""

    track_threshold: float = 0.5
    new_track_threshold: float = 0.6
    min_iou: float = 0.3
    max_tracks: int = 512

    def __post_init__(self) -> None:
        for name in ("track_threshold", "new_track_threshold", "min_iou"):
            object.__setattr__(self, name, _score(getattr(self, name), name))
        if self.track_threshold > self.new_track_threshold:
            raise ValueError("track_threshold must be <= new_track_threshold")
        if type(self.max_tracks) is not int or self.max_tracks < 1:
            raise ValueError("max_tracks must be an integer >= 1")


@dataclass
class _Track:
    numeric_id: int
    category: str
    confidence: float
    box: NormalizedBox
    model_class_id: str | int | None

    @property
    def track_id(self) -> str:
        return f"iou-{self.numeric_id:06d}"


def _associate(
    tracks: tuple[_Track, ...],
    detections: tuple[DetectionCandidate, ...],
    minimum: float,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]]:
    """Deterministic maximum-cardinality same-category raw-IoU association."""

    adjacency: dict[int, tuple[int, ...]] = {}
    for track_index, track in enumerate(tracks):
        candidates: list[tuple[float, float, float, float, float, int]] = []
        for detection_index, detection in enumerate(detections):
            if track.category != detection.category:
                continue
            value = _iou(track.box, detection.box)
            if value >= minimum:
                candidates.append(
                    (
                        -value,
                        detection.box.x_min,
                        detection.box.y_min,
                        detection.box.x_max,
                        detection.box.y_max,
                        detection_index,
                    )
                )
        candidates.sort()
        adjacency[track_index] = tuple(item[-1] for item in candidates)

    by_detection: dict[int, int] = {}

    def augment(track_index: int, seen: set[int]) -> bool:
        for detection_index in adjacency.get(track_index, ()):
            if detection_index in seen:
                continue
            seen.add(detection_index)
            incumbent = by_detection.get(detection_index)
            if incumbent is None or augment(incumbent, seen):
                by_detection[detection_index] = track_index
                return True
        return False

    for track_index in sorted(
        adjacency,
        key=lambda index: (len(adjacency[index]), tracks[index].numeric_id),
    ):
        augment(track_index, set())

    matches = tuple(
        sorted(
            (track_index, detection_index)
            for detection_index, track_index in by_detection.items()
        )
    )
    matched_tracks = {item[0] for item in matches}
    matched_detections = {item[1] for item in matches}
    return (
        matches,
        tuple(index for index in range(len(tracks)) if index not in matched_tracks),
        tuple(
            index
            for index in range(len(detections))
            if index not in matched_detections
        ),
    )


class SimpleIoUAssociationBackend:
    """One-frame-memory raw-IoU control; unmatched tracks expire immediately."""

    def __init__(self, config: SimpleIoUConfig | None = None) -> None:
        self.config = config or SimpleIoUConfig()
        if not isinstance(self.config, SimpleIoUConfig):
            raise ValueError("config must be SimpleIoUConfig")
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._last_frame = -1

    @property
    def retained_tracks(self) -> int:
        return len(self._tracks)

    def update(
        self,
        frame_index: int,
        timestamp_ms: int,
        detections: tuple[DetectionCandidate, ...],
    ) -> Iterable[TrackedDetection]:
        del timestamp_ms  # TrackingSession owns timestamp monotonicity.
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if frame_index <= self._last_frame:
            raise ValueError("tracker frames must be strictly increasing")
        if not isinstance(detections, tuple) or any(
            not isinstance(item, DetectionCandidate) for item in detections
        ):
            raise ValueError("detections must be a tuple of DetectionCandidate values")
        if len(detections) > self.config.max_tracks:
            raise RuntimeError("candidate count exceeds simple-IoU capacity")

        high = tuple(
            item
            for item in detections
            if item.confidence >= self.config.track_threshold
        )
        active = tuple(
            sorted(self._tracks.values(), key=lambda track: track.numeric_id)
        )
        matches, _, unmatched_detection_indexes = _associate(
            active,
            high,
            self.config.min_iou,
        )

        current: dict[int, _Track] = {}
        for track_index, detection_index in matches:
            prior = active[track_index]
            detection = high[detection_index]
            current[prior.numeric_id] = _Track(
                prior.numeric_id,
                detection.category,
                detection.confidence,
                detection.box,
                detection.model_class_id,
            )

        for detection_index in unmatched_detection_indexes:
            detection = high[detection_index]
            if detection.confidence < self.config.new_track_threshold:
                continue
            if len(current) >= self.config.max_tracks:
                raise RuntimeError("track capacity exceeded")
            numeric_id = self._next_id
            self._next_id += 1
            current[numeric_id] = _Track(
                numeric_id,
                detection.category,
                detection.confidence,
                detection.box,
                detection.model_class_id,
            )

        self._tracks = current
        self._last_frame = frame_index
        return tuple(
            TrackedDetection(
                track.track_id,
                track.category,
                track.confidence,
                track.box,
                track.model_class_id,
            )
            for track in sorted(current.values(), key=lambda value: value.numeric_id)
        )
