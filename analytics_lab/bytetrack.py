"""Portable ByteTrack-derived association backend.

Adapted from FoundationVision/ByteTrack at commit
``d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`` (MIT, Yifu Zhang, 2021).

This is deliberately the smallest platform-neutral donor slice needed to test
ByteTrack's defining high-confidence -> low-confidence association behavior
behind :mod:`analytics_lab.tracking`. It does *not* claim parity with the full
upstream tracker: the upstream Kalman/LAP/native-extension path stays outside
this first slice until dependency/runtime evidence justifies admitting it.
Track IDs are session-local association identifiers, never identities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .tracking import DetectionCandidate, NormalizedBox, TrackedDetection

BYTE_TRACK_REVISION = "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99"


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
class ByteTrackAssociationConfig:
    """Bounded thresholds matching the upstream association stages by default.

    ``primary_min_similarity`` represents the donor's IoU/score fused similarity
    after converting its default 0.8 cost threshold to similarity. The secondary
    stage uses raw IoU, as upstream does for lower-confidence detections.
    """

    track_threshold: float = 0.5
    low_threshold: float = 0.1
    new_track_threshold: float = 0.6
    primary_min_similarity: float = 0.2
    secondary_min_iou: float = 0.5
    unconfirmed_min_similarity: float = 0.3
    track_buffer_frames: int = 30
    max_tracks: int = 512

    def __post_init__(self) -> None:
        for name in (
            "track_threshold", "low_threshold", "new_track_threshold",
            "primary_min_similarity", "secondary_min_iou", "unconfirmed_min_similarity",
        ):
            object.__setattr__(self, name, _score(getattr(self, name), name))
        if not self.low_threshold < self.track_threshold <= self.new_track_threshold:
            raise ValueError("require low_threshold < track_threshold <= new_track_threshold")
        for name in ("track_buffer_frames", "max_tracks"):
            value = getattr(self, name)
            minimum = 0 if name == "track_buffer_frames" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass
class _Track:
    numeric_id: int
    category: str
    confidence: float
    box: NormalizedBox
    model_class_id: str | int | None
    start_frame: int
    last_frame: int
    state: str

    @property
    def track_id(self) -> str:
        return f"bt-{self.numeric_id:06d}"


def _box_order(box: NormalizedBox, index: int) -> tuple[float, float, float, float, int]:
    return (box.x_min, box.y_min, box.x_max, box.y_max, index)


def _associate(
    tracks: tuple[_Track, ...],
    detections: tuple[DetectionCandidate, ...],
    similarity: Callable[[_Track, DetectionCandidate], float],
    minimum: float,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]]:
    """Deterministic max-cardinality association with donor-aligned score ordering.

    This intentionally avoids upstream ``lap``/``cython_bbox`` admission in the
    first portable slice. Edges are donor-style similarity gated, then ordered
    highest-first; an augmenting path preserves the maximum feasible number of
    existing tracks. Full LAP/Kalman parity remains a separately measured step.
    """
    adjacency: dict[int, tuple[int, ...]] = {}
    for track_index, track in enumerate(tracks):
        candidates: list[tuple[float, tuple[float, float, float, float, int], int]] = []
        for detection_index, detection in enumerate(detections):
            if track.category != detection.category:
                continue
            value = similarity(track, detection)
            if value >= minimum:
                candidates.append((-value, _box_order(detection.box, detection_index), detection_index))
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

    for track_index in sorted(adjacency, key=lambda i: (len(adjacency[i]), tracks[i].numeric_id)):
        augment(track_index, set())

    matches = tuple(sorted(((track_index, detection_index) for detection_index, track_index in by_detection.items())))
    matched_tracks = {item[0] for item in matches}
    matched_detections = {item[1] for item in matches}
    return (
        matches,
        tuple(i for i in range(len(tracks)) if i not in matched_tracks),
        tuple(i for i in range(len(detections)) if i not in matched_detections),
    )


class ByteTrackAssociationBackend:
    """Dependency-free, platform-neutral first ByteTrack donor slice.

    It preserves the donor's defining two-stage use of lower-confidence
    detections and lost-track recovery, but intentionally omits the upstream
    Kalman/LAP/native-extension stack. This backend is therefore an engineering
    integration baseline, not a claim of upstream ByteTrack accuracy.
    """

    def __init__(self, config: ByteTrackAssociationConfig | None = None) -> None:
        self.config = config or ByteTrackAssociationConfig()
        if not isinstance(self.config, ByteTrackAssociationConfig):
            raise ValueError("config must be ByteTrackAssociationConfig")
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._last_frame = -1
        self._first_frame: int | None = None

    @property
    def retained_tracks(self) -> int:
        return len(self._tracks)

    def _update_track(self, track: _Track, detection: DetectionCandidate, frame_index: int) -> None:
        track.box = detection.box
        track.confidence = detection.confidence
        track.model_class_id = detection.model_class_id
        track.last_frame = frame_index
        track.state = "tracked"

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
        if not isinstance(detections, tuple) or any(not isinstance(item, DetectionCandidate) for item in detections):
            raise ValueError("detections must be a tuple of DetectionCandidate values")
        if len(detections) > self.config.max_tracks:
            raise RuntimeError("candidate count exceeds ByteTrack association capacity")

        if self._first_frame is None:
            self._first_frame = frame_index

        # Unconfirmed tracks are allowed exactly one following frame, matching
        # the donor's short probation path. Lost tracks must remain eligible for
        # the current frame's first association before age-based removal; the
        # donor removes an over-age lost track only after that recovery chance.
        self._tracks = {
            track_id: track for track_id, track in self._tracks.items()
            if not (track.state == "unconfirmed" and frame_index - track.last_frame > 1)
        }

        high = tuple(item for item in detections if item.confidence >= self.config.track_threshold)
        low = tuple(item for item in detections if self.config.low_threshold < item.confidence < self.config.track_threshold)

        confirmed = tuple(
            sorted(
                (track for track in self._tracks.values() if track.state in {"tracked", "lost"}),
                key=lambda track: track.numeric_id,
            )
        )

        # Donor first association: IoU similarity fused with detection score.
        matches, unmatched_track_indexes, unmatched_high_indexes = _associate(
            confirmed,
            high,
            lambda track, detection: _iou(track.box, detection.box) * detection.confidence,
            self.config.primary_min_similarity,
        )
        for track_index, detection_index in matches:
            self._update_track(confirmed[track_index], high[detection_index], frame_index)

        # Donor second association: only still-tracked objects may use lower-score
        # detections, and the gate is raw IoU.
        unmatched_confirmed = tuple(confirmed[index] for index in unmatched_track_indexes)
        second_tracks = tuple(track for track in unmatched_confirmed if track.state == "tracked")
        matches_low, _, _ = _associate(
            second_tracks,
            low,
            lambda track, detection: _iou(track.box, detection.box),
            self.config.secondary_min_iou,
        )
        matched_second_ids: set[int] = set()
        for track_index, detection_index in matches_low:
            track = second_tracks[track_index]
            self._update_track(track, low[detection_index], frame_index)
            matched_second_ids.add(track.numeric_id)
        for track in second_tracks:
            if track.numeric_id not in matched_second_ids:
                track.state = "lost"
        # Previously-lost tracks unmatched in stage one remain lost.

        remaining_high = tuple(high[index] for index in unmatched_high_indexes)
        unconfirmed = tuple(
            sorted((track for track in self._tracks.values() if track.state == "unconfirmed"), key=lambda track: track.numeric_id)
        )
        matches_new, unmatched_unconfirmed, unmatched_remaining_high = _associate(
            unconfirmed,
            remaining_high,
            lambda track, detection: _iou(track.box, detection.box) * detection.confidence,
            self.config.unconfirmed_min_similarity,
        )
        for track_index, detection_index in matches_new:
            self._update_track(unconfirmed[track_index], remaining_high[detection_index], frame_index)
        for track_index in unmatched_unconfirmed:
            self._tracks.pop(unconfirmed[track_index].numeric_id, None)

        for detection_index in unmatched_remaining_high:
            detection = remaining_high[detection_index]
            if detection.confidence < self.config.new_track_threshold:
                continue
            if len(self._tracks) >= self.config.max_tracks:
                raise RuntimeError("track capacity exceeded")
            numeric_id = self._next_id
            self._next_id += 1
            state = "tracked" if frame_index == self._first_frame else "unconfirmed"
            self._tracks[numeric_id] = _Track(
                numeric_id=numeric_id,
                category=detection.category,
                confidence=detection.confidence,
                box=detection.box,
                model_class_id=detection.model_class_id,
                start_frame=frame_index,
                last_frame=frame_index,
                state=state,
            )

        # Match upstream lost-buffer order: only now remove tracks that remained
        # lost beyond the configured frame budget.
        self._tracks = {
            track_id: track for track_id, track in self._tracks.items()
            if not (track.state == "lost" and frame_index - track.last_frame > self.config.track_buffer_frames)
        }

        self._last_frame = frame_index
        current = sorted(
            (track for track in self._tracks.values() if track.state == "tracked" and track.last_frame == frame_index),
            key=lambda track: track.numeric_id,
        )
        return tuple(
            TrackedDetection(
                track.track_id,
                track.category,
                track.confidence,
                track.box,
                track.model_class_id,
            )
            for track in current
        )
