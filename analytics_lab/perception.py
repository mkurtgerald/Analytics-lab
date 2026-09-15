"""Model-neutral pose semantics, temporary tracking, and posture observations.

No model runtime, downloader, biometric identity, ReID, or medical inference is
implemented here. A reviewed backend supplies per-frame pose candidates; this
module assigns temporary session-local track IDs and conservatively maps pose
geometry to upright/down/other/unknown observations for temporal reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Protocol

from .video import PerceptionObservation


def _finite(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _score(value: float, name: str) -> float:
    result = _finite(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must have positive width and height")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BBox") -> float:
        if not isinstance(other, BBox):
            raise ValueError("other must be a BBox")
        left = max(self.x1, other.x1)
        top = max(self.y1, other.y1)
        right = min(self.x2, other.x2)
        bottom = min(self.y2, other.y2)
        if right <= left or bottom <= top:
            return 0.0
        intersection = (right - left) * (bottom - top)
        return intersection / (self.area + other.area - intersection)


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or len(self.name) > 64:
            raise ValueError("keypoint name is required")
        object.__setattr__(self, "x", _finite(self.x, "keypoint x"))
        object.__setattr__(self, "y", _finite(self.y, "keypoint y"))
        object.__setattr__(self, "confidence", _score(self.confidence, "keypoint confidence"))


@dataclass(frozen=True)
class PoseCandidate:
    bbox: BBox
    keypoints: tuple[Keypoint, ...]
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.bbox, BBox):
            raise ValueError("pose bbox must be a BBox")
        if not isinstance(self.keypoints, tuple):
            raise ValueError("pose keypoints must be a tuple")
        names: set[str] = set()
        for item in self.keypoints:
            if not isinstance(item, Keypoint):
                raise ValueError("pose keypoints must contain Keypoint values")
            if item.name in names:
                raise ValueError("duplicate keypoint name")
            names.add(item.name)
        object.__setattr__(self, "confidence", _score(self.confidence, "pose confidence"))

    def keypoint(self, name: str) -> Keypoint | None:
        for item in self.keypoints:
            if item.name == name:
                return item
        return None


class PoseBackend(Protocol):
    def __call__(self, image: Any, frame_index: int, timestamp_ms: int) -> Iterable[PoseCandidate]: ...


@dataclass(frozen=True)
class TrackerConfig:
    # Low enough to survive upright->horizontal bbox rotation; still a temporary
    # geometric association threshold and not a validated tracking operating point.
    min_iou: float = 0.15
    max_missed_frames: int = 2
    max_tracks: int = 256

    def __post_init__(self) -> None:
        _score(self.min_iou, "min_iou")
        if self.min_iou <= 0:
            raise ValueError("min_iou must be > 0")
        for name in ("max_missed_frames", "max_tracks"):
            value = getattr(self, name)
            minimum = 0 if name == "max_missed_frames" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass
class _TrackState:
    bbox: BBox
    last_frame: int


class IoUTracker:
    """Deterministic, temporary within-session box association; not identity/ReID."""

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        if not isinstance(self.config, TrackerConfig):
            raise ValueError("config must be a TrackerConfig")
        self._tracks: dict[str, _TrackState] = {}
        self._next_id = 1
        self._last_frame = -1

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def update(self, frame_index: int, boxes: tuple[BBox, ...]) -> tuple[str, ...]:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if frame_index <= self._last_frame:
            raise ValueError("tracker frames must be strictly increasing")
        if not isinstance(boxes, tuple) or any(not isinstance(box, BBox) for box in boxes):
            raise ValueError("boxes must be a tuple of BBox values")
        if len(boxes) > self.config.max_tracks:
            raise RuntimeError("candidate count exceeds tracker capacity")

        # Work against a copy so a capacity or validation failure cannot partially
        # consume a frame or alter association state.
        active = {track_id: _TrackState(state.bbox, state.last_frame)
                  for track_id, state in self._tracks.items()
                  if frame_index - state.last_frame - 1 <= self.config.max_missed_frames}
        result: list[str | None] = [None] * len(boxes)
        pairs: list[tuple[float, str, int]] = []
        for track_id, state in active.items():
            for index, box in enumerate(boxes):
                score = state.bbox.iou(box)
                if score >= self.config.min_iou:
                    pairs.append((-score, track_id, index))
        pairs.sort()
        used_tracks: set[str] = set()
        used_boxes: set[int] = set()
        for _negative_score, track_id, index in pairs:
            if track_id in used_tracks or index in used_boxes:
                continue
            result[index] = track_id
            used_tracks.add(track_id)
            used_boxes.add(index)

        unmatched = [index for index, track_id in enumerate(result) if track_id is None]
        if len(active) + len(unmatched) > self.config.max_tracks:
            raise RuntimeError("track capacity exceeded")

        for index, track_id in enumerate(result):
            if track_id is not None:
                active[track_id] = _TrackState(boxes[index], frame_index)
        unmatched.sort(key=lambda i: (boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2, i))
        next_id = self._next_id
        for index in unmatched:
            track_id = f"track-{next_id:06d}"
            next_id += 1
            active[track_id] = _TrackState(boxes[index], frame_index)
            result[index] = track_id

        self._tracks = active
        self._next_id = next_id
        self._last_frame = frame_index
        return tuple(track_id for track_id in result if track_id is not None)


@dataclass(frozen=True)
class PostureConfig:
    min_keypoint_confidence: float = 0.35
    min_pose_confidence: float = 0.35
    orientation_threshold: float = 0.70
    upright_aspect_min: float = 1.15
    down_aspect_min: float = 1.15
    min_torso_fraction: float = 0.10

    def __post_init__(self) -> None:
        for name in ("min_keypoint_confidence", "min_pose_confidence", "orientation_threshold"):
            _score(getattr(self, name), name)
        if self.orientation_threshold <= 0.5:
            raise ValueError("orientation_threshold must exceed 0.5")
        for name in ("upright_aspect_min", "down_aspect_min", "min_torso_fraction"):
            value = _finite(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0")


@dataclass(frozen=True)
class PostureResult:
    posture: str
    confidence: float
    basis: str

    def __post_init__(self) -> None:
        if self.posture not in {"upright", "down", "other", "unknown"}:
            raise ValueError("unsupported posture")
        _score(self.confidence, "posture confidence")
        if not isinstance(self.basis, str) or not self.basis:
            raise ValueError("posture basis is required")


_REQUIRED = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def classify_posture(pose: PoseCandidate, config: PostureConfig | None = None) -> PostureResult:
    """Conservative geometric baseline; thresholds are engineering fixtures, not calibrated probabilities."""
    if not isinstance(pose, PoseCandidate):
        raise ValueError("pose must be a PoseCandidate")
    cfg = config or PostureConfig()
    if not isinstance(cfg, PostureConfig):
        raise ValueError("config must be a PostureConfig")
    if pose.confidence < cfg.min_pose_confidence:
        return PostureResult("unknown", pose.confidence, "pose_confidence_below_threshold")

    points = [pose.keypoint(name) for name in _REQUIRED]
    if any(point is None for point in points):
        return PostureResult("unknown", 0.0, "required_keypoints_missing")
    keypoints = [point for point in points if point is not None]
    minimum = min(point.confidence for point in keypoints)
    confidence = min(pose.confidence, minimum)
    if minimum < cfg.min_keypoint_confidence:
        return PostureResult("unknown", confidence, "required_keypoint_confidence_below_threshold")

    ls, rs, lh, rh = keypoints
    shoulder_x = (ls.x + rs.x) / 2.0
    shoulder_y = (ls.y + rs.y) / 2.0
    hip_x = (lh.x + rh.x) / 2.0
    hip_y = (lh.y + rh.y) / 2.0
    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y
    torso = math.hypot(dx, dy)
    diagonal = math.hypot(pose.bbox.width, pose.bbox.height)
    if torso < diagonal * cfg.min_torso_fraction:
        return PostureResult("unknown", confidence, "torso_geometry_too_small")

    vertical = abs(dy) / torso
    horizontal = abs(dx) / torso
    height_over_width = pose.bbox.height / pose.bbox.width
    width_over_height = pose.bbox.width / pose.bbox.height
    if vertical >= cfg.orientation_threshold and height_over_width >= cfg.upright_aspect_min:
        return PostureResult("upright", confidence, "vertical_torso_and_tall_bbox")
    if horizontal >= cfg.orientation_threshold and width_over_height >= cfg.down_aspect_min:
        return PostureResult("down", confidence, "horizontal_torso_and_wide_bbox")
    return PostureResult("other", confidence, "geometry_not_decisive")


@dataclass(frozen=True)
class PosePerceptionConfig:
    max_candidates_per_frame: int = 256

    def __post_init__(self) -> None:
        if type(self.max_candidates_per_frame) is not int or self.max_candidates_per_frame < 1:
            raise ValueError("max_candidates_per_frame must be an integer >= 1")


class PosePerceptionAdapter:
    """Bridge reviewed pose candidates to stable temporary observations."""

    def __init__(
        self,
        backend: PoseBackend,
        *,
        tracker: IoUTracker | None = None,
        posture_config: PostureConfig | None = None,
        config: PosePerceptionConfig | None = None,
    ) -> None:
        if not callable(backend):
            raise ValueError("backend must be callable")
        self.backend = backend
        self.tracker = tracker or IoUTracker()
        if not isinstance(self.tracker, IoUTracker):
            raise ValueError("tracker must be an IoUTracker")
        self.posture_config = posture_config or PostureConfig()
        if not isinstance(self.posture_config, PostureConfig):
            raise ValueError("posture_config must be a PostureConfig")
        self.config = config or PosePerceptionConfig()
        if not isinstance(self.config, PosePerceptionConfig):
            raise ValueError("config must be a PosePerceptionConfig")

    def __call__(self, image: Any, frame_index: int, timestamp_ms: int) -> tuple[PerceptionObservation, ...]:
        raw = self.backend(image, frame_index, timestamp_ms)
        if raw is None:
            raise ValueError("pose backend must return an iterable, not None")
        try:
            iterator = iter(raw)
        except TypeError as exc:
            raise ValueError("pose backend must return an iterable") from exc
        poses: list[PoseCandidate] = []
        for item in iterator:
            if len(poses) >= self.config.max_candidates_per_frame:
                raise RuntimeError("pose candidate limit exceeded")
            if not isinstance(item, PoseCandidate):
                raise ValueError("pose backend returned an unsupported candidate")
            poses.append(item)
        ids = self.tracker.update(frame_index, tuple(pose.bbox for pose in poses))
        return tuple(
            PerceptionObservation(track_id, result.posture, result.confidence)
            for track_id, pose in zip(ids, poses)
            for result in (classify_posture(pose, self.posture_config),)
        )
