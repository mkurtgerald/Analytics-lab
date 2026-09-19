"""Platform-neutral detection/tracking contracts.

This module intentionally contains no detector, model downloader, biometric
identity, re-identification model, or tracker algorithm. Reviewed detector and
tracker donors plug in behind this boundary. Track IDs are session-local
association identifiers only and must never be presented as identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Protocol

_MAX_TIMESTAMP_MS = 253402300799999


def _finite(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _score(value: float, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


def _category(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("category must be a string")
    result = value.strip()
    if not result or len(result) > 64:
        raise ValueError("category must contain 1-64 non-whitespace characters")
    return result


@dataclass(frozen=True)
class NormalizedBox:
    """Inclusive-normalized image coordinates in detector-independent space."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "y_min", "x_max", "y_max"):
            value = _score(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("normalized box must have positive area")


@dataclass(frozen=True)
class DetectionCandidate:
    """One reviewed detector output before temporal association."""

    category: str
    confidence: float
    box: NormalizedBox
    model_class_id: str | int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", _category(self.category))
        object.__setattr__(self, "confidence", _score(self.confidence, "confidence"))
        if not isinstance(self.box, NormalizedBox):
            raise ValueError("box must be a NormalizedBox")
        if self.model_class_id is not None:
            if isinstance(self.model_class_id, str):
                if not self.model_class_id or len(self.model_class_id) > 128:
                    raise ValueError("model_class_id string is invalid")
            elif type(self.model_class_id) is int:
                if self.model_class_id < 0:
                    raise ValueError("model_class_id integer must be nonnegative")
            else:
                raise ValueError("model_class_id must be a string, integer, or None")


@dataclass(frozen=True)
class TrackedDetection:
    """One session-local tracked object; track_id is not an identity claim."""

    track_id: str
    category: str
    confidence: float
    box: NormalizedBox
    model_class_id: str | int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or not self.track_id or len(self.track_id) > 128:
            raise ValueError("track_id must contain 1-128 characters")
        object.__setattr__(self, "category", _category(self.category))
        object.__setattr__(self, "confidence", _score(self.confidence, "confidence"))
        if not isinstance(self.box, NormalizedBox):
            raise ValueError("box must be a NormalizedBox")
        DetectionCandidate(self.category, self.confidence, self.box, self.model_class_id)


class TrackingBackend(Protocol):
    """Replaceable donor adapter interface for one session."""

    def update(
        self,
        frame_index: int,
        timestamp_ms: int,
        detections: tuple[DetectionCandidate, ...],
    ) -> Iterable[TrackedDetection]: ...


@dataclass(frozen=True)
class TrackingSessionConfig:
    max_detections_per_frame: int = 512
    max_tracks_per_frame: int = 512

    def __post_init__(self) -> None:
        for name in ("max_detections_per_frame", "max_tracks_per_frame"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")


class TrackingSession:
    """Bound and validate a detector-agnostic tracker backend.

    This wrapper does not evaluate tracking accuracy. It standardizes input and
    output semantics so ByteTrack/Norfair/other reviewed donors can be measured
    behind one contract without leaking model-specific shapes into analytics.
    """

    def __init__(self, backend: TrackingBackend, config: TrackingSessionConfig | None = None) -> None:
        if not callable(getattr(backend, "update", None)):
            raise ValueError("tracking backend must expose update")
        self.backend = backend
        self.config = config or TrackingSessionConfig()
        if not isinstance(self.config, TrackingSessionConfig):
            raise ValueError("config must be TrackingSessionConfig")
        self._last_frame = -1
        self._last_timestamp = -1

    def update(
        self,
        frame_index: int,
        timestamp_ms: int,
        detections: Iterable[DetectionCandidate],
    ) -> tuple[TrackedDetection, ...]:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if type(timestamp_ms) is not int or not 0 <= timestamp_ms <= _MAX_TIMESTAMP_MS:
            raise ValueError("timestamp_ms is outside the supported UTC range")
        if frame_index <= self._last_frame or timestamp_ms <= self._last_timestamp:
            raise ValueError("frames and timestamps must be strictly increasing")

        try:
            values = tuple(detections)
        except TypeError as exc:
            raise ValueError("detections must be iterable") from exc
        if len(values) > self.config.max_detections_per_frame:
            raise RuntimeError("detection count exceeds configured bound")
        if any(not isinstance(item, DetectionCandidate) for item in values):
            raise ValueError("detections must contain DetectionCandidate values")

        raw = self.backend.update(frame_index, timestamp_ms, values)
        if raw is None:
            raise ValueError("tracking backend must return an iterable, not None")
        try:
            tracks = tuple(raw)
        except TypeError as exc:
            raise ValueError("tracking backend must return an iterable") from exc
        if len(tracks) > self.config.max_tracks_per_frame:
            raise RuntimeError("track count exceeds configured bound")
        if any(not isinstance(item, TrackedDetection) for item in tracks):
            raise ValueError("tracking backend returned an unsupported value")
        ids = [item.track_id for item in tracks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate track_id in one frame")

        self._last_frame = frame_index
        self._last_timestamp = timestamp_ms
        return tracks
