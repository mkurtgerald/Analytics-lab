"""Bounded local-video ingestion and perception-to-temporal bridge.

This module intentionally contains no detector, pose model, tracker, model
download, URL reader, or webcam connector. A separately reviewed perception
adapter supplies temporary track IDs and posture observations for decoded
frames. The bridge owns timestamps and feeds those observations into the
model-independent temporal engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

from .temporal import Config, Observation, PersonDownEngine

_MAX_TIMESTAMP_MS = 253402300799999


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_ms: int
    image: Any

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("frame index must be a nonnegative integer")
        if type(self.timestamp_ms) is not int or not 0 <= self.timestamp_ms <= _MAX_TIMESTAMP_MS:
            raise ValueError("frame timestamp_ms is outside the supported UTC range")
        if self.image is None:
            raise ValueError("frame image is required")


@dataclass(frozen=True)
class PerceptionObservation:
    """One perception result for one frame; track IDs are session-local only."""

    track_id: str
    posture: str
    confidence: float

    def bind(self, timestamp_ms: int) -> Observation:
        return Observation(timestamp_ms, self.track_id, self.posture, self.confidence)


@dataclass(frozen=True)
class VideoRunConfig:
    max_frames: int = 100_000
    max_observations_per_frame: int = 256
    max_events: int = 4_096

    def __post_init__(self) -> None:
        for name in ("max_frames", "max_observations_per_frame", "max_events"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")


@dataclass(frozen=True)
class VideoRunResult:
    frames_processed: int
    observations_processed: int
    events: tuple[dict, ...]


class PerceptionAdapter(Protocol):
    def __call__(self, image: Any, frame_index: int, timestamp_ms: int) -> Iterable[PerceptionObservation]: ...


def run_frame_source(
    frames: Iterable[Frame],
    perception: PerceptionAdapter,
    engine: PersonDownEngine,
    config: VideoRunConfig | None = None,
) -> VideoRunResult:
    """Run a bounded, already-decoded frame source through perception + temporal logic."""
    if not callable(perception):
        raise ValueError("perception must be callable")
    if not isinstance(engine, PersonDownEngine):
        raise ValueError("engine must be a PersonDownEngine")
    cfg = config or VideoRunConfig()
    events: list[dict] = []
    frames_processed = 0
    observations_processed = 0
    last_index = -1
    last_timestamp = -1

    for frame in frames:
        if frames_processed >= cfg.max_frames:
            raise RuntimeError("video frame limit exceeded")
        if not isinstance(frame, Frame):
            raise ValueError("frame source must yield Frame objects")
        if frame.index <= last_index or frame.timestamp_ms <= last_timestamp:
            raise ValueError("frame source must provide strictly increasing indices and timestamps")
        last_index = frame.index
        last_timestamp = frame.timestamp_ms

        raw = perception(frame.image, frame.index, frame.timestamp_ms)
        if raw is None:
            raise ValueError("perception adapter must return an iterable, not None")
        seen_tracks: set[str] = set()
        per_frame = 0
        try:
            iterator = iter(raw)
        except TypeError as exc:
            raise ValueError("perception adapter must return an iterable") from exc
        for item in iterator:
            per_frame += 1
            if per_frame > cfg.max_observations_per_frame:
                raise RuntimeError("per-frame observation limit exceeded")
            if not isinstance(item, PerceptionObservation):
                raise ValueError("perception adapter returned an unsupported observation")
            if item.track_id in seen_tracks:
                raise ValueError("duplicate track observation in one frame")
            seen_tracks.add(item.track_id)
            observations_processed += 1
            for event in engine.observe(item.bind(frame.timestamp_ms)):
                events.append(event)
                if len(events) > cfg.max_events:
                    raise RuntimeError("video event limit exceeded")
        frames_processed += 1

    return VideoRunResult(frames_processed, observations_processed, tuple(events))


class OpenCVVideoFileSource:
    """Decode one bounded local file. Network/device capture is deliberately unsupported."""

    def __init__(
        self,
        path: str | Path,
        *,
        start_timestamp_ms: int,
        max_file_bytes: int = 512 * 1024 * 1024,
        max_fps: float = 240.0,
    ) -> None:
        raw = str(path)
        if "://" in raw or raw.startswith(("/dev/", "\\\\.\\")):
            raise ValueError("only ordinary local video files are supported")
        self.path = Path(path)
        if type(start_timestamp_ms) is not int or not 0 <= start_timestamp_ms <= _MAX_TIMESTAMP_MS:
            raise ValueError("start_timestamp_ms is outside the supported UTC range")
        if type(max_file_bytes) is not int or max_file_bytes < 1:
            raise ValueError("max_file_bytes must be an integer >= 1")
        if type(max_fps) not in (int, float) or not math.isfinite(max_fps) or max_fps <= 0:
            raise ValueError("max_fps must be finite and > 0")
        self.start_timestamp_ms = start_timestamp_ms
        self.max_file_bytes = max_file_bytes
        self.max_fps = float(max_fps)
        self._capture = None
        self._cv2 = None
        self._fps = None

    def __enter__(self) -> "OpenCVVideoFileSource":
        if not self.path.is_file():
            raise ValueError("video path must identify an existing regular file")
        if self.path.stat().st_size > self.max_file_bytes:
            raise ValueError("video file exceeds configured size limit")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for local video decoding") from exc
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise ValueError("local video file could not be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0 or fps > self.max_fps:
            capture.release()
            raise ValueError("video FPS is invalid or exceeds configured limit")
        self._cv2 = cv2
        self._capture = capture
        self._fps = fps
        return self

    def __iter__(self) -> Iterator[Frame]:
        if self._capture is None or self._fps is None:
            raise RuntimeError("video source must be used as a context manager")
        index = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                break
            elapsed_ms = round(index * 1000.0 / self._fps)
            timestamp_ms = self.start_timestamp_ms + elapsed_ms
            if timestamp_ms > _MAX_TIMESTAMP_MS:
                raise ValueError("video timestamp exceeds supported UTC range")
            yield Frame(index, timestamp_ms, image)
            index += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None


def run_local_video(
    path: str | Path,
    *,
    start_timestamp_ms: int,
    source_id: str,
    session_id: str,
    perception: PerceptionAdapter,
    temporal_config: Config | None = None,
    run_config: VideoRunConfig | None = None,
) -> VideoRunResult:
    """Decode one local file and bridge a reviewed perception adapter to the engine."""
    engine = PersonDownEngine(source_id, session_id, temporal_config)
    with OpenCVVideoFileSource(path, start_timestamp_ms=start_timestamp_ms) as frames:
        return run_frame_source(frames, perception, engine, run_config)
