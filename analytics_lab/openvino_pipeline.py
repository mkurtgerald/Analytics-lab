"""Composition boundary for the reviewed OpenVINO/Open Model Zoo perception path.

This module adds no downloader, network source, device capture, identity, ReID,
or release behavior. It composes already-reviewed local model artifacts and
runtime inference with temporary tracking, posture classification, and the
existing temporal candidate-event engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .detector_continuity import DetectionBox
from .detector_orientation_fallback import select_orientation_fallback
from .openvino_omz import (
    OpenVINOOMZConfig, OpenVINOOMZPoseBackend, PersonDetection, RuntimeFactory, _image_size,
)
from .perception import (
    BBox,
    IoUTracker,
    PosePerceptionAdapter,
    PosePerceptionConfig,
    PostureConfig,
    TrackerConfig,
)
from .temporal import Config, PersonDownEngine
from .video import (
    Frame,
    PerceptionObservation,
    VideoRunConfig,
    VideoRunResult,
    run_frame_source,
    run_local_video,
)


@dataclass(frozen=True)
class OpenVINOOMZPipelineConfig:
    openvino: OpenVINOOMZConfig = field(default_factory=OpenVINOOMZConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    perception: PosePerceptionConfig = field(default_factory=PosePerceptionConfig)
    temporal: Config = field(default_factory=Config)
    video: VideoRunConfig = field(default_factory=VideoRunConfig)
    orientation_recovery: bool = False

    def __post_init__(self) -> None:
        expected = (
            ("openvino", OpenVINOOMZConfig),
            ("tracker", TrackerConfig),
            ("posture", PostureConfig),
            ("perception", PosePerceptionConfig),
            ("temporal", Config),
            ("video", VideoRunConfig),
        )
        for name, kind in expected:
            if not isinstance(getattr(self, name), kind):
                raise ValueError(f"{name} must be a {kind.__name__}")
        if type(self.orientation_recovery) is not bool:
            raise ValueError("orientation_recovery must be boolean")
        if self.orientation_recovery and self.openvino.detector_model != "person-detection-0200":
            raise ValueError("orientation recovery requires person-detection-0200")


@dataclass(frozen=True)
class OpenVINOOMZRunResult:
    runtime_version: str
    device: str
    video: VideoRunResult

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_version, str) or not self.runtime_version.strip() or len(self.runtime_version) > 128:
            raise ValueError("runtime_version must be a bounded nonempty string")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device is required")
        if not isinstance(self.video, VideoRunResult):
            raise ValueError("video must be a VideoRunResult")


class _OrientationRecoveryBackend:
    """One clip's guarded recovery state; shared model backends remain stateless."""

    def __init__(self, backend: OpenVINOOMZPoseBackend, config: OpenVINOOMZPipelineConfig) -> None:
        self.backend = backend
        self.min_iou = config.tracker.min_iou
        self.max_gap_ms = config.temporal.max_gap_ms
        self.previous: tuple[int, int, int, int, PersonDetection] | None = None

    def clear(self) -> None:
        self.previous = None

    @staticmethod
    def _normalized(detection: PersonDetection, width: int, height: int) -> DetectionBox:
        box = detection.bbox
        return DetectionBox(detection.confidence, box.x1 / width, box.y1 / height,
                            box.x2 / width, box.y2 / height)

    def __call__(self, image: Any, frame_index: int, timestamp_ms: int) -> tuple:
        prior = self.previous
        self.clear()
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if type(timestamp_ms) is not int or timestamp_ms < 0:
            raise ValueError("timestamp_ms must be a nonnegative integer")
        width, height = _image_size(image)
        detections = self.backend.detect_people(image)
        if (not detections and prior is not None and frame_index == prior[0] + 1
                and 0 < timestamp_ms - prior[1] <= self.max_gap_ms
                and (width, height) == prior[2:4]):
            rotated = {
                turn: tuple(self._normalized(item, height, width)
                            for item in self.backend.detect_people(image, quarter_turn=turn))
                for turn in (-1, 1)
            }
            selected = select_orientation_fallback(
                self._normalized(prior[4], width, height), rotated,
                min_confidence=self.backend.config.detection_threshold,
                min_link_iou=self.min_iou,
            ).selected
            if selected is not None:
                detections = (PersonDetection(BBox(
                    selected.x_min * width, selected.y_min * height,
                    selected.x_max * width, selected.y_max * height,
                ), selected.confidence),)
        poses = self.backend.poses_from_detections(image, detections)
        if len(detections) == 1:
            self.previous = frame_index, timestamp_ms, width, height, detections[0]
        return poses


class OpenVINOOMZPerceptionAdapter:
    """Verified OMZ models -> pose -> temporary track -> posture observations."""

    def __init__(
        self,
        artifact_root: str | Path | None,
        *,
        config: OpenVINOOMZPipelineConfig | None = None,
        runtime_factory: RuntimeFactory | None = None,
        backend: OpenVINOOMZPoseBackend | None = None,
    ) -> None:
        self.config = config or OpenVINOOMZPipelineConfig()
        if not isinstance(self.config, OpenVINOOMZPipelineConfig):
            raise ValueError("config must be an OpenVINOOMZPipelineConfig")
        self.config.__post_init__()
        if backend is not None:
            if runtime_factory is not None:
                raise ValueError("runtime_factory cannot be combined with a prepared backend")
            if not isinstance(backend, OpenVINOOMZPoseBackend):
                raise ValueError("backend must be an OpenVINOOMZPoseBackend")
            if backend.config != self.config.openvino:
                raise ValueError("prepared backend configuration does not match the pipeline")
            self.backend = backend
        else:
            if artifact_root is None:
                raise ValueError("artifact_root is required when no prepared backend is supplied")
            self.backend = OpenVINOOMZPoseBackend(
                artifact_root,
                config=self.config.openvino,
                runtime_factory=runtime_factory,
            )
        self.runtime_version = self.backend.runtime_version
        self.device = self.config.openvino.device
        self._recovery = (_OrientationRecoveryBackend(self.backend, self.config)
                          if self.config.orientation_recovery else None)
        self._adapter = PosePerceptionAdapter(
            self._recovery if self._recovery is not None else self.backend,
            tracker=IoUTracker(self.config.tracker),
            posture_config=self.config.posture,
            config=self.config.perception,
        )

    def __call__(self, image: Any, frame_index: int, timestamp_ms: int) -> tuple[PerceptionObservation, ...]:
        try:
            return self._adapter(image, frame_index, timestamp_ms)
        except Exception:
            if self._recovery is not None:
                self._recovery.clear()
            raise


def run_frame_source_openvino_omz(
    frames: Iterable[Frame],
    *,
    artifact_root: str | Path,
    source_id: str,
    session_id: str,
    config: OpenVINOOMZPipelineConfig | None = None,
    runtime_factory: RuntimeFactory | None = None,
) -> OpenVINOOMZRunResult:
    """Run already-decoded frames through the complete reviewed OMZ baseline."""
    cfg = config or OpenVINOOMZPipelineConfig()
    if not isinstance(cfg, OpenVINOOMZPipelineConfig):
        raise ValueError("config must be an OpenVINOOMZPipelineConfig")
    perception = OpenVINOOMZPerceptionAdapter(
        artifact_root,
        config=cfg,
        runtime_factory=runtime_factory,
    )
    engine = PersonDownEngine(source_id, session_id, cfg.temporal)
    result = run_frame_source(frames, perception, engine, cfg.video)
    return OpenVINOOMZRunResult(perception.runtime_version, perception.device, result)


def run_local_video_openvino_omz(
    path: str | Path,
    *,
    artifact_root: str | Path,
    start_timestamp_ms: int,
    source_id: str,
    session_id: str,
    config: OpenVINOOMZPipelineConfig | None = None,
    runtime_factory: RuntimeFactory | None = None,
    backend: OpenVINOOMZPoseBackend | None = None,
) -> OpenVINOOMZRunResult:
    """Run one bounded ordinary local video file through the reviewed OMZ path.

    A caller may supply a previously verified/compiled backend to amortize model
    initialization across independent clips. A fresh tracking/posture adapter is
    still created for every call so session-local tracks cannot bleed between
    validation samples.
    """
    cfg = config or OpenVINOOMZPipelineConfig()
    if not isinstance(cfg, OpenVINOOMZPipelineConfig):
        raise ValueError("config must be an OpenVINOOMZPipelineConfig")
    perception = OpenVINOOMZPerceptionAdapter(
        artifact_root if backend is None else None,
        config=cfg,
        runtime_factory=runtime_factory,
        backend=backend,
    )
    result = run_local_video(
        path,
        start_timestamp_ms=start_timestamp_ms,
        source_id=source_id,
        session_id=session_id,
        perception=perception,
        temporal_config=cfg.temporal,
        run_config=cfg.video,
    )
    return OpenVINOOMZRunResult(perception.runtime_version, perception.device, result)
