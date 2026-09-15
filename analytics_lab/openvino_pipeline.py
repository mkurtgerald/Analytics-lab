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

from .openvino_omz import OpenVINOOMZConfig, OpenVINOOMZPoseBackend, RuntimeFactory
from .perception import (
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


class OpenVINOOMZPerceptionAdapter:
    """Verified OMZ models -> pose -> temporary track -> posture observations."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        config: OpenVINOOMZPipelineConfig | None = None,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self.config = config or OpenVINOOMZPipelineConfig()
        if not isinstance(self.config, OpenVINOOMZPipelineConfig):
            raise ValueError("config must be an OpenVINOOMZPipelineConfig")
        self.backend = OpenVINOOMZPoseBackend(
            artifact_root,
            config=self.config.openvino,
            runtime_factory=runtime_factory,
        )
        self.runtime_version = self.backend.runtime_version
        self.device = self.config.openvino.device
        self._adapter = PosePerceptionAdapter(
            self.backend,
            tracker=IoUTracker(self.config.tracker),
            posture_config=self.config.posture,
            config=self.config.perception,
        )

    def __call__(self, image: Any, frame_index: int, timestamp_ms: int) -> tuple[PerceptionObservation, ...]:
        return self._adapter(image, frame_index, timestamp_ms)


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
) -> OpenVINOOMZRunResult:
    """Run one bounded ordinary local video file through the reviewed OMZ path."""
    cfg = config or OpenVINOOMZPipelineConfig()
    if not isinstance(cfg, OpenVINOOMZPipelineConfig):
        raise ValueError("config must be an OpenVINOOMZPipelineConfig")
    perception = OpenVINOOMZPerceptionAdapter(
        artifact_root,
        config=cfg,
        runtime_factory=runtime_factory,
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
