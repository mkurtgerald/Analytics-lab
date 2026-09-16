"""Rights-bound, bounded execution of the reviewed person-down validation path.

This module does not download models or media, open network streams, retain
frames, or claim that candidate alerts establish a fall or injury. It binds
already-authorized local samples to the exact reviewed model-artifact manifest,
runtime/device identity, measured execution time, and deterministic evaluation
metrics so later commercial validation is reproducible rather than ad hoc.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable

from .artifacts import OPENVINO_OMZ_2023_FP16
from .evaluation import (
    EvaluationAggregate,
    EvaluationConfig,
    EvaluationResult,
    EvaluationSample,
    LabeledPersonDown,
    aggregate_person_down_evaluations,
    evaluate_person_down_candidates,
)
from .openvino_pipeline import (
    OpenVINOOMZPipelineConfig,
    OpenVINOOMZRunResult,
    run_local_video_openvino_omz,
)

_MAX_TIMESTAMP_MS = 253402300799999
_MAX_SAMPLES = 256
_MAX_REFERENCE_LENGTH = 256
_DEVICE = re.compile(r"[A-Za-z0-9_.:-]{1,64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _bounded_text(value: str, name: str, *, maximum: int = 128) -> None:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded nonempty string")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{name} must not contain control characters")


def _timestamp(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_TIMESTAMP_MS:
        raise ValueError(f"{name} must be an integer timestamp in the supported UTC range")


@dataclass(frozen=True)
class ModelArtifactIdentity:
    component: str
    relative_path: str
    size_bytes: int
    sha384: str
    license_id: str


@dataclass(frozen=True)
class ValidationSampleSpec:
    """One authorized, labeled, ordinary local-video validation sample."""

    sample_id: str
    site_id: str
    camera_id: str
    authorization_ref: str
    video_path: Path
    media_sha256: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    labels: tuple[LabeledPersonDown, ...] = ()

    def __post_init__(self) -> None:
        for name in ("sample_id", "site_id", "camera_id"):
            _bounded_text(getattr(self, name), name)
        _bounded_text(self.authorization_ref, "authorization_ref", maximum=_MAX_REFERENCE_LENGTH)
        if "://" in self.authorization_ref or self.authorization_ref.lower().startswith(("http:", "https:", "rtsp:", "rtsps:")):
            raise ValueError("authorization_ref must be an opaque non-URL rights reference")
        raw = str(self.video_path)
        if not raw or "://" in raw or raw.lower().startswith(("http:", "https:", "rtsp:", "rtsps:")) or raw.startswith(("/dev/", "\\\\.\\")):
            raise ValueError("video_path must identify an ordinary local file")
        object.__setattr__(self, "video_path", Path(self.video_path))
        if not isinstance(self.media_sha256, str) or not _SHA256.fullmatch(self.media_sha256):
            raise ValueError("media_sha256 must be lowercase 64-character hex")
        _timestamp(self.start_timestamp_ms, "start_timestamp_ms")
        _timestamp(self.end_timestamp_ms, "end_timestamp_ms")
        if self.end_timestamp_ms <= self.start_timestamp_ms:
            raise ValueError("sample interval must have positive duration")
        if not isinstance(self.labels, tuple) or any(not isinstance(item, LabeledPersonDown) for item in self.labels):
            raise ValueError("labels must be a tuple of LabeledPersonDown values")
        label_ids = [item.label_id for item in self.labels]
        if len(set(label_ids)) != len(label_ids):
            raise ValueError("label_id values must be unique within a sample")
        for label in self.labels:
            if label.start_timestamp_ms < self.start_timestamp_ms or label.end_timestamp_ms > self.end_timestamp_ms:
                raise ValueError("label lies outside the declared sample interval")


@dataclass(frozen=True)
class ValidationSuiteConfig:
    max_samples: int = 64
    required_device: str = "CPU"
    max_total_video_bytes: int = 2 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if type(self.max_samples) is not int or not 1 <= self.max_samples <= _MAX_SAMPLES:
            raise ValueError("max_samples must be an integer in [1, 256]")
        if not isinstance(self.required_device, str) or not _DEVICE.fullmatch(self.required_device):
            raise ValueError("required_device must be a simple device name")
        if type(self.max_total_video_bytes) is not int or self.max_total_video_bytes < 1:
            raise ValueError("max_total_video_bytes must be a positive integer")


@dataclass(frozen=True)
class ValidationSampleRun:
    sample_id: str
    site_id: str
    camera_id: str
    authorization_ref: str
    media_sha256: str
    runtime_version: str
    device: str
    frames_processed: int
    observations_processed: int
    elapsed_ms: float
    frames_per_second: float | None
    evaluation: EvaluationResult


@dataclass(frozen=True)
class ValidationSuiteResult:
    runtime_version: str
    device: str
    model_artifacts: tuple[ModelArtifactIdentity, ...]
    sample_runs: tuple[ValidationSampleRun, ...]
    aggregate: EvaluationAggregate
    total_elapsed_ms: float
    total_frames_processed: int
    frames_per_second: float | None


def _artifact_identity() -> tuple[ModelArtifactIdentity, ...]:
    return tuple(
        ModelArtifactIdentity(spec.component, spec.relative_path, spec.size_bytes, spec.sha384, spec.license_id)
        for spec in OPENVINO_OMZ_2023_FP16
    )


def _preflight_samples(samples: list[ValidationSampleSpec], cfg: ValidationSuiteConfig) -> None:
    if not samples:
        raise ValueError("at least one validation sample is required")
    if len(samples) > cfg.max_samples:
        raise RuntimeError("validation sample limit exceeded")
    ids = [item.sample_id for item in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("sample_id values must be unique")

    total_bytes = 0
    by_camera: dict[str, list[ValidationSampleSpec]] = {}
    for item in samples:
        path = item.video_path
        if path.is_symlink() or not path.is_file():
            raise ValueError("every video_path must be an existing non-symlink regular file")
        total_bytes += path.stat().st_size
        if total_bytes > cfg.max_total_video_bytes:
            raise RuntimeError("validation media exceeds the configured total byte budget")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        if digest.hexdigest() != item.media_sha256:
            raise ValueError("validation media checksum mismatch")
        by_camera.setdefault(item.camera_id, []).append(item)
    for camera_id, group in by_camera.items():
        ordered = sorted(group, key=lambda item: (item.start_timestamp_ms, item.end_timestamp_ms, item.sample_id))
        previous_end: int | None = None
        for item in ordered:
            if previous_end is not None and item.start_timestamp_ms < previous_end:
                raise ValueError(f"overlapping validation intervals for camera {camera_id}")
            previous_end = item.end_timestamp_ms


def run_validation_suite(
    samples: Iterable[ValidationSampleSpec],
    *,
    artifact_root: str | Path,
    pipeline_config: OpenVINOOMZPipelineConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
    config: ValidationSuiteConfig | None = None,
    runtime_factory: Any = None,
    runner: Callable[..., OpenVINOOMZRunResult] = run_local_video_openvino_omz,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> ValidationSuiteResult:
    """Run a bounded comparable validation suite without retaining media.

    All sample rights references, local paths and camera intervals are validated
    before the first inference call. The suite is restricted to one runtime
    version and one required device so aggregate latency/accuracy evidence is
    not silently mixed across incomparable execution environments.
    """
    items = list(samples)
    if any(not isinstance(item, ValidationSampleSpec) for item in items):
        raise ValueError("samples must contain ValidationSampleSpec values")
    suite_cfg = config or ValidationSuiteConfig()
    if not isinstance(suite_cfg, ValidationSuiteConfig):
        raise ValueError("config must be a ValidationSuiteConfig")
    if pipeline_config is not None and not isinstance(pipeline_config, OpenVINOOMZPipelineConfig):
        raise ValueError("pipeline_config must be an OpenVINOOMZPipelineConfig")
    if evaluation_config is not None and not isinstance(evaluation_config, EvaluationConfig):
        raise ValueError("evaluation_config must be an EvaluationConfig")
    if not callable(runner) or not callable(clock_ns):
        raise ValueError("runner and clock_ns must be callable")
    _preflight_samples(items, suite_cfg)

    sample_runs: list[ValidationSampleRun] = []
    evaluation_samples: list[EvaluationSample] = []
    expected_runtime: str | None = None
    total_elapsed_ns = 0
    total_frames = 0

    for item in items:
        before = clock_ns()
        if type(before) is not int or before < 0:
            raise RuntimeError("clock_ns must return a nonnegative integer")
        result = runner(
            item.video_path,
            artifact_root=artifact_root,
            start_timestamp_ms=item.start_timestamp_ms,
            source_id=item.camera_id,
            session_id=item.sample_id,
            config=pipeline_config,
            runtime_factory=runtime_factory,
        )
        after = clock_ns()
        if type(after) is not int or after < before:
            raise RuntimeError("clock_ns must be monotonic nondecreasing integer nanoseconds")
        if not isinstance(result, OpenVINOOMZRunResult):
            raise RuntimeError("runner returned an unsupported result")
        if result.device != suite_cfg.required_device:
            raise RuntimeError("validation result device does not match the required device")
        if expected_runtime is None:
            expected_runtime = result.runtime_version
        elif result.runtime_version != expected_runtime:
            raise RuntimeError("validation suite mixed OpenVINO runtime versions")
        if result.video.frames_processed < 1:
            raise RuntimeError("validation sample decoded zero frames")

        elapsed_ns = after - before
        elapsed_ms = elapsed_ns / 1_000_000.0
        fps = (result.video.frames_processed * 1_000_000_000.0 / elapsed_ns) if elapsed_ns else None
        if fps is not None and not math.isfinite(fps):
            raise RuntimeError("non-finite validation throughput")
        evaluation = evaluate_person_down_candidates(
            result.video.events,
            item.labels,
            video_start_timestamp_ms=item.start_timestamp_ms,
            video_end_timestamp_ms=item.end_timestamp_ms,
            config=evaluation_config,
        )
        evaluation_sample = EvaluationSample(
            item.sample_id,
            item.site_id,
            item.camera_id,
            item.start_timestamp_ms,
            item.end_timestamp_ms,
            evaluation,
        )
        evaluation_samples.append(evaluation_sample)
        sample_runs.append(ValidationSampleRun(
            sample_id=item.sample_id,
            site_id=item.site_id,
            camera_id=item.camera_id,
            authorization_ref=item.authorization_ref,
            media_sha256=item.media_sha256,
            runtime_version=result.runtime_version,
            device=result.device,
            frames_processed=result.video.frames_processed,
            observations_processed=result.video.observations_processed,
            elapsed_ms=elapsed_ms,
            frames_per_second=fps,
            evaluation=evaluation,
        ))
        total_elapsed_ns += elapsed_ns
        total_frames += result.video.frames_processed

    aggregate = aggregate_person_down_evaluations(evaluation_samples)
    total_elapsed_ms = total_elapsed_ns / 1_000_000.0
    total_fps = (total_frames * 1_000_000_000.0 / total_elapsed_ns) if total_elapsed_ns else None
    if total_fps is not None and not math.isfinite(total_fps):
        raise RuntimeError("non-finite aggregate validation throughput")
    if expected_runtime is None:
        raise RuntimeError("validation suite produced no runtime identity")
    return ValidationSuiteResult(
        runtime_version=expected_runtime,
        device=suite_cfg.required_device,
        model_artifacts=_artifact_identity(),
        sample_runs=tuple(sample_runs),
        aggregate=aggregate,
        total_elapsed_ms=total_elapsed_ms,
        total_frames_processed=total_frames,
        frames_per_second=total_fps,
    )
