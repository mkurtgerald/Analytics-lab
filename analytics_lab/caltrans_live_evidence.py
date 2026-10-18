"""Bounded public Caltrans traffic compatibility probe, without media retention.

Two verified local TS segments are decoded separately. Vehicle-only inference
shares one compiled model; each segment gets a fresh portable ByteTrack session.
All times are media-relative or explicitly HTTP receipt times, never inferred
camera capture times. Unlabelled observations cannot establish accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Callable

from .artifacts import verify_artifact_set
from .bytetrack import BYTE_TRACK_REVISION, ByteTrackAssociationBackend, ByteTrackAssociationConfig
from .public_live_hls import CAMERA_NAME, LiveDeadline, bounded_fetch, capture_live_window as _capture
from .tracking import DetectionCandidate, TrackingSession, TrackingSessionConfig
from .video import OpenCVVideoFileSource

_MAX_MEDIA_BYTES = 4 * 1024 * 1024
_MAX_DURATION_SECONDS = 35.0
_MAX_DECODED = 2400
_MAX_SAMPLES = 100
_SAMPLE_PERIOD_MS = 360
_EVIDENCE_SECONDS = 180.0
_CLEANUP_SECONDS = 5.0


def _phase(name: str, **values: Any) -> None:
    print(json.dumps({"phase": name, **values}, sort_keys=True), flush=True)


def _file_hash(path: Path, deadline: Any | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if deadline is not None:
                deadline.check()
            block = handle.read(64 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_hosted_linux() -> None:
    if (platform.system() != "Linux" or os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
            or os.environ.get("RUNNER_OS") != "Linux"
            or os.environ.get("GITHUB_REPOSITORY") != "mkurtgerald/Analytics-lab"
            or os.environ.get("GITHUB_EVENT_NAME") != "pull_request"
            or not os.environ.get("GITHUB_HEAD_REF", "").startswith("evidence/public-live-caltrans-")):
        raise RuntimeError("Caltrans evidence requires its GitHub-hosted Linux PR lane")


def _work_directory(path: Path) -> Path:
    value = os.environ.get("RUNNER_TEMP")
    if not value:
        raise RuntimeError("RUNNER_TEMP is required")
    root, candidate = Path(value).absolute(), path.absolute()
    if (not root.is_dir() or root.is_symlink() or root.resolve() != root
            or candidate.parent != root or candidate.resolve() != candidate
            or candidate.is_symlink() or candidate.exists()):
        raise RuntimeError("Caltrans evidence requires a new direct RUNNER_TEMP child")
    return candidate


def _runtime_versions() -> dict[str, str]:
    import cv2
    import numpy
    import openvino
    if (cv2.__version__ != "4.12.0" or numpy.__version__ != "2.2.6"
            or not openvino.__version__.startswith("2026.3.1")):
        raise RuntimeError("Caltrans evidence runtime differs from the reviewed hash-locked stack")
    return {"python": platform.python_version(), "opencv": cv2.__version__,
            "numpy": numpy.__version__, "openvino": openvino.__version__}


def _acquire_models(work_dir: Path, deadline: Any, opener: Any) -> list[dict[str, Any]]:
    from .lpr_ocr import omz_vehicle_artifacts
    specs = omz_vehicle_artifacts()
    if len(specs) != 2 or sum(spec.size_bytes for spec in specs) != 1_612_224:
        raise RuntimeError("Caltrans evidence requires only the pinned 0106 XML/BIN pair")
    root = work_dir / "models"
    root.mkdir()
    allowed = frozenset(spec.source_url for spec in specs)
    for spec in specs:
        fetched = bounded_fetch(spec.source_url, max_bytes=spec.size_bytes, deadline=deadline,
                                allowed_urls=allowed, opener=opener)
        if (len(fetched.payload) != spec.size_bytes
                or hashlib.sha384(fetched.payload).hexdigest() != spec.sha384):
            raise RuntimeError("Caltrans evidence model identity mismatch")
        target = root.joinpath(*Path(spec.relative_path).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(fetched.payload)
        deadline.check()
    verified = verify_artifact_set(root, specs)
    return [asdict(item.spec) for item in verified]


def _detector_factory(root: Path) -> Any:
    from .lpr_ocr import OpenVINOOMZVehicleDetector
    return OpenVINOOMZVehicleDetector(root, confidence_threshold=.50, max_vehicles=64)


def _validate_window(window: Any, capture_root: Path, deadline: Any) -> None:
    if (len(window.segments) != 2 or window.metadata.get("sequence_advanced") is not True
            or window.metadata.get("source_id") != "caltrans-d1-cctv-88"
            or window.metadata.get("camera_name") != CAMERA_NAME
            or window.metadata.get("camera_capture_utc", "missing") is not None):
        raise RuntimeError("Caltrans window lacks two segments and explicit sequence/unknown-UTC proof")
    total_bytes, total_duration = 0, 0.0
    previous = None
    for segment in window.segments:
        if (type(segment.sequence) is not int or segment.sequence < 0
                or (previous is not None and segment.sequence != previous + 1)
                or type(segment.duration_seconds) not in (int, float)
                or not math.isfinite(segment.duration_seconds) or segment.duration_seconds <= 0
                or type(segment.size_bytes) is not int or segment.size_bytes <= 0):
            raise RuntimeError("Caltrans segment metadata is outside the frozen scope")
        path = segment.path
        if (not isinstance(path, Path) or path.is_symlink() or not path.is_file()
                or path.parent != capture_root or path.resolve() != path
                or path.stat().st_size != segment.size_bytes
                or _file_hash(path, deadline) != segment.sha256):
            raise RuntimeError("Caltrans segment file identity/boundary mismatch")
        previous = segment.sequence
        total_bytes += segment.size_bytes
        total_duration += segment.duration_seconds
    if total_bytes > _MAX_MEDIA_BYTES or total_duration > _MAX_DURATION_SECONDS:
        raise RuntimeError("Caltrans segment window exceeds the frozen byte/duration scope")


def _process_segment(segment: Any, detector: Any, *, deadline: Any,
                     decoded_budget: int, sample_budget: int) -> dict[str, Any]:
    # This source accepts only ordinary local paths. Never pass HLS/network URLs
    # to VideoCapture, and never stitch segments across unknown PTS boundaries.
    session = TrackingSession(ByteTrackAssociationBackend(), TrackingSessionConfig(64, 64))
    chain = hashlib.sha256()
    decoded, sampled, detections_count, tracks_count = 0, 0, 0, 0
    track_ids: set[str] = set()
    samples: list[dict[str, Any]] = []
    first_ms = last_ms = None
    next_sample_ms = 0
    with OpenCVVideoFileSource(segment.path, start_timestamp_ms=0,
                               max_file_bytes=_MAX_MEDIA_BYTES, max_fps=60.0) as source:
        fps = source._fps
        if type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= 60:
            raise RuntimeError("Caltrans decoded FPS is outside the frozen bound")
        for frame in source:
            deadline.check()
            if decoded >= decoded_budget:
                raise RuntimeError("Caltrans decoded-frame budget exceeded")
            image = frame.image
            if tuple(getattr(image, "shape", ())) != (240, 320, 3) or str(getattr(image, "dtype", "")) != "uint8":
                raise RuntimeError("Caltrans decoded frame is not 320x240 uint8 BGR")
            payload = image.tobytes(order="C")
            if len(payload) != 320 * 240 * 3:
                raise RuntimeError("Caltrans decoded frame byte count changed")
            digest = hashlib.sha256(payload).hexdigest()
            chain.update(frame.index.to_bytes(8, "big") + frame.timestamp_ms.to_bytes(8, "big")
                         + bytes.fromhex(digest))
            decoded += 1
            first_ms = frame.timestamp_ms if first_ms is None else first_ms
            last_ms = frame.timestamp_ms
            if frame.timestamp_ms < next_sample_ms:
                continue
            if sampled >= sample_budget:
                raise RuntimeError("Caltrans sampled-frame budget exceeded")
            deadline.check()
            detections = detector.detect(image)
            deadline.check()
            if (not isinstance(detections, tuple) or len(detections) > 64
                    or any(not isinstance(item, DetectionCandidate) or item.category != "vehicle"
                           or item.model_class_id != 1 or item.confidence < .50 for item in detections)):
                raise RuntimeError("Caltrans detector escaped the fixed vehicle-only contract")
            if hashlib.sha256(image.tobytes(order="C")).hexdigest() != digest:
                raise RuntimeError("Caltrans detector mutated the decoded input")
            # ByteTrack frame-buffer/confirmation units are updates of this
            # sampled stream. Decoded frame gaps must not expire every newly
            # arriving unconfirmed vehicle before its next sampled observation.
            tracks = session.update(sampled, frame.timestamp_ms, detections)
            sampled += 1
            detections_count += len(detections)
            tracks_count += len(tracks)
            track_ids.update(item.track_id for item in tracks)
            samples.append({"frame_index": frame.index, "segment_relative_ms": frame.timestamp_ms,
                            "tracking_update_index": sampled - 1,
                            "decoded_bgr24_sha256": digest, "detections": len(detections),
                            "track_observations": len(tracks)})
            next_sample_ms = frame.timestamp_ms + _SAMPLE_PERIOD_MS
    if decoded == 0 or sampled == 0:
        raise RuntimeError("Caltrans segment produced no decoded/sample frames")
    return {"sequence": segment.sequence, "source_sha256": segment.sha256, "source_bytes": segment.size_bytes,
            "declared_duration_seconds": segment.duration_seconds, "decoded_frames": decoded,
            "sampled_frames": sampled, "detections": detections_count, "track_observations": tracks_count,
            "within_segment_unique_ids": sorted(track_ids), "decoded_span_ms": [first_ms, last_ms],
            "nominal_fps": fps, "samples": samples, "decoded_bgr24_chain_sha256": chain.hexdigest(),
            "time_basis": "segment-relative decoder PTS with FPS fallback; camera UTC unknown",
            "tracking_frame_basis": "sample ordinal",
            "decode_eof_is_complete_media_proof": False}


def _cpu_metadata() -> dict[str, Any]:
    model = "unknown"
    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    model = line.partition(":")[2].strip() or "unknown"
                    break
    except OSError:
        pass
    return {"platform": platform.platform(), "machine": platform.machine(),
            "logical_cpus": os.cpu_count(), "linux_model_name": model}


def run(work_dir: Path, *, opener: Any = None,
        clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    _require_hosted_linux()
    work_dir = _work_directory(work_dir)
    started = clock()
    deadline = LiveDeadline(clock=clock, total_seconds=_EVIDENCE_SECONDS - _CLEANUP_SECONDS)
    versions = _runtime_versions()
    durations: dict[str, float] = {}

    def phase(name: str, cap: float, operation: Callable[[], Any]) -> Any:
        _phase(name)
        deadline.check()
        before = clock()
        result = operation()
        elapsed = clock() - before
        deadline.check()
        if elapsed > cap:
            raise TimeoutError(f"Caltrans {name} exceeded its cooperative phase budget")
        durations[name] = round(elapsed, 6)
        return result

    work_dir.mkdir()
    try:
        capture_root = work_dir / "capture"
        capture_root.mkdir()
        window = phase("capture", 75.0, lambda: _capture(capture_root, opener=opener, clock=clock, deadline=deadline))
        _validate_window(window, capture_root, deadline)
        models = phase("model_acquisition", 30.0, lambda: _acquire_models(work_dir, deadline, opener))
        detector = phase("compile", 30.0, lambda: _detector_factory(work_dir / "models"))
        if not str(detector.runtime_version).startswith("2026.3.1"):
            raise RuntimeError("Caltrans vehicle runtime version mismatch")
        rows: list[dict[str, Any]] = []
        decoded = sampled = 0
        for segment in window.segments:
            _validate_window(window, capture_root, deadline)
            row = phase(f"segment_{segment.sequence}", 30.0,
                        lambda: _process_segment(segment, detector, deadline=deadline,
                                                 decoded_budget=_MAX_DECODED - decoded,
                                                 sample_budget=_MAX_SAMPLES - sampled))
            decoded += row["decoded_frames"]
            sampled += row["sampled_frames"]
            if decoded > _MAX_DECODED or sampled > _MAX_SAMPLES:
                raise RuntimeError("Caltrans aggregate frame budget exceeded")
            rows.append(row)
        code_files = ("caltrans_live_evidence.py", "public_live_hls.py", "lpr_ocr.py", "video.py",
                      "bytetrack.py", "tracking.py", "association.py")
        report = {
            "evidence": "caltrans-public-live-vehicle-compatibility-v1", "source": window.metadata,
            "segments": rows, "models": models, "runtime": versions, "cpu": _cpu_metadata(),
            "project_source_sha256": {name: _file_hash(Path(__file__).with_name(name), deadline) for name in code_files},
            "github_checkout_sha": os.environ.get("GITHUB_SHA", "unknown"),
            "detector": {"model": "vehicle-license-plate-detection-barrier-0106/FP16", "class_id": 1,
                         "category": "vehicle", "confidence_threshold": .50, "max_vehicles": 64,
                         "instances": 1, "compile_count": 1,
                         "compile_count_basis": "one reviewed adapter construction; no per-frame compilation",
                         "model_domain": "front-facing barrier model; public wide-traffic domain stress"},
            "tracking": {"backend": "portable ByteTrack association slice", "revision": BYTE_TRACK_REVISION,
                         "config": asdict(ByteTrackAssociationConfig()), "sessions": 2,
                         "frame_basis": "sample ordinal", "buffer_units": "sampled updates",
                         "cross_segment_association": False, "identity_claim": False},
            "limits": {"media_bytes": _MAX_MEDIA_BYTES, "duration_seconds": _MAX_DURATION_SECONDS,
                       "decoded_frames": _MAX_DECODED, "sampled_frames": _MAX_SAMPLES,
                       "sample_period_ms": _SAMPLE_PERIOD_MS, "evidence_seconds": _EVIDENCE_SECONDS,
                       "cleanup_reserve_seconds": _CLEANUP_SECONDS, "hard_hosted_job_seconds": 300,
                       "semantics": "cooperative elapsed checks; capped network inactivity; native calls not preempted"},
            "decoded_frames": decoded, "sampled_frames": sampled,
            "detections": sum(row["detections"] for row in rows),
            "track_observations": sum(row["track_observations"] for row in rows),
            "phase_elapsed_seconds": durations, "camera_capture_utc": None,
            "precision": None, "recall": None, "false_positive_rate": None, "misses": None,
            "independent_labels": False, "positive_qualified": False, "negative_qualified": False,
            "accuracy_gate_passed": False, "accuracy_claim": False, "performance_claim": False,
            "training": False, "recognition": False, "deployment": False,
            "retained_media": False, "exported_media": False,
        }
        deadline.check()
    finally:
        try:
            _phase("cleanup")
        finally:
            shutil.rmtree(work_dir)
    elapsed = clock() - started
    if work_dir.exists() or elapsed > _EVIDENCE_SECONDS:
        raise RuntimeError("Caltrans cleanup/elapsed qualification failed")
    report["temporary_assets_removed"] = True
    report["evidence_elapsed_seconds"] = round(elapsed, 6)
    report["execution_complete"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.work_dir), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
