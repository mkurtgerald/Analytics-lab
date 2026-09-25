"""One bounded two-source real Weapons engineering measurement.

This evidence-only path re-verifies the already-admitted standard OIDv4 SSD
frozen graph and exactly two admitted CC0 Wikimedia sources, decodes each source
only in memory with EXIF auto-orientation disabled, compiles the reviewed
OpenVINO model exactly once, and executes exactly one frozen-threshold detection
per source through the project-owned Weapons adapter.

The source roles are selection metadata, not independently authored detector
ground truth. Misses and false positives remain unscored. No threshold/class
mapping is tuned from results, and no model or media artifact is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import face_donor_oid_ssd_admission as model_admission
from . import face_donor_oid_ssd_openvino_probe as model_probe
from . import weapons_real_cc0_admission as source_admission
from .face_real_cc0_measurement import _bounded_work_dir, _stream_verified_graph
from .tracking import DetectionCandidate
from .weapons_oid_ssd import OpenVINOOIDSSDWeaponRuntime

_FIXED_THRESHOLD = 0.50
_EXPECTED_OPENVINO_PREFIX = "2026.3.1"
_EXPECTED_OPENCV_VERSION = "4.12.0"
_USER_AGENT = "Analytics-lab bounded Weapons real measurement/1.0"
_CHUNK = 64 * 1024


def _download_verified_source(
    source: source_admission._Source,
    opener: Callable = urlopen,
) -> bytes:
    if source.sha256 is None:
        raise RuntimeError("Weapons measurement source SHA-256 is not pinned")
    expected = urlparse(source.url)
    if expected.scheme != "https" or expected.netloc != "upload.wikimedia.org":
        raise RuntimeError("unapproved Weapons measurement source URL")
    request = Request(source.url, headers={"User-Agent": _USER_AGENT})
    payload = bytearray()
    with opener(request, timeout=30) as response:
        final = urlparse(response.geturl())
        if (
            final.scheme != "https"
            or final.netloc != expected.netloc
            or final.path != expected.path
        ):
            raise RuntimeError("Weapons measurement source redirected outside reviewed asset")
        declared = response.headers.get("Content-Length")
        if declared is not None:
            try:
                advertised = int(declared)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("invalid Weapons measurement Content-Length") from exc
            if advertised != source.expected_size:
                raise RuntimeError("Weapons measurement Content-Length changed")
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray)):
                raise ValueError("Weapons measurement response must yield bytes")
            payload.extend(chunk)
            if len(payload) > source.max_bytes:
                raise RuntimeError("Weapons measurement source exceeds bounded size")
    data = bytes(payload)
    if len(data) != source.expected_size:
        raise RuntimeError("Weapons measurement source byte length changed")
    if hashlib.sha1(data).hexdigest() != source.expected_sha1:
        raise RuntimeError("Weapons measurement source SHA-1 changed")
    if hashlib.sha256(data).hexdigest() != source.sha256:
        raise RuntimeError("Weapons measurement source SHA-256 changed")
    return data


def _serialize(detections: tuple[DetectionCandidate, ...]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for item in detections:
        if not isinstance(item, DetectionCandidate):
            raise ValueError("unsupported Weapons detection value")
        output.append(
            {
                "class_id": item.model_class_id,
                "category": item.category,
                "confidence": float(item.confidence),
                "box_xyxy_normalized": [
                    float(item.box.x_min),
                    float(item.box.y_min),
                    float(item.box.x_max),
                    float(item.box.y_max),
                ],
            }
        )
    return output


def _measure_source(
    runtime: OpenVINOOIDSSDWeaponRuntime,
    source: source_admission._Source,
    *,
    opener: Callable = urlopen,
) -> dict[str, object]:
    payload = _download_verified_source(source, opener)

    import cv2
    import numpy as np

    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    encoded = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(encoded, flags)
    if (
        frame is None
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or int(frame.shape[2]) != 3
    ):
        raise RuntimeError("admitted Weapons source did not decode to HxWx3 uint8 BGR")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if (width, height) != (source.width, source.height):
        raise RuntimeError("Weapons source decoded dimensions changed")

    before = hashlib.sha256(frame.tobytes()).hexdigest()
    detections = runtime.detect(frame)
    after = hashlib.sha256(frame.tobytes()).hexdigest()
    if before != after:
        raise RuntimeError("Weapons detector mutated admitted input pixels")

    return {
        "name": source.name,
        "role": source.role,
        "source_page": source.page,
        "source_url": source.url,
        "source_license": source.license,
        "source_size": source.expected_size,
        "source_sha1": source.expected_sha1,
        "source_sha256": source.sha256,
        "published_dimensions": [source.width, source.height],
        "decode_flags": int(flags),
        "input_immutable": True,
        "detection_count": len(detections),
        "detections": _serialize(detections),
        "ground_truth_available": False,
        "misses": "unscored_without_independent_ground_truth",
        "false_positives": "unscored_without_independent_ground_truth",
        "media_artifact_retained": False,
    }


def run(work_dir: str | Path, *, opener: Callable = urlopen) -> dict[str, object]:
    root = _bounded_work_dir(work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    graph = _stream_verified_graph()

    import cv2
    import openvino as ov

    openvino_version = str(getattr(ov, "__version__", ""))
    if not openvino_version.startswith(_EXPECTED_OPENVINO_PREFIX):
        raise RuntimeError("unexpected OpenVINO runtime version")
    opencv_version = str(getattr(cv2, "__version__", ""))
    if opencv_version != _EXPECTED_OPENCV_VERSION:
        raise RuntimeError("unexpected OpenCV runtime version")

    try:
        with tempfile.TemporaryDirectory(prefix="weapons-real-cc0-", dir=root) as temporary:
            model_path = Path(temporary) / "frozen_inference_graph.pb"
            model_path.write_bytes(graph)
            runtime = OpenVINOOIDSSDWeaponRuntime.from_tensorflow_graph(
                model_path,
                confidence_threshold=_FIXED_THRESHOLD,
            )
            observations = tuple(
                _measure_source(runtime, source, opener=opener)
                for source in source_admission._SOURCES
            )

        if runtime.compile_count != 1:
            raise RuntimeError("Weapons measurement did not preserve compile-once lifecycle")
        if runtime.inference_count != len(source_admission._SOURCES):
            raise RuntimeError("Weapons measurement inference count mismatch")

        return {
            "evidence": "weapons-real-cc0-measurement-v1",
            "model_archive_sha256": model_admission._EXPECTED_ARCHIVE_SHA256,
            "model_graph_sha256": model_probe._expected_model_identity()[1],
            "confidence_threshold": _FIXED_THRESHOLD,
            "openvino_version": openvino_version,
            "opencv_version": opencv_version,
            "compile_count": runtime.compile_count,
            "inference_count": runtime.inference_count,
            "source_count": len(observations),
            "sources": list(observations),
            "ground_truth_available": False,
            "threshold_tuned_after_result": False,
            "model_artifact_retained": False,
            "media_artifact_retained": False,
            "claim": (
                "two-source engineering observation only; source roles are selection "
                "metadata, not independent detection ground truth; not weapon-detection "
                "accuracy, product performance, or commercial readiness"
            ),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.work_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
