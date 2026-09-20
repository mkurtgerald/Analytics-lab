"""Bounded public-domain LPR smoke evidence for the first runnable baseline.

This evidence-only module admits one intentionally fictional/sample Chinese plate
image whose author released it to the public domain, then (only after an exact
SHA-256 is pinned) runs the already-reviewed OMZ 0106 plate detector on CPU.
It emits aggregate smoke metrics only. The source is a staged plate graphic,
not real-world accuracy evidence and never a commercial-accuracy claim.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .lpr_ocr import OpenVINOOMZPlateDetector

_RIGHTS_PAGE = (
    "https://commons.wikimedia.org/w/index.php?title="
    "File:China_license_plate-Chongqing_%E6%B8%9DA_92518.png&oldid=856647637"
)
_SOURCE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/6/63/"
    "China_license_plate-Chongqing_%E6%B8%9DA_92518.png"
)
_SOURCE_SIZE = 61_465
_SOURCE_SHA1 = "d365a117631a8fa5a2a0fb7a8d2a03fe2e9b73bc"
_SOURCE_SHA256 = "47ea02127b3c22856ac548164c822b104c74f7afb711a0cb43458080a11d5433"
_WIDTH = 680
_HEIGHT = 144
_EXPECTED_TEXT_NORMALIZED = "A92518"
_MAX_SOURCE_BYTES = 65_536

_MODEL_FILES = (
    (
        "vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml",
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml",
        325_452,
        "b0a70cfc4ccbe85dd5ba5549c2d142d7f922a0a1116657cdcc9f48cb6afc7cc40a596a655b9e6566396c48b72232ff65",
    ),
    (
        "vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.bin",
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.bin",
        1_286_772,
        "0406e296c9822f4a5e8ee300ad40e5c0a0eeeaf9206f829556cb836cb8f10a4f8a4dd94d1bb16a60562bf1568fd12618",
    ),
)


def _fetch_bytes(url: str, *, max_bytes: int, expected_host: str) -> bytes:
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("evidence URL must use https")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    request = Request(url, headers={"User-Agent": "Analytics-Lab-bounded-evidence/1"})
    with urlopen(request, timeout=20) as response:
        final = urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != expected_host:
            raise RuntimeError("evidence redirect escaped reviewed host")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > max_bytes:
            raise RuntimeError("evidence payload exceeds byte cap")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise RuntimeError("evidence payload exceeds byte cap")
    return payload


def validate_source_bytes(payload: bytes, *, required_sha256: str | None = _SOURCE_SHA256) -> dict[str, str | int]:
    if not isinstance(payload, bytes):
        raise ValueError("source payload must be bytes")
    if len(payload) != _SOURCE_SIZE:
        raise RuntimeError("source size mismatch")
    sha1 = hashlib.sha1(payload).hexdigest()
    if sha1 != _SOURCE_SHA1:
        raise RuntimeError("source SHA-1 mismatch")
    sha256 = hashlib.sha256(payload).hexdigest()
    if required_sha256 is not None:
        if not isinstance(required_sha256, str) or len(required_sha256) != 64:
            raise ValueError("required SHA-256 must be lowercase hex")
        if sha256 != required_sha256:
            raise RuntimeError("source SHA-256 mismatch")
    return {"bytes": len(payload), "sha1": sha1, "sha256": sha256}


def _download_model_files(root: Path) -> None:
    for relative_path, url, size_bytes, sha384 in _MODEL_FILES:
        payload = _fetch_bytes(
            url,
            max_bytes=size_bytes,
            expected_host="storage.openvinotoolkit.org",
        )
        if len(payload) != size_bytes:
            raise RuntimeError("OMZ artifact size mismatch")
        if hashlib.sha384(payload).hexdigest() != sha384:
            raise RuntimeError("OMZ artifact SHA-384 mismatch")
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


def _iou_full_frame(plate: object) -> float:
    values = tuple(float(getattr(plate, name)) for name in ("x1", "y1", "x2", "y2"))
    x1, y1, x2, y2 = values
    if not all(math.isfinite(value) for value in values):
        raise ValueError("plate box must be finite")
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return min(1.0, max(0.0, area))


def run() -> dict[str, object]:
    source = _fetch_bytes(
        _SOURCE_URL,
        max_bytes=_MAX_SOURCE_BYTES,
        expected_host="upload.wikimedia.org",
    )
    identity = validate_source_bytes(source)
    common: dict[str, object] = {
        "evidence_scope": "staged_public_domain_smoke_not_commercial_accuracy",
        "rights_source": _RIGHTS_PAGE,
        "source_url": _SOURCE_URL,
        "source_bytes": identity["bytes"],
        "source_sha1": identity["sha1"],
        "source_sha256": identity["sha256"],
        "expected_dimensions": [_WIDTH, _HEIGHT],
        "expected_text_normalized": _EXPECTED_TEXT_NORMALIZED,
    }
    if _SOURCE_SHA256 is None:
        return {**common, "admission_only": True, "inference_run": False}

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for LPR evidence") from exc
    image = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or tuple(int(value) for value in image.shape[:2]) != (_HEIGHT, _WIDTH):
        raise RuntimeError("decoded source dimensions changed")

    with tempfile.TemporaryDirectory(prefix="analytics-lpr-") as directory:
        root = Path(directory)
        _download_model_files(root)
        detector = OpenVINOOMZPlateDetector(root)
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        detections = detector(image)
        cpu_seconds = time.process_time() - cpu_start
        elapsed_seconds = time.perf_counter() - wall_start

    best = detections[0] if detections else None
    best_iou = _iou_full_frame(best) if best is not None else 0.0
    return {
        **common,
        "admission_only": False,
        "inference_run": True,
        "runtime_version": detector.runtime_version,
        "plate_detections": len(detections),
        "best_plate_confidence": float(best.confidence) if best is not None else None,
        "best_plate_iou_full_frame": best_iou,
        "elapsed_seconds": elapsed_seconds,
        "cpu_seconds": cpu_seconds,
        "throughput_fps": (1.0 / elapsed_seconds) if elapsed_seconds > 0.0 else 0.0,
        "latency_ms": elapsed_seconds * 1000.0,
    }


def main() -> int:
    result = run()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
