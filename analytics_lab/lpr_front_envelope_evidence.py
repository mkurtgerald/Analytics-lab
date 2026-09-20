"""Bounded CC0 front-envelope smoke for the frozen LPR plate detector.

The source was selected before detector output because it is a genuinely
front-facing vehicle photograph, its front plate is plainly wider than the
reviewed detector's documented 96-pixel minimum, and the photographer released
the work under CC0.  The first evidence head is admission-only: it verifies the
published byte size/SHA-1 and discovers SHA-256.  Only a later head with that
SHA-256 pinned may execute the already-reviewed OMZ 0106 detector.

This remains one engineering smoke diagnostic, never commercial accuracy or
geographic-generalization evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .lpr_ocr import OpenVINOOMZPlateDetector

_RIGHTS_PAGE = (
    "https://commons.wikimedia.org/w/index.php?title="
    "File:Land_Rover_Defender_110_(L316),_front_view.jpg&oldid=1230217363"
)
_SOURCE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/3/3e/"
    "Land_Rover_Defender_110_%28L316%29%2C_front_view.jpg"
)
_SOURCE_SIZE = 2_680_061
_SOURCE_SHA1 = "8cdb3acc024e67267dabf6d3ac793d0c54924e85"
_SOURCE_SHA256: str | None = "32e5637e39b54c26192c011c1cc5516bd6d35573582b8e09d7bc9aae90ef1db4"
_WIDTH = 4_032
_HEIGHT = 3_024
_EXPECTED_TEXT_NORMALIZED = "MPR318"
_DOCUMENTED_MIN_PLATE_WIDTH_PX = 96
# Conservative visual lower bound authored before detector output.  The plate
# spans roughly two-fifths of the 4032 px image width; 1000 px is deliberately
# far below that observation and is not a detector-derived box/ground truth.
_PREDECLARED_PLATE_WIDTH_LOWER_BOUND_PX = 1_000
_MAX_SOURCE_BYTES = 3 * 1024 * 1024

_MODEL_FILES = (
    (
        "vehicle-license-plate-detection-barrier-0106/FP16/"
        "vehicle-license-plate-detection-barrier-0106.xml",
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
        "models_bin/1/vehicle-license-plate-detection-barrier-0106/FP16/"
        "vehicle-license-plate-detection-barrier-0106.xml",
        325_452,
        "b0a70cfc4ccbe85dd5ba5549c2d142d7f922a0a1116657cdcc9f48cb6afc7cc"
        "40a596a655b9e6566396c48b72232ff65",
    ),
    (
        "vehicle-license-plate-detection-barrier-0106/FP16/"
        "vehicle-license-plate-detection-barrier-0106.bin",
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
        "models_bin/1/vehicle-license-plate-detection-barrier-0106/FP16/"
        "vehicle-license-plate-detection-barrier-0106.bin",
        1_286_772,
        "0406e296c9822f4a5e8ee300ad40e5c0a0eeeaf9206f829556cb836cb8f10a4f"
        "8a4dd94d1bb16a60562bf1568fd12618",
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


def validate_source_bytes(
    payload: bytes,
    *,
    required_sha1: str | None = _SOURCE_SHA1,
    required_sha256: str | None = _SOURCE_SHA256,
) -> dict[str, str | int]:
    if not isinstance(payload, bytes):
        raise ValueError("source payload must be bytes")
    if len(payload) != _SOURCE_SIZE:
        raise RuntimeError("source size mismatch")
    sha1 = hashlib.sha1(payload).hexdigest()
    sha256 = hashlib.sha256(payload).hexdigest()
    if required_sha1 is not None:
        if (
            not isinstance(required_sha1, str)
            or len(required_sha1) != 40
            or required_sha1.lower() != required_sha1
        ):
            raise ValueError("required SHA-1 must be lowercase 40-character hex")
        if sha1 != required_sha1:
            raise RuntimeError("source SHA-1 mismatch")
    if required_sha256 is not None:
        if (
            not isinstance(required_sha256, str)
            or len(required_sha256) != 64
            or required_sha256.lower() != required_sha256
        ):
            raise ValueError("required SHA-256 must be lowercase 64-character hex")
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


def run() -> dict[str, object]:
    source = _fetch_bytes(
        _SOURCE_URL,
        max_bytes=_MAX_SOURCE_BYTES,
        expected_host="upload.wikimedia.org",
    )
    identity = validate_source_bytes(source)
    common: dict[str, object] = {
        "evidence_scope": "single_cc0_front_envelope_smoke_not_commercial_accuracy",
        "rights_source": _RIGHTS_PAGE,
        "source_url": _SOURCE_URL,
        "source_bytes": identity["bytes"],
        "source_sha1": identity["sha1"],
        "source_sha256": identity["sha256"],
        "expected_dimensions": [_WIDTH, _HEIGHT],
        "expected_text_normalized": _EXPECTED_TEXT_NORMALIZED,
        "documented_min_plate_width_px": _DOCUMENTED_MIN_PLATE_WIDTH_PX,
        "predeclared_plate_width_lower_bound_px": _PREDECLARED_PLATE_WIDTH_LOWER_BOUND_PX,
        "within_documented_plate_size_envelope": (
            _PREDECLARED_PLATE_WIDTH_LOWER_BOUND_PX >= _DOCUMENTED_MIN_PLATE_WIDTH_PX
        ),
    }
    if _SOURCE_SHA256 is None:
        return {**common, "admission_only": True, "inference_run": False}

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV and NumPy are required for LPR front-envelope evidence"
        ) from exc
    image = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or tuple(int(value) for value in image.shape[:2]) != (
        _HEIGHT,
        _WIDTH,
    ):
        raise RuntimeError("decoded source dimensions changed")

    with tempfile.TemporaryDirectory(prefix="analytics-lpr-front-envelope-") as directory:
        root = Path(directory)
        _download_model_files(root)
        detector = OpenVINOOMZPlateDetector(root)
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        detections = detector(image)
        cpu_seconds = time.process_time() - cpu_start
        elapsed_seconds = time.perf_counter() - wall_start

    best = detections[0] if detections else None
    return {
        **common,
        "admission_only": False,
        "inference_run": True,
        "runtime_version": detector.runtime_version,
        "plate_detections": len(detections),
        "best_plate_confidence": float(best.confidence) if best is not None else None,
        "elapsed_seconds": elapsed_seconds,
        "cpu_seconds": cpu_seconds,
        "throughput_fps": (1.0 / elapsed_seconds) if elapsed_seconds > 0.0 else 0.0,
        "latency_ms": elapsed_seconds * 1000.0,
    }


def main() -> int:
    print(json.dumps(run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
