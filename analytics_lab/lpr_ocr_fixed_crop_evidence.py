"""Bind one independently authored CC0 plate crop before OCR.

This evidence lane deliberately does not use detector/proposal output to define
the crop. It downloads only the already-reviewed CC0 Land Rover source, verifies
its immutable byte identity, decodes it ephemerally, extracts one manually
pre-registered plate rectangle, and emits only source/crop identities plus the
known visible normalized text. It performs no OCR and makes no accuracy claim.
"""
from __future__ import annotations

import hashlib
import json
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
_SOURCE_SHA256 = "32e5637e39b54c26192c011c1cc5516bd6d35573582b8e09d7bc9aae90ef1db4"
_WIDTH = 4_032
_HEIGHT = 3_024
_EXPECTED_TEXT_NORMALIZED = "MPR318"
_MAX_SOURCE_BYTES = 3 * 1024 * 1024

# Manually authored before OCR from independent visual inspection of the exact
# source. Intentionally loose around the complete physical plate; no detector,
# proposal score, OCR result, or model output was used to choose these pixels.
_PLATE_BOX_PX = (960, 2_020, 2_740, 2_470)


def _fetch_source() -> bytes:
    request = Request(_SOURCE_URL, headers={"User-Agent": "Analytics-Lab-bounded-evidence/1"})
    with urlopen(request, timeout=20) as response:
        final = urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != "upload.wikimedia.org":
            raise RuntimeError("evidence redirect escaped reviewed host")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > _MAX_SOURCE_BYTES:
            raise RuntimeError("evidence payload exceeds byte cap")
        payload = response.read(_MAX_SOURCE_BYTES + 1)
    if len(payload) > _MAX_SOURCE_BYTES:
        raise RuntimeError("evidence payload exceeds byte cap")
    return payload


def validate_source(payload: bytes) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise ValueError("source payload must be bytes")
    if len(payload) != _SOURCE_SIZE:
        raise RuntimeError("source size mismatch")
    sha1 = hashlib.sha1(payload).hexdigest()
    sha256 = hashlib.sha256(payload).hexdigest()
    if sha1 != _SOURCE_SHA1:
        raise RuntimeError("source SHA-1 mismatch")
    if sha256 != _SOURCE_SHA256:
        raise RuntimeError("source SHA-256 mismatch")
    return {"bytes": len(payload), "sha1": sha1, "sha256": sha256}


def validate_plate_box(
    box: tuple[int, int, int, int] = _PLATE_BOX_PX,
    *,
    width: int = _WIDTH,
    height: int = _HEIGHT,
) -> tuple[int, int, int, int]:
    if (
        not isinstance(box, tuple)
        or len(box) != 4
        or any(type(value) is not int for value in box)
    ):
        raise ValueError("plate box must contain four integer pixel coordinates")
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError("image dimensions must be positive integers")
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("plate box must be inside the source image")
    if (right - left) < 96 or (bottom - top) < 24:
        raise ValueError("plate box is implausibly small")
    return box


def crop_identity(image: object) -> dict[str, object]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for fixed-crop evidence") from exc

    shape = getattr(image, "shape", None)
    if shape is None or tuple(int(value) for value in shape[:2]) != (_HEIGHT, _WIDTH):
        raise RuntimeError("decoded source dimensions changed")
    if len(shape) != 3 or int(shape[2]) != 3:
        raise RuntimeError("decoded source must be three-channel BGR")

    left, top, right, bottom = validate_plate_box()
    crop = image[top:bottom, left:right]
    if tuple(int(value) for value in crop.shape[:2]) != (bottom - top, right - left):
        raise RuntimeError("fixed plate crop dimensions changed")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb_bytes = bytes(rgb.tobytes(order="C"))
    digest = hashlib.sha256(rgb_bytes).hexdigest()
    return {
        "plate_box_px": [left, top, right, bottom],
        "plate_box_normalized": [
            left / _WIDTH,
            top / _HEIGHT,
            right / _WIDTH,
            bottom / _HEIGHT,
        ],
        "crop_width": right - left,
        "crop_height": bottom - top,
        "crop_rgb24_bytes": len(rgb_bytes),
        "crop_rgb24_sha256": digest,
    }


def run() -> dict[str, object]:
    payload = _fetch_source()
    source_identity = validate_source(payload)
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for fixed-crop evidence") from exc

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("source decode failed")
    crop = crop_identity(image)
    return {
        "evidence_scope": "single_cc0_manual_plate_crop_binding_not_commercial_accuracy",
        "rights_source": _RIGHTS_PAGE,
        "source_url": _SOURCE_URL,
        "source_bytes": source_identity["bytes"],
        "source_sha1": source_identity["sha1"],
        "source_sha256": source_identity["sha256"],
        "expected_dimensions": [_WIDTH, _HEIGHT],
        "expected_text_normalized": _EXPECTED_TEXT_NORMALIZED,
        "crop_authorship": "independent_manual_pre_ocr",
        "detector_or_proposal_used_for_crop": False,
        "ocr_run": False,
        **crop,
    }


def main() -> int:
    print(json.dumps(run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
