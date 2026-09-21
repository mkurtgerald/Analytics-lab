"""Bounded second-source LPR/OCR evidence on one CC0 South Carolina plate.

The source, expected serial and crop are predeclared before OCR. First head is
admission-only: verify published size/SHA-1, discover SHA-256, decode ephemerally,
and bind the manually authored RGB24 crop. Only after those identities are pinned
may the unchanged retained grayscale->Otsu->RGB24 + exact Tesseract 5.5.3 path run.
This single sample is engineering evidence only, never commercial accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .lpr_ocr import LPROCRConfig, TesseractPlateRecognizer, verify_tessdata_fast_eng
from .lpr_ocr_exact_evidence import (
    _OfficialWindowsPackageRunner,
    _TESSERACT_INSTALLER_URL,
    _TESSERACT_INSTALLER_SIZE,
    _TESSERACT_INSTALLER_SHA256,
    _TESSERACT_RELEASE,
    _TESSDATA_URL,
    _TESSDATA_SIZE,
    _TESSDATA_GIT_BLOB_SHA1,
    _download,
    _git_blob_sha1,
    _ppm,
)

_RIGHTS_PAGE = (
    "https://commons.wikimedia.org/w/index.php?title="
    "File:2023_South_Carolina_mail-out_series_passenger_car_rear_license_plate.png"
    "&oldid=1078900343"
)
_SOURCE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/d/d6/"
    "2023_South_Carolina_mail-out_series_passenger_car_rear_license_plate.png"
)
_SOURCE_SIZE = 1_046_799
_SOURCE_SHA1 = "b3d301e18457b16e0729f89be1abd438f935aa55"
_SOURCE_SHA256: str | None = None
_SOURCE_WIDTH = 1_286
_SOURCE_HEIGHT = 634

# Independently authored from the exact CC0 source before OCR/model output.
# Excludes the slogan/footer while preserving the complete serial and center mark.
_PLATE_TEXT_BOX_PX = (45, 145, 1_220, 455)
_EXPECTED_TEXT = "354AVV"
_CROP_RGB24_SHA256: str | None = None

_PREPROCESS_PLAN = "rgb24->grayscale->global-otsu-binary->rgb24;native-size;no-invert"
_TESSERACT_PSM = 7
_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_MAX_INSTALLER_BYTES = 27 * 1024 * 1024
_MAX_TESSDATA_BYTES = 5 * 1024 * 1024


def _verify_source(payload: bytes) -> tuple[str, str]:
    if len(payload) != _SOURCE_SIZE:
        raise RuntimeError("source size mismatch")
    sha1 = hashlib.sha1(payload).hexdigest()
    sha256 = hashlib.sha256(payload).hexdigest()
    if sha1 != _SOURCE_SHA1:
        raise RuntimeError("source SHA-1 mismatch")
    if _SOURCE_SHA256 is not None and sha256 != _SOURCE_SHA256:
        raise RuntimeError("source SHA-256 mismatch")
    return sha1, sha256


def _bounded_work_dir(path: Path) -> Path:
    root_value = os.environ.get("RUNNER_TEMP")
    if not root_value:
        raise RuntimeError("RUNNER_TEMP is required")
    root = Path(root_value).resolve()
    candidate = path.resolve()
    if candidate == root or root not in candidate.parents:
        raise RuntimeError("work directory must be below RUNNER_TEMP")
    return candidate


def _crop_identity(image: object) -> tuple[bytes, int, int, str]:
    import cv2  # type: ignore

    shape = getattr(image, "shape", None)
    if shape is None or tuple(int(v) for v in shape[:2]) != (_SOURCE_HEIGHT, _SOURCE_WIDTH):
        raise RuntimeError("source decode dimensions mismatch")
    left, top, right, bottom = _PLATE_TEXT_BOX_PX
    if not (0 <= left < right <= _SOURCE_WIDTH and 0 <= top < bottom <= _SOURCE_HEIGHT):
        raise RuntimeError("predeclared crop is outside source")
    crop = image[top:bottom, left:right]
    height, width = int(crop.shape[0]), int(crop.shape[1])
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).tobytes(order="C")
    digest = hashlib.sha256(rgb).hexdigest()
    if _CROP_RGB24_SHA256 is not None and digest != _CROP_RGB24_SHA256:
        raise RuntimeError("crop RGB24 identity mismatch")
    return rgb, width, height, digest


def run(work_dir: Path) -> dict[str, object]:
    if os.name != "nt":
        raise RuntimeError("South Carolina exact OCR evidence requires Windows")
    work_dir = _bounded_work_dir(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        source = _download(
            _SOURCE_URL,
            allowed_hosts=frozenset({"upload.wikimedia.org"}),
            max_bytes=_MAX_SOURCE_BYTES,
        )
        source_sha1, source_sha256 = _verify_source(source)

        import cv2  # type: ignore
        import numpy  # type: ignore

        image = cv2.imdecode(numpy.frombuffer(source, dtype=numpy.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("source decode failed")
        rgb, crop_width, crop_height, crop_sha256 = _crop_identity(image)

        common: dict[str, object] = {
            "evidence": "lpr-ocr-sc-cc0",
            "rights_page": _RIGHTS_PAGE,
            "source_url": _SOURCE_URL,
            "source_bytes": len(source),
            "source_sha1": source_sha1,
            "source_sha256": source_sha256,
            "source_dimensions": [_SOURCE_WIDTH, _SOURCE_HEIGHT],
            "crop_box_px": list(_PLATE_TEXT_BOX_PX),
            "crop_width": crop_width,
            "crop_height": crop_height,
            "crop_rgb24_bytes": len(rgb),
            "crop_rgb24_sha256": crop_sha256,
            "expected_text": _EXPECTED_TEXT,
            "preprocess_plan": _PREPROCESS_PLAN,
            "claim": "single CC0 plate engineering evidence only; not commercial accuracy",
        }
        if _SOURCE_SHA256 is None or _CROP_RGB24_SHA256 is None:
            return {
                **common,
                "admission_only": True,
                "ocr_run": False,
                "total_elapsed_seconds": round(time.perf_counter() - started, 6),
                "total_cpu_seconds": round(time.process_time() - cpu_started, 6),
            }

        canonical = numpy.frombuffer(rgb, dtype=numpy.uint8).reshape((crop_height, crop_width, 3))
        gray = cv2.cvtColor(canonical, cv2.COLOR_RGB2GRAY)
        threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ocr_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB).tobytes(order="C")
        preprocessed_sha256 = hashlib.sha256(ocr_rgb).hexdigest()
        ppm = _ppm(ocr_rgb, width=crop_width, height=crop_height)

        installer = _download(
            _TESSERACT_INSTALLER_URL,
            allowed_hosts=frozenset({"github.com", "release-assets.githubusercontent.com"}),
            max_bytes=_MAX_INSTALLER_BYTES,
        )
        if len(installer) != _TESSERACT_INSTALLER_SIZE or hashlib.sha256(installer).hexdigest() != _TESSERACT_INSTALLER_SHA256:
            raise RuntimeError("Tesseract installer identity mismatch")

        tessdata = _download(
            _TESSDATA_URL,
            allowed_hosts=frozenset({"raw.githubusercontent.com"}),
            max_bytes=_MAX_TESSDATA_BYTES,
        )
        if len(tessdata) != _TESSDATA_SIZE or _git_blob_sha1(tessdata) != _TESSDATA_GIT_BLOB_SHA1:
            raise RuntimeError("tessdata identity mismatch")

        installer_path = work_dir / "tesseract-ocr-w64-setup-5.5.3.20260724.exe"
        installer_path.write_bytes(installer)
        install_dir = work_dir / "tesseract"
        result = subprocess.run(
            [str(installer_path), "/S", f"/D={install_dir}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Tesseract installer failed")
        binaries = tuple(install_dir.rglob("tesseract.exe"))
        if len(binaries) != 1:
            raise RuntimeError("Tesseract executable was not uniquely installed")

        tessdata_dir = work_dir / "tessdata"
        tessdata_dir.mkdir()
        traineddata_path = tessdata_dir / "eng.traineddata"
        traineddata_path.write_bytes(tessdata)
        artifact = verify_tessdata_fast_eng(traineddata_path)

        package_runner = _OfficialWindowsPackageRunner(str(binaries[0]))
        recognizer = TesseractPlateRecognizer(
            traineddata_path,
            binary=str(binaries[0]),
            config=LPROCRConfig(tesseract_timeout_seconds=15.0),
            runner=package_runner,
        )
        ocr_started = time.perf_counter()
        ocr_cpu_started = time.process_time()
        observed = recognizer(ppm)
        ocr_elapsed = time.perf_counter() - ocr_started
        ocr_cpu = time.process_time() - ocr_cpu_started
        if package_runner.reported_version is None:
            raise RuntimeError("Tesseract package version was not observed")

        return {
            **common,
            "admission_only": False,
            "ocr_run": True,
            "preprocessed_rgb24_sha256": preprocessed_sha256,
            "otsu_threshold": float(threshold),
            "tesseract_psm": _TESSERACT_PSM,
            "tesseract_version": _TESSERACT_RELEASE,
            "tesseract_reported_version": package_runner.reported_version,
            "tesseract_installer_sha256": _TESSERACT_INSTALLER_SHA256,
            "tessdata_git_blob_sha1": artifact.git_blob_sha1,
            "tessdata_sha256": artifact.sha256,
            "observed_text": None if observed is None else observed.text,
            "observed_confidence": None if observed is None else observed.confidence,
            "exact_match": bool(observed is not None and observed.text == _EXPECTED_TEXT),
            "ocr_elapsed_seconds": round(ocr_elapsed, 6),
            "ocr_cpu_seconds": round(ocr_cpu, 6),
            "total_elapsed_seconds": round(time.perf_counter() - started, 6),
            "total_cpu_seconds": round(time.process_time() - cpu_started, 6),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.work_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
