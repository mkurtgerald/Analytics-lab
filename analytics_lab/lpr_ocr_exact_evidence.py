"""Run one bounded OCR measurement on the pre-bound CC0 plate crop.

The lane is intentionally narrow: it downloads only the already-reviewed source,
the official Tesseract 5.5.3 Windows release installer, and the pinned Apache-2.0
`tessdata_fast` English model. Every payload is fail-closed on immutable identity,
all temporary files stay under RUNNER_TEMP, no image/crop/model artifact is
uploaded, and each explicitly pre-registered branch performs exactly one OCR
observation without an accuracy claim.
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
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .lpr_ocr import LPROCRConfig, TesseractPlateRecognizer, verify_tessdata_fast_eng

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
_SOURCE_WIDTH = 4_032
_SOURCE_HEIGHT = 3_024
_PLATE_BOX_PX = (960, 2_020, 2_740, 2_470)
_CROP_RGB24_SHA256 = "0d89606f174889fdeab9fd969c3cfbe0a8e033c88eac6c0a4d0385fd45f56599"
_EXPECTED_TEXT = "MPR318"
_PREPROCESS_BRANCH_PREFIX = "evidence/lpr-ocr-exact-preprocess-"
_PREPROCESS_PLAN = "rgb24->grayscale->global-otsu-binary->rgb24;native-size;no-invert"
_TESSERACT_PSM = 7

_TESSERACT_RELEASE = "5.5.3"
_TESSERACT_WINDOWS_VERSION_LINE = "tesseract v5.5.3.20260724"
_TESSERACT_INSTALLER_URL = (
    "https://github.com/tesseract-ocr/tesseract/releases/download/5.5.3/"
    "tesseract-ocr-w64-setup-5.5.3.20260724.exe"
)
_TESSERACT_INSTALLER_SIZE = 26_573_224
_TESSERACT_INSTALLER_SHA256 = "bee9e3434bd94fd65387d9be28cd467a41f61b1275383b55b0f59a1331270ae4"

_TESSDATA_COMMIT = "87416418657359cb625c412a48b6e1d6d41c29bd"
_TESSDATA_URL = (
    "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/"
    f"{_TESSDATA_COMMIT}/eng.traineddata"
)
_TESSDATA_SIZE = 4_113_088
_TESSDATA_GIT_BLOB_SHA1 = "bbef4675053b5b468cdb477053e28b1c698ba08e"

_MAX_SOURCE_BYTES = 3 * 1024 * 1024
_MAX_INSTALLER_BYTES = 27 * 1024 * 1024
_MAX_TESSDATA_BYTES = 5 * 1024 * 1024
_USER_AGENT = "Analytics-Lab-bounded-evidence/1"


def _git_blob_sha1(payload: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _download(url: str, *, allowed_hosts: frozenset[str], max_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=30) as response:
        final = urlparse(response.geturl())
        if final.scheme != "https" or final.hostname not in allowed_hosts:
            raise RuntimeError("evidence redirect escaped reviewed hosts")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > max_bytes:
            raise RuntimeError("evidence payload exceeds byte cap")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise RuntimeError("evidence payload exceeds byte cap")
    return payload


def _verify_sha256(payload: bytes, *, size: int, sha256: str, name: str) -> str:
    if len(payload) != size:
        raise RuntimeError(f"{name} size mismatch")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != sha256:
        raise RuntimeError(f"{name} SHA-256 mismatch")
    return actual


def _verify_source(payload: bytes) -> None:
    if len(payload) != _SOURCE_SIZE:
        raise RuntimeError("source size mismatch")
    if hashlib.sha1(payload).hexdigest() != _SOURCE_SHA1:
        raise RuntimeError("source SHA-1 mismatch")
    if hashlib.sha256(payload).hexdigest() != _SOURCE_SHA256:
        raise RuntimeError("source SHA-256 mismatch")


def _verify_tessdata_bytes(payload: bytes) -> str:
    if len(payload) != _TESSDATA_SIZE:
        raise RuntimeError("tessdata size mismatch")
    if _git_blob_sha1(payload) != _TESSDATA_GIT_BLOB_SHA1:
        raise RuntimeError("tessdata Git blob identity mismatch")
    return hashlib.sha256(payload).hexdigest()


def _ppm(rgb: bytes, *, width: int, height: int) -> bytes:
    expected = width * height * 3
    if len(rgb) != expected:
        raise ValueError("RGB24 crop byte count mismatch")
    return f"P6\n{width} {height}\n255\n".encode("ascii") + rgb


def _selected_preprocess(head_ref: str) -> str:
    """Select the one preregistered comparison without exposing a tuning knob."""
    if not isinstance(head_ref, str):
        raise ValueError("head ref must be text")
    if head_ref.startswith(_PREPROCESS_BRANCH_PREFIX):
        return "gray-otsu"
    return "raw"


def _bounded_work_dir(path: Path) -> tuple[Path, Path]:
    root_value = os.environ.get("RUNNER_TEMP")
    if not root_value:
        raise RuntimeError("RUNNER_TEMP is required")
    root = Path(root_value).resolve()
    candidate = path.resolve()
    if candidate == root or root not in candidate.parents:
        raise RuntimeError("work directory must be below RUNNER_TEMP")
    return root, candidate


BaseRunner = Callable[..., subprocess.CompletedProcess[Any]]


class _OfficialWindowsPackageRunner:
    """Bridge one immutable Windows package version string to the core semver gate.

    Tesseract's official 5.5.3 Windows package reports
    ``tesseract v5.5.3.20260724`` while the platform-neutral adapter intentionally
    admits only semantic version 5.5.3. This evidence-only runner accepts exactly
    that one reviewed package string, records it, and presents the already-pinned
    semantic version to the unchanged adapter. Any other package string fails
    closed before OCR.
    """

    def __init__(
        self,
        binary: str,
        *,
        runner: BaseRunner = subprocess.run,
    ) -> None:
        self.binary = binary
        self._runner = runner
        self.reported_version: str | None = None

    def __call__(self, args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        result = self._runner(args, **kwargs)
        if list(args) != [self.binary, "--version"]:
            return result
        if getattr(result, "returncode", 1) != 0:
            return result
        stdout = getattr(result, "stdout", b"")
        text = stdout.decode("utf-8", "strict") if isinstance(stdout, bytes) else str(stdout)
        first = text.splitlines()[0].strip() if text.splitlines() else ""
        if first != _TESSERACT_WINDOWS_VERSION_LINE:
            raise RuntimeError("Tesseract Windows package version mismatch")
        self.reported_version = first
        normalized: bytes | str
        if isinstance(stdout, bytes):
            normalized = f"tesseract {_TESSERACT_RELEASE}\n".encode("ascii")
        else:
            normalized = f"tesseract {_TESSERACT_RELEASE}\n"
        return subprocess.CompletedProcess(
            result.args,
            result.returncode,
            stdout=normalized,
            stderr=getattr(result, "stderr", None),
        )


def run(work_dir: Path) -> dict[str, object]:
    if os.name != "nt":
        raise RuntimeError("exact Tesseract evidence lane requires Windows")
    _, work_dir = _bounded_work_dir(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    preprocess = _selected_preprocess(os.environ.get("GITHUB_HEAD_REF", ""))
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        source = _download(
            _SOURCE_URL,
            allowed_hosts=frozenset({"upload.wikimedia.org"}),
            max_bytes=_MAX_SOURCE_BYTES,
        )
        _verify_source(source)

        installer = _download(
            _TESSERACT_INSTALLER_URL,
            allowed_hosts=frozenset({"github.com", "release-assets.githubusercontent.com"}),
            max_bytes=_MAX_INSTALLER_BYTES,
        )
        installer_sha256 = _verify_sha256(
            installer,
            size=_TESSERACT_INSTALLER_SIZE,
            sha256=_TESSERACT_INSTALLER_SHA256,
            name="Tesseract installer",
        )

        tessdata = _download(
            _TESSDATA_URL,
            allowed_hosts=frozenset({"raw.githubusercontent.com"}),
            max_bytes=_MAX_TESSDATA_BYTES,
        )
        tessdata_sha256 = _verify_tessdata_bytes(tessdata)

        import cv2  # type: ignore
        import numpy  # type: ignore

        encoded = numpy.frombuffer(source, dtype=numpy.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or tuple(int(v) for v in image.shape[:2]) != (_SOURCE_HEIGHT, _SOURCE_WIDTH):
            raise RuntimeError("source decode dimensions mismatch")
        left, top, right, bottom = _PLATE_BOX_PX
        crop = image[top:bottom, left:right]
        crop_height, crop_width = (int(crop.shape[0]), int(crop.shape[1]))
        if (crop_width, crop_height) != (1_780, 450):
            raise RuntimeError("fixed crop dimensions mismatch")
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).tobytes(order="C")
        crop_sha256 = hashlib.sha256(rgb).hexdigest()
        if crop_sha256 != _CROP_RGB24_SHA256:
            raise RuntimeError("fixed crop RGB24 identity mismatch")

        ocr_rgb = rgb
        otsu_threshold: float | None = None
        if preprocess == "gray-otsu":
            canonical = numpy.frombuffer(rgb, dtype=numpy.uint8).reshape((crop_height, crop_width, 3))
            gray = cv2.cvtColor(canonical, cv2.COLOR_RGB2GRAY)
            threshold, binary = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU,
            )
            otsu_threshold = float(threshold)
            ocr_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB).tobytes(order="C")
        preprocessed_sha256 = hashlib.sha256(ocr_rgb).hexdigest()
        ppm = _ppm(ocr_rgb, width=crop_width, height=crop_height)

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
        binary = binaries[0]

        tessdata_dir = work_dir / "tessdata"
        tessdata_dir.mkdir()
        traineddata_path = tessdata_dir / "eng.traineddata"
        traineddata_path.write_bytes(tessdata)
        artifact = verify_tessdata_fast_eng(traineddata_path)
        if artifact.sha256 != tessdata_sha256:
            raise RuntimeError("tessdata verification disagreement")

        package_runner = _OfficialWindowsPackageRunner(str(binary))
        recognizer = TesseractPlateRecognizer(
            traineddata_path,
            binary=str(binary),
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

        evidence_name = (
            "lpr-ocr-exact-gray-otsu-first-attempt"
            if preprocess == "gray-otsu"
            else "lpr-ocr-exact-first-attempt"
        )
        return {
            "evidence": evidence_name,
            "rights_page": _RIGHTS_PAGE,
            "source_sha256": _SOURCE_SHA256,
            "crop_box_px": list(_PLATE_BOX_PX),
            "crop_rgb24_sha256": crop_sha256,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "crop_rgb24_bytes": len(rgb),
            "preprocess": preprocess,
            "preprocess_plan": _PREPROCESS_PLAN if preprocess == "gray-otsu" else "raw-rgb24",
            "preprocessed_rgb24_sha256": preprocessed_sha256,
            "otsu_threshold": otsu_threshold,
            "tesseract_psm": _TESSERACT_PSM,
            "tesseract_version": _TESSERACT_RELEASE,
            "tesseract_reported_version": package_runner.reported_version,
            "tesseract_installer_bytes": len(installer),
            "tesseract_installer_sha256": installer_sha256,
            "tessdata_commit": _TESSDATA_COMMIT,
            "tessdata_git_blob_sha1": artifact.git_blob_sha1,
            "tessdata_sha256": tessdata_sha256,
            "expected_text": _EXPECTED_TEXT,
            "observed_text": None if observed is None else observed.text,
            "observed_confidence": None if observed is None else observed.confidence,
            "exact_match": bool(observed is not None and observed.text == _EXPECTED_TEXT),
            "ocr_elapsed_seconds": round(ocr_elapsed, 6),
            "ocr_cpu_seconds": round(ocr_cpu, 6),
            "total_elapsed_seconds": round(time.perf_counter() - started, 6),
            "total_cpu_seconds": round(time.process_time() - cpu_started, 6),
            "claim": "single-image engineering smoke only; not commercial accuracy",
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.work_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
