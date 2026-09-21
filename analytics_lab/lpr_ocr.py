"""First runnable, platform-neutral LPR/OCR engineering baseline.

The module deliberately keeps donor/runtime boundaries replaceable. It can run a
reviewed Open Model Zoo plate detector locally and send only the resulting plate
crop to a locally provisioned Tesseract 5.5.3 executable. No network access,
model download, identity/ReID behavior, training, or accuracy claim occurs here.

This is an engineering integration baseline, not a commercial-accuracy claim or
release approval. Product packaging still requires the native dependency and
model/data provenance review recorded in THIRD_PARTY.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterable, Protocol, Sequence

_OMZ_COMMIT = "6697dead54ed1cdd664b0313189c2cb52ee6335e"
_OMZ_LICENSE = f"https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/{_OMZ_COMMIT}/LICENSE"
_OMZ_XML = "vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml"
_OMZ_BIN = "vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.bin"
_TESSERACT_VERSION = "5.5.3"
_TESSDATA_ENG_SIZE = 4_113_088
_TESSDATA_ENG_GIT_BLOB_SHA1 = "bbef4675053b5b468cdb477053e28b1c698ba08e"
_SAFE_BINARY = re.compile(r"[A-Za-z0-9_./:\\-]{1,512}\Z")
_SAFE_TEXT = re.compile(r"[A-Z0-9]{1,16}\Z")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PlateDetection:
    """Normalized plate rectangle emitted by the detector boundary."""

    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        confidence = _finite(self.confidence, "confidence")
        coords = tuple(_finite(getattr(self, name), name) for name in ("x1", "y1", "x2", "y2"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError("plate coordinates must be normalized")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("plate rectangle must have positive area")


@dataclass(frozen=True)
class OCRText:
    text: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not _SAFE_TEXT.fullmatch(self.text):
            raise ValueError("OCR text must be normalized A-Z/0-9 and 1-16 characters")
        confidence = _finite(self.confidence, "OCR confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("OCR confidence must be in [0, 1]")


@dataclass(frozen=True)
class LPRObservation:
    plate: PlateDetection
    text: OCRText

    def __post_init__(self) -> None:
        if not isinstance(self.plate, PlateDetection) or not isinstance(self.text, OCRText):
            raise ValueError("LPR observation requires a plate and OCR text")


@dataclass(frozen=True)
class TessdataIdentity:
    size_bytes: int
    git_blob_sha1: str
    sha256: str


@dataclass(frozen=True)
class LPROCRConfig:
    plate_threshold: float = 0.50
    max_plates: int = 16
    tesseract_timeout_seconds: float = 3.0
    min_ocr_confidence: float = 0.0

    def __post_init__(self) -> None:
        for name in ("plate_threshold", "min_ocr_confidence"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if type(self.max_plates) is not int or not 1 <= self.max_plates <= 64:
            raise ValueError("max_plates must be an integer in [1, 64]")
        timeout = _finite(self.tesseract_timeout_seconds, "tesseract_timeout_seconds")
        if not 0.1 <= timeout <= 30.0:
            raise ValueError("tesseract_timeout_seconds must be in [0.1, 30]")


def parse_omz_plate_rows(
    rows: Iterable[Sequence[float]], *, threshold: float = 0.50, max_plates: int = 16
) -> tuple[PlateDetection, ...]:
    """Parse OMZ SSD rows, admitting label 2 (license plate) only."""
    cutoff = _finite(threshold, "threshold")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if type(max_plates) is not int or not 1 <= max_plates <= 64:
        raise ValueError("max_plates must be an integer in [1, 64]")
    result: list[PlateDetection] = []
    count = 0
    for row in rows:
        count += 1
        if count > 4096:
            raise RuntimeError("detector output row limit exceeded")
        if isinstance(row, (str, bytes)):
            raise ValueError("detector row must contain seven numeric values")
        try:
            if len(row) != 7:
                raise ValueError("detector row must contain seven numeric values")
            image_id, label, confidence, x1, y1, x2, y2 = (
                _finite(row[index], "detector value") for index in range(7)
            )
        except (TypeError, IndexError, KeyError) as exc:
            raise ValueError("detector row must contain seven numeric values") from exc
        if image_id < 0:
            break
        if label != 2 or confidence < cutoff:
            continue
        left = min(1.0, max(0.0, x1))
        top = min(1.0, max(0.0, y1))
        right = min(1.0, max(0.0, x2))
        bottom = min(1.0, max(0.0, y2))
        if right <= left or bottom <= top:
            continue
        result.append(PlateDetection(confidence, left, top, right, bottom))
    result.sort(key=lambda item: (-item.confidence, item.x1, item.y1, item.x2, item.y2))
    return tuple(result[:max_plates])


def verify_tessdata_artifact(
    path: str | Path, *, expected_size: int, expected_git_blob_sha1: str
) -> TessdataIdentity:
    """Verify a regular traineddata file against its immutable Git object identity."""
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("traineddata must be a regular non-symlink file")
    if type(expected_size) is not int or expected_size < 1:
        raise ValueError("expected_size must be positive")
    if not isinstance(expected_git_blob_sha1, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_git_blob_sha1):
        raise ValueError("expected_git_blob_sha1 must be lowercase SHA-1 hex")
    actual_size = candidate.stat().st_size
    if actual_size != expected_size:
        raise ValueError("traineddata size mismatch")
    git_sha = hashlib.sha1()
    git_sha.update(f"blob {actual_size}\0".encode("ascii"))
    sha256 = hashlib.sha256()
    with candidate.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            git_sha.update(block)
            sha256.update(block)
    if git_sha.hexdigest() != expected_git_blob_sha1:
        raise ValueError("traineddata Git blob identity mismatch")
    return TessdataIdentity(actual_size, git_sha.hexdigest(), sha256.hexdigest())


def verify_tessdata_fast_eng(path: str | Path) -> TessdataIdentity:
    """Verify the pinned tessdata_fast English model used by this baseline."""
    return verify_tessdata_artifact(
        path,
        expected_size=_TESSDATA_ENG_SIZE,
        expected_git_blob_sha1=_TESSDATA_ENG_GIT_BLOB_SHA1,
    )


def _image_size(image: Any) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if shape is not None:
        if len(shape) != 3 or int(shape[2]) != 3:
            raise ValueError("image must be HxWx3 BGR")
        height, width = int(shape[0]), int(shape[1])
    else:
        try:
            height = len(image)
            width = len(image[0]) if height else 0
        except (TypeError, IndexError) as exc:
            raise ValueError("image must be HxWx3 BGR") from exc
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    return width, height


def encode_plate_crop_ppm(image: Any, plate: PlateDetection) -> bytes:
    """Encode one BGR plate crop as binary PPM using only Python primitives."""
    if not isinstance(plate, PlateDetection):
        raise ValueError("plate must be a PlateDetection")
    width, height = _image_size(image)
    left = max(0, min(width - 1, int(math.floor(plate.x1 * width))))
    top = max(0, min(height - 1, int(math.floor(plate.y1 * height))))
    right = max(left + 1, min(width, int(math.ceil(plate.x2 * width))))
    bottom = max(top + 1, min(height, int(math.ceil(plate.y2 * height))))
    pixels = bytearray()
    for y in range(top, bottom):
        for x in range(left, right):
            pixel = image[y][x]
            try:
                if len(pixel) != 3:
                    raise ValueError("image must be HxWx3 BGR")
                b, g, r = (int(pixel[index]) for index in range(3))
            except (TypeError, IndexError, ValueError) as exc:
                raise ValueError("image pixels must contain three byte channels") from exc
            if any(value < 0 or value > 255 for value in (b, g, r)):
                raise ValueError("image channels must be bytes")
            pixels.extend((r, g, b))
    crop_width = right - left
    crop_height = bottom - top
    return f"P6\n{crop_width} {crop_height}\n255\n".encode("ascii") + bytes(pixels)


def normalize_plate_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("OCR text must be a string")
    return "".join(character for character in value.upper() if "A" <= character <= "Z" or "0" <= character <= "9")[:16]


def parse_tesseract_tsv(tsv: str, *, min_confidence: float = 0.0) -> OCRText | None:
    """Collapse Tesseract TSV word rows into one bounded plate text result."""
    cutoff = _finite(min_confidence, "min_confidence")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if not isinstance(tsv, str) or len(tsv) > 1_000_000:
        raise ValueError("Tesseract TSV must be bounded text")
    lines = tsv.splitlines()
    if not lines:
        return None
    required = ("level", "conf", "text")
    header_index: int | None = None
    header: list[str] | None = None
    # Tesseract may emit bounded informational preamble lines before the TSV
    # table. Accept only a real tab-delimited header containing the required
    # columns, and search only a small prefix so malformed output still fails
    # closed instead of being treated as arbitrary text.
    for index, line in enumerate(lines[:32]):
        columns = line.split("\t")
        if all(name in columns for name in required):
            header_index = index
            header = columns
            break
    if header_index is None or header is None:
        raise ValueError("Tesseract TSV header is incomplete")
    indexes = {name: header.index(name) for name in required}
    words: list[str] = []
    confidences: list[float] = []
    for line in lines[header_index + 1:]:
        columns = line.split("\t")
        if len(columns) <= max(indexes.values()):
            continue
        try:
            level = int(columns[indexes["level"]])
            confidence = float(columns[indexes["conf"]])
        except ValueError:
            continue
        if level != 5 or not math.isfinite(confidence) or confidence < 0.0:
            continue
        normalized = normalize_plate_text(columns[indexes["text"]])
        if not normalized:
            continue
        confidence01 = min(1.0, max(0.0, confidence / 100.0))
        if confidence01 < cutoff:
            continue
        words.append(normalized)
        confidences.append(confidence01)
    text = normalize_plate_text("".join(words))
    if not text:
        return None
    return OCRText(text, sum(confidences) / len(confidences))


Runner = Callable[..., subprocess.CompletedProcess[Any]]


class TesseractPlateRecognizer:
    """Fail-closed adapter around a locally provisioned Tesseract 5.5.3 CLI."""

    def __init__(
        self,
        traineddata_path: str | Path,
        *,
        binary: str = "tesseract",
        config: LPROCRConfig | None = None,
        runner: Runner = subprocess.run,
    ) -> None:
        self.config = config or LPROCRConfig()
        if not isinstance(self.config, LPROCRConfig):
            raise ValueError("config must be LPROCRConfig")
        if not isinstance(binary, str) or not _SAFE_BINARY.fullmatch(binary):
            raise ValueError("binary must be a bounded simple path")
        if not callable(runner):
            raise ValueError("runner must be callable")
        path = Path(traineddata_path)
        if path.name != "eng.traineddata":
            raise ValueError("baseline requires eng.traineddata")
        self.artifact_identity = verify_tessdata_fast_eng(path)
        self._tessdata_dir = path.parent
        self._binary = binary
        self._runner = runner
        self._version_checked = False

    def _check_version(self) -> None:
        if self._version_checked:
            return
        result = self._runner(
            [self._binary, "--version"], capture_output=True, timeout=self.config.tesseract_timeout_seconds, check=False
        )
        if getattr(result, "returncode", 1) != 0:
            raise RuntimeError("Tesseract version probe failed")
        stdout = getattr(result, "stdout", b"")
        text = stdout.decode("utf-8", "strict") if isinstance(stdout, bytes) else str(stdout)
        first = text.splitlines()[0].strip() if text.splitlines() else ""
        if first != f"tesseract {_TESSERACT_VERSION}":
            raise RuntimeError("Tesseract runtime version does not match pinned baseline")
        self._version_checked = True

    def __call__(self, ppm_crop: bytes) -> OCRText | None:
        if not isinstance(ppm_crop, bytes) or not ppm_crop.startswith(b"P6\n"):
            raise ValueError("recognizer input must be a binary PPM crop")
        if len(ppm_crop) > 16 * 1024 * 1024:
            raise ValueError("plate crop exceeds bounded OCR input")
        self._check_version()
        command = [
            self._binary, "stdin", "stdout", "--tessdata-dir", str(self._tessdata_dir),
            "-l", "eng", "--psm", "7", "-c", "tessedit_create_tsv=1",
        ]
        result = self._runner(
            command,
            input=ppm_crop,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.config.tesseract_timeout_seconds,
            check=False,
        )
        if getattr(result, "returncode", 1) != 0:
            raise RuntimeError("Tesseract OCR failed")
        stdout = getattr(result, "stdout", b"")
        text = stdout.decode("utf-8", "strict") if isinstance(stdout, bytes) else str(stdout)
        return parse_tesseract_tsv(text, min_confidence=self.config.min_ocr_confidence)


class PlateDetector(Protocol):
    def __call__(self, image: Any) -> tuple[PlateDetection, ...]: ...


class PlateRecognizer(Protocol):
    def __call__(self, ppm_crop: bytes) -> OCRText | None: ...


def run_lpr_ocr(image: Any, detector: PlateDetector, recognizer: PlateRecognizer) -> tuple[LPRObservation, ...]:
    """Run one frame through plate detection then OCR, without retaining imagery."""
    if not callable(detector) or not callable(recognizer):
        raise ValueError("detector and recognizer must be callable")
    plates = detector(image)
    if not isinstance(plates, tuple) or any(not isinstance(item, PlateDetection) for item in plates):
        raise ValueError("detector must return a tuple of PlateDetection values")
    if len(plates) > 64:
        raise ValueError("detector returned too many plates")
    observations: list[LPRObservation] = []
    for plate in plates:
        text = recognizer(encode_plate_crop_ppm(image, plate))
        if text is not None:
            observations.append(LPRObservation(plate, text))
    return tuple(observations)


class OpenVINOOMZPlateDetector:
    """Optional local CPU adapter for the pinned OMZ 0106 FP16 plate detector."""

    def __init__(self, artifact_root: str | Path, *, config: LPROCRConfig | None = None, device: str = "CPU") -> None:
        self.config = config or LPROCRConfig()
        if not isinstance(self.config, LPROCRConfig):
            raise ValueError("config must be LPROCRConfig")
        if device != "CPU":
            raise ValueError("first LPR baseline is CPU-only")
        try:
            import cv2
            import numpy as np
            import openvino as ov
            from .artifacts import ArtifactSpec, verify_artifact_set
        except ImportError as exc:
            raise RuntimeError("OpenVINO, NumPy and OpenCV are required for the OMZ plate detector") from exc
        specs = (
            ArtifactSpec(
                component="open-model-zoo/vehicle-license-plate-detection-barrier-0106/fp16",
                relative_path=_OMZ_XML,
                size_bytes=325452,
                sha384="b0a70cfc4ccbe85dd5ba5549c2d142d7f922a0a1116657cdcc9f48cb6afc7cc40a596a655b9e6566396c48b72232ff65",
                source_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml",
                license_id="Apache-2.0", license_url=_OMZ_LICENSE,
            ),
            ArtifactSpec(
                component="open-model-zoo/vehicle-license-plate-detection-barrier-0106/fp16",
                relative_path=_OMZ_BIN,
                size_bytes=1286772,
                sha384="0406e296c9822f4a5e8ee300ad40e5c0a0eeeaf9206f829556cb836cb8f10a4f8a4dd94d1bb16a60562bf1568fd12618",
                source_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.bin",
                license_id="Apache-2.0", license_url=_OMZ_LICENSE,
            ),
        )
        verified = verify_artifact_set(artifact_root, specs)
        paths = {item.spec.relative_path: item.path for item in verified}
        core = ov.Core()
        model = core.read_model(model=str(paths[_OMZ_XML]), weights=str(paths[_OMZ_BIN]))
        compiled = core.compile_model(model, device)
        inputs = tuple(compiled.inputs)
        if len(inputs) != 1 or tuple(int(value) for value in inputs[0].shape) != (1, 300, 300, 3):
            raise RuntimeError("OMZ plate detector input shape changed")
        self._cv2 = cv2
        self._np = np
        self._compiled = compiled
        self.runtime_version = str(getattr(ov, "__version__", "unknown"))
        if not self.runtime_version.startswith("2026.3.1"):
            raise RuntimeError("OpenVINO runtime version does not match reviewed release")

    def __call__(self, image: Any) -> tuple[PlateDetection, ...]:
        _image_size(image)
        resized = self._cv2.resize(image, (300, 300), interpolation=self._cv2.INTER_LINEAR)
        blob = self._np.ascontiguousarray(resized[None], dtype=self._np.float32)
        result = self._compiled([blob])
        output = self._np.asarray(result[self._compiled.output(0)])
        if output.size % 7:
            raise RuntimeError("OMZ plate detector output is not seven-value rows")
        return parse_omz_plate_rows(
            output.reshape((-1, 7)), threshold=self.config.plate_threshold, max_plates=self.config.max_plates
        )
