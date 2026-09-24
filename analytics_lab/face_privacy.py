"""Face-detection integration and privacy-blur enforcement.

This module does not ship a face model or perform face recognition. A reviewed
detector can plug in behind the normalized contract. Privacy defaults to blur;
an unblur request is honored only when the caller supplies an authorization
decision from the owning product. No biometric identity is produced here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

from .tracking import NormalizedBox

_MAX_FACES = 64
_MAX_MODEL_BYTES = 16 * 1024 * 1024
_MAX_DIRECT_GAUSSIAN_SIGMA = 32.0


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return result


@dataclass(frozen=True)
class FaceDetection:
    confidence: float
    box: NormalizedBox

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _score(self.confidence, "confidence"))
        if not isinstance(self.box, NormalizedBox):
            raise ValueError("box must be NormalizedBox")


class FaceDetectorBackend(Protocol):
    def detect(self, frame_bgr: Any) -> Iterable[FaceDetection]: ...


@dataclass(frozen=True)
class FaceModelIdentity:
    size_bytes: int
    sha256: str


def verify_face_model_artifact(
    path: str | Path, *, expected_sha256: str, max_bytes: int = _MAX_MODEL_BYTES
) -> FaceModelIdentity:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("face model must be a regular non-symlink file")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected_sha256 must be 64 lowercase hex characters")
    try:
        int(expected_sha256, 16)
    except ValueError as exc:
        raise ValueError("expected_sha256 must be lowercase hex") from exc
    if expected_sha256 != expected_sha256.lower():
        raise ValueError("expected_sha256 must be lowercase hex")
    if type(max_bytes) is not int or not 1 <= max_bytes <= _MAX_MODEL_BYTES:
        raise ValueError("max_bytes is outside the supported bound")
    size = candidate.stat().st_size
    if not 1 <= size <= max_bytes:
        raise ValueError("face model size is outside the supported bound")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError("face model SHA-256 mismatch")
    return FaceModelIdentity(size, actual)


def _frame_size(frame: Any) -> tuple[int, int]:
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) != 3 or int(shape[2]) != 3:
        raise ValueError("frame must be HxWx3 BGR")
    height, width = int(shape[0]), int(shape[1])
    if not 1 <= width <= 16384 or not 1 <= height <= 16384:
        raise ValueError("frame dimensions outside supported bounds")
    return width, height


def parse_yunet_rows(
    rows: Iterable[Sequence[float]] | None,
    *,
    width: int,
    height: int,
    confidence_threshold: float = 0.90,
    max_faces: int = _MAX_FACES,
) -> tuple[FaceDetection, ...]:
    cutoff = _score(confidence_threshold, "confidence_threshold")
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError("width and height must be positive integers")
    if type(max_faces) is not int or not 1 <= max_faces <= _MAX_FACES:
        raise ValueError("max_faces is outside the supported bound")
    if rows is None:
        return ()
    result: list[FaceDetection] = []
    seen = 0
    for row in rows:
        seen += 1
        if seen > 4096:
            raise RuntimeError("YuNet output row limit exceeded")
        if isinstance(row, (str, bytes)):
            raise ValueError("YuNet row must be numeric")
        try:
            if len(row) < 15:
                raise ValueError("YuNet row must contain bbox, landmarks and score")
            x, y, w, h = (float(row[index]) for index in range(4))
            confidence = float(row[14])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("YuNet row is malformed") from exc
        if not all(math.isfinite(v) for v in (x, y, w, h, confidence)):
            raise ValueError("YuNet row must be finite")
        if confidence < cutoff or w <= 0.0 or h <= 0.0:
            continue
        left = max(0.0, min(1.0, x / width))
        top = max(0.0, min(1.0, y / height))
        right = max(0.0, min(1.0, (x + w) / width))
        bottom = max(0.0, min(1.0, (y + h) / height))
        if right <= left or bottom <= top:
            continue
        result.append(FaceDetection(confidence, NormalizedBox(left, top, right, bottom)))
    result.sort(key=lambda item: (-item.confidence, item.box.x_min, item.box.y_min))
    return tuple(result[:max_faces])


DetectorFactory = Callable[[str, str, tuple[int, int], float, float, int], Any]


def _opencv_yunet_factory(
    model: str, config: str, input_size: tuple[int, int],
    score_threshold: float, nms_threshold: float, top_k: int,
) -> Any:
    import cv2
    cls = getattr(cv2, "FaceDetectorYN", None)
    if cls is not None and callable(getattr(cls, "create", None)):
        return cls.create(model, config, input_size, score_threshold, nms_threshold, top_k)
    legacy = getattr(cv2, "FaceDetectorYN_create", None)
    if callable(legacy):
        return legacy(model, config, input_size, score_threshold, nms_threshold, top_k)
    raise RuntimeError("OpenCV FaceDetectorYN is unavailable")


class OpenCVYuNetDetector:
    """FaceDetectorYN-compatible adapter with mandatory artifact verification."""

    def __init__(
        self, model_path: str | Path, *, expected_sha256: str,
        confidence_threshold: float = 0.90, nms_threshold: float = 0.30,
        top_k: int = 5000, max_faces: int = _MAX_FACES,
        factory: DetectorFactory | None = None,
    ) -> None:
        self.identity = verify_face_model_artifact(model_path, expected_sha256=expected_sha256)
        self.model_path = str(Path(model_path))
        self.confidence_threshold = _score(confidence_threshold, "confidence_threshold")
        self.nms_threshold = _score(nms_threshold, "nms_threshold")
        if type(top_k) is not int or not 1 <= top_k <= 10000:
            raise ValueError("top_k is outside the supported bound")
        if type(max_faces) is not int or not 1 <= max_faces <= _MAX_FACES:
            raise ValueError("max_faces is outside the supported bound")
        self.top_k = top_k
        self.max_faces = max_faces
        self._factory = factory or _opencv_yunet_factory
        self._detector: Any | None = None
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame_bgr: Any) -> tuple[FaceDetection, ...]:
        width, height = _frame_size(frame_bgr)
        size = (width, height)
        if self._detector is None:
            self._detector = self._factory(
                self.model_path, "", size, self.confidence_threshold,
                self.nms_threshold, self.top_k,
            )
            self._input_size = size
        elif self._input_size != size:
            setter = getattr(self._detector, "setInputSize", None)
            if not callable(setter):
                raise RuntimeError("face detector cannot change input size")
            setter(size)
            self._input_size = size
        output = self._detector.detect(frame_bgr)
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("face detector returned unsupported output")
        _, rows = output
        return parse_yunet_rows(
            rows, width=width, height=height,
            confidence_threshold=self.confidence_threshold, max_faces=self.max_faces,
        )


@dataclass(frozen=True)
class FacePrivacyConfig:
    default_blur: bool = True
    allow_authorized_unblur: bool = True
    expand_ratio: float = 0.12
    sigma_ratio: float = 0.16
    min_sigma: float = 5.0
    max_faces: int = _MAX_FACES

    def __post_init__(self) -> None:
        if type(self.default_blur) is not bool or type(self.allow_authorized_unblur) is not bool:
            raise ValueError("privacy toggles must be booleans")
        for name, low, high in (
            ("expand_ratio", 0.0, 0.5),
            ("sigma_ratio", 0.02, 0.5),
            ("min_sigma", 1.0, 64.0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or type(value) not in (int, float):
                raise ValueError(f"{name} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or not low <= numeric <= high:
                raise ValueError(f"{name} is outside the supported range")
        if type(self.max_faces) is not int or not 1 <= self.max_faces <= _MAX_FACES:
            raise ValueError("max_faces is outside the supported bound")


@dataclass(frozen=True)
class FacePrivacyAudit:
    action: str
    requested_unblur: bool
    authorized_unblur: bool
    face_count: int

    def __post_init__(self) -> None:
        if self.action not in {"blur", "unblur", "unblur_denied", "policy_passthrough"}:
            raise ValueError("unsupported privacy action")
        if type(self.requested_unblur) is not bool or type(self.authorized_unblur) is not bool:
            raise ValueError("audit flags must be booleans")
        if type(self.face_count) is not int or not 0 <= self.face_count <= _MAX_FACES:
            raise ValueError("face_count is outside the supported bound")


@dataclass(frozen=True)
class FacePrivacyResult:
    frame_bgr: Any
    detections: tuple[FaceDetection, ...]
    audit: FacePrivacyAudit


def _expanded_pixels(box: NormalizedBox, *, width: int, height: int, ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.x_min * width, box.y_min * height, box.x_max * width, box.y_max * height
    pad_x, pad_y = (x2 - x1) * ratio, (y2 - y1) * ratio
    left = max(0, min(width - 1, int(math.floor(x1 - pad_x))))
    top = max(0, min(height - 1, int(math.floor(y1 - pad_y))))
    right = max(left + 1, min(width, int(math.ceil(x2 + pad_x))))
    bottom = max(top + 1, min(height, int(math.ceil(y2 + pad_y))))
    return left, top, right, bottom


def _bounded_blur_plan(
    *, width: int, height: int, sigma: float
) -> tuple[int, int, float, float]:
    """Preserve requested Gaussian strength while bounding direct kernel work.

    Large full-resolution face regions can imply Gaussian sigmas in the hundreds
    of pixels. OpenCV expands an unconstrained direct kernel accordingly, which
    can consume the entire CI budget. Downsampling before the Gaussian and
    restoring afterward keeps the effective sigma in source-pixel units while
    discarding detail before the blur; it does not create a weaker unblur path.
    """
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError("blur dimensions must be positive integers")
    numeric = float(sigma)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("blur sigma must be positive and finite")
    if numeric <= _MAX_DIRECT_GAUSSIAN_SIGMA:
        return width, height, numeric, numeric

    scale = _MAX_DIRECT_GAUSSIAN_SIGMA / numeric
    target_width = max(1, int(math.floor(width * scale)))
    target_height = max(1, int(math.floor(height * scale)))
    sigma_x = numeric * (target_width / width)
    sigma_y = numeric * (target_height / height)
    return target_width, target_height, sigma_x, sigma_y


def _gaussian_blur_bounded(region: Any, sigma: float) -> Any:
    import cv2

    shape = getattr(region, "shape", None)
    if shape is None or len(shape) != 3 or int(shape[2]) != 3:
        raise ValueError("blur region must be HxWx3")
    height, width = int(shape[0]), int(shape[1])
    target_width, target_height, sigma_x, sigma_y = _bounded_blur_plan(
        width=width, height=height, sigma=sigma
    )
    if target_width == width and target_height == height:
        return cv2.GaussianBlur(
            region, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y
        )

    reduced = cv2.resize(
        region,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    blurred = cv2.GaussianBlur(
        reduced,
        (0, 0),
        sigmaX=sigma_x,
        sigmaY=sigma_y,
    )
    return cv2.resize(
        blurred,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def _blur_regions(frame_bgr: Any, detections: tuple[FaceDetection, ...], config: FacePrivacyConfig) -> Any:
    width, height = _frame_size(frame_bgr)
    result = frame_bgr.copy()
    for item in detections:
        left, top, right, bottom = _expanded_pixels(
            item.box, width=width, height=height, ratio=float(config.expand_ratio)
        )
        region = result[top:bottom, left:right]
        if getattr(region, "size", 0) == 0:
            continue
        minimum = max(1, min(right - left, bottom - top))
        sigma = max(float(config.min_sigma), minimum * float(config.sigma_ratio))
        result[top:bottom, left:right] = _gaussian_blur_bounded(region, sigma)
    return result


def apply_face_privacy(
    frame_bgr: Any, detector: FaceDetectorBackend, *,
    config: FacePrivacyConfig | None = None,
    request_unblur: bool = False, authorized_unblur: bool = False,
) -> FacePrivacyResult:
    if type(request_unblur) is not bool or type(authorized_unblur) is not bool:
        raise ValueError("unblur flags must be booleans")
    if not callable(getattr(detector, "detect", None)):
        raise ValueError("detector must expose detect")
    settings = config or FacePrivacyConfig()
    if not isinstance(settings, FacePrivacyConfig):
        raise ValueError("config must be FacePrivacyConfig")
    _frame_size(frame_bgr)
    raw = detector.detect(frame_bgr)
    if raw is None:
        raise ValueError("detector must return an iterable")
    try:
        detections = tuple(raw)
    except TypeError as exc:
        raise ValueError("detector must return an iterable") from exc
    if len(detections) > settings.max_faces:
        raise RuntimeError("face count exceeds configured bound")
    if any(not isinstance(item, FaceDetection) for item in detections):
        raise ValueError("detector returned unsupported face value")

    if request_unblur:
        if authorized_unblur and settings.allow_authorized_unblur:
            return FacePrivacyResult(
                frame_bgr.copy(), detections,
                FacePrivacyAudit("unblur", True, True, len(detections)),
            )
        return FacePrivacyResult(
            _blur_regions(frame_bgr, detections, settings), detections,
            FacePrivacyAudit("unblur_denied", True, authorized_unblur, len(detections)),
        )
    if settings.default_blur:
        return FacePrivacyResult(
            _blur_regions(frame_bgr, detections, settings), detections,
            FacePrivacyAudit("blur", False, authorized_unblur, len(detections)),
        )
    return FacePrivacyResult(
        frame_bgr.copy(), detections,
        FacePrivacyAudit("policy_passthrough", False, authorized_unblur, len(detections)),
    )
