"""Optional OpenVINO/Open Model Zoo detector + pose backend.

No network access or model download occurs here. The backend verifies the exact
locally provisioned Open Model Zoo artifacts before constructing a runtime. It
uses the reviewed person detector to isolate one person per crop and extracts
only shoulder/hip peaks from the reviewed OpenPose-style heatmaps. The latter
is deliberately conservative and does not claim the full upstream PAF decoder's
accuracy; it is an integration baseline for measured video evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Protocol, Sequence

from .artifacts import OPENVINO_OMZ_2023_FP16, VerifiedArtifact, verify_artifact_set
from .perception import BBox, Keypoint, PoseCandidate

_DETECTOR_XML = "person-detection-retail-0013/FP16/person-detection-retail-0013.xml"
_DETECTOR_BIN = "person-detection-retail-0013/FP16/person-detection-retail-0013.bin"
_POSE_XML = "human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml"
_POSE_BIN = "human-pose-estimation-0001/FP16/human-pose-estimation-0001.bin"
# Channel IDs and quarter-pixel peak refinement are adapted from Open Model
# Zoo's Apache-2.0 OpenPose decoder at commit
# 6697dead54ed1cdd664b0313189c2cb52ee6335e. We intentionally do not copy the
# full PAF grouping decoder; detector-isolated crops are an evaluation baseline.
_REQUIRED_CHANNELS = (
    ("right_shoulder", 2),
    ("left_shoulder", 5),
    ("right_hip", 8),
    ("left_hip", 11),
)
_DEVICE = re.compile(r"[A-Za-z0-9_.:-]{1,64}\Z")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _score(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True)
class OpenVINOOMZConfig:
    detection_threshold: float = 0.50
    max_people: int = 32
    device: str = "CPU"
    runtime_version_prefix: str = "2026.3.1"

    def __post_init__(self) -> None:
        _score(self.detection_threshold, "detection_threshold")
        if self.detection_threshold <= 0:
            raise ValueError("detection_threshold must be > 0")
        if type(self.max_people) is not int or not 1 <= self.max_people <= 256:
            raise ValueError("max_people must be an integer in [1, 256]")
        if not isinstance(self.device, str) or not _DEVICE.fullmatch(self.device):
            raise ValueError("device must be a simple OpenVINO device name")
        if (not isinstance(self.runtime_version_prefix, str) or not self.runtime_version_prefix
                or len(self.runtime_version_prefix) > 32 or not re.fullmatch(r"[0-9.]+", self.runtime_version_prefix)):
            raise ValueError("runtime_version_prefix must be a bounded numeric release prefix")


@dataclass(frozen=True)
class PersonDetection:
    bbox: BBox
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.bbox, BBox):
            raise ValueError("detection bbox must be a BBox")
        object.__setattr__(self, "confidence", _score(self.confidence, "detection confidence"))


def parse_person_detections(
    rows: Iterable[Sequence[float]],
    *,
    image_width: int,
    image_height: int,
    threshold: float,
    max_people: int,
) -> tuple[PersonDetection, ...]:
    """Parse OMZ SSD rows `[image_id,label,confidence,x1,y1,x2,y2]`."""
    if type(image_width) is not int or type(image_height) is not int or image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive integers")
    cutoff = _score(threshold, "threshold")
    if type(max_people) is not int or not 1 <= max_people <= 256:
        raise ValueError("max_people must be an integer in [1, 256]")
    result: list[PersonDetection] = []
    count = 0
    for raw in rows:
        count += 1
        if count > 4096:
            raise RuntimeError("detector output row limit exceeded")
        if isinstance(raw, (str, bytes)):
            raise ValueError("detector row must contain seven numeric values")
        try:
            if len(raw) != 7:
                raise ValueError("detector row must contain seven numeric values")
            values = [raw[index] for index in range(7)]
        except (TypeError, IndexError, KeyError) as exc:
            raise ValueError("detector row must contain seven numeric values") from exc
        image_id, label, confidence, x1, y1, x2, y2 = (_finite(v, "detector value") for v in values)
        confidence = _score(confidence, "detector confidence")
        if image_id < 0:
            break
        if label != 1 or confidence < cutoff:
            continue
        left = min(1.0, max(0.0, x1))
        top = min(1.0, max(0.0, y1))
        right = min(1.0, max(0.0, x2))
        bottom = min(1.0, max(0.0, y2))
        if right <= left or bottom <= top:
            continue
        result.append(PersonDetection(
            BBox(left * image_width, top * image_height, right * image_width, bottom * image_height),
            confidence,
        ))
    result.sort(key=lambda item: (-item.confidence, item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2))
    return tuple(result[:max_people])


def _matrix_shape(matrix: Any) -> tuple[int, int]:
    try:
        height = len(matrix)
        width = len(matrix[0]) if height else 0
    except (TypeError, IndexError) as exc:
        raise ValueError("heatmap must be a nonempty rectangular matrix") from exc
    if height < 1 or width < 1:
        raise ValueError("heatmap must be a nonempty rectangular matrix")
    for row in matrix:
        if len(row) != width:
            raise ValueError("heatmap must be rectangular")
    return height, width


def _heatmap_peak(matrix: Any) -> tuple[float, float, float]:
    """Peak plus donor-inspired quarter-pixel refinement, without a PAF decoder."""
    height, width = _matrix_shape(matrix)
    best_score = -math.inf
    best_x = best_y = 0
    for y in range(height):
        for x in range(width):
            value = _finite(matrix[y][x], "heatmap value")
            if value > best_score:
                best_score, best_x, best_y = value, x, y
    x = float(best_x)
    y = float(best_y)
    if 0 < best_x < width - 1:
        delta = _finite(matrix[best_y][best_x + 1], "heatmap value") - _finite(matrix[best_y][best_x - 1], "heatmap value")
        x += 0.25 if delta > 0 else (-0.25 if delta < 0 else 0.0)
    if 0 < best_y < height - 1:
        delta = _finite(matrix[best_y + 1][best_x], "heatmap value") - _finite(matrix[best_y - 1][best_x], "heatmap value")
        y += 0.25 if delta > 0 else (-0.25 if delta < 0 else 0.0)
    return x, y, min(1.0, max(0.0, best_score))


def pose_candidate_from_heatmaps(heatmaps: Any, detection: PersonDetection) -> PoseCandidate:
    """Extract the four posture keypoints from one detector-isolated pose crop."""
    if not isinstance(detection, PersonDetection):
        raise ValueError("detection must be a PersonDetection")
    try:
        channels = len(heatmaps)
    except TypeError as exc:
        raise ValueError("pose heatmaps must be channel-first") from exc
    if channels < 19:
        raise ValueError("pose heatmaps must contain at least 19 channels")
    keypoints: list[Keypoint] = []
    for name, channel in _REQUIRED_CHANNELS:
        matrix = heatmaps[channel]
        height, width = _matrix_shape(matrix)
        x, y, confidence = _heatmap_peak(matrix)
        px = detection.bbox.x1 + ((x + 0.5) / width) * detection.bbox.width
        py = detection.bbox.y1 + ((y + 0.5) / height) * detection.bbox.height
        keypoints.append(Keypoint(name, px, py, confidence))
    return PoseCandidate(detection.bbox, tuple(keypoints), detection.confidence)


def _image_size(image: Any) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("image must expose height/width through shape")
    height, width = int(shape[0]), int(shape[1])
    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")
    if len(shape) >= 3 and int(shape[2]) != 3:
        raise ValueError("Open Model Zoo baseline requires a three-channel BGR image")
    return width, height


class OMZRuntime(Protocol):
    runtime_version: str

    def detect_rows(self, image: Any) -> Iterable[Sequence[float]]: ...

    def pose_heatmaps(self, image: Any, bbox: BBox) -> Any: ...


class _OpenVINORuntime:
    """Thin optional OpenVINO runtime; instantiated only after artifact verification."""

    def __init__(self, artifacts: tuple[VerifiedArtifact, ...], device: str) -> None:
        try:
            import cv2
            import numpy as np
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("OpenVINO, NumPy and OpenCV are required for OMZ inference") from exc
        self._cv2 = cv2
        self._np = np
        self.runtime_version = str(getattr(ov, "__version__", "unknown"))
        paths = {item.spec.relative_path: item.path for item in artifacts}
        for required in (_DETECTOR_XML, _DETECTOR_BIN, _POSE_XML, _POSE_BIN):
            if required not in paths:
                raise ValueError("verified artifact set is incomplete")
        core = ov.Core()
        detector_model = core.read_model(model=str(paths[_DETECTOR_XML]), weights=str(paths[_DETECTOR_BIN]))
        pose_model = core.read_model(model=str(paths[_POSE_XML]), weights=str(paths[_POSE_BIN]))
        self._detector = core.compile_model(detector_model, device)
        self._pose = core.compile_model(pose_model, device)
        self._validate_model(self._detector, (1, 3, 320, 544), "detector")
        self._validate_model(self._pose, (1, 3, 256, 456), "pose")
        pose_outputs = tuple(self._pose.outputs)
        heatmap_outputs = [port for port in pose_outputs if tuple(int(v) for v in port.shape)[1:] == (19, 32, 57)]
        if len(heatmap_outputs) != 1:
            raise RuntimeError("pose model must expose exactly one 19x32x57 heatmap output")
        self._pose_heatmap_port = heatmap_outputs[0]

    @staticmethod
    def _validate_model(compiled: Any, expected: tuple[int, ...], name: str) -> None:
        try:
            inputs = tuple(compiled.inputs)
            shape = tuple(int(v) for v in inputs[0].shape)
        except Exception as exc:
            raise RuntimeError(f"{name} model input metadata is unavailable") from exc
        if len(inputs) != 1 or shape != expected:
            raise RuntimeError(f"{name} model input shape does not match the reviewed artifact")

    def _blob(self, image: Any, width: int, height: int) -> Any:
        resized = self._cv2.resize(image, (width, height), interpolation=self._cv2.INTER_LINEAR)
        return self._np.ascontiguousarray(resized.transpose((2, 0, 1))[None], dtype=self._np.float32)

    def detect_rows(self, image: Any) -> Iterable[Sequence[float]]:
        blob = self._blob(image, 544, 320)
        results = self._detector([blob])
        output = self._np.asarray(results[self._detector.output(0)])
        if output.size % 7:
            raise RuntimeError("detector output does not contain seven-value rows")
        return output.reshape((-1, 7))

    def pose_heatmaps(self, image: Any, bbox: BBox) -> Any:
        width, height = _image_size(image)
        left = max(0, min(width - 1, int(math.floor(bbox.x1))))
        top = max(0, min(height - 1, int(math.floor(bbox.y1))))
        right = max(left + 1, min(width, int(math.ceil(bbox.x2))))
        bottom = max(top + 1, min(height, int(math.ceil(bbox.y2))))
        crop = image[top:bottom, left:right]
        if getattr(crop, "size", 0) == 0:
            raise RuntimeError("detector crop is empty")
        blob = self._blob(crop, 456, 256)
        results = self._pose([blob])
        heatmaps = self._np.asarray(results[self._pose_heatmap_port])
        if tuple(heatmaps.shape) != (1, 19, 32, 57):
            raise RuntimeError("pose heatmap output shape changed")
        return heatmaps[0]


RuntimeFactory = Callable[[tuple[VerifiedArtifact, ...], str], OMZRuntime]


class OpenVINOOMZPoseBackend:
    """Verified local OMZ artifacts -> person detections -> posture keypoint candidates."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        config: OpenVINOOMZConfig | None = None,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self.config = config or OpenVINOOMZConfig()
        if not isinstance(self.config, OpenVINOOMZConfig):
            raise ValueError("config must be an OpenVINOOMZConfig")
        verified = verify_artifact_set(artifact_root, OPENVINO_OMZ_2023_FP16)
        factory = runtime_factory or _OpenVINORuntime
        if not callable(factory):
            raise ValueError("runtime_factory must be callable")
        self._runtime = factory(verified, self.config.device)
        version = getattr(self._runtime, "runtime_version", None)
        if not isinstance(version, str) or not version.strip() or len(version) > 128:
            raise RuntimeError("runtime must expose a bounded version string")
        if not version.startswith(self.config.runtime_version_prefix):
            raise RuntimeError("OpenVINO runtime version does not match the reviewed release prefix")
        self.runtime_version = version

    def __call__(self, image: Any, frame_index: int, timestamp_ms: int) -> tuple[PoseCandidate, ...]:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if type(timestamp_ms) is not int or timestamp_ms < 0:
            raise ValueError("timestamp_ms must be a nonnegative integer")
        width, height = _image_size(image)
        detections = parse_person_detections(
            self._runtime.detect_rows(image), image_width=width, image_height=height,
            threshold=self.config.detection_threshold, max_people=self.config.max_people,
        )
        return tuple(
            pose_candidate_from_heatmaps(self._runtime.pose_heatmaps(image, detection.bbox), detection)
            for detection in detections
        )
