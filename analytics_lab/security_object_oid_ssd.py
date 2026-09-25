"""Bounded Open Images V4 security-object adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .face_oid_ssd import (
    OID_V4_BOXES_OUTPUT,
    OID_V4_CLASSES_OUTPUT,
    OID_V4_INPUT_NAME,
    OID_V4_MODEL_INPUT_HEIGHT,
    OID_V4_MODEL_INPUT_WIDTH,
    OID_V4_NUM_DETECTIONS_OUTPUT,
    OID_V4_SCORES_OUTPUT,
    _batch_row,
    _finite_number,
    _port_by_name,
)
from .tracking import DetectionCandidate, NormalizedBox

OID_V4_SECURITY_OBJECT_CLASSES = {
    285: "knife",
    325: "kitchen_knife",
    351: "rifle",
    361: "shotgun",
    365: "sword",
    408: "weapon",
    533: "handgun",
}

_MAX_MODEL_DETECTIONS = 100
_MAX_SECURITY_OBJECTS = 64
_DEFAULT_CONFIDENCE_THRESHOLD = 0.50


def parse_oid_v4_security_object_detections(
    boxes: Any,
    scores: Any,
    classes: Any,
    num_detections: Any,
    *,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    max_objects: int = _MAX_SECURITY_OBJECTS,
) -> tuple[DetectionCandidate, ...]:
    cutoff = _finite_number(confidence_threshold, "confidence_threshold")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("confidence_threshold must be within [0, 1]")
    if type(max_objects) is not int or not 1 <= max_objects <= _MAX_SECURITY_OBJECTS:
        raise ValueError("max_objects is outside the supported bound")

    if isinstance(num_detections, (str, bytes)):
        raise ValueError("num_detections must be a batch-size-1 vector")
    try:
        if len(num_detections) != 1:
            raise ValueError("num_detections must have batch size 1")
        count_value = _finite_number(num_detections[0], "num_detections")
    except (TypeError, IndexError) as exc:
        raise ValueError("num_detections must be a batch-size-1 vector") from exc
    count = int(count_value)
    if count_value != count or not 0 <= count <= _MAX_MODEL_DETECTIONS:
        raise ValueError("num_detections is outside the supported bound")

    box_rows = _batch_row(boxes, "boxes")
    score_row = _batch_row(scores, "scores")
    class_row = _batch_row(classes, "classes")
    if any(len(row) < count for row in (box_rows, score_row, class_row)):
        raise ValueError("OIDv4 output tensor is shorter than num_detections")
    if any(len(row) > _MAX_MODEL_DETECTIONS for row in (box_rows, score_row, class_row)):
        raise ValueError("OIDv4 output tensor exceeds supported bound")

    detections = []
    for index in range(count):
        class_value = _finite_number(class_row[index], "class_id")
        class_id = int(class_value)
        if class_value != class_id:
            raise ValueError("class_id must be integral")
        score = _finite_number(score_row[index], "score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be within [0, 1]")
        category = OID_V4_SECURITY_OBJECT_CLASSES.get(class_id)
        if category is None or score < cutoff:
            continue

        row = box_rows[index]
        if isinstance(row, (str, bytes)):
            raise ValueError("box row must be numeric")
        try:
            if len(row) != 4:
                raise ValueError("box row must contain four coordinates")
            y_min, x_min, y_max, x_max = (
                _finite_number(row[pos], "box coordinate") for pos in range(4)
            )
        except (TypeError, IndexError) as exc:
            raise ValueError("box row must contain four coordinates") from exc

        left = max(0.0, min(1.0, x_min))
        top = max(0.0, min(1.0, y_min))
        right = max(0.0, min(1.0, x_max))
        bottom = max(0.0, min(1.0, y_max))
        if right <= left or bottom <= top:
            continue
        detections.append(
            DetectionCandidate(
                category,
                score,
                NormalizedBox(left, top, right, bottom),
                class_id,
            )
        )

    detections.sort(
        key=lambda item: (
            -item.confidence,
            item.category,
            item.box.x_min,
            item.box.y_min,
            item.box.x_max,
            item.box.y_max,
        )
    )
    return tuple(detections[:max_objects])


class OpenVINOOIDSSDSecurityObjectDetector:
    def __init__(
        self,
        compiled_model: Any,
        *,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        max_objects: int = _MAX_SECURITY_OBJECTS,
    ) -> None:
        if not callable(compiled_model):
            raise ValueError("compiled_model must be callable")
        cutoff = _finite_number(confidence_threshold, "confidence_threshold")
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if type(max_objects) is not int or not 1 <= max_objects <= _MAX_SECURITY_OBJECTS:
            raise ValueError("max_objects is outside the supported bound")
        self._compiled_model = compiled_model
        self.confidence_threshold = cutoff
        self.max_objects = max_objects
        self._input = _port_by_name(compiled_model.inputs, OID_V4_INPUT_NAME)
        self._boxes = _port_by_name(compiled_model.outputs, OID_V4_BOXES_OUTPUT)
        self._classes = _port_by_name(compiled_model.outputs, OID_V4_CLASSES_OUTPUT)
        self._scores = _port_by_name(compiled_model.outputs, OID_V4_SCORES_OUTPUT)
        self._count = _port_by_name(compiled_model.outputs, OID_V4_NUM_DETECTIONS_OUTPUT)

    def detect(self, frame_bgr: Any) -> tuple[DetectionCandidate, ...]:
        import cv2
        import numpy as np

        frame = np.asarray(frame_bgr)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be HxWx3 uint8 BGR")
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if not 1 <= width <= 16384 or not 1 <= height <= 16384:
            raise ValueError("frame dimensions outside supported bounds")
        if height == OID_V4_MODEL_INPUT_HEIGHT and width == OID_V4_MODEL_INPUT_WIDTH:
            model_frame = frame
        else:
            model_frame = cv2.resize(
                frame,
                (OID_V4_MODEL_INPUT_WIDTH, OID_V4_MODEL_INPUT_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
        rgb_batch = np.ascontiguousarray(model_frame[:, :, ::-1][None, ...])
        outputs = self._compiled_model({self._input: rgb_batch})
        try:
            boxes = outputs[self._boxes]
            classes = outputs[self._classes]
            scores = outputs[self._scores]
            count = outputs[self._count]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("OIDv4 compiled model returned unsupported outputs") from exc
        return parse_oid_v4_security_object_detections(
            boxes,
            scores,
            classes,
            count,
            confidence_threshold=self.confidence_threshold,
            max_objects=self.max_objects,
        )


DetectorFactory = Callable[..., Any]


class OpenVINOOIDSSDSecurityObjectRuntime:
    def __init__(self, detector: Any, *, compile_count: int) -> None:
        if not callable(getattr(detector, "detect", None)):
            raise ValueError("detector must expose detect")
        if type(compile_count) is not int or compile_count != 1:
            raise ValueError("runtime must represent exactly one model compilation")
        self._detector = detector
        self._compile_count = compile_count
        self._inference_count = 0

    @classmethod
    def from_tensorflow_graph(
        cls,
        model_path: str | Path,
        *,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        max_objects: int = _MAX_SECURITY_OBJECTS,
        core: Any | None = None,
        convert_model: Callable[[str], Any] | None = None,
        detector_factory: DetectorFactory | None = None,
    ) -> "OpenVINOOIDSSDSecurityObjectRuntime":
        candidate = Path(model_path)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("OIDv4 graph must be a regular non-symlink file")
        if core is None or convert_model is None:
            import openvino as ov
            if core is None:
                core = ov.Core()
            if convert_model is None:
                convert_model = ov.convert_model
        compiler = getattr(core, "compile_model", None)
        if not callable(compiler):
            raise ValueError("core must expose compile_model")
        if not callable(convert_model):
            raise ValueError("convert_model must be callable")
        converted = convert_model(str(candidate))
        compiled = compiler(converted, "CPU")
        factory = detector_factory or OpenVINOOIDSSDSecurityObjectDetector
        detector = factory(
            compiled,
            confidence_threshold=confidence_threshold,
            max_objects=max_objects,
        )
        return cls(detector, compile_count=1)

    @property
    def compile_count(self) -> int:
        return self._compile_count

    @property
    def inference_count(self) -> int:
        return self._inference_count

    def detect(self, frame_bgr: Any) -> tuple[DetectionCandidate, ...]:
        detections = self._detector.detect(frame_bgr)
        try:
            normalized = tuple(detections)
        except TypeError as exc:
            raise ValueError("detector must return an iterable") from exc
        if any(not isinstance(item, DetectionCandidate) for item in normalized):
            raise ValueError("detector returned unsupported security-object value")
        self._inference_count += 1
        return normalized
