"""Standard Open Images V4 SSD face adapter.

This module binds the admitted Google/TensorFlow standard OIDv4 SSD output
contract to Analytics Lab's normalized ``FaceDetection`` boundary. It admits
only the ``Human face`` class (OIDv4 id 502); it performs no recognition,
embedding, ReID, or identity matching.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .face_privacy import FaceDetection
from .tracking import NormalizedBox

OID_V4_HUMAN_FACE_CLASS_ID = 502
OID_V4_INPUT_NAME = "image_tensor:0"
OID_V4_MODEL_INPUT_HEIGHT = 300
OID_V4_MODEL_INPUT_WIDTH = 300
OID_V4_BOXES_OUTPUT = (
    "Postprocessor/BatchMultiClassNonMaxSuppression/map/"
    "TensorArrayStack/TensorArrayGatherV3:0"
)
OID_V4_CLASSES_OUTPUT = "add:0"
OID_V4_SCORES_OUTPUT = (
    "Postprocessor/BatchMultiClassNonMaxSuppression/map/"
    "TensorArrayStack_1/TensorArrayGatherV3:0"
)
OID_V4_NUM_DETECTIONS_OUTPUT = "Postprocessor/ToFloat_3:0"

_MAX_MODEL_DETECTIONS = 100
_MAX_FACES = 64
_DEFAULT_CONFIDENCE_THRESHOLD = 0.50


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _batch_row(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a single-batch tensor")
    try:
        if len(value) != 1:
            raise ValueError(f"{name} must have batch size 1")
        row = value[0]
        len(row)
    except (TypeError, IndexError) as exc:
        raise ValueError(f"{name} must be a single-batch tensor") from exc
    if isinstance(row, (str, bytes)):
        raise ValueError(f"{name} batch row must be numeric")
    return row


def parse_oid_v4_detections(
    boxes: Any,
    scores: Any,
    classes: Any,
    num_detections: Any,
    *,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    max_faces: int = _MAX_FACES,
) -> tuple[FaceDetection, ...]:
    """Filter exact OIDv4 Human-face detections into normalized face boxes.

    The TensorFlow Object Detection API box convention is
    ``[y_min, x_min, y_max, x_max]``. Only class id 502 is admitted.
    """
    cutoff = _finite_number(confidence_threshold, "confidence_threshold")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("confidence_threshold must be within [0, 1]")
    if type(max_faces) is not int or not 1 <= max_faces <= _MAX_FACES:
        raise ValueError("max_faces is outside the supported bound")

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

    detections: list[FaceDetection] = []
    for index in range(count):
        class_value = _finite_number(class_row[index], "class_id")
        class_id = int(class_value)
        if class_value != class_id:
            raise ValueError("class_id must be integral")
        score = _finite_number(score_row[index], "score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be within [0, 1]")
        if class_id != OID_V4_HUMAN_FACE_CLASS_ID or score < cutoff:
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
            FaceDetection(score, NormalizedBox(left, top, right, bottom))
        )

    detections.sort(
        key=lambda item: (
            -item.confidence,
            item.box.x_min,
            item.box.y_min,
            item.box.x_max,
            item.box.y_max,
        )
    )
    return tuple(detections[:max_faces])


def _port_by_name(ports: Any, exact_name: str) -> Any:
    try:
        candidates = tuple(ports)
    except TypeError as exc:
        raise ValueError("compiled model ports are unavailable") from exc
    for port in candidates:
        try:
            names = set(port.get_names())
        except Exception:
            names = set()
        if exact_name in names:
            return port
        try:
            if port.get_any_name() == exact_name:
                return port
        except Exception:
            pass
    raise RuntimeError("required OIDv4 model port is missing")


class OpenVINOOIDSSDDetector:
    """OpenVINO adapter for the admitted standard OIDv4 SSD graph."""

    def __init__(
        self,
        compiled_model: Any,
        *,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        max_faces: int = _MAX_FACES,
    ) -> None:
        if not callable(compiled_model):
            raise ValueError("compiled_model must be callable")
        cutoff = _finite_number(confidence_threshold, "confidence_threshold")
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if type(max_faces) is not int or not 1 <= max_faces <= _MAX_FACES:
            raise ValueError("max_faces is outside the supported bound")

        self._compiled_model = compiled_model
        self.confidence_threshold = cutoff
        self.max_faces = max_faces
        self._input = _port_by_name(compiled_model.inputs, OID_V4_INPUT_NAME)
        self._boxes = _port_by_name(compiled_model.outputs, OID_V4_BOXES_OUTPUT)
        self._classes = _port_by_name(compiled_model.outputs, OID_V4_CLASSES_OUTPUT)
        self._scores = _port_by_name(compiled_model.outputs, OID_V4_SCORES_OUTPUT)
        self._count = _port_by_name(
            compiled_model.outputs, OID_V4_NUM_DETECTIONS_OUTPUT
        )

    def detect(self, frame_bgr: Any) -> tuple[FaceDetection, ...]:
        """Run the pinned detector preprocessing while preserving source geometry.

        The admitted TensorFlow model's exact pipeline config fixes inference to
        300x300. The model therefore receives a 300x300 RGB tensor while its
        normalized output boxes remain applicable to the untouched source frame
        used by the privacy-blur path.
        """
        import cv2
        import numpy as np

        frame = np.asarray(frame_bgr)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be HxWx3 uint8 BGR")
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if not 1 <= width <= 16384 or not 1 <= height <= 16384:
            raise ValueError("frame dimensions outside supported bounds")

        if (
            height == OID_V4_MODEL_INPUT_HEIGHT
            and width == OID_V4_MODEL_INPUT_WIDTH
        ):
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

        return parse_oid_v4_detections(
            boxes,
            scores,
            classes,
            count,
            confidence_threshold=self.confidence_threshold,
            max_faces=self.max_faces,
        )
