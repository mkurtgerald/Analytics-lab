"""Security-analytics adapter for an exact reviewed subset of OIDv4 classes.

This module is for passive video detection/alerting only. It reuses the
already-admitted Google/TensorFlow standard OIDv4 SSD MobileNetV2 detector and
adds no second detector, runtime, model, media source, or training data.
"""
from __future__ import annotations

from typing import Any

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


OID_V4_WEAPON_CLASS_IDS = frozenset({285, 325, 351, 361, 365, 408, 533})
_MAX_MODEL_DETECTIONS = 100
_MAX_DETECTIONS = 64
_DEFAULT_CONFIDENCE_THRESHOLD = 0.50


def parse_oid_v4_weapon_detections(
    boxes: Any,
    scores: Any,
    classes: Any,
    num_detections: Any,
    *,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    max_detections: int = _MAX_DETECTIONS,
) -> tuple[DetectionCandidate, ...]:
    """Return only exact allowlisted security-object classes."""

    cutoff = _finite_number(confidence_threshold, "confidence_threshold")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("confidence_threshold must be within [0, 1]")
    if type(max_detections) is not int or not 1 <= max_detections <= _MAX_DETECTIONS:
        raise ValueError("max_detections is outside the supported bound")

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

    detections: list[DetectionCandidate] = []
    for index in range(count):
        class_value = _finite_number(class_row[index], "class_id")
        class_id = int(class_value)
        if class_value != class_id:
            raise ValueError("class_id must be integral")
        score = _finite_number(score_row[index], "score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be within [0, 1]")
        if class_id not in OID_V4_WEAPON_CLASS_IDS or score < cutoff:
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
                category="weapon",
                confidence=score,
                box=NormalizedBox(left, top, right, bottom),
                model_class_id=class_id,
            )
        )

    detections.sort(
        key=lambda item: (
            -item.confidence,
            int(item.model_class_id),
            item.box.x_min,
            item.box.y_min,
            item.box.x_max,
            item.box.y_max,
        )
    )
    return tuple(detections[:max_detections])


class OpenVINOOIDSSDWeaponDetector:
    """Passive security-analytics adapter for the shared compiled OIDv4 graph."""

    def __init__(
        self,
        compiled_model: Any,
        *,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        max_detections: int = _MAX_DETECTIONS,
    ) -> None:
        if not callable(compiled_model):
            raise ValueError("compiled_model must be callable")
        cutoff = _finite_number(confidence_threshold, "confidence_threshold")
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if type(max_detections) is not int or not 1 <= max_detections <= _MAX_DETECTIONS:
            raise ValueError("max_detections is outside the supported bound")
        self._compiled_model = compiled_model
        self.confidence_threshold = cutoff
        self.max_detections = max_detections
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

        return parse_oid_v4_weapon_detections(
            boxes,
            scores,
            classes,
            count,
            confidence_threshold=self.confidence_threshold,
            max_detections=self.max_detections,
        )
