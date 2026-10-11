"""Warm CPU adapter for the pinned OIDv4 SSD ResNet101-FPN graph.

The graph owns its 512x512 resize and RGB preprocessing. The adapter supplies
original-size RGB uint8 pixels and preserves all admitted proposals, up to the
model's 100-proposal bound. It adds no acquisition, identity or alert policy.
"""
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
import time
from typing import Any, Callable

from .face_oid_ssd import _batch_row
from .tracking import DetectionCandidate
from .weapons_oid_ssd import parse_oid_v4_weapon_detections

CONFIDENCE_THRESHOLD = 0.50
MAX_PROPOSALS = 100
INPUT_NAME = "image_tensor:0"
OUTPUT_NAMES = (
    "detection_boxes:0", "detection_scores:0",
    "detection_classes:0", "num_detections:0",
)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def parse_fpn_weapon_detections(boxes, scores, classes, num_detections) -> tuple[DetectionCandidate, ...]:
    if isinstance(num_detections, (str, bytes)):
        raise ValueError("num_detections must be a batch-size-1 vector")
    try:
        if len(num_detections) != 1:
            raise ValueError("num_detections must have batch size 1")
        count_value = _number(num_detections[0], "num_detections")
    except (TypeError, IndexError) as exc:
        raise ValueError("num_detections must be a batch-size-1 vector") from exc
    count = int(count_value)
    if count != count_value or not 0 <= count <= MAX_PROPOSALS:
        raise ValueError("num_detections exceeds the FPN proposal bound")
    rows = (_batch_row(boxes, "boxes"), _batch_row(scores, "scores"), _batch_row(classes, "classes"))
    if len({len(row) for row in rows}) != 1 or not count <= len(rows[0]) <= MAX_PROPOSALS:
        raise ValueError("FPN output tensors disagree or exceed the proposal bound")
    detections = []
    for index in range(count):
        score = _number(rows[1][index], "score")
        model_class = _number(rows[2][index], "class_id")
        if model_class != int(model_class) or not 1 <= model_class <= 601:
            raise ValueError("active FPN class_id must be integral and within [1, 601]")
        # Each invocation contains one proposal, so the existing parser's
        # configurable 64-candidate cap cannot discard any FPN proposal.
        detections.extend(parse_oid_v4_weapon_detections(
            [[rows[0][index]]], [[score]], [[model_class]], [1.0],
            confidence_threshold=CONFIDENCE_THRESHOLD, max_detections=1,
        ))
    detections.sort(key=lambda item: (
        -item.confidence, item.category, item.box.x_min, item.box.y_min,
        item.box.x_max, item.box.y_max, item.model_class_id,
    ))
    return tuple(detections)


def _port_description(port: Any) -> dict:
    shape = port.get_partial_shape()
    if not shape.rank.is_static:
        raise RuntimeError("FPN port rank is unknown")
    dimensions = [int(dim.get_length()) if dim.is_static else None for dim in shape]
    return {
        "shape": dimensions,
        "dtype": port.get_element_type().get_type_name(),
        "names": sorted(port.get_names()),
    }


def _validate_ports(inputs, outputs) -> dict:
    inputs, outputs = tuple(inputs), tuple(outputs)
    if len(inputs) != 1 or len(outputs) != 4:
        raise RuntimeError("FPN model must expose one input and four outputs")
    source = _port_description(inputs[0])
    if source["dtype"] != "u8" or source["shape"] != [1, None, None, 3]:
        raise RuntimeError("FPN input must preserve dynamic HxW uint8 RGB geometry")
    descriptions = [_port_description(port) for port in outputs]
    expected_ranks = (3, 2, 2, 1)
    proposal_dimensions = []
    for index, (description, rank) in enumerate(zip(descriptions, expected_ranks)):
        shape = description["shape"]
        if description["dtype"] != "f32" or len(shape) != rank or shape[0] != 1:
            raise RuntimeError("FPN output type, rank or batch is unsupported")
        if index < 3:
            proposals = shape[1]
            if proposals is not None and not 0 <= proposals <= MAX_PROPOSALS:
                raise RuntimeError("FPN output exceeds the proposal bound")
            proposal_dimensions.append(proposals)
        if index == 0 and shape[2] != 4:
            raise RuntimeError("FPN boxes must have four coordinates")
    known = {dim for dim in proposal_dimensions if dim is not None}
    if len(known) > 1:
        raise RuntimeError("FPN proposal dimensions disagree")
    return {"input": source, "outputs": descriptions, "semantic_output_order": list(OUTPUT_NAMES)}


class OpenVINOOIDFPNWeaponRuntime:
    """One conversion/CPU compilation, reused for original-size source frames."""

    def __init__(self, compiled_model: Any, *, ports: dict, phase_seconds: dict,
                 progress: Callable[[str], None] | None = None) -> None:
        if not callable(compiled_model):
            raise ValueError("compiled_model must be callable")
        self.compiled_ports = _validate_ports(compiled_model.inputs, compiled_model.outputs)
        self.converted_ports = ports
        self._compiled_model = compiled_model
        self._input = compiled_model.input(0)
        self._outputs = tuple(compiled_model.output(index) for index in range(4))
        self._phase_seconds = dict(phase_seconds)
        self._progress = progress or (lambda phase: None)
        self._inference_count = 0

    @classmethod
    def from_tensorflow_graph(
        cls, model_path: str | Path, *, core: Any | None = None,
        frontend: Any | None = None, shape_factory: Callable | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> "OpenVINOOIDFPNWeaponRuntime":
        candidate = Path(model_path)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("FPN graph must be a regular non-symlink file")
        candidate = candidate.resolve()
        if progress is not None and not callable(progress):
            raise ValueError("progress must be callable")
        if core is None or frontend is None or shape_factory is None:
            import openvino as ov
            from openvino.frontend import FrontEndManager
            core = ov.Core() if core is None else core
            frontend = FrontEndManager().load_by_framework("tf") if frontend is None else frontend
            shape_factory = ov.PartialShape if shape_factory is None else shape_factory
        notify = progress or (lambda phase: None)
        phases = {}
        notify("frontend_load")
        started = time.perf_counter()
        source = frontend.load(str(candidate))
        places = [source.get_place_by_tensor_name(name) for name in OUTPUT_NAMES]
        if any(place is None for place in places):
            raise RuntimeError("FPN semantic output endpoint is missing")
        source.override_all_outputs(places)
        image = source.get_place_by_tensor_name(INPUT_NAME)
        if image is None:
            raise RuntimeError("FPN image input is missing")
        source.set_partial_shape(image, shape_factory([1, -1, -1, 3]))
        phases["frontend_load"] = time.perf_counter() - started
        notify("conversion")
        started = time.perf_counter()
        converted = frontend.convert(source)
        phases["conversion"] = time.perf_counter() - started
        if tuple(result.get_friendly_name() for result in converted.get_results()) != OUTPUT_NAMES:
            raise RuntimeError("FPN converted Result identities changed")
        ports = _validate_ports(converted.inputs, converted.outputs)
        notify("cpu_compile")
        started = time.perf_counter()
        compiled = core.compile_model(converted, "CPU")
        phases["cpu_compile"] = time.perf_counter() - started
        return cls(compiled, ports=ports, phase_seconds=phases, progress=notify)

    @property
    def compile_count(self) -> int:
        return 1

    @property
    def inference_count(self) -> int:
        return self._inference_count

    @property
    def phase_seconds(self) -> dict:
        return dict(self._phase_seconds)

    def detect(self, frame_bgr: Any) -> tuple[DetectionCandidate, ...]:
        import numpy as np
        frame = np.asarray(frame_bgr)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be HxWx3 uint8 BGR")
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if not 1 <= height <= 16384 or not 1 <= width <= 16384:
            raise ValueError("frame dimensions outside supported bounds")
        # Copy unconditionally: reversing an already reversed caller view can
        # produce a contiguous view that ascontiguousarray would alias.
        # The graph owns resizing of the untouched rectangular source.
        batch = np.array(frame[:, :, ::-1][None, ...], copy=True, order="C")
        self._progress("native_inference_" + str(self._inference_count + 1))
        outputs = self._compiled_model({self._input: batch})
        try:
            values = tuple(outputs[port] for port in self._outputs)
        except (KeyError, TypeError) as exc:
            raise RuntimeError("FPN inference outputs are incomplete") from exc
        for index, value in enumerate(values):
            array = np.asarray(value)
            rank = (3, 2, 2, 1)[index]
            if array.dtype != np.float32 or array.ndim != rank or array.shape[0] != 1:
                raise RuntimeError("FPN inference output type/rank/batch changed")
            if index < 3 and not 0 <= array.shape[1] <= MAX_PROPOSALS:
                raise RuntimeError("FPN inference output exceeds the proposal bound")
            if index == 0 and array.shape[2] != 4:
                raise RuntimeError("FPN inference box geometry changed")
        detections = parse_fpn_weapon_detections(*values)
        self._inference_count += 1
        return detections
