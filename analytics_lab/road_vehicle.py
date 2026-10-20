"""Optional local OMZ crossroad-0078 vehicle-only CPU adapter.

Model and preprocessing semantics are pinned to Open Model Zoo
6697dead54ed1cdd664b0313189c2cb52ee6335e. This module does not acquire models,
emit person/bike detections, or qualify road-scene accuracy.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from .artifacts import ArtifactSpec, verify_artifact_set
from .lpr_ocr import _finite, parse_omz_vehicle_rows
from .tracking import DetectionCandidate

_MODEL = "person-vehicle-bike-detection-crossroad-0078"
_BASE = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/" + _MODEL + "/"
_LICENSE = "https://github.com/openvinotoolkit/open_model_zoo/blob/6697dead54ed1cdd664b0313189c2cb52ee6335e/LICENSE"


def road_vehicle_artifacts() -> tuple[ArtifactSpec, ...]:
    """The reviewed FP16 IR pair; callers provision it outside this adapter."""
    return tuple(ArtifactSpec(
        component="open-model-zoo/" + _MODEL + "/fp16", relative_path="FP16/" + _MODEL + extension,
        size_bytes=size, sha384=digest, source_url=_BASE + "FP16/" + _MODEL + extension,
        license_id="Apache-2.0", license_url=_LICENSE,
    ) for extension, size, digest in (
        (".xml", 526260, "7997f14109662b3d422ee5b9c7a812690eb00227b17e366ab6098fb5924a9a8c5077b326e3dcc3d2e2cbce91797bab4d"),
        (".bin", 2357042, "9fa3031c27af7e934a61ff02a01820d990b257a21ea0f9e195d79941fcbe97fac9c731bd404bd0c8dddfa4c1e1937cb4"),
    ))


def parse_road_vehicle_rows(
    rows: Iterable[Sequence[float]], *, threshold: float = .50, max_vehicles: int = 64,
) -> tuple[DetectionCandidate, ...]:
    """Reuse bounded SSD validation, selecting only donor vehicle label 2."""
    def mapped_rows():
        for row in rows:
            if isinstance(row, (str, bytes)):
                raise ValueError("detector row must contain seven numeric values")
            try:
                if len(row) != 7:
                    raise ValueError("detector row must contain seven numeric values")
                label = _finite(row[1], "detector label")
                # Validate before mapping: malformed ignored classes must not
                # disappear into a valid numeric background label.
                yield (row[0], 1 if label == 2 else 0, row[2], row[3], row[4], row[5], row[6])
            except (TypeError, IndexError, KeyError) as exc:
                raise ValueError("detector row must contain seven numeric values") from exc

    return tuple(replace(item, model_class_id=2) for item in parse_omz_vehicle_rows(
        mapped_rows(), threshold=threshold, max_vehicles=max_vehicles,
    ))


def _validate_ports(inputs: tuple[Any, ...], outputs: tuple[Any, ...]) -> None:
    if len(inputs) != 1 or tuple(int(value) for value in inputs[0].shape) != (1, 3, 1024, 1024):
        raise RuntimeError("crossroad detector input shape changed")
    if "data" not in inputs[0].get_names() or inputs[0].get_element_type().get_type_name() != "f32":
        raise RuntimeError("crossroad detector requires named data float32 input")
    if len(outputs) != 1 or tuple(int(value) for value in outputs[0].shape) != (1, 1, 200, 7):
        raise RuntimeError("crossroad detector output shape changed")
    if outputs[0].get_element_type().get_type_name() != "f32":
        raise RuntimeError("crossroad detector output must be float32")


class OpenVINORoadVehicleDetector:
    def __init__(
        self, artifact_root: str | Path, *, confidence_threshold: float = .50,
        max_vehicles: int = 64, device: str = "CPU",
    ) -> None:
        cutoff = _finite(confidence_threshold, "confidence_threshold")
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if type(max_vehicles) is not int or not 1 <= max_vehicles <= 64:
            raise ValueError("max_vehicles must be an integer within [1, 64]")
        if device != "CPU":
            raise ValueError("crossroad detector is CPU-only")
        specs = road_vehicle_artifacts()
        verified = verify_artifact_set(artifact_root, specs)
        paths = {item.spec.relative_path: item.path for item in verified}
        try:
            import cv2
            import numpy as np
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("OpenVINO, NumPy and OpenCV are required for crossroad detection") from exc
        self.runtime_version = str(getattr(ov, "__version__", "unknown"))
        if not self.runtime_version.startswith("2026.3.1"):
            raise RuntimeError("crossroad OpenVINO runtime version changed")
        core = ov.Core()
        model = core.read_model(model=str(paths[specs[0].relative_path]), weights=str(paths[specs[1].relative_path]))
        _validate_ports(tuple(model.inputs), tuple(model.outputs))
        compiled = core.compile_model(model, "CPU")
        _validate_ports(tuple(compiled.inputs), tuple(compiled.outputs))
        self._cv2, self._np, self._compiled = cv2, np, compiled
        self._output = compiled.output(0)
        self.confidence_threshold, self.max_vehicles = cutoff, max_vehicles
        self.compile_count, self.inference_count = 1, 0

    def infer_rows(self, image: Any) -> Any:
        if (not isinstance(image, self._np.ndarray) or image.dtype != self._np.uint8 or image.ndim != 3
                or image.shape[2] != 3 or min(image.shape[:2]) < 1):
            raise ValueError("crossroad image must be nonempty HxWx3 uint8 BGR")
        resized = self._cv2.resize(image, (1024, 1024), interpolation=self._cv2.INTER_LINEAR)
        blob = self._np.array(resized.transpose((2, 0, 1))[None], dtype=self._np.float32, copy=True, order="C")
        self.inference_count += 1
        result = self._compiled([blob])
        output = self._np.asarray(result[self._output])
        if output.shape != (1, 1, 200, 7) or output.dtype != self._np.float32:
            raise RuntimeError("crossroad inference output contract changed")
        return output.reshape((200, 7))

    def detect(self, image: Any) -> tuple[DetectionCandidate, ...]:
        return parse_road_vehicle_rows(self.infer_rows(image), threshold=self.confidence_threshold,
                                       max_vehicles=self.max_vehicles)
