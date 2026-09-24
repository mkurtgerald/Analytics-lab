"""Reusable OpenVINO lifecycle for the admitted standard OIDv4 face detector.

The runtime owns one compiled model for its lifetime. Repeated ``detect`` calls
reuse that compiled model; conversion and compilation occur only in the
``from_tensorflow_graph`` constructor. No recognition, embedding, ReID or
identity matching is implemented here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .face_oid_ssd import OpenVINOOIDSSDDetector
from .face_privacy import FaceDetection


DetectorFactory = Callable[..., Any]


class OpenVINOOIDSSDRuntime:
    """Own one compiled OIDv4 detector and reuse it across frames."""

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
        confidence_threshold: float = 0.50,
        max_faces: int = 64,
        core: Any | None = None,
        convert_model: Callable[[str], Any] | None = None,
        detector_factory: DetectorFactory | None = None,
    ) -> "OpenVINOOIDSSDRuntime":
        """Convert and compile the graph once, then bind one reusable detector."""
        candidate = Path(model_path)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("face graph must be a regular non-symlink file")

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
        factory = detector_factory or OpenVINOOIDSSDDetector
        detector = factory(
            compiled,
            confidence_threshold=confidence_threshold,
            max_faces=max_faces,
        )
        return cls(detector, compile_count=1)

    @property
    def compile_count(self) -> int:
        return self._compile_count

    @property
    def inference_count(self) -> int:
        return self._inference_count

    def detect(self, frame_bgr: Any) -> tuple[FaceDetection, ...]:
        detections = self._detector.detect(frame_bgr)
        try:
            normalized = tuple(detections)
        except TypeError as exc:
            raise ValueError("detector must return an iterable") from exc
        if any(not isinstance(item, FaceDetection) for item in normalized):
            raise ValueError("detector returned unsupported face value")
        self._inference_count += 1
        return normalized
