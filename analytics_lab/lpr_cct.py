"""Local, identity-pinned CCT-S-v2 OCR through the existing PPM boundary.

Preprocessing and slot order follow fast-plate-ocr at
9ce7a5b64a939aa421c243b331d42e6bc25ffd44. No model acquisition occurs here.
The conservative complete-read cutoff is an engineering policy, not calibrated
plate accuracy. Region output is deliberately unused.
"""
from __future__ import annotations

import hashlib
from itertools import islice
import math
from numbers import Real
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Sequence

from .lpr_ocr import OCRText

CCT_MODEL_SIZE = 5_262_230
CCT_MODEL_SHA256 = "384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b"
CCT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
CCT_MAX_INPUT_BYTES = 16 * 1024 * 1024
_PPM_HEADER = re.compile(rb"P6\n([1-9][0-9]{0,7}) ([1-9][0-9]{0,7})\n255\n")


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def parse_cct_plate_probabilities(
    rows: Iterable[Sequence[float]], *, min_confidence: float = .90,
) -> OCRText | None:
    """Decode ten ordered 37-class probability rows without partial reads."""
    cutoff = _probability(min_confidence, "min_confidence")
    if isinstance(rows, (str, bytes)):
        raise ValueError("CCT output must contain ten probability rows")
    try:
        bounded_rows = tuple(islice(iter(rows), 11))
    except TypeError as exc:
        raise ValueError("CCT output must contain ten probability rows") from exc
    if len(bounded_rows) != 10:
        raise ValueError("CCT output must contain ten probability rows")
    chars: list[str] = []
    confidences: list[float] = []
    for row in bounded_rows:
        if isinstance(row, (str, bytes)):
            raise ValueError("CCT row must contain 37 probabilities")
        try:
            values = tuple(_probability(value, "CCT probability") for value in islice(iter(row), 38))
        except TypeError as exc:
            raise ValueError("CCT row must contain 37 probabilities") from exc
        if len(values) != 37:
            raise ValueError("CCT row must contain 37 probabilities")
        if abs(math.fsum(values) - 1.0) > 1e-3:
            raise ValueError("CCT probability row must sum to one")
        best = max(values)
        confidences.append(best)
        chars.append(CCT_ALPHABET[values.index(best)])
    confidence = min(confidences)
    if confidence < cutoff:
        return None
    text = "".join(chars).rstrip("_")
    if not text or "_" in text:
        return None
    return OCRText(text, confidence)


def _ppm_pixels(payload: bytes) -> tuple[int, int, memoryview]:
    if not isinstance(payload, bytes) or len(payload) > CCT_MAX_INPUT_BYTES:
        raise ValueError("CCT input must be bounded canonical RGB PPM bytes")
    match = _PPM_HEADER.match(payload)
    if match is None:
        raise ValueError("CCT input must be canonical P6 RGB PPM")
    width, height = (int(value) for value in match.groups())
    if len(payload) - match.end() != width * height * 3:
        raise ValueError("CCT PPM pixel byte count mismatch")
    return width, height, memoryview(payload)[match.end():]


def _verify_model(path: Path) -> None:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError("CCT model must be a regular non-symlink file")
    if before.st_size != CCT_MODEL_SIZE:
        raise ValueError("CCT model size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if identity(before) != identity(after) or digest.hexdigest() != CCT_MODEL_SHA256:
        raise ValueError("CCT model identity mismatch")


def _shape(port: Any) -> tuple[int | None, ...]:
    shape = port.get_partial_shape()
    if shape.rank.is_dynamic:
        raise RuntimeError("CCT port rank must be fixed")
    return tuple(None if dimension.is_dynamic else dimension.get_length() for dimension in shape)


def _validate_input(port: Any) -> None:
    if "input" not in port.get_names() or _shape(port) not in ((None, 64, 128, 3), (1, 64, 128, 3)):
        raise RuntimeError("CCT input name or shape changed")
    if port.get_element_type().get_type_name() != "u8":
        raise RuntimeError("CCT input must be uint8")


def _validate_plate(port: Any) -> None:
    shape = _shape(port)
    if not shape or shape[0] not in (None, 1) or shape[1:] not in ((370,), (10, 37)):
        raise RuntimeError("CCT plate output shape changed")
    if port.get_element_type().get_type_name() != "f32":
        raise RuntimeError("CCT plate output must be float32")


def _plate_output_index(outputs: tuple[Any, ...]) -> int:
    named = tuple(index for index, port in enumerate(outputs) if "plate" in port.get_names())
    if len(named) == 1:
        _validate_plate(outputs[named[0]])
        return named[0]
    if named:
        raise RuntimeError("CCT plate output name is ambiguous")
    compatible = []
    for index, port in enumerate(outputs):
        try:
            _validate_plate(port)
        except RuntimeError:
            continue
        compatible.append(index)
    if len(compatible) != 1:
        raise RuntimeError("CCT requires one uniquely compatible plate output")
    return compatible[0]


class OpenVINOCCTPlateRecognizer:
    """Compile the verified local CCT-S-v2 model once for CPU OCR."""

    def __init__(self, model_path: str | Path, *, min_confidence: float = .90) -> None:
        self.min_confidence = _probability(min_confidence, "min_confidence")
        path = Path(model_path)
        _verify_model(path)
        try:
            import cv2
            import numpy as np
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("OpenVINO, NumPy and OpenCV are required for CCT OCR") from exc
        self.runtime_version = str(getattr(ov, "__version__", "unknown"))
        if not self.runtime_version.startswith("2026.3.1"):
            raise RuntimeError("OpenVINO runtime does not match reviewed CCT release")
        core = ov.Core()
        model = core.read_model(str(path))
        inputs = tuple(model.inputs)
        if len(inputs) != 1:
            raise RuntimeError("CCT requires one input")
        _validate_input(inputs[0])
        plate_index = _plate_output_index(tuple(model.outputs))
        compiled = core.compile_model(model, "CPU")
        compiled_inputs = tuple(compiled.inputs)
        if len(compiled_inputs) != 1:
            raise RuntimeError("CCT compiled input count changed")
        _validate_input(compiled_inputs[0])
        plate_output = compiled.output(plate_index)
        _validate_plate(plate_output)
        self._np, self._cv2 = np, cv2
        self._compiled, self._plate_output = compiled, plate_output
        self.compile_count = 1
        self.inference_count = 0

    def __call__(self, ppm_bytes: bytes) -> OCRText | None:
        width, height, pixels = _ppm_pixels(ppm_bytes)
        rgb = self._np.frombuffer(pixels, dtype=self._np.uint8).reshape((height, width, 3))
        resized = self._cv2.resize(rgb, (128, 64), interpolation=self._cv2.INTER_LINEAR)
        # Own the input storage; RGB order and integer values are retained.
        blob = self._np.array(resized[None], dtype=self._np.uint8, copy=True, order="C")
        self.inference_count += 1
        result = self._compiled([blob])
        output = self._np.asarray(result[self._plate_output])
        if output.shape not in ((1, 370), (1, 10, 37)) or output.dtype != self._np.float32:
            raise RuntimeError("CCT inference must return one float32 plate with 370 probabilities")
        return parse_cct_plate_probabilities(output.reshape((10, 37)), min_confidence=self.min_confidence)
