"""Bounded real-plate comparison for the project-authored OpenCV glyph OCR fallback.

Uses only the three already-bound CC0 plate crops. It downloads no model or
training dataset, retains no media, and compares total edit distance against the
preserved Tesseract observations. This is engineering evidence only.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from . import lpr_ocr_exact_evidence as land_rover
from . import lpr_ocr_sc_evidence as south_carolina
from . import lpr_ocr_wa_evidence as washington
from .lpr_ocr_exact_evidence import _download
from .lpr_ocr_glyph import recognize_plate_glyphs

_BASELINE_OBSERVED = {
    "land-rover": "IFMPR318J",
    "south-carolina": "354AVIV",
    "washington": "CPU404",
}


def _edit_distance(left: str, right: str) -> int:
    if not isinstance(left, str) or not isinstance(right, str):
        raise ValueError("edit-distance inputs must be strings")
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def _sample(name: str, module: Any) -> dict[str, object]:
    import cv2
    import numpy as np

    expected_sha256 = getattr(module, "_SOURCE_SHA256", None)
    crop_sha256 = getattr(module, "_CROP_RGB24_SHA256", None)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError("source SHA-256 must already be pinned")
    if not isinstance(crop_sha256, str) or len(crop_sha256) != 64:
        raise RuntimeError("crop SHA-256 must already be pinned")

    source = _download(
        module._SOURCE_URL,
        allowed_hosts=frozenset({"upload.wikimedia.org"}),
        max_bytes=int(module._SOURCE_SIZE) + 1024,
    )
    if len(source) != int(module._SOURCE_SIZE):
        raise RuntimeError("source size mismatch")
    if hashlib.sha1(source).hexdigest() != module._SOURCE_SHA1:
        raise RuntimeError("source SHA-1 mismatch")
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise RuntimeError("source SHA-256 mismatch")

    image = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or tuple(int(v) for v in image.shape[:2]) != (
        int(module._SOURCE_HEIGHT), int(module._SOURCE_WIDTH)
    ):
        raise RuntimeError("source decode dimensions mismatch")
    left, top, right, bottom = module._PLATE_BOX_PX
    crop = image[top:bottom, left:right]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb_bytes = rgb.tobytes(order="C")
    if hashlib.sha256(rgb_bytes).hexdigest() != crop_sha256:
        raise RuntimeError("crop RGB24 identity mismatch")

    started = time.perf_counter()
    cpu_started = time.process_time()
    result = recognize_plate_glyphs(rgb)
    elapsed = time.perf_counter() - started
    cpu = time.process_time() - cpu_started
    expected = module._EXPECTED_TEXT
    observed = result.text
    distance = _edit_distance(expected, observed)
    baseline_observed = _BASELINE_OBSERVED[name]
    baseline_distance = _edit_distance(expected, baseline_observed)
    denominator = max(1, len(expected), len(observed))
    return {
        "name": name,
        "source_sha256": expected_sha256,
        "crop_rgb24_sha256": crop_sha256,
        "expected_text": expected,
        "observed_text": observed,
        "mean_template_score": round(result.mean_score, 6),
        "candidate_components": result.candidate_components,
        "accepted_components": result.accepted_components,
        "edit_distance": distance,
        "character_accuracy": round(max(0.0, 1.0 - distance / denominator), 6),
        "baseline_observed_text": baseline_observed,
        "baseline_edit_distance": baseline_distance,
        "elapsed_seconds": round(elapsed, 6),
        "cpu_seconds": round(cpu, 6),
    }


def run() -> dict[str, object]:
    samples = (
        _sample("land-rover", land_rover),
        _sample("south-carolina", south_carolina),
        _sample("washington", washington),
    )
    candidate_total = sum(int(item["edit_distance"]) for item in samples)
    baseline_total = sum(int(item["baseline_edit_distance"]) for item in samples)
    return {
        "evidence": "lpr-ocr-project-glyph-first-attempt",
        "candidate": "project-authored-opencv-hershey-template-bank",
        "training_data": "none",
        "external_weights": "none",
        "samples": list(samples),
        "candidate_total_edit_distance": candidate_total,
        "baseline_total_edit_distance": baseline_total,
        "material_improvement": candidate_total < baseline_total,
        "claim": "three bound CC0 engineering samples only; not commercial accuracy",
    }


def main() -> int:
    result = run()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["material_improvement"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
