"""Bounded threshold-sensitivity evidence for the best measured detector candidate.

Runs person-detection-0200 once per frame on the existing rights-bound two-clip
GMDCSA-24 seed, then derives several fixed confidence thresholds from the same
raw outputs. Output is aggregate-only; no frames or media are retained.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

from .artifacts import verify_artifact_set
from .detector_shootout import (
    DETECTOR_CANDIDATES,
    DetectorCandidate,
    _CompiledDetector,
    _download_exact,
    _sha256,
    _window,
)
from .validation import ValidationSampleSpec
from .validation_cli import load_manifest
from .video import OpenCVVideoFileSource

_THRESHOLDS = (0.50, 0.40, 0.30, 0.20, 0.10)
_TARGET_NAME = "person-detection-0200"
_MIN_DURING_COVERAGE = 0.55
_MIN_NEGATIVE_COVERAGE = 0.90
_MAX_DUPLICATE_FRAME_RATE = 0.05


@dataclass(frozen=True)
class ThresholdTotals:
    frames: int
    frames_with_detection: int
    detections: int
    duplicate_frames: int
    duplicate_excess: int

    @property
    def coverage(self) -> float:
        return self.frames_with_detection / self.frames if self.frames else 0.0

    @property
    def duplicate_frame_rate(self) -> float:
        return self.duplicate_frames / self.frames if self.frames else 0.0


class _Accumulator:
    def __init__(self) -> None:
        self.frames = 0
        self.frames_with_detection = 0
        self.detections = 0
        self.duplicate_frames = 0
        self.duplicate_excess = 0

    def add(self, detections: int) -> None:
        if type(detections) is not int or detections < 0:
            raise ValueError("detections must be a nonnegative integer")
        self.frames += 1
        if detections:
            self.frames_with_detection += 1
        self.detections += detections
        if detections > 1:
            self.duplicate_frames += 1
            self.duplicate_excess += detections - 1

    def freeze(self) -> ThresholdTotals:
        return ThresholdTotals(
            self.frames,
            self.frames_with_detection,
            self.detections,
            self.duplicate_frames,
            self.duplicate_excess,
        )


def choose_threshold(rows: list[dict[str, float]]) -> float | None:
    """Choose the highest threshold clearing the predeclared evidence bars."""
    eligible = [
        row for row in rows
        if float(row["positive_during_coverage"]) >= _MIN_DURING_COVERAGE
        and float(row["negative_coverage"]) >= _MIN_NEGATIVE_COVERAGE
        and float(row["positive_duplicate_frame_rate"]) <= _MAX_DUPLICATE_FRAME_RATE
        and float(row["negative_duplicate_frame_rate"]) <= _MAX_DUPLICATE_FRAME_RATE
    ]
    if not eligible:
        return None
    return max(float(row["threshold"]) for row in eligible)


def _target_candidate() -> DetectorCandidate:
    matches = [item for item in DETECTOR_CANDIDATES if item.name == _TARGET_NAME]
    if len(matches) != 1 or matches[0].output_kind != "ssd":
        raise RuntimeError("reviewed threshold candidate changed")
    return matches[0]


def _provision(root: Path, candidate: DetectorCandidate) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("candidate root must be a regular directory")
    for spec in candidate.specs:
        target = root.joinpath(*Path(spec.relative_path).parts)
        _download_exact(spec.source_url, target, spec.size_bytes)
    verify_artifact_set(root, candidate.specs)


def _person_confidences(detector: _CompiledDetector, image: Any) -> tuple[float, ...]:
    candidate = detector.candidate
    if candidate.output_kind != "ssd":
        raise RuntimeError("threshold sensitivity requires SSD output")
    result = detector.compiled([detector._blob(image)])
    output = detector.np.asarray(result[detector.compiled.output(0)])
    if output.size % 7:
        raise RuntimeError("SSD detector output shape changed")
    values: list[float] = []
    for raw in output.reshape((-1, 7)):
        image_id = float(raw[0])
        if image_id < 0:
            break
        label = int(round(float(raw[1])))
        confidence = float(raw[2])
        if label == candidate.person_label and 0.0 <= confidence <= 1.0:
            values.append(confidence)
    return tuple(values)


def _summary(acc: _Accumulator) -> dict[str, float | int]:
    frozen = acc.freeze()
    return asdict(frozen) | {
        "coverage": frozen.coverage,
        "duplicate_frame_rate": frozen.duplicate_frame_rate,
    }


def _sample_scan(sample: ValidationSampleSpec, detector: _CompiledDetector) -> tuple[dict[str, Any], float]:
    path = sample.video_path
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("threshold media identity mismatch")
    totals = {threshold: _Accumulator() for threshold in _THRESHOLDS}
    windows = {
        threshold: {name: _Accumulator() for name in ("before", "during", "after")}
        for threshold in _THRESHOLDS
    }
    inference_ms = 0.0
    with OpenCVVideoFileSource(path, start_timestamp_ms=sample.start_timestamp_ms) as frames:
        for frame in frames:
            started = time.perf_counter()
            confidences = _person_confidences(detector, frame.image)
            inference_ms += (time.perf_counter() - started) * 1000.0
            window_name = _window(sample, frame.timestamp_ms)
            for threshold in _THRESHOLDS:
                count = sum(1 for value in confidences if value >= threshold)
                totals[threshold].add(count)
                if window_name is not None:
                    windows[threshold][window_name].add(count)
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("threshold media changed during execution")
    rows: dict[str, Any] = {}
    for threshold in _THRESHOLDS:
        item: dict[str, Any] = {"overall": _summary(totals[threshold])}
        if len(sample.labels) == 1:
            item["windows"] = {name: _summary(acc) for name, acc in windows[threshold].items()}
        rows[f"{threshold:.2f}"] = item
    return {"sample_id": sample.sample_id, "thresholds": rows}, inference_ms


def run_threshold_sensitivity(manifest: str | Path, candidate_root: str | Path) -> dict[str, Any]:
    _artifact_root, samples, config = load_manifest(manifest)
    if config.required_device != "CPU" or len(samples) != 2:
        raise RuntimeError("threshold sensitivity requires the bounded two-clip CPU seed")
    candidate = _target_candidate()
    root = Path(candidate_root)
    _provision(root, candidate)
    prepared = time.perf_counter()
    detector = _CompiledDetector(candidate, root)
    preparation_ms = (time.perf_counter() - prepared) * 1000.0
    sample_results = []
    inference_ms = 0.0
    for sample in samples:
        result, elapsed = _sample_scan(sample, detector)
        sample_results.append(result)
        inference_ms += elapsed
    positive = next(item for item in sample_results if "windows" in next(iter(item["thresholds"].values())))
    negative = next(item for item in sample_results if "windows" not in next(iter(item["thresholds"].values())))
    selection_rows: list[dict[str, float]] = []
    for threshold in _THRESHOLDS:
        key = f"{threshold:.2f}"
        pos = positive["thresholds"][key]
        neg = negative["thresholds"][key]
        selection_rows.append({
            "threshold": threshold,
            "positive_during_coverage": float(pos["windows"]["during"]["coverage"]),
            "positive_duplicate_frame_rate": float(pos["overall"]["duplicate_frame_rate"]),
            "negative_coverage": float(neg["overall"]["coverage"]),
            "negative_duplicate_frame_rate": float(neg["overall"]["duplicate_frame_rate"]),
        })
    frames = sum(int(next(iter(item["thresholds"].values()))["overall"]["frames"]) for item in sample_results)
    return {
        "schema_version": 1,
        "model": candidate.name,
        "thresholds": list(_THRESHOLDS),
        "selection_bars": {
            "min_positive_during_coverage": _MIN_DURING_COVERAGE,
            "min_negative_coverage": _MIN_NEGATIVE_COVERAGE,
            "max_duplicate_frame_rate": _MAX_DUPLICATE_FRAME_RATE,
        },
        "samples": sample_results,
        "selection_rows": selection_rows,
        "recommended_threshold": choose_threshold(selection_rows),
        "recommendation_is_evidence_only": True,
        "preparation_ms": preparation_ms,
        "detector_inference_ms": inference_ms,
        "detector_fps": (frames * 1000.0 / inference_ms) if inference_ms > 0 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_threshold_sensitivity(args.manifest, args.candidate_dir)
        print(json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError):
        print("Detector threshold sensitivity rejected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
