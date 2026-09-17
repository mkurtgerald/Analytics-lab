"""Bounded real-video detector comparison on the rights-reviewed validation seed.

This evidence-only command downloads at most three already-reviewed Open Model
Zoo detector alternatives, verifies their exact size/SHA-384 identities, and
compares them with the current detector on the same two GMDCSA-24 clips and CPU
runtime. It emits aggregate counts/timing only and retains no decoded frames.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence
import urllib.request

from .artifacts import ArtifactSpec, OPENVINO_OMZ_2023_FP16, verify_artifact_set
from .validation import ValidationSampleSpec
from .validation_cli import load_manifest
from .video import OpenCVVideoFileSource

_OMZ_COMMIT = "6697dead54ed1cdd664b0313189c2cb52ee6335e"
_OMZ_LICENSE = f"https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/{_OMZ_COMMIT}/LICENSE"
_MAX_ITEM_BYTES = 8 * 1024 * 1024
_THRESHOLD = 0.50
_RUNTIME_PREFIX = "2026.3.1"


@dataclass(frozen=True)
class DetectorCandidate:
    name: str
    input_height: int
    input_width: int
    output_kind: str
    person_label: int
    xml: ArtifactSpec
    weights: ArtifactSpec

    @property
    def specs(self) -> tuple[ArtifactSpec, ArtifactSpec]:
        return (self.xml, self.weights)


@dataclass(frozen=True)
class DetectionTotals:
    frames: int
    frames_with_detection: int
    detections: int

    @property
    def coverage(self) -> float:
        return (self.frames_with_detection / self.frames) if self.frames else 0.0


class _Accumulator:
    def __init__(self) -> None:
        self.frames = 0
        self.frames_with_detection = 0
        self.detections = 0

    def add(self, detections: int) -> None:
        self.frames += 1
        if detections:
            self.frames_with_detection += 1
        self.detections += detections

    def freeze(self) -> DetectionTotals:
        return DetectionTotals(self.frames, self.frames_with_detection, self.detections)


def _candidate(name: str, height: int, width: int, output_kind: str, person_label: int,
               xml_size: int, xml_sha384: str, bin_size: int, bin_sha384: str) -> DetectorCandidate:
    prefix = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{name}/FP16/{name}"
    return DetectorCandidate(
        name=name,
        input_height=height,
        input_width=width,
        output_kind=output_kind,
        person_label=person_label,
        xml=ArtifactSpec(
            component=f"open-model-zoo/{name}/fp16",
            relative_path=f"{name}/FP16/{name}.xml",
            size_bytes=xml_size,
            sha384=xml_sha384,
            source_url=prefix + ".xml",
            license_id="Apache-2.0",
            license_url=_OMZ_LICENSE,
        ),
        weights=ArtifactSpec(
            component=f"open-model-zoo/{name}/fp16",
            relative_path=f"{name}/FP16/{name}.bin",
            size_bytes=bin_size,
            sha384=bin_sha384,
            source_url=prefix + ".bin",
            license_id="Apache-2.0",
            license_url=_OMZ_LICENSE,
        ),
    )


DETECTOR_CANDIDATES = (
    _candidate(
        "person-detection-0200", 256, 256, "ssd", 0,
        254_619, "654a515935f6dffc0440cffdeeb6a889bc25bf5d64e425ce2337070cdbcce1b1fbbcf42e4b2761265eb65ec07c1cd6f7",
        3_634_654, "10b6f79b495ad1ef13748938c452379c6b6cd825d11426ffa68be3eeed6c48a0af04e0b913c843e6e7431aa1c13f5471",
    ),
    _candidate(
        "person-detection-0202", 512, 512, "ssd", 0,
        254_889, "7746ed6534bb59c9d16e7af4dbcbc768772288d48aca7e081c059eb1924ea956c67f23381a860a0bf0e187d66b7f1955",
        3_634_654, "2a149bc8c2f02965c59a998d7ca7868e4ba537c223ae492a8a177386f98997d8c15204f1c7fe847a51c660d9d4116d10",
    ),
    _candidate(
        "person-detection-0203", 480, 864, "atss", 1,
        1_216_181, "086b17b4fc8b5454e4c892bbc26cdd120b517f89a86eab7ba6bb2a0e4ed5437cd011e65c299b89346f62538cbcc7b735",
        3_902_528, "906e38b168001ab43117c6cc5a737e5d596d4aae441ca93dc5ea22411d6808fb63c61666bc626411c6cc60fd603ebab8",
    ),
)

_BASELINE = DetectorCandidate(
    name="person-detection-retail-0013",
    input_height=320,
    input_width=544,
    output_kind="ssd",
    person_label=1,
    xml=OPENVINO_OMZ_2023_FP16[0],
    weights=OPENVINO_OMZ_2023_FP16[1],
)


def count_ssd_people(rows: Iterable[Sequence[float]], *, person_label: int, threshold: float = _THRESHOLD) -> int:
    count = 0
    for raw in rows:
        if len(raw) != 7:
            raise RuntimeError("SSD detector output row shape changed")
        image_id, label, confidence = float(raw[0]), int(round(float(raw[1]))), float(raw[2])
        if image_id < 0:
            break
        if label == person_label and confidence >= threshold:
            count += 1
    return count


def count_atss_people(boxes: Iterable[Sequence[float]], labels: Iterable[float], *,
                      person_label: int = 1, threshold: float = _THRESHOLD) -> int:
    count = 0
    box_rows = list(boxes)
    label_rows = list(labels)
    if len(box_rows) != len(label_rows):
        raise RuntimeError("ATSS detector output lengths changed")
    for box, label in zip(box_rows, label_rows):
        if len(box) != 5:
            raise RuntimeError("ATSS box output shape changed")
        if int(round(float(label))) == person_label and float(box[4]) >= threshold:
            count += 1
    return count


def choose_material_candidate(results: dict[str, dict[str, float]], *, baseline: str) -> str | None:
    """Choose the smallest model that clears a measured prone-recall improvement bar.

    This is only an evidence recommendation. It does not alter the production
    backend. Candidates must gain at least 25 percentage points during the
    labeled fall interval and preserve at least 90% person coverage on the hard
    normal clip. Among qualifying candidates, prefer fewer model bytes, then
    faster measured detector FPS.
    """
    base = results[baseline]
    required = min(1.0, float(base["positive_during_coverage"]) + 0.25)
    eligible = [
        (name, data) for name, data in results.items() if name != baseline
        and float(data["positive_during_coverage"]) >= required
        and float(data["negative_coverage"]) >= 0.90
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda item: (int(item[1]["artifact_bytes"]), -float(item[1]["detector_fps"]), item[0]))
    return eligible[0][0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download_exact(url: str, target: Path, expected_size: int) -> None:
    if expected_size < 1 or expected_size > _MAX_ITEM_BYTES:
        raise ValueError("detector artifact exceeds bounded item budget")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("detector artifact path may not be a symlink")
    if target.exists():
        if not target.is_file() or target.stat().st_size != expected_size:
            raise ValueError("existing detector artifact has wrong size")
        return
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        raise ValueError("stale detector artifact partial exists")
    request = urllib.request.Request(url, headers={"User-Agent": "Analytics-lab-detector-shootout/1"})
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=30) as response, partial.open("xb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > expected_size:
                    raise RuntimeError("detector artifact exceeded pinned byte count")
                handle.write(block)
        if total != expected_size:
            raise RuntimeError("detector artifact byte count mismatch")
        os.replace(partial, target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _provision_candidates(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("candidate root must be a regular directory")
    for candidate in DETECTOR_CANDIDATES:
        for spec in candidate.specs:
            target = root.joinpath(*Path(spec.relative_path).parts)
            _download_exact(spec.source_url, target, spec.size_bytes)
        verify_artifact_set(root, candidate.specs)


def _window(sample: ValidationSampleSpec, timestamp_ms: int) -> str | None:
    if len(sample.labels) != 1:
        return None
    label = sample.labels[0]
    if timestamp_ms < label.start_timestamp_ms:
        return "before"
    if timestamp_ms <= label.end_timestamp_ms:
        return "during"
    return "after"


class _CompiledDetector:
    def __init__(self, candidate: DetectorCandidate, artifact_root: Path, device: str = "CPU") -> None:
        try:
            import cv2
            import numpy as np
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("OpenVINO, NumPy and OpenCV are required") from exc
        self.cv2 = cv2
        self.np = np
        self.candidate = candidate
        if not str(getattr(ov, "__version__", "")).startswith(_RUNTIME_PREFIX):
            raise RuntimeError("OpenVINO runtime version does not match reviewed release")
        verified = verify_artifact_set(artifact_root, candidate.specs)
        paths = {item.spec.relative_path: item.path for item in verified}
        core = ov.Core()
        model = core.read_model(str(paths[candidate.xml.relative_path]), str(paths[candidate.weights.relative_path]))
        self.compiled = core.compile_model(model, device)
        inputs = tuple(self.compiled.inputs)
        if len(inputs) != 1 or tuple(int(v) for v in inputs[0].shape) != (1, 3, candidate.input_height, candidate.input_width):
            raise RuntimeError("candidate detector input shape changed")

    def _blob(self, image: Any) -> Any:
        resized = self.cv2.resize(
            image, (self.candidate.input_width, self.candidate.input_height), interpolation=self.cv2.INTER_LINEAR
        )
        return self.np.ascontiguousarray(resized.transpose((2, 0, 1))[None], dtype=self.np.float32)

    def detect_count(self, image: Any) -> int:
        result = self.compiled([self._blob(image)])
        if self.candidate.output_kind == "ssd":
            output = self.np.asarray(result[self.compiled.output(0)])
            if output.size % 7:
                raise RuntimeError("SSD detector output shape changed")
            return count_ssd_people(output.reshape((-1, 7)), person_label=self.candidate.person_label)
        by_name: dict[str, Any] = {}
        for port in self.compiled.outputs:
            try:
                name = port.any_name
            except Exception:
                name = ""
            if name:
                by_name[name] = self.np.asarray(result[port])
        boxes = by_name.get("boxes")
        labels = by_name.get("labels")
        if boxes is None or labels is None:
            arrays = [self.np.asarray(result[port]) for port in self.compiled.outputs]
            boxes = next((item for item in arrays if item.ndim >= 2 and item.shape[-1] == 5), None)
            labels = next((item for item in arrays if item is not boxes and item.size == (boxes.size // 5 if boxes is not None else -1)), None)
        if boxes is None or labels is None:
            raise RuntimeError("ATSS detector outputs changed")
        return count_atss_people(boxes.reshape((-1, 5)), labels.reshape((-1,)), person_label=self.candidate.person_label)


def _sample_result(sample: ValidationSampleSpec, detector: _CompiledDetector) -> tuple[dict[str, Any], float]:
    path = sample.video_path
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("shootout media identity mismatch")
    overall = _Accumulator()
    windows = {name: _Accumulator() for name in ("before", "during", "after")}
    elapsed_ms = 0.0
    with OpenCVVideoFileSource(path, start_timestamp_ms=sample.start_timestamp_ms) as frames:
        for frame in frames:
            start = time.perf_counter()
            detections = detector.detect_count(frame.image)
            elapsed_ms += (time.perf_counter() - start) * 1000.0
            overall.add(detections)
            name = _window(sample, frame.timestamp_ms)
            if name is not None:
                windows[name].add(detections)
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("shootout media changed during execution")
    result: dict[str, Any] = {"sample_id": sample.sample_id, "overall": asdict(overall.freeze()) | {"coverage": overall.freeze().coverage}}
    if len(sample.labels) == 1:
        result["windows"] = {
            name: asdict(acc.freeze()) | {"coverage": acc.freeze().coverage} for name, acc in windows.items()
        }
    return result, elapsed_ms


def run_shootout(manifest: str | Path, candidate_root: str | Path) -> dict[str, Any]:
    artifact_root, samples, config = load_manifest(manifest)
    if config.required_device != "CPU" or len(samples) != 2:
        raise RuntimeError("detector shootout requires the bounded two-clip CPU seed")
    verify_artifact_set(artifact_root, OPENVINO_OMZ_2023_FP16)
    candidate_dir = Path(candidate_root)
    _provision_candidates(candidate_dir)

    models = (_BASELINE,) + DETECTOR_CANDIDATES
    output_models: list[dict[str, Any]] = []
    selection_inputs: dict[str, dict[str, float]] = {}
    for candidate in models:
        root = artifact_root if candidate is _BASELINE else candidate_dir
        prep_start = time.perf_counter()
        detector = _CompiledDetector(candidate, root)
        preparation_ms = (time.perf_counter() - prep_start) * 1000.0
        sample_results = []
        inference_ms = 0.0
        for sample in samples:
            item, elapsed = _sample_result(sample, detector)
            sample_results.append(item)
            inference_ms += elapsed
        frames = sum(int(item["overall"]["frames"]) for item in sample_results)
        detector_fps = (frames * 1000.0 / inference_ms) if inference_ms > 0 else 0.0
        artifact_bytes = candidate.xml.size_bytes + candidate.weights.size_bytes
        output_models.append({
            "name": candidate.name,
            "input_shape": [1, 3, candidate.input_height, candidate.input_width],
            "artifact_bytes": artifact_bytes,
            "preparation_ms": preparation_ms,
            "detector_inference_ms": inference_ms,
            "detector_fps": detector_fps,
            "samples": sample_results,
        })
        positive = next(item for item in sample_results if "windows" in item)
        negative = next(item for item in sample_results if "windows" not in item)
        selection_inputs[candidate.name] = {
            "positive_during_coverage": float(positive["windows"]["during"]["coverage"]),
            "negative_coverage": float(negative["overall"]["coverage"]),
            "artifact_bytes": float(artifact_bytes),
            "detector_fps": float(detector_fps),
        }

    recommendation = choose_material_candidate(selection_inputs, baseline=_BASELINE.name)
    return {
        "schema_version": 1,
        "threshold": _THRESHOLD,
        "baseline": _BASELINE.name,
        "candidate_count": len(DETECTOR_CANDIDATES),
        "models": output_models,
        "selection_inputs": selection_inputs,
        "recommended_candidate": recommendation,
        "recommendation_is_evidence_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_shootout(args.manifest, args.candidate_dir)
        print(json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        print(f"Detector shootout rejected ({type(exc).__name__}).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
