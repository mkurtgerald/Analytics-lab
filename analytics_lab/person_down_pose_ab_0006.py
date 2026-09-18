"""Bounded staged-real A/B for retained OpenPose versus pinned OMZ pose 0006.

This is a resolution-step experiment after the lower-resolution EfficientHRNet
0005 candidate failed its predeclared same-video retention rule. 0005, 0006 and
0007 share the same pinned FP16 weight blob in Open Model Zoo; 0006 raises only
the reviewed static network input from 288x288 to 352x352. Detector, orientation
recovery, association, posture semantics, temporal policy, evaluator, media and
CPU device remain unchanged. Candidate bytes remain in ephemeral evidence space.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

from .artifacts import ArtifactSpec, OPENVINO_OMZ_2023_FP16, verify_artifact_set
from . import person_down_pose_ab as base
from .detector_shootout import _CompiledDetector
from .detector_thresholds import _provision, _target_candidate
from .evaluation import aggregate_person_down_evaluations
from .openpose_diagnostics import _ReferenceRuntime
from .openpose_reference import ReferenceOpenPoseDecoder
from .openvino_omz import _OpenVINORuntime
from .validation_cli import load_manifest

_RUNTIME_PREFIX = "2026.3.1"
_OMZ_COMMIT = "6697dead54ed1cdd664b0313189c2cb52ee6335e"
_OMZ_LICENSE = f"https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/{_OMZ_COMMIT}/LICENSE"
_MODEL_BASE = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/human-pose-estimation-0006/FP16"
_MAX_CANDIDATE_ARTIFACT_BYTES = 20 * 1024 * 1024
_INPUT = 352
_POSE_0006 = (
    ArtifactSpec(
        component="open-model-zoo/human-pose-estimation-0006/fp16",
        relative_path="human-pose-estimation-0006/FP16/human-pose-estimation-0006.xml",
        size_bytes=1_064_076,
        sha384="567d2f921e5af960384fddc30bfe8d7110d90222dae50ce36a13d6a0ca5113ff83b9adfe7c505a403d9f46d17285da39",
        source_url=_MODEL_BASE + "/human-pose-estimation-0006.xml",
        license_id="Apache-2.0",
        license_url=_OMZ_LICENSE,
    ),
    ArtifactSpec(
        component="open-model-zoo/human-pose-estimation-0006/fp16",
        relative_path="human-pose-estimation-0006/FP16/human-pose-estimation-0006.bin",
        size_bytes=19_039_904,
        sha384="ef4ab20cd0695a4b86789607acc6eb636d07fbb6786f30300ea713bb94a9110ed1ecec5db5577398ddc13663cc2ce690",
        source_url=_MODEL_BASE + "/human-pose-estimation-0006.bin",
        license_id="Apache-2.0",
        license_url=_OMZ_LICENSE,
    ),
)


def _download_pose_artifact(spec: ArtifactSpec, target: Path) -> None:
    if spec not in _POSE_0006:
        raise ValueError("unreviewed pose artifact is not accepted")
    expected_size = spec.size_bytes
    if expected_size < 1 or expected_size > _MAX_CANDIDATE_ARTIFACT_BYTES:
        raise ValueError("pose candidate artifact exceeds bounded item budget")
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.stat().st_size != expected_size:
            raise ValueError("existing pose candidate artifact has wrong size or type")
        return
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        raise ValueError("stale pose candidate partial exists")
    request = urllib.request.Request(spec.source_url, headers={"User-Agent": "Analytics-lab-pose-ab/1"})
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=30) as response, partial.open("xb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > expected_size:
                    raise RuntimeError("pose candidate artifact exceeded pinned byte count")
                handle.write(block)
        if total != expected_size:
            raise RuntimeError("pose candidate artifact byte count mismatch")
        os.replace(partial, target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _provision_pose_0006(root: Path) -> tuple[Any, ...]:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("candidate artifact root must be a non-symlink directory")
    root = root.resolve(strict=True)
    for spec in _POSE_0006:
        target = base._target(root, spec.relative_path)
        _download_pose_artifact(spec, target)
    return verify_artifact_set(root, _POSE_0006)


class _Pose0006Runtime:
    """Exact pinned 0006 OpenVINO CPU runtime with distortion-free padding."""

    def __init__(self, artifacts: tuple[Any, ...], device: str = "CPU") -> None:
        try:
            import cv2
            import numpy as np
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("OpenVINO, NumPy and OpenCV are required for pose A/B evidence") from exc
        self.cv2 = cv2
        self.np = np
        self.runtime_version = str(getattr(ov, "__version__", "unknown"))
        paths = {item.spec.relative_path: item.path for item in artifacts}
        xml = "human-pose-estimation-0006/FP16/human-pose-estimation-0006.xml"
        binary = "human-pose-estimation-0006/FP16/human-pose-estimation-0006.bin"
        if xml not in paths or binary not in paths:
            raise RuntimeError("pose 0006 artifact set is incomplete")
        core = ov.Core()
        model = core.read_model(model=str(paths[xml]), weights=str(paths[binary]))
        self.compiled = core.compile_model(model, device)
        inputs = tuple(self.compiled.inputs)
        if len(inputs) != 1 or tuple(int(v) for v in inputs[0].shape) != (1, 3, 352, 352):
            raise RuntimeError("pose 0006 input shape does not match reviewed artifact")
        heatmaps = []
        embeddings = []
        for port in self.compiled.outputs:
            shape = tuple(int(v) for v in port.shape)
            if shape == (1, 17, 176, 176):
                heatmaps.append(port)
            elif shape == (1, 17, 176, 176, 1):
                embeddings.append(port)
        if len(heatmaps) != 1 or len(embeddings) != 1:
            raise RuntimeError("pose 0006 output contract changed")
        self.heatmap_port = heatmaps[0]
        self.embedding_port = embeddings[0]

    def infer(self, image: Any) -> tuple[Any, Any, int, int]:
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) != 3 or int(shape[2]) != 3:
            raise ValueError("pose 0006 input must be a three-channel BGR image")
        height, width = int(shape[0]), int(shape[1])
        if height < 1 or width < 1:
            raise ValueError("pose 0006 input dimensions must be positive")
        scale = min(_INPUT / width, _INPUT / height)
        resized_width = max(1, min(_INPUT, int(round(width * scale))))
        resized_height = max(1, min(_INPUT, int(round(height * scale))))
        resized = self.cv2.resize(image, (resized_width, resized_height), interpolation=self.cv2.INTER_LINEAR)
        padded = self.np.pad(
            resized,
            ((0, _INPUT - resized_height), (0, _INPUT - resized_width), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        blob = self.np.ascontiguousarray(padded.transpose((2, 0, 1))[None], dtype=self.np.float32)
        results = self.compiled([blob])
        heatmaps = self.np.asarray(results[self.heatmap_port])
        embeddings = self.np.asarray(results[self.embedding_port])
        if tuple(heatmaps.shape) != (1, 17, 176, 176):
            raise RuntimeError("pose 0006 heatmap output shape changed")
        if tuple(embeddings.shape) != (1, 17, 176, 176, 1):
            raise RuntimeError("pose 0006 embedding output shape changed")
        return heatmaps, embeddings, resized_width, resized_height


class _Pose0006Decoder(base._Pose0005Decoder):
    """Same reviewed AE decoder and 2x-to-8x geometry compatibility mapping."""


def _run_path(samples: tuple[Any, ...], detector: _CompiledDetector, runtime: Any, decoder: Any) -> dict[str, Any]:
    results = []
    evaluated = []
    for sample in samples:
        result, sample_evaluation = base._scan_sample(sample, detector, runtime, decoder)
        results.append(result)
        evaluated.append(sample_evaluation)
    total_frames = sum(int(item["frames_processed"]) for item in results)
    total_elapsed_ms = sum(float(item["elapsed_ms"]) for item in results)
    return {
        "samples": results,
        "aggregate": asdict(aggregate_person_down_evaluations(evaluated)),
        "total_frames_processed": total_frames,
        "total_elapsed_ms": total_elapsed_ms,
        "throughput_fps": total_frames * 1000.0 / total_elapsed_ms if total_elapsed_ms > 0 else 0.0,
        "pose_inference_ms": sum(float(item["pose_inference_ms"]) for item in results),
        "pose_decode_ms": sum(float(item["pose_decode_ms"]) for item in results),
    }


def run_pose_ab(manifest: str | Path, candidate_root: str | Path) -> dict[str, Any]:
    artifact_root, samples, config = load_manifest(manifest)
    if config.required_device != "CPU" or len(samples) != 2:
        raise RuntimeError("pose A/B requires the bounded two-clip CPU subset")

    preparation_started = time.perf_counter()
    candidate_root = Path(candidate_root)
    detector_candidate = _target_candidate()
    _provision(candidate_root, detector_candidate)
    detector = _CompiledDetector(detector_candidate, candidate_root)

    current_verified = verify_artifact_set(artifact_root, OPENVINO_OMZ_2023_FP16)
    current_base = _OpenVINORuntime(current_verified, "CPU")
    if not current_base.runtime_version.startswith(_RUNTIME_PREFIX):
        raise RuntimeError("retained OpenVINO runtime does not match reviewed release")
    current_runtime = _ReferenceRuntime(current_base)
    current_decoder = ReferenceOpenPoseDecoder()

    candidate_verified = base._stage_call(
        "candidate_provision", _provision_pose_0006, candidate_root / "pose-0006"
    )
    candidate_runtime = base._stage_call(
        "candidate_runtime_contract", _Pose0006Runtime, candidate_verified, "CPU"
    )
    if not candidate_runtime.runtime_version.startswith(_RUNTIME_PREFIX):
        raise base._PoseABStageError("candidate_runtime_contract")
    candidate_decoder = _Pose0006Decoder()
    preparation_elapsed_ms = (time.perf_counter() - preparation_started) * 1000.0

    baseline = base._stage_call(
        "baseline_path", _run_path, samples, detector, current_runtime, current_decoder
    )
    candidate = base._stage_call(
        "candidate_downstream",
        _run_path,
        samples,
        detector,
        base._StageRuntime(candidate_runtime),
        base._StageDecoder(candidate_decoder),
    )
    return {
        "schema_version": 1,
        "diagnostic": "person_down_pose_model_ab",
        "comparison": {
            "baseline_pose_model": "human-pose-estimation-0001",
            "candidate_pose_model": "human-pose-estimation-0006",
            "omz_commit": _OMZ_COMMIT,
            "candidate_decoder": "associative_embedding",
            "candidate_input": [1, 3, 352, 352],
            "same_weights_as_0005": True,
            "same_detector": detector_candidate.name,
            "same_orientation_recovery": True,
            "same_association_policy": True,
            "same_posture_policy": True,
            "same_temporal_policy": True,
            "same_media": True,
            "device": "CPU",
        },
        "preparation_elapsed_ms": preparation_elapsed_ms,
        "baseline": baseline,
        "candidate": candidate,
        "evidence_only": True,
        "production_promoted": False,
        "commercial_accuracy_claim": False,
        "inferences_not_supported": ["fall", "injury", "cause", "fault", "intent"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_pose_ab(args.manifest, args.candidate_dir)
    except base._PoseABStageError as exc:
        print(f"Person-down pose A/B diagnostic rejected at stage: {exc.stage}.", file=sys.stderr)
        return 2
    except Exception:
        print("Person-down pose A/B diagnostic rejected at stage: unknown.", file=sys.stderr)
        return 2
    try:
        payload = json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        print("Person-down pose A/B diagnostic rejected at stage: serialization.", file=sys.stderr)
        return 2
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
