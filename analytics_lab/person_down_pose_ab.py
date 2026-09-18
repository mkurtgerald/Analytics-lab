"""Bounded staged-real A/B for retained OpenPose versus pinned OMZ pose 0005.

The experiment changes only the pose model/decoder. It reuses the exact retained
person detector, orientation recovery, continuity tracking, safe pose association,
posture semantics, temporal policy, evaluator, media and CPU device. Candidate
media/model bytes remain in the caller's ephemeral evidence directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

from .artifacts import ArtifactSpec, OPENVINO_OMZ_2023_FP16, verify_artifact_set
from .associative_embedding_reference import AssociativeEmbeddingDecoder
from .detector_shootout import _CompiledDetector
from .detector_thresholds import _provision, _target_candidate
from .evaluation import EvaluationSample, aggregate_person_down_evaluations
from .openpose_diagnostics import _ReferenceRuntime
from .openpose_reference import ReferenceOpenPoseDecoder
from .openvino_omz import _OpenVINORuntime
from .person_down_orientation_diagnostics import _scan_sample
from .validation_cli import load_manifest
from .validation_seed import _download_exact

_RUNTIME_PREFIX = "2026.3.1"
_OMZ_COMMIT = "6697dead54ed1cdd664b0313189c2cb52ee6335e"
_OMZ_LICENSE = f"https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/{_OMZ_COMMIT}/LICENSE"
_MODEL_BASE = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/human-pose-estimation-0005/FP16"
_POSE_0005 = (
    ArtifactSpec(
        component="open-model-zoo/human-pose-estimation-0005/fp16",
        relative_path="human-pose-estimation-0005/FP16/human-pose-estimation-0005.xml",
        size_bytes=1_063_570,
        sha384="37595cec7cb044266eb7cb934fcf596d5b0382b12a03f5462f046a52aba3f9c96097377773618ca957d7b6941a12334b",
        source_url=_MODEL_BASE + "/human-pose-estimation-0005.xml",
        license_id="Apache-2.0",
        license_url=_OMZ_LICENSE,
    ),
    ArtifactSpec(
        component="open-model-zoo/human-pose-estimation-0005/fp16",
        relative_path="human-pose-estimation-0005/FP16/human-pose-estimation-0005.bin",
        size_bytes=19_039_904,
        sha384="ef4ab20cd0695a4b86789607acc6eb636d07fbb6786f30300ea713bb94a9110ed1ecec5db5577398ddc13663cc2ce690",
        source_url=_MODEL_BASE + "/human-pose-estimation-0005.bin",
        license_id="Apache-2.0",
        license_url=_OMZ_LICENSE,
    ),
)
_INPUT = 288
_OUTPUT_SCALE_COMPATIBILITY = 2.0 / 8.0


def _target(root: Path, relative_path: str) -> Path:
    target = root.joinpath(*relative_path.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("candidate artifact path may not be a symlink")
    resolved_root = root.resolve(strict=True)
    target.parent.resolve(strict=True).relative_to(resolved_root)
    return target


def _provision_pose_0005(root: Path) -> tuple[Any, ...]:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("candidate artifact root must be a non-symlink directory")
    root = root.resolve(strict=True)
    for spec in _POSE_0005:
        target = _target(root, spec.relative_path)
        _download_exact(spec.source_url, target, expected_size=spec.size_bytes)
    return verify_artifact_set(root, _POSE_0005)


class _Pose0005Runtime:
    """Exact pinned 0005 OpenVINO CPU runtime with distortion-free padding."""

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
        xml = "human-pose-estimation-0005/FP16/human-pose-estimation-0005.xml"
        binary = "human-pose-estimation-0005/FP16/human-pose-estimation-0005.bin"
        if xml not in paths or binary not in paths:
            raise RuntimeError("pose 0005 artifact set is incomplete")
        core = ov.Core()
        model = core.read_model(model=str(paths[xml]), weights=str(paths[binary]))
        self.compiled = core.compile_model(model, device)
        inputs = tuple(self.compiled.inputs)
        if len(inputs) != 1 or tuple(int(v) for v in inputs[0].shape) != (1, 3, 288, 288):
            raise RuntimeError("pose 0005 input shape does not match reviewed artifact")
        heatmaps = []
        embeddings = []
        for port in self.compiled.outputs:
            shape = tuple(int(v) for v in port.shape)
            if shape == (1, 17, 144, 144):
                heatmaps.append(port)
            elif shape == (1, 17, 144, 144, 1):
                embeddings.append(port)
        if len(heatmaps) != 1 or len(embeddings) != 1:
            raise RuntimeError("pose 0005 output contract changed")
        self.heatmap_port = heatmaps[0]
        self.embedding_port = embeddings[0]

    def infer(self, image: Any) -> tuple[Any, Any, int, int]:
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) != 3 or int(shape[2]) != 3:
            raise ValueError("pose 0005 input must be a three-channel BGR image")
        height, width = int(shape[0]), int(shape[1])
        if height < 1 or width < 1:
            raise ValueError("pose 0005 input dimensions must be positive")
        scale = min(_INPUT / width, _INPUT / height)
        resized_width = max(1, min(_INPUT, int(round(width * scale))))
        resized_height = max(1, min(_INPUT, int(round(height * scale))))
        resized = self.cv2.resize(
            image, (resized_width, resized_height), interpolation=self.cv2.INTER_LINEAR
        )
        padded = self.np.pad(
            resized,
            ((0, _INPUT - resized_height), (0, _INPUT - resized_width), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        blob = self.np.ascontiguousarray(
            padded.transpose((2, 0, 1))[None], dtype=self.np.float32
        )
        results = self.compiled([blob])
        heatmaps = self.np.asarray(results[self.heatmap_port])
        embeddings = self.np.asarray(results[self.embedding_port])
        if tuple(heatmaps.shape) != (1, 17, 144, 144):
            raise RuntimeError("pose 0005 heatmap output shape changed")
        if tuple(embeddings.shape) != (1, 17, 144, 144, 1):
            raise RuntimeError("pose 0005 embedding output shape changed")
        return heatmaps, embeddings, resized_width, resized_height


class _Pose0005Decoder:
    """AE decoder adapted to the retained geometry adapter's 8x output scale."""

    def __init__(self) -> None:
        self.decoder = AssociativeEmbeddingDecoder()

    def __call__(self, heatmaps: Any, embeddings: Any) -> tuple[Any, Any]:
        poses, scores = self.decoder(heatmaps, embeddings, nms_heatmaps=heatmaps)
        poses = poses.copy()
        if poses.size:
            # Retained geometry maps 0001 coordinates with an 8x output stride.
            # Pose 0005 has a 2x stride, so scaling decoded x/y by 2/8 keeps the
            # existing source-pixel mapping and every downstream association rule unchanged.
            poses[..., :2] *= _OUTPUT_SCALE_COMPATIBILITY
        return poses, scores


def _aggregate(results: list[dict[str, Any]], evaluated: list[EvaluationSample]) -> dict[str, Any]:
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


def _run_path(samples: tuple[Any, ...], detector: _CompiledDetector, runtime: Any, decoder: Any) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    evaluated: list[EvaluationSample] = []
    for sample in samples:
        result, sample_evaluation = _scan_sample(sample, detector, runtime, decoder)
        results.append(result)
        evaluated.append(sample_evaluation)
    return _aggregate(results, evaluated)


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

    candidate_verified = _provision_pose_0005(candidate_root / "pose-0005")
    candidate_runtime = _Pose0005Runtime(candidate_verified, "CPU")
    if not candidate_runtime.runtime_version.startswith(_RUNTIME_PREFIX):
        raise RuntimeError("candidate OpenVINO runtime does not match reviewed release")
    candidate_decoder = _Pose0005Decoder()
    preparation_elapsed_ms = (time.perf_counter() - preparation_started) * 1000.0

    baseline = _run_path(samples, detector, current_runtime, current_decoder)
    candidate = _run_path(samples, detector, candidate_runtime, candidate_decoder)
    return {
        "schema_version": 1,
        "diagnostic": "person_down_pose_model_ab",
        "comparison": {
            "baseline_pose_model": "human-pose-estimation-0001",
            "candidate_pose_model": "human-pose-estimation-0005",
            "omz_commit": _OMZ_COMMIT,
            "candidate_decoder": "associative_embedding",
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
        print(json.dumps(
            run_pose_ab(args.manifest, args.candidate_dir),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError):
        print("Person-down pose A/B diagnostic rejected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
