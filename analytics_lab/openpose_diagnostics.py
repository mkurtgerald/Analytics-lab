"""Evidence-only full-frame OpenPose reference diagnostic.

Runs the pinned reviewed human-pose-estimation-0001 model with the upstream
Open Model Zoo preserve-aspect/full-frame preprocessing and the pinned NMS +
PAF grouping decoder. The continuity-selected detector box is used only to
associate one decoded pose to the measured person track. Media stays local and
only aggregate evidence is emitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

from .artifacts import OPENVINO_OMZ_2023_FP16, verify_artifact_set
from .detector_continuity import ContinuityAccumulator, DetectionBox
from .detector_shootout import _CompiledDetector, _sha256, _window
from .detector_thresholds import (
    _CONTINUITY_MIN_LINK_IOU,
    _CONTINUITY_THRESHOLD,
    _person_detections,
    _provision,
    _target_candidate,
)
from .openpose_reference import DecodedReferencePose, ReferenceOpenPoseDecoder, select_reference_pose
from .openvino_omz import _OpenVINORuntime
from .perception import BBox, PostureConfig, classify_posture
from .validation import ValidationSampleSpec
from .validation_cli import load_manifest
from .video import OpenCVVideoFileSource

_RUNTIME_PREFIX = "2026.3.1"
_INPUT_HEIGHT = 256
_INPUT_WIDTH = 456
_KEYPOINT_FLOOR = 0.10


def _pixel_bbox(box: DetectionBox, image: Any) -> BBox | None:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("image shape is required")
    height, width = int(shape[0]), int(shape[1])
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    left = min(1.0, max(0.0, float(box.x_min)))
    top = min(1.0, max(0.0, float(box.y_min)))
    right = min(1.0, max(0.0, float(box.x_max)))
    bottom = min(1.0, max(0.0, float(box.y_max)))
    if right <= left or bottom <= top:
        return None
    return BBox(left * width, top * height, right * width, bottom * height)


class _ReferenceRuntime:
    """Read-only diagnostic view over the already-verified OpenVINO runtime."""

    def __init__(self, runtime: _OpenVINORuntime) -> None:
        self.runtime = runtime
        outputs = tuple(runtime._pose.outputs)  # diagnostic-only internal adapter
        paf = [port for port in outputs if tuple(int(v) for v in port.shape)[1:] == (38, 32, 57)]
        if len(paf) != 1:
            raise RuntimeError("pose model must expose exactly one reviewed PAF output")
        self.paf_port = paf[0]

    def infer(self, image: Any) -> tuple[Any, Any, int, int]:
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) != 3 or int(shape[2]) != 3:
            raise ValueError("reference OpenPose input must be a three-channel BGR image")
        height, width = int(shape[0]), int(shape[1])
        if height < 1 or width < 1:
            raise ValueError("reference OpenPose input dimensions must be positive")
        scale = _INPUT_HEIGHT / height
        resized_width = max(1, int(round(width * scale)))
        if resized_width > _INPUT_WIDTH:
            raise RuntimeError("source aspect ratio exceeds reviewed OpenPose input")
        cv2 = self.runtime._cv2
        np = self.runtime._np
        resized = cv2.resize(image, (resized_width, _INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        padded = np.pad(
            resized,
            ((0, 0), (0, _INPUT_WIDTH - resized_width), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        blob = np.ascontiguousarray(padded.transpose((2, 0, 1))[None], dtype=np.float32)
        results = self.runtime._pose([blob])
        heatmaps = np.asarray(results[self.runtime._pose_heatmap_port])
        pafs = np.asarray(results[self.paf_port])
        if tuple(heatmaps.shape) != (1, 19, 32, 57):
            raise RuntimeError("pose heatmap output shape changed")
        if tuple(pafs.shape) != (1, 38, 32, 57):
            raise RuntimeError("pose PAF output shape changed")
        return heatmaps, pafs, resized_width, _INPUT_HEIGHT


class _Accumulator:
    def __init__(self) -> None:
        self.frames = 0
        self.selected_frames = 0
        self.inferred_frames = 0
        self.frames_with_decoded_pose = 0
        self.total_decoded_poses = 0
        self.multi_pose_frames = 0
        self.associated_frames = 0
        self.postures: Counter[str] = Counter()
        self.bases: Counter[str] = Counter()
        self.required_points: Counter[int] = Counter()
        self.required_points_inside: Counter[int] = Counter()

    def add(
        self,
        *,
        selected: DetectionBox | None,
        decoded_count: int | None = None,
        associated: DecodedReferencePose | None = None,
    ) -> None:
        self.frames += 1
        if selected is None:
            return
        self.selected_frames += 1
        if decoded_count is None:
            return
        self.inferred_frames += 1
        self.total_decoded_poses += decoded_count
        if decoded_count > 0:
            self.frames_with_decoded_pose += 1
        if decoded_count > 1:
            self.multi_pose_frames += 1
        if associated is None:
            return
        self.associated_frames += 1
        self.required_points[associated.required_points] += 1
        self.required_points_inside[associated.required_points_inside_selection] += 1
        posture = classify_posture(
            associated.candidate,
            PostureConfig(min_pose_confidence=0.0, min_keypoint_confidence=_KEYPOINT_FLOOR),
        )
        self.postures[posture.posture] += 1
        self.bases[posture.basis] += 1

    def freeze(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "selected_frames": self.selected_frames,
            "selection_coverage": self.selected_frames / self.frames if self.frames else 0.0,
            "inferred_frames": self.inferred_frames,
            "frames_with_decoded_pose": self.frames_with_decoded_pose,
            "total_decoded_poses": self.total_decoded_poses,
            "multi_pose_frames": self.multi_pose_frames,
            "associated_frames": self.associated_frames,
            "association_fraction": self.associated_frames / self.inferred_frames if self.inferred_frames else 0.0,
            "posture_counts": dict(sorted(self.postures.items())),
            "posture_fractions_of_inferred": {
                posture: self.postures[posture] / self.inferred_frames if self.inferred_frames else 0.0
                for posture in ("upright", "down", "other", "unknown")
            },
            "classification_basis_counts": dict(sorted(self.bases.items())),
            "required_points_histogram": {str(k): v for k, v in sorted(self.required_points.items())},
            "required_points_inside_selection_histogram": {
                str(k): v for k, v in sorted(self.required_points_inside.items())
            },
        }


def _scan_sample(
    sample: ValidationSampleSpec,
    detector: _CompiledDetector,
    runtime: _ReferenceRuntime,
    decoder: ReferenceOpenPoseDecoder,
) -> tuple[dict[str, Any], float, float]:
    path = sample.video_path
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("OpenPose diagnostic media identity mismatch")
    continuity = ContinuityAccumulator(min_link_iou=_CONTINUITY_MIN_LINK_IOU)
    overall = _Accumulator()
    windows = {name: _Accumulator() for name in ("before", "during", "after")}
    inference_ms = 0.0
    decode_ms = 0.0
    with OpenCVVideoFileSource(path, start_timestamp_ms=sample.start_timestamp_ms) as frames:
        for frame in frames:
            detections = _person_detections(detector, frame.image)
            selected = continuity.add(
                item for item in detections if float(item.confidence) >= _CONTINUITY_THRESHOLD
            )
            decoded_count = None
            associated = None
            if selected is not None:
                bbox = _pixel_bbox(selected, frame.image)
                if bbox is not None:
                    started = time.perf_counter()
                    heatmaps, pafs, resized_width, resized_height = runtime.infer(frame.image)
                    inference_ms += (time.perf_counter() - started) * 1000.0
                    started = time.perf_counter()
                    poses, scores = decoder(heatmaps, pafs)
                    shape = frame.image.shape
                    associated = select_reference_pose(
                        poses,
                        scores,
                        selection_bbox=bbox,
                        frame_width=int(shape[1]),
                        frame_height=int(shape[0]),
                        resized_width=resized_width,
                        resized_height=resized_height,
                    )
                    decode_ms += (time.perf_counter() - started) * 1000.0
                    decoded_count = len(poses)
            overall.add(selected=selected, decoded_count=decoded_count, associated=associated)
            window_name = _window(sample, frame.timestamp_ms)
            if window_name is not None:
                windows[window_name].add(
                    selected=selected, decoded_count=decoded_count, associated=associated
                )
    if path.is_symlink() or not path.is_file() or _sha256(path) != sample.media_sha256:
        raise RuntimeError("OpenPose diagnostic media changed during execution")
    result: dict[str, Any] = {"sample_id": sample.sample_id, "overall": overall.freeze()}
    if len(sample.labels) == 1:
        result["windows"] = {name: accumulator.freeze() for name, accumulator in windows.items()}
    return result, inference_ms, decode_ms


def run_reference_diagnostic(manifest: str | Path, candidate_root: str | Path) -> dict[str, Any]:
    artifact_root, samples, config = load_manifest(manifest)
    if config.required_device != "CPU" or len(samples) != 2:
        raise RuntimeError("OpenPose diagnostics require the bounded two-clip CPU seed")
    candidate = _target_candidate()
    candidate_dir = Path(candidate_root)
    _provision(candidate_dir, candidate)
    detector = _CompiledDetector(candidate, candidate_dir)
    verified = verify_artifact_set(artifact_root, OPENVINO_OMZ_2023_FP16)
    base_runtime = _OpenVINORuntime(verified, "CPU")
    if not base_runtime.runtime_version.startswith(_RUNTIME_PREFIX):
        raise RuntimeError("OpenVINO runtime version does not match reviewed release")
    runtime = _ReferenceRuntime(base_runtime)
    decoder = ReferenceOpenPoseDecoder()
    results = []
    inference_ms = 0.0
    decode_ms = 0.0
    for sample in samples:
        item, item_inference_ms, item_decode_ms = _scan_sample(sample, detector, runtime, decoder)
        results.append(item)
        inference_ms += item_inference_ms
        decode_ms += item_decode_ms
    inferred_frames = sum(int(item["overall"]["inferred_frames"]) for item in results)
    return {
        "schema_version": 1,
        "detector": candidate.name,
        "detector_threshold": _CONTINUITY_THRESHOLD,
        "continuity_min_link_iou": _CONTINUITY_MIN_LINK_IOU,
        "pose_model": "human-pose-estimation-0001",
        "decoder": {
            "source": "Open Model Zoo demos/common/python/model_zoo/model_api/models/open_pose.py",
            "commit": "6697dead54ed1cdd664b0313189c2cb52ee6335e",
            "preprocess": "full_frame_preserve_aspect_pad_right",
            "keypoint_nms": "3x3_max_pool_equivalent",
            "grouping": "part_affinity_fields",
            "keypoint_score_threshold": 0.1,
            "posture_pose_confidence_gate_neutralized": True,
            "posture_required_keypoint_floor": _KEYPOINT_FLOOR,
        },
        "samples": results,
        "inference_ms": inference_ms,
        "decode_ms": decode_ms,
        "inferred_frames": inferred_frames,
        "pose_model_fps": inferred_frames * 1000.0 / inference_ms if inference_ms > 0 else 0.0,
        "evidence_only": True,
        "commercial_accuracy_claim": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(
            run_reference_diagnostic(args.manifest, args.candidate_dir),
            allow_nan=False, sort_keys=True, separators=(",", ":"),
        ))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError):
        print("OpenPose diagnostic rejected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
