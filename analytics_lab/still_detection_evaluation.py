"""Offline box matching for independently labeled still images.

Inputs contain metadata only. Frame hashes bind caller-supplied labels and
predictions; this scorer cannot authenticate their provenance or label quality.
It measures raw detections, never tracker identity or background true negatives.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .tracking import DetectionCandidate, TrackedDetection
from .tracking_evaluation import (
    GroundTruthObject,
    TrackingEvaluationConfig,
    TrackingEvaluationFrame,
    evaluate_tracking,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class StillDetectionFrame:
    image_id: str
    frame_sha256: str
    ground_truth: tuple[GroundTruthObject, ...]
    detections: tuple[DetectionCandidate, ...]

    def __post_init__(self) -> None:
        if (type(self.image_id) is not str or not self.image_id.strip() or len(self.image_id) > 128
                or any(ord(character) < 32 for character in self.image_id)):
            raise ValueError("image_id must be bounded nonempty text")
        if type(self.frame_sha256) is not str or _SHA256.fullmatch(self.frame_sha256) is None:
            raise ValueError("frame_sha256 must be lowercase SHA-256")
        if type(self.ground_truth) is not tuple or any(type(item) is not GroundTruthObject for item in self.ground_truth):
            raise ValueError("ground_truth must be a tuple of GroundTruthObject values")
        if type(self.detections) is not tuple or any(type(item) is not DetectionCandidate for item in self.detections):
            raise ValueError("detections must be a tuple of raw DetectionCandidate values")
        if len(self.ground_truth) > 512 or len(self.detections) > 512:
            raise ValueError("still image exceeds the 512-object bound")
        if len({item.object_id for item in self.ground_truth}) != len(self.ground_truth):
            raise ValueError("duplicate ground-truth object_id within image")


@dataclass(frozen=True)
class StillDetectionImageResult:
    image_id: str
    frame_sha256: str
    ground_truth_count: int
    detection_count: int
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class StillDetectionResult:
    image_count: int
    ground_truth_count: int
    detection_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    false_detections_per_image: float | None
    min_iou: float
    per_image: tuple[StillDetectionImageResult, ...]


def evaluate_still_detections(
    frames: Iterable[StillDetectionFrame], *, min_iou: float = .5,
    max_images: int = 64, max_objects_per_image: int = 512,
) -> StillDetectionResult:
    """Match exact categories one-to-one at fixed IoU, independently per image.

    Every supplied detection is scored: no confidence filtering, deduplication
    or tracker suppression occurs here. Undefined ratios are None. There is no
    object false-positive rate because background true negatives are undefined.
    """
    if type(max_images) is not int or not 1 <= max_images <= 64:
        raise ValueError("max_images must be within [1, 64]")
    if type(max_objects_per_image) is not int or not 1 <= max_objects_per_image <= 512:
        raise ValueError("max_objects_per_image must be within [1, 512]")
    config = TrackingEvaluationConfig(min_iou=min_iou, max_frames=1, max_objects_per_frame=max_objects_per_image)
    try:
        iterator = iter(frames)
    except TypeError as exc:
        raise ValueError("frames must be iterable") from exc
    results = []
    image_ids, hashes = set(), set()
    for frame in iterator:
        if len(results) == max_images:
            raise RuntimeError("still image count exceeds configured bound")
        if type(frame) is not StillDetectionFrame:
            raise ValueError("frames must contain StillDetectionFrame values")
        if frame.image_id in image_ids or frame.frame_sha256 in hashes:
            raise ValueError("duplicate still image id or frame hash")
        image_ids.add(frame.image_id)
        hashes.add(frame.frame_sha256)
        # One evaluator call per image ensures labels and proposal ordinals
        # cannot imply associations between independent pictures.
        tracks = tuple(TrackedDetection(f"prediction-{index}", item.category, item.confidence,
                                        item.box, item.model_class_id)
                       for index, item in enumerate(frame.detections))
        metrics = evaluate_tracking((TrackingEvaluationFrame(0, frame.ground_truth, tracks),), config)
        results.append(StillDetectionImageResult(
            frame.image_id, frame.frame_sha256, len(frame.ground_truth), len(frame.detections),
            metrics.matched_observations, metrics.false_track_observations, metrics.misses,
        ))
    tp = sum(item.true_positives for item in results)
    fp = sum(item.false_positives for item in results)
    fn = sum(item.false_negatives for item in results)
    return StillDetectionResult(
        len(results), tp + fn, tp + fp, tp, fp, fn,
        tp / (tp + fp) if tp + fp else None,
        tp / (tp + fn) if tp + fn else None,
        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        fp / len(results) if results else None, config.min_iou, tuple(results),
    )
