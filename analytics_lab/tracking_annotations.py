"""Canonical binding for independently authored multi-object tracking labels.

The benchmark plan fixes *where* to measure before tracker output inspection.
This module fixes *what the ground truth says* after independent annotation,
without storing image/video bytes in GitHub. It is intentionally an evidence
integrity primitive, not an annotation tool and not an accuracy claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

from .tracking_benchmark_plan import TrackingBenchmarkPlan
from .tracking_evaluation import GroundTruthObject


@dataclass(frozen=True)
class TrackingGroundTruthFrame:
    """Exhaustive ground-truth objects for exactly one benchmark frame."""

    frame_index: int
    objects: tuple[GroundTruthObject, ...]

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if not isinstance(self.objects, tuple) or any(
            not isinstance(item, GroundTruthObject) for item in self.objects
        ):
            raise ValueError("objects must be a tuple of GroundTruthObject values")
        ids = [item.object_id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate object_id in one annotation frame")


def _box_hex(item: GroundTruthObject) -> tuple[str, str, str, str]:
    box = item.box
    return (
        box.x_min.hex(),
        box.y_min.hex(),
        box.x_max.hex(),
        box.y_max.hex(),
    )


def canonical_tracking_annotations_bytes(
    plan: TrackingBenchmarkPlan,
    frames: Iterable[TrackingGroundTruthFrame],
    *,
    max_objects_per_frame: int = 512,
    max_canonical_bytes: int = 2_000_000,
) -> bytes:
    """Return deterministic bytes for one complete pre-registered label set.

    Every frame in the benchmark plan must be present exactly once, including
    frames with zero measured objects. Object ordering is canonicalized by the
    dataset-local ``object_id`` because tuple order has no annotation meaning.
    Coordinates are encoded with ``float.hex`` so hashes are stable across
    JSON float formatting differences.
    """

    if not isinstance(plan, TrackingBenchmarkPlan):
        raise ValueError("plan must be a TrackingBenchmarkPlan")
    for name, value in (
        ("max_objects_per_frame", max_objects_per_frame),
        ("max_canonical_bytes", max_canonical_bytes),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")

    try:
        values = tuple(frames)
    except TypeError as exc:
        raise ValueError("frames must be iterable") from exc
    if any(not isinstance(frame, TrackingGroundTruthFrame) for frame in values):
        raise ValueError("frames must contain TrackingGroundTruthFrame values")

    expected = plan.frame_indices
    actual = tuple(frame.frame_index for frame in values)
    if actual != expected:
        raise ValueError("annotation frames must exactly match the pre-registered frame window")

    payload_frames: list[dict[str, object]] = []
    for frame in values:
        if len(frame.objects) > max_objects_per_frame:
            raise RuntimeError("annotation object count exceeds configured bound")
        if any(item.category != plan.object_class for item in frame.objects):
            raise ValueError("annotation object category must match the benchmark plan")
        ordered = sorted(frame.objects, key=lambda item: item.object_id)
        payload_frames.append(
            {
                "frame_index": frame.frame_index,
                "objects": [
                    {
                        "object_id": item.object_id,
                        "category": item.category,
                        "box_hex": _box_hex(item),
                    }
                    for item in ordered
                ],
            }
        )

    payload = {
        "schema_version": 1,
        "benchmark_plan_sha256": plan.canonical_sha256(),
        "source_sha256": plan.source_sha256,
        "object_class": plan.object_class,
        "frames": payload_frames,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if len(encoded) > max_canonical_bytes:
        raise RuntimeError("canonical annotation payload exceeds configured bound")
    return encoded


def canonical_tracking_annotations_sha256(
    plan: TrackingBenchmarkPlan,
    frames: Iterable[TrackingGroundTruthFrame],
    *,
    max_objects_per_frame: int = 512,
    max_canonical_bytes: int = 2_000_000,
) -> str:
    """Return the SHA-256 bound to the exact plan and exhaustive labels."""

    encoded = canonical_tracking_annotations_bytes(
        plan,
        frames,
        max_objects_per_frame=max_objects_per_frame,
        max_canonical_bytes=max_canonical_bytes,
    )
    return hashlib.sha256(encoded).hexdigest()
