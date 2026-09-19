"""Fail-closed exchange format for independently authored tracking ground truth.

The label authoring tool stays outside the public repository and source media
remains ephemeral. This module accepts only a complete, plan-bound JSON package
containing exhaustive labels plus canonical per-frame RGB24 digests. It does
not inspect tracker outputs and does not establish real-world accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .tracking import NormalizedBox
from .tracking_annotations import (
    TrackingGroundTruthFrame,
    canonical_tracking_annotations_sha256,
)
from .tracking_benchmark_plan import TrackingBenchmarkPlan
from .tracking_evaluation import GroundTruthObject
from .tracking_evidence import FrameDigest, canonical_frame_manifest_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COORDINATE_SPACE = "normalized_xyxy"
_COMPLETE_STATUS = "complete_exhaustive"
_TOP_LEVEL_KEYS = {
    "schema_version",
    "annotation_status",
    "benchmark_plan_sha256",
    "source_sha256",
    "coordinate_space",
    "frames",
}
_FRAME_KEYS = {"frame_index", "frame_sha256", "objects"}
_OBJECT_KEYS = {"object_id", "category", "box"}


@dataclass(frozen=True)
class TrackingGroundTruthPackage:
    """Validated labels and frame identities for one fixed benchmark plan."""

    frames: tuple[TrackingGroundTruthFrame, ...]
    frame_digests: tuple[FrameDigest, ...]
    annotation_sha256: str
    frame_manifest_sha256: str
    object_observations: int

    def __post_init__(self) -> None:
        if not isinstance(self.frames, tuple) or any(
            not isinstance(item, TrackingGroundTruthFrame) for item in self.frames
        ):
            raise ValueError("frames must be TrackingGroundTruthFrame values")
        if not isinstance(self.frame_digests, tuple) or any(
            not isinstance(item, FrameDigest) for item in self.frame_digests
        ):
            raise ValueError("frame_digests must be FrameDigest values")
        for name in ("annotation_sha256", "frame_manifest_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if type(self.object_observations) is not int or self.object_observations < 0:
            raise ValueError("object_observations must be a nonnegative integer")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must contain exactly the required fields")
    return value


def canonical_rgb24_frame_sha256(
    plan: TrackingBenchmarkPlan,
    frame_index: int,
    rgb24: bytes,
) -> str:
    """Hash one decoded RGB24 frame independently of image-container encoding."""

    if not isinstance(plan, TrackingBenchmarkPlan):
        raise ValueError("plan must be a TrackingBenchmarkPlan")
    if type(frame_index) is not int or frame_index not in plan.frame_indices:
        raise ValueError("frame_index is outside the benchmark plan")
    if not isinstance(rgb24, bytes):
        raise ValueError("rgb24 must be bytes")
    expected = plan.image_width * plan.image_height * 3
    if len(rgb24) != expected:
        raise ValueError("rgb24 byte length does not match benchmark dimensions")
    digest = hashlib.sha256()
    digest.update(b"analytics-lab-rgb24-frame-v1\n")
    digest.update(
        f"{frame_index}\n{plan.image_width}\n{plan.image_height}\n".encode("ascii")
    )
    digest.update(rgb24)
    return digest.hexdigest()


def tracking_ground_truth_template_bytes(plan: TrackingBenchmarkPlan) -> bytes:
    """Return a deterministic human-authoring template that cannot pass validation."""

    if not isinstance(plan, TrackingBenchmarkPlan):
        raise ValueError("plan must be a TrackingBenchmarkPlan")
    payload = {
        "schema_version": 1,
        "annotation_status": "incomplete",
        "benchmark_plan_sha256": plan.canonical_sha256(),
        "source_sha256": plan.source_sha256,
        "coordinate_space": _COORDINATE_SPACE,
        "frames": [
            {"frame_index": index, "frame_sha256": "", "objects": []}
            for index in plan.frame_indices
        ],
    }
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("ascii")


def parse_tracking_ground_truth_package(
    plan: TrackingBenchmarkPlan,
    payload: bytes | str,
    *,
    max_payload_bytes: int = 2_000_000,
    max_objects_per_frame: int = 512,
) -> TrackingGroundTruthPackage:
    """Validate one completed exhaustive annotation package and return its bindings."""

    if not isinstance(plan, TrackingBenchmarkPlan):
        raise ValueError("plan must be a TrackingBenchmarkPlan")
    for name, value in (
        ("max_payload_bytes", max_payload_bytes),
        ("max_objects_per_frame", max_objects_per_frame),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")

    if isinstance(payload, str):
        try:
            raw = payload.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("payload must be UTF-8") from exc
    elif isinstance(payload, bytes):
        raw = payload
    else:
        raise ValueError("payload must be bytes or str")
    if len(raw) > max_payload_bytes:
        raise RuntimeError("annotation package exceeds configured byte bound")

    try:
        text = raw.decode("utf-8")
        document = json.loads(text, object_pairs_hook=_json_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("annotation package must be valid UTF-8 JSON") from exc

    root = _exact_keys(document, _TOP_LEVEL_KEYS, "annotation package")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ValueError("unsupported annotation package schema version")
    if root["annotation_status"] != _COMPLETE_STATUS:
        raise ValueError("annotation package must be explicitly marked complete_exhaustive")
    if root["benchmark_plan_sha256"] != plan.canonical_sha256():
        raise ValueError("annotation package benchmark plan does not match")
    if root["source_sha256"] != plan.source_sha256:
        raise ValueError("annotation package source SHA-256 does not match")
    if root["coordinate_space"] != _COORDINATE_SPACE:
        raise ValueError("unsupported annotation coordinate space")

    raw_frames = root["frames"]
    if not isinstance(raw_frames, list) or len(raw_frames) != plan.frame_count:
        raise ValueError("annotation package must contain every benchmark frame exactly once")

    frames: list[TrackingGroundTruthFrame] = []
    frame_digests: list[FrameDigest] = []
    observations = 0

    for expected_index, raw_frame in zip(plan.frame_indices, raw_frames, strict=True):
        frame = _exact_keys(raw_frame, _FRAME_KEYS, "annotation frame")
        if type(frame["frame_index"]) is not int or frame["frame_index"] != expected_index:
            raise ValueError("annotation frames must exactly match benchmark frame order")
        frame_sha256 = frame["frame_sha256"]
        if not isinstance(frame_sha256, str) or not _SHA256.fullmatch(frame_sha256):
            raise ValueError("frame_sha256 must be a lowercase SHA-256 digest")

        raw_objects = frame["objects"]
        if not isinstance(raw_objects, list):
            raise ValueError("objects must be a JSON array")
        if len(raw_objects) > max_objects_per_frame:
            raise RuntimeError("annotation object count exceeds configured bound")

        objects: list[GroundTruthObject] = []
        for raw_object in raw_objects:
            item = _exact_keys(raw_object, _OBJECT_KEYS, "annotation object")
            box = item["box"]
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("box must be a four-value JSON array")
            objects.append(
                GroundTruthObject(
                    item["object_id"],
                    item["category"],
                    NormalizedBox(*box),
                )
            )
        observations += len(objects)
        frames.append(TrackingGroundTruthFrame(expected_index, tuple(objects)))
        frame_digests.append(FrameDigest(expected_index, frame_sha256))

    frame_values = tuple(frames)
    digest_values = tuple(frame_digests)
    annotation_sha256 = canonical_tracking_annotations_sha256(
        plan,
        frame_values,
        max_objects_per_frame=max_objects_per_frame,
    )
    frame_manifest_sha256 = canonical_frame_manifest_sha256(
        digest_values,
        max_frames=plan.frame_count,
    )
    return TrackingGroundTruthPackage(
        frame_values,
        digest_values,
        annotation_sha256,
        frame_manifest_sha256,
        observations,
    )