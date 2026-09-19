"""Fail-closed adapter for the reviewed UVify/NCSOFT tracking labels.

The reviewed source is ``uvify-public/human_tracking_dataset`` pinned to
``eb3af0cfe49de018a0c4736581daadd8eb860883`` under CC-BY-4.0. This module
parses the dataset's comma-separated ground-truth rows into Analytics Lab's
platform-neutral tracking-evaluation contract.

The adapter deliberately maps ``tracking_id`` to ``GroundTruthObject.object_id``.
It does not expose the dataset's ``person_id`` as an identity primitive and does
not perform face recognition, re-identification, biometric matching, media
acquisition, or accuracy scoring by itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .tracking import NormalizedBox
from .tracking_evaluation import GroundTruthObject

UVIFY_REPOSITORY = "uvify-public/human_tracking_dataset"
UVIFY_REVISION = "eb3af0cfe49de018a0c4736581daadd8eb860883"
UVIFY_LICENSE = "CC-BY-4.0"


@dataclass(frozen=True)
class UvifyGroundTruthFrame:
    """One dataset frame containing valid human tracking annotations."""

    frame_index: int
    ground_truth: tuple[GroundTruthObject, ...]

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if not isinstance(self.ground_truth, tuple) or any(
            not isinstance(item, GroundTruthObject) for item in self.ground_truth
        ):
            raise ValueError("ground_truth must be a tuple of GroundTruthObject values")
        ids = [item.object_id for item in self.ground_truth]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate tracking_id in one frame")


def _integer(token: str, name: str, line_number: int) -> int:
    value = token.strip()
    if not value or not value.lstrip("-").isdigit():
        raise ValueError(f"line {line_number}: {name} must be an integer")
    return int(value)


def _finite_float(token: str, name: str, line_number: int) -> float:
    try:
        value = float(token.strip())
    except ValueError as exc:
        raise ValueError(f"line {line_number}: {name} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"line {line_number}: {name} must be finite")
    return value


def parse_uvify_ground_truth(
    text: str,
    image_width: int,
    image_height: int,
    *,
    max_rows: int = 250_000,
    max_objects_per_frame: int = 512,
) -> tuple[UvifyGroundTruthFrame, ...]:
    """Parse reviewed UVify labels into normalized person boxes.

    Expected columns, as published by the source repository, are::

        frame_number, person_id, tracking_id, x, y, width, height,
        is_valid, pose_class, occlusion, truncated, visibility

    Only rows with ``is_valid == 1`` are admitted. ``person_id`` is parsed only
    to validate the source row shape; it is intentionally not propagated.
    Boxes must lie fully inside the declared image dimensions. No clamping or
    silent repair is performed because evidence evaluation should fail closed on
    malformed labels.
    """

    if not isinstance(text, str):
        raise ValueError("text must be a string")
    for name, value in (("image_width", image_width), ("image_height", image_height)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    for name, value in (("max_rows", max_rows), ("max_objects_per_frame", max_objects_per_frame)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(raw_lines) > max_rows:
        raise RuntimeError("UVify label row count exceeds configured bound")

    grouped: dict[int, list[GroundTruthObject]] = {}
    for line_number, line in enumerate(raw_lines, start=1):
        tokens = [token.strip() for token in line.split(",")]
        if len(tokens) != 12:
            raise ValueError(f"line {line_number}: expected 12 comma-separated columns")

        frame_number = _integer(tokens[0], "frame_number", line_number)
        person_id = _integer(tokens[1], "person_id", line_number)
        tracking_id = _integer(tokens[2], "tracking_id", line_number)
        x = _integer(tokens[3], "box_top_left_x", line_number)
        y = _integer(tokens[4], "box_top_left_y", line_number)
        width = _integer(tokens[5], "box_width", line_number)
        height = _integer(tokens[6], "box_height", line_number)
        is_valid = _integer(tokens[7], "is_valid", line_number)
        pose_class = _integer(tokens[8], "pose_class", line_number)
        occlusion = _integer(tokens[9], "occlusion", line_number)
        truncated = _integer(tokens[10], "truncated", line_number)
        visibility = _finite_float(tokens[11], "visibility", line_number)

        if frame_number < 0 or person_id < 0 or tracking_id < 0:
            raise ValueError(f"line {line_number}: identifiers must be nonnegative")
        if is_valid not in (0, 1):
            raise ValueError(f"line {line_number}: is_valid must be 0 or 1")
        if pose_class not in (0, 1, 2):
            raise ValueError(f"line {line_number}: pose_class must be 0, 1, or 2")
        if occlusion not in (0, 1) or truncated not in (0, 1):
            raise ValueError(f"line {line_number}: occlusion/truncated must be 0 or 1")
        if not 0.0 <= visibility <= 1.0:
            raise ValueError(f"line {line_number}: visibility must be within [0, 1]")
        if width <= 0 or height <= 0 or x < 0 or y < 0:
            raise ValueError(f"line {line_number}: box coordinates must define positive in-frame area")
        if x + width > image_width or y + height > image_height:
            raise ValueError(f"line {line_number}: box extends outside declared image dimensions")

        if is_valid == 0:
            continue

        box = NormalizedBox(
            x / image_width,
            y / image_height,
            (x + width) / image_width,
            (y + height) / image_height,
        )
        object_id = f"uvify-track:{tracking_id}"
        values = grouped.setdefault(frame_number, [])
        if any(item.object_id == object_id for item in values):
            raise ValueError(f"line {line_number}: duplicate tracking_id in one frame")
        if len(values) >= max_objects_per_frame:
            raise RuntimeError("UVify object count exceeds configured per-frame bound")
        values.append(GroundTruthObject(object_id=object_id, category="person", box=box))

    if not grouped:
        raise ValueError("no valid UVify tracking annotations were admitted")

    return tuple(
        UvifyGroundTruthFrame(frame_index=frame_index, ground_truth=tuple(grouped[frame_index]))
        for frame_index in sorted(grouped)
    )
