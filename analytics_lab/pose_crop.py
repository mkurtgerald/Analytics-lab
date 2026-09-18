"""Bounded selected-person crop planning for evidence-only pose recovery.

The crop does not change detector identity, pose-model identity, posture rules or
temporal policy. It is intended only as a second view of the same already-
selected person when full-frame pose association fails.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

from .perception import BBox

DEFAULT_PADDING_FRACTION = 0.20
OPENPOSE_MAX_ASPECT_RATIO = 456.0 / 256.0


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if allow_zero:
        if result < 0:
            raise ValueError(f"{name} must be nonnegative")
    elif result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class PoseCropPlan:
    left: int
    top: int
    right: int
    bottom: int
    pad_top: int
    pad_bottom: int
    selection_bbox: BBox

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom", "pad_top", "pad_bottom"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("crop bounds must have positive area")
        if not isinstance(self.selection_bbox, BBox):
            raise ValueError("selection_bbox must be a BBox")

    @property
    def output_width(self) -> int:
        return self.right - self.left

    @property
    def output_height(self) -> int:
        return (self.bottom - self.top) + self.pad_top + self.pad_bottom


def plan_person_crop(
    selection_bbox: BBox,
    *,
    frame_width: int,
    frame_height: int,
    padding_fraction: float = DEFAULT_PADDING_FRACTION,
    max_aspect_ratio: float = OPENPOSE_MAX_ASPECT_RATIO,
) -> PoseCropPlan:
    """Plan a bounded crop around one selected detector box.

    The box is expanded by a fixed fraction for limb/context margin. If a wide
    crop would exceed the reviewed OpenPose input aspect ratio, the plan first
    consumes available source-frame vertical context, then adds only the
    minimum symmetric black padding needed to stay inside that ratio. No image
    resampling or geometric distortion is introduced here.
    """
    if not isinstance(selection_bbox, BBox):
        raise ValueError("selection_bbox must be a BBox")
    frame_width = _positive_int(frame_width, "frame_width")
    frame_height = _positive_int(frame_height, "frame_height")
    padding_fraction = _finite_positive(
        padding_fraction, "padding_fraction", allow_zero=True
    )
    max_aspect_ratio = _finite_positive(max_aspect_ratio, "max_aspect_ratio")

    x1 = min(float(frame_width), max(0.0, float(selection_bbox.x1)))
    y1 = min(float(frame_height), max(0.0, float(selection_bbox.y1)))
    x2 = min(float(frame_width), max(0.0, float(selection_bbox.x2)))
    y2 = min(float(frame_height), max(0.0, float(selection_bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("selection bbox must overlap the source frame")

    width = x2 - x1
    height = y2 - y1
    pad_x = width * padding_fraction
    pad_y = height * padding_fraction

    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(frame_width, int(math.ceil(x2 + pad_x)))
    bottom = min(frame_height, int(math.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        raise ValueError("planned crop has no source pixels")

    crop_width = right - left
    crop_height = bottom - top
    target_height = int(math.ceil(crop_width / max_aspect_ratio))
    if target_height > crop_height:
        needed = target_height - crop_height
        grow_top = min(top, needed // 2)
        top -= grow_top
        needed -= grow_top
        grow_bottom = min(frame_height - bottom, needed)
        bottom += grow_bottom
        needed -= grow_bottom
        if needed:
            grow_top = min(top, needed)
            top -= grow_top
            needed -= grow_top
        source_height = bottom - top
        remaining = max(0, target_height - source_height)
    else:
        remaining = 0

    pad_top = remaining // 2
    pad_bottom = remaining - pad_top
    mapped = BBox(
        x1 - left,
        y1 - top + pad_top,
        x2 - left,
        y2 - top + pad_top,
    )
    plan = PoseCropPlan(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        selection_bbox=mapped,
    )
    if plan.output_width / plan.output_height > max_aspect_ratio + 1e-12:
        raise RuntimeError("planned crop exceeds reviewed OpenPose aspect ratio")
    return plan


def extract_person_crop(image: Any, plan: PoseCropPlan, np_module: Any) -> Any:
    """Extract one planned crop and apply only the plan's bounded black padding."""
    if not isinstance(plan, PoseCropPlan):
        raise ValueError("plan must be a PoseCropPlan")
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) != 3 or int(shape[2]) != 3:
        raise ValueError("source image must be a three-channel image")
    frame_height, frame_width = int(shape[0]), int(shape[1])
    if plan.right > frame_width or plan.bottom > frame_height:
        raise ValueError("crop plan exceeds source image bounds")
    crop = image[plan.top:plan.bottom, plan.left:plan.right]
    if plan.pad_top or plan.pad_bottom:
        crop = np_module.pad(
            crop,
            ((plan.pad_top, plan.pad_bottom), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    crop = np_module.ascontiguousarray(crop)
    crop_shape = getattr(crop, "shape", None)
    if crop_shape is None or tuple(int(v) for v in crop_shape[:2]) != (
        plan.output_height,
        plan.output_width,
    ):
        raise RuntimeError("extracted crop shape does not match plan")
    return crop
