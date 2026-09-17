"""Evidence-only full-frame pose geometry and detector-track association.

The pinned Open Model Zoo decoder returns source-independent COCO keypoints.
This adapter derives posture geometry from the decoded pose itself and uses the
continuity-selected detector box only to associate that pose to the measured
person track. It is deliberately diagnostic-only and does not alter the
production perception backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Sequence

from .perception import BBox, Keypoint, PoseCandidate

_REQUIRED_COCO = (
    ("left_shoulder", 5),
    ("right_shoulder", 6),
    ("left_hip", 11),
    ("right_hip", 12),
)
_OUTPUT_SCALE = 8.0


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bbox_iou(a: BBox, b: BBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class AssociatedReferencePose:
    candidate: PoseCandidate
    decoder_score: float
    valid_points: int
    all_points_inside_selection: int
    required_points: int
    required_points_inside_selection: int
    selection_iou: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, PoseCandidate):
            raise ValueError("candidate must be a PoseCandidate")
        object.__setattr__(self, "decoder_score", _finite(self.decoder_score, "decoder score"))
        object.__setattr__(self, "selection_iou", _finite(self.selection_iou, "selection IoU"))
        if self.decoder_score < 0:
            raise ValueError("decoder score must be nonnegative")
        if not 0 <= self.selection_iou <= 1:
            raise ValueError("selection IoU must be in [0, 1]")
        if type(self.valid_points) is not int or not 0 <= self.valid_points <= 17:
            raise ValueError("valid_points must be bounded")
        if type(self.all_points_inside_selection) is not int or not 0 <= self.all_points_inside_selection <= self.valid_points:
            raise ValueError("all_points_inside_selection must be bounded")
        if type(self.required_points) is not int or not 0 <= self.required_points <= 4:
            raise ValueError("required_points must be bounded")
        if type(self.required_points_inside_selection) is not int or not 0 <= self.required_points_inside_selection <= self.required_points:
            raise ValueError("required_points_inside_selection must be bounded")


def candidate_from_coco_pose(
    pose: Sequence[Sequence[float]],
    *,
    decoder_score: float,
    selection_bbox: BBox,
    frame_width: int,
    frame_height: int,
    resized_width: int,
    resized_height: int,
) -> AssociatedReferencePose:
    """Map a decoded pose to source pixels and derive its own posture bbox."""
    if not isinstance(selection_bbox, BBox):
        raise ValueError("selection_bbox must be a BBox")
    frame_width = _positive_int(frame_width, "frame_width")
    frame_height = _positive_int(frame_height, "frame_height")
    resized_width = _positive_int(resized_width, "resized_width")
    resized_height = _positive_int(resized_height, "resized_height")
    if len(pose) < 17:
        raise ValueError("decoded pose must contain 17 COCO keypoints")
    scale_x = (frame_width / resized_width) * _OUTPUT_SCALE
    scale_y = (frame_height / resized_height) * _OUTPUT_SCALE
    mapped: list[tuple[int, float, float, float]] = []
    for index in range(17):
        raw = pose[index]
        if len(raw) < 3:
            raise ValueError("decoded keypoint must contain x, y and confidence")
        x = _finite(raw[0], "decoded keypoint x")
        y = _finite(raw[1], "decoded keypoint y")
        confidence = _finite(raw[2], "decoded keypoint confidence")
        if confidence <= 0:
            continue
        px = min(float(frame_width), max(0.0, x * scale_x))
        py = min(float(frame_height), max(0.0, y * scale_y))
        mapped.append((index, px, py, min(1.0, confidence)))
    if len(mapped) < 2:
        raise ValueError("decoded pose must contain at least two valid keypoints")
    xs = [item[1] for item in mapped]
    ys = [item[2] for item in mapped]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("decoded pose keypoints do not span a valid bbox")
    pose_bbox = BBox(x1, y1, x2, y2)
    required_by_index = {index: name for name, index in _REQUIRED_COCO}
    keypoints: list[Keypoint] = []
    required_inside = 0
    all_inside = 0
    for index, px, py, confidence in mapped:
        inside = (
            selection_bbox.x1 <= px <= selection_bbox.x2
            and selection_bbox.y1 <= py <= selection_bbox.y2
        )
        if inside:
            all_inside += 1
        name = required_by_index.get(index)
        if name is None:
            continue
        keypoints.append(Keypoint(name, px, py, confidence))
        if inside:
            required_inside += 1
    return AssociatedReferencePose(
        candidate=PoseCandidate(pose_bbox, tuple(keypoints), 1.0),
        decoder_score=decoder_score,
        valid_points=len(mapped),
        all_points_inside_selection=all_inside,
        required_points=len(keypoints),
        required_points_inside_selection=required_inside,
        selection_iou=_bbox_iou(pose_bbox, selection_bbox),
    )


def select_reference_pose(
    poses: Sequence[Sequence[Sequence[float]]],
    scores: Sequence[float],
    *,
    selection_bbox: BBox,
    frame_width: int,
    frame_height: int,
    resized_width: int,
    resized_height: int,
) -> AssociatedReferencePose | None:
    """Associate the decoded pose that spatially overlaps the selected track."""
    if len(poses) != len(scores):
        raise ValueError("pose and score counts must match")
    candidates: list[AssociatedReferencePose] = []
    for pose, score in zip(poses, scores):
        try:
            item = candidate_from_coco_pose(
                pose,
                decoder_score=float(score),
                selection_bbox=selection_bbox,
                frame_width=frame_width,
                frame_height=frame_height,
                resized_width=resized_width,
                resized_height=resized_height,
            )
        except ValueError:
            continue
        if item.required_points > 0 and (
            item.selection_iou > 0.0 or item.all_points_inside_selection > 0
        ):
            candidates.append(item)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -item.selection_iou,
            -item.all_points_inside_selection,
            -item.required_points,
            -item.decoder_score,
        )
    )
    return candidates[0]
