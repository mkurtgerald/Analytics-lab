"""Evidence-only continuity selection for low-confidence person detections.

This module does not change the production candidate detector. It provides a
small deterministic selector used to test whether person-detection-0200's
measured sub-threshold prone-person signal can be recovered without accepting
multiple boxes per frame. The selector prefers spatial continuity when a prior
box exists and otherwise reacquires the highest-confidence person box.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class DetectionBox:
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.confidence, self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise ValueError("detection values must be finite numbers")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if not float(self.x_max) > float(self.x_min) or not float(self.y_max) > float(self.y_min):
            raise ValueError("detection box must have positive area")


def box_iou(left: DetectionBox, right: DetectionBox) -> float:
    inter_left = max(float(left.x_min), float(right.x_min))
    inter_top = max(float(left.y_min), float(right.y_min))
    inter_right = min(float(left.x_max), float(right.x_max))
    inter_bottom = min(float(left.y_max), float(right.y_max))
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0
    intersection = (inter_right - inter_left) * (inter_bottom - inter_top)
    left_area = (float(left.x_max) - float(left.x_min)) * (float(left.y_max) - float(left.y_min))
    right_area = (float(right.x_max) - float(right.x_min)) * (float(right.y_max) - float(right.y_min))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def select_detection(
    previous: DetectionBox | None,
    candidates: Iterable[DetectionBox],
    *,
    min_link_iou: float = 0.05,
) -> tuple[DetectionBox | None, bool]:
    """Select one box and report whether it linked spatially to the prior box.

    When a prior box exists, any candidate clearing ``min_link_iou`` is ranked
    first by IoU and then confidence. If none clears the continuity floor, the
    highest-confidence candidate is returned as a reacquisition and ``linked``
    is false. With no prior box, the highest-confidence candidate starts a new
    track and is not considered linked.
    """
    if not 0.0 <= float(min_link_iou) <= 1.0:
        raise ValueError("min_link_iou must be within [0, 1]")
    options = tuple(candidates)
    if not options:
        return None, False
    if previous is None:
        return max(options, key=lambda item: float(item.confidence)), False
    scored = [(box_iou(previous, item), float(item.confidence), item) for item in options]
    linked = [item for item in scored if item[0] >= min_link_iou]
    if linked:
        _iou, _confidence, chosen = max(linked, key=lambda item: (item[0], item[1]))
        return chosen, True
    return max(options, key=lambda item: float(item.confidence)), False


@dataclass(frozen=True)
class ContinuityTotals:
    frames: int
    selected_frames: int
    transitions: int
    linked_transitions: int
    resets: int
    mean_confidence: float
    min_confidence: float | None

    @property
    def coverage(self) -> float:
        return self.selected_frames / self.frames if self.frames else 0.0

    @property
    def link_rate(self) -> float:
        return self.linked_transitions / self.transitions if self.transitions else 0.0

    @property
    def reset_rate(self) -> float:
        return self.resets / self.transitions if self.transitions else 0.0


class ContinuityAccumulator:
    """Accumulate one-box-per-frame continuity statistics without retaining media."""

    def __init__(self, *, min_link_iou: float = 0.05) -> None:
        if not 0.0 <= float(min_link_iou) <= 1.0:
            raise ValueError("min_link_iou must be within [0, 1]")
        self.min_link_iou = float(min_link_iou)
        self.previous: DetectionBox | None = None
        self.frames = 0
        self.selected_frames = 0
        self.transitions = 0
        self.linked_transitions = 0
        self.resets = 0
        self._confidence_sum = 0.0
        self._min_confidence: float | None = None

    def add(self, candidates: Iterable[DetectionBox]) -> DetectionBox | None:
        options = tuple(candidates)
        prior = self.previous
        selected, linked = select_detection(prior, options, min_link_iou=self.min_link_iou)
        self.frames += 1
        if selected is None:
            self.previous = None
            return None
        self.selected_frames += 1
        confidence = float(selected.confidence)
        self._confidence_sum += confidence
        self._min_confidence = confidence if self._min_confidence is None else min(self._min_confidence, confidence)
        if prior is not None:
            self.transitions += 1
            if linked:
                self.linked_transitions += 1
            else:
                self.resets += 1
        self.previous = selected
        return selected

    def freeze(self) -> ContinuityTotals:
        mean = self._confidence_sum / self.selected_frames if self.selected_frames else 0.0
        return ContinuityTotals(
            frames=self.frames,
            selected_frames=self.selected_frames,
            transitions=self.transitions,
            linked_transitions=self.linked_transitions,
            resets=self.resets,
            mean_confidence=mean,
            min_confidence=self._min_confidence,
        )
