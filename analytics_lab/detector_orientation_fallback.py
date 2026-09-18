"""Evidence-only orientation fallback for an upright-biased person detector.

The reviewed person-detection-0200 model is retained unchanged. When the normal
orientation produces no admitted person box and a prior continuity box exists,
this helper maps detections from bounded +/-90-degree views back to the source
frame and admits only boxes that still link spatially to that prior box.

This is not a second detector, training path, or production promotion. It is a
small measured test of whether orientation bias explains held-out prone-person
recall loss. Primary-orientation detections always take precedence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .detector_continuity import DetectionBox, box_iou, select_detection


_QUARTER_TURNS = (-1, 1)


def unrotate_detection(box: DetectionBox, quarter_turn: int) -> DetectionBox:
    """Map a normalized box from a ``numpy.rot90`` view to the source frame.

    ``quarter_turn=1`` is 90 degrees counter-clockwise and ``-1`` is clockwise.
    Only the two bounded quarter-turn views used by the evidence experiment are
    accepted.
    """
    if quarter_turn not in _QUARTER_TURNS:
        raise ValueError("quarter_turn must be -1 or 1")
    if quarter_turn == 1:
        # CCW view: x' = y, y' = 1 - x. Invert the rotated rectangle.
        return DetectionBox(
            confidence=box.confidence,
            x_min=1.0 - box.y_max,
            y_min=box.x_min,
            x_max=1.0 - box.y_min,
            y_max=box.x_max,
        )
    # CW view: x' = 1 - y, y' = x. Invert the rotated rectangle.
    return DetectionBox(
        confidence=box.confidence,
        x_min=box.y_min,
        y_min=1.0 - box.x_max,
        x_max=box.y_max,
        y_max=1.0 - box.x_min,
    )


@dataclass(frozen=True)
class OrientationFallbackSelection:
    selected: DetectionBox | None
    linked_candidate_count: int
    mapped_candidate_count: int


def select_orientation_fallback(
    previous: DetectionBox,
    rotated: Mapping[int, Iterable[DetectionBox]],
    *,
    min_confidence: float,
    min_link_iou: float,
) -> OrientationFallbackSelection:
    """Select one mapped rotated-view box only when it links to ``previous``.

    Both quarter-turn views may propose boxes. Every proposal is mapped back to
    source coordinates, filtered by the existing confidence and continuity
    floors, then ranked by the same IoU-first selector as the normal path.
    Nothing is reacquired without a prior box; the caller only invokes this
    helper on a primary-orientation miss.
    """
    if not 0.0 <= float(min_confidence) <= 1.0:
        raise ValueError("min_confidence must be within [0, 1]")
    if not 0.0 <= float(min_link_iou) <= 1.0:
        raise ValueError("min_link_iou must be within [0, 1]")

    mapped: list[DetectionBox] = []
    for turn, boxes in rotated.items():
        if turn not in _QUARTER_TURNS:
            raise ValueError("rotated views must use only +/-1 quarter turns")
        for box in boxes:
            mapped_box = unrotate_detection(box, turn)
            if float(mapped_box.confidence) >= min_confidence:
                mapped.append(mapped_box)

    linked = tuple(box for box in mapped if box_iou(previous, box) >= min_link_iou)
    selected, is_linked = select_detection(previous, linked, min_link_iou=min_link_iou)
    if selected is not None and not is_linked:
        raise RuntimeError("orientation fallback selected an unlinked detection")
    return OrientationFallbackSelection(
        selected=selected,
        linked_candidate_count=len(linked),
        mapped_candidate_count=len(mapped),
    )
