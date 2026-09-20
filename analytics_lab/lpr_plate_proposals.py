"""Small, model-free plate proposal fallback for bounded LPR engineering evidence.

This module exists because the first reviewed OMZ plate detector missed an
independently pre-registered front-facing plate well inside its documented
geometry.  The fallback intentionally adds no trained weights, training data,
or second ML framework.  It uses only generic OpenCV image morphology already
present in the bounded evidence environment and returns proposal scores, not
probabilistic accuracy claims.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterable


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PlateProposalConfig:
    max_working_width: int = 960
    min_aspect_ratio: float = 2.0
    max_aspect_ratio: float = 6.5
    target_aspect_ratio: float = 4.2
    min_width_fraction: float = 0.03
    max_width_fraction: float = 0.80
    min_height_fraction: float = 0.01
    max_height_fraction: float = 0.25
    min_box_area_fraction: float = 0.0003
    max_box_area_fraction: float = 0.20
    min_rectangularity: float = 0.20
    nms_iou: float = 0.50
    max_plates: int = 16

    def __post_init__(self) -> None:
        if type(self.max_working_width) is not int or not 320 <= self.max_working_width <= 1920:
            raise ValueError("max_working_width must be an integer in [320, 1920]")
        if type(self.max_plates) is not int or not 1 <= self.max_plates <= 64:
            raise ValueError("max_plates must be an integer in [1, 64]")
        for name in (
            "min_aspect_ratio", "max_aspect_ratio", "target_aspect_ratio",
            "min_width_fraction", "max_width_fraction", "min_height_fraction",
            "max_height_fraction", "min_box_area_fraction", "max_box_area_fraction",
            "min_rectangularity", "nms_iou",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not 1.0 < self.min_aspect_ratio < self.target_aspect_ratio < self.max_aspect_ratio <= 10.0:
            raise ValueError("aspect-ratio bounds are invalid")
        for low, high, label in (
            (self.min_width_fraction, self.max_width_fraction, "width"),
            (self.min_height_fraction, self.max_height_fraction, "height"),
            (self.min_box_area_fraction, self.max_box_area_fraction, "area"),
        ):
            if not 0.0 < low < high <= 1.0:
                raise ValueError(f"{label} fraction bounds are invalid")
        if not 0.0 <= self.min_rectangularity <= 1.0:
            raise ValueError("min_rectangularity must be in [0, 1]")
        if not 0.0 < self.nms_iou < 1.0:
            raise ValueError("nms_iou must be in (0, 1)")


@dataclass(frozen=True)
class PlateProposal:
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = tuple(_finite(getattr(self, name), name) for name in ("score", "x1", "y1", "x2", "y2"))
        if not 0.0 <= values[0] <= 1.0:
            raise ValueError("proposal score must be in [0, 1]")
        if any(value < 0.0 or value > 1.0 for value in values[1:]):
            raise ValueError("proposal coordinates must be normalized")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("proposal rectangle must have positive area")


def _iou(left: PlateProposal, right: PlateProposal) -> float:
    ix1 = max(left.x1, right.x1)
    iy1 = max(left.y1, right.y1)
    ix2 = min(left.x2, right.x2)
    iy2 = min(left.y2, right.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    return inter / (left_area + right_area - inter)


def rank_plate_rectangles(
    rectangles: Iterable[tuple[int, int, int, int, float]],
    *,
    image_width: int,
    image_height: int,
    config: PlateProposalConfig | None = None,
) -> tuple[PlateProposal, ...]:
    """Filter/rank contour rectangles with frozen geometry-only heuristics.

    ``rectangles`` entries are ``(x, y, width, height, contour_area)`` in the
    working image.  The function is dependency-free and deterministic so its
    filtering semantics can be regression-tested on Linux and Windows without
    installing OpenCV.
    """
    cfg = config or PlateProposalConfig()
    if not isinstance(cfg, PlateProposalConfig):
        raise ValueError("config must be PlateProposalConfig")
    if type(image_width) is not int or type(image_height) is not int or image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive integers")
    proposals: list[PlateProposal] = []
    count = 0
    image_area = float(image_width * image_height)
    for item in rectangles:
        count += 1
        if count > 4096:
            raise RuntimeError("plate proposal rectangle limit exceeded")
        if not isinstance(item, tuple) or len(item) != 5:
            raise ValueError("rectangle must be (x, y, width, height, contour_area)")
        x, y, width, height, contour_area_raw = item
        if any(type(value) is not int for value in (x, y, width, height)):
            raise ValueError("rectangle geometry must use integers")
        contour_area = _finite(contour_area_raw, "contour_area")
        if x < 0 or y < 0 or width < 1 or height < 1 or x + width > image_width or y + height > image_height:
            continue
        if contour_area < 0.0:
            continue
        aspect = width / height
        width_fraction = width / image_width
        height_fraction = height / image_height
        box_area_fraction = (width * height) / image_area
        rectangularity = min(1.0, contour_area / float(width * height))
        if not cfg.min_aspect_ratio <= aspect <= cfg.max_aspect_ratio:
            continue
        if not cfg.min_width_fraction <= width_fraction <= cfg.max_width_fraction:
            continue
        if not cfg.min_height_fraction <= height_fraction <= cfg.max_height_fraction:
            continue
        if not cfg.min_box_area_fraction <= box_area_fraction <= cfg.max_box_area_fraction:
            continue
        if rectangularity < cfg.min_rectangularity:
            continue
        aspect_span = max(cfg.target_aspect_ratio - cfg.min_aspect_ratio, cfg.max_aspect_ratio - cfg.target_aspect_ratio)
        aspect_score = max(0.0, 1.0 - abs(aspect - cfg.target_aspect_ratio) / aspect_span)
        rectangularity_score = min(1.0, rectangularity / 0.80)
        size_score = min(1.0, width_fraction / 0.20)
        score = min(1.0, max(0.0, 0.45 * aspect_score + 0.35 * rectangularity_score + 0.20 * size_score))
        proposals.append(PlateProposal(
            score,
            x / image_width,
            y / image_height,
            (x + width) / image_width,
            (y + height) / image_height,
        ))
    proposals.sort(key=lambda item: (-item.score, item.x1, item.y1, item.x2, item.y2))
    kept: list[PlateProposal] = []
    for proposal in proposals:
        if any(_iou(proposal, prior) > cfg.nms_iou for prior in kept):
            continue
        kept.append(proposal)
        if len(kept) >= cfg.max_plates:
            break
    return tuple(kept)


class OpenCVPlateProposalDetector:
    """Model-free OpenCV morphology fallback with frozen pre-measurement defaults."""

    def __init__(self, *, config: PlateProposalConfig | None = None, cv2_module: Any | None = None) -> None:
        self.config = config or PlateProposalConfig()
        if not isinstance(self.config, PlateProposalConfig):
            raise ValueError("config must be PlateProposalConfig")
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as exc:
                raise RuntimeError("OpenCV is required for the plate proposal fallback") from exc
        version = str(getattr(cv2_module, "__version__", ""))
        if version != "4.12.0":
            raise RuntimeError("OpenCV runtime version does not match reviewed 4.12.0 release")
        self._cv2 = cv2_module
        self.runtime_version = version

    def __call__(self, image: Any) -> tuple[Any, ...]:
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) != 3 or int(shape[2]) != 3:
            raise ValueError("image must be HxWx3 BGR")
        height, width = int(shape[0]), int(shape[1])
        if width < 1 or height < 1:
            raise ValueError("image dimensions must be positive")
        cv2 = self._cv2
        scale = min(1.0, self.config.max_working_width / width)
        if scale < 1.0:
            working_width = self.config.max_working_width
            working_height = max(1, int(round(height * scale)))
            working = cv2.resize(image, (working_width, working_height), interpolation=cv2.INTER_AREA)
        else:
            working = image
            working_width, working_height = width, height
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gradient = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gradient = cv2.convertScaleAbs(gradient)
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours_result = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2]
        if len(contours) > 4096:
            raise RuntimeError("plate proposal contour limit exceeded")
        rectangles = []
        for contour in contours:
            x, y, rect_width, rect_height = (int(value) for value in cv2.boundingRect(contour))
            rectangles.append((x, y, rect_width, rect_height, float(cv2.contourArea(contour))))
        proposals = rank_plate_rectangles(
            rectangles,
            image_width=working_width,
            image_height=working_height,
            config=self.config,
        )
        from .lpr_ocr import PlateDetection
        return tuple(PlateDetection(item.score, item.x1, item.y1, item.x2, item.y2) for item in proposals)
