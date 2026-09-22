from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any

_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_CANVAS_W = 32
_CANVAS_H = 48
_MIN_MATCH_SCORE = 0.30
_MAX_COMPONENTS = 24
_FONT_IDS = (0, 2, 3, 4)
_THICKNESSES = (1, 2, 3)
_ANGLES = (-4.0, 0.0, 4.0)
_STRETCHES = (0.85, 1.0, 1.15)


@dataclass(frozen=True)
class GlyphOCRResult:
    text: str
    mean_score: float
    accepted_components: int
    candidate_components: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or len(self.text) > 16 or any(c not in _ALPHABET for c in self.text):
            raise ValueError("glyph OCR text must be bounded uppercase alphanumeric")
        if not math.isfinite(float(self.mean_score)) or not 0.0 <= float(self.mean_score) <= 1.0:
            raise ValueError("glyph OCR mean_score must be in [0, 1]")
        for name in ("accepted_components", "candidate_components"):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > _MAX_COMPONENTS:
                raise ValueError(name + " is outside bounded component count")
        if self.accepted_components > self.candidate_components:
            raise ValueError("accepted components cannot exceed candidates")


def _require_rgb(image: Any) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) != 3 or int(shape[2]) != 3:
        raise ValueError("image must be HxWx3 RGB")
    h, w = int(shape[0]), int(shape[1])
    if h < 16 or w < 16 or h > 4096 or w > 8192:
        raise ValueError("image dimensions outside glyph OCR bounds")
    return w, h


def _normalise(mask: Any) -> Any:
    import cv2
    import numpy as np

    if mask is None or getattr(mask, "ndim", 0) != 2:
        raise ValueError("glyph mask must be 2D")
    points = cv2.findNonZero(mask)
    if points is None:
        return np.zeros((_CANVAS_H, _CANVAS_W), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(points)
    crop = mask[y:y + h, x:x + w]
    inner_w, inner_h = _CANVAS_W - 6, _CANVAS_H - 6
    scale = min(inner_w / max(1, w), inner_h / max(1, h))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((_CANVAS_H, _CANVAS_W), dtype=np.uint8)
    left = (_CANVAS_W - out_w) // 2
    top = (_CANVAS_H - out_h) // 2
    canvas[top:top + out_h, left:left + out_w] = resized
    return canvas


@lru_cache(maxsize=1)
def _template_bank() -> dict[str, tuple[Any, ...]]:
    import cv2
    import numpy as np

    bank: dict[str, list[Any]] = {char: [] for char in _ALPHABET}
    for char in _ALPHABET:
        for font in _FONT_IDS:
            for thickness in _THICKNESSES:
                base = np.zeros((80, 80), dtype=np.uint8)
                scale = 1.8
                (tw, th), baseline = cv2.getTextSize(char, font, scale, thickness)
                x = max(1, (80 - tw) // 2)
                y = max(th + 1, (80 + th) // 2 - baseline // 2)
                cv2.putText(base, char, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
                _, base = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                for stretch in _STRETCHES:
                    stretched = cv2.resize(
                        base,
                        (max(8, int(round(80 * stretch))), 80),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for angle in _ANGLES:
                        h, w = stretched.shape
                        center = (w / 2.0, h / 2.0)
                        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(
                            stretched, matrix, (w, h),
                            flags=cv2.INTER_LINEAR, borderValue=0,
                        )
                        _, rotated = cv2.threshold(
                            rotated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                        )
                        bank[char].append(_normalise(rotated))
    return {key: tuple(value) for key, value in bank.items()}


def _cosine(a: Any, b: Any) -> float:
    import numpy as np

    av = a.astype(np.float32).reshape(-1) / 255.0
    bv = b.astype(np.float32).reshape(-1) / 255.0
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(av, bv) / denom)))


def _segment(binary_inv: Any, width: int, height: int) -> list[tuple[int, int, int, int, Any]]:
    import cv2
    import numpy as np

    kernel = np.ones((2, 2), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    candidates: list[tuple[int, int, int, int, Any]] = []
    frame_area = float(width * height)
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if not (0.18 * height <= h <= 0.90 * height):
            continue
        if not (0.006 * width <= w <= 0.22 * width):
            continue
        if not (0.00045 * frame_area <= area <= 0.16 * frame_area):
            continue
        if y + h < 0.25 * height or y > 0.78 * height:
            continue
        mask = np.where(labels[y:y + h, x:x + w] == label, 255, 0).astype(np.uint8)
        candidates.append((x, y, w, h, mask))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (-item[3], -item[2] * item[3], item[0]))
    reference = float(np.median([c[3] for c in candidates[:min(12, len(candidates))]]))
    filtered = [c for c in candidates if 0.62 * reference <= c[3] <= 1.45 * reference]
    filtered.sort(key=lambda item: item[0])
    return filtered[:_MAX_COMPONENTS]


def recognize_plate_glyphs(
    rgb_image: Any, *, min_match_score: float = _MIN_MATCH_SCORE
) -> GlyphOCRResult:
    import cv2

    if not isinstance(min_match_score, (int, float)) or isinstance(min_match_score, bool):
        raise ValueError("min_match_score must be numeric")
    threshold = float(min_match_score)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_match_score must be in [0, 1]")
    width, height = _require_rgb(rgb_image)
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    _, binary_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    components = _segment(binary_inv, width, height)
    bank = _template_bank()
    output: list[str] = []
    scores: list[float] = []
    for _, _, _, _, mask in components:
        glyph = _normalise(mask)
        best_char = ""
        best_score = -1.0
        for char, templates in bank.items():
            score = max(_cosine(glyph, template) for template in templates)
            if score > best_score or (score == best_score and char < best_char):
                best_char, best_score = char, score
        if best_score >= threshold:
            output.append(best_char)
            scores.append(best_score)
        if len(output) >= 16:
            break
    mean = sum(scores) / len(scores) if scores else 0.0
    return GlyphOCRResult("".join(output), mean, len(output), len(components))
