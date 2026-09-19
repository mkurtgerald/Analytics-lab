"""Pre-registered real-video tracking benchmark windows.

This module binds the exact source asset and measured frame window *before*
benchmark detector/tracker outputs are inspected. It prevents silent window
cherry-picking while keeping source media outside GitHub.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import re

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TrackingBenchmarkPlan:
    source_id: str
    source_sha256: str
    start_frame: int
    frame_count: int
    fps_numerator: int
    fps_denominator: int
    image_width: int
    image_height: int
    object_class: str
    purpose: str
    selection_basis: str

    def __post_init__(self) -> None:
        for name, maximum in (
            ("source_id", 240),
            ("object_class", 64),
            ("purpose", 160),
            ("selection_basis", 512),
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            value = value.strip()
            if not value or len(value) > maximum:
                raise ValueError(f"{name} must contain 1-{maximum} non-whitespace characters")
            object.__setattr__(self, name, value)
        if not isinstance(self.source_sha256, str) or not _SHA256.fullmatch(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")
        for name in (
            "start_frame",
            "frame_count",
            "fps_numerator",
            "fps_denominator",
            "image_width",
            "image_height",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.start_frame < 0:
            raise ValueError("start_frame must be nonnegative")
        if not 1 <= self.frame_count <= 500:
            raise ValueError("frame_count must be between 1 and 500")
        if self.start_frame + self.frame_count > 100_000:
            raise RuntimeError("benchmark window exceeds configured frame bound")
        if self.fps_numerator < 1 or self.fps_denominator < 1:
            raise ValueError("frame rate components must be positive")
        if self.image_width < 1 or self.image_height < 1:
            raise ValueError("image dimensions must be positive")
        if self.image_width > 32_768 or self.image_height > 32_768:
            raise RuntimeError("image dimensions exceed configured evidence bound")

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start_frame, self.start_frame + self.frame_count))

    @property
    def start_seconds(self) -> Fraction:
        return Fraction(self.start_frame * self.fps_denominator, self.fps_numerator)

    @property
    def last_frame_seconds(self) -> Fraction:
        return Fraction(
            (self.start_frame + self.frame_count - 1) * self.fps_denominator,
            self.fps_numerator,
        )

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


WIKIMEDIA_PEDESTRIAN_SOURCE_SHA256 = (
    "bfadaa62cccb42db875d50bb842aa0964fbf72040432e4097c1df59e043e0c26"
)


def wikimedia_pedestrian_smoke_plan() -> TrackingBenchmarkPlan:
    """Return the first fixed real-video tracker-comparison window.

    The one-second window (source frames 150-174 inclusive) was selected from
    the already admitted 25 fps CC0 source before inspecting any detector or
    tracker output. This is acceptance/smoke evidence only, never a commercial
    accuracy claim.
    """
    return TrackingBenchmarkPlan(
        source_id="wikimedia-cc0-pedestrian-area@oldid-1196486240",
        source_sha256=WIKIMEDIA_PEDESTRIAN_SOURCE_SHA256,
        start_frame=150,
        frame_count=25,
        fps_numerator=25,
        fps_denominator=1,
        image_width=1920,
        image_height=1080,
        object_class="person",
        purpose="first real-video simple-iou-vs-bytetrack smoke comparison",
        selection_basis=(
            "Fixed source frames 150-174 (6.00s through 6.96s at 25 fps) before "
            "benchmark detector/tracker output inspection; expand only after this "
            "pre-registered smoke comparison is measured."
        ),
    )
