"""Evidence-only posture-quality diagnostics on the bounded staged-real seed.

This reuses the measured detector continuity, bounded pose association and pinned
OpenPose path. It adds aggregate posture-failure geometry only; no media, frame,
model artifact or identity data is emitted, and production perception is not
changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

from . import openpose_association_diagnostics as association
from .openpose_diagnostics import _Accumulator as _BaseAccumulator, _KEYPOINT_FLOOR
from .openpose_pose_geometry import AssociatedReferencePose
from .perception import PostureConfig, classify_posture

_REQUIRED = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
_POSTURE_CONFIG = PostureConfig(
    min_pose_confidence=0.0,
    min_keypoint_confidence=_KEYPOINT_FLOOR,
)
_METRICS = (
    "min_required_confidence",
    "torso_fraction",
    "vertical_fraction",
    "horizontal_fraction",
    "width_over_height",
    "height_over_width",
)


def _pose_metrics(item: AssociatedReferencePose) -> dict[str, float]:
    pose = item.candidate
    points = [pose.keypoint(name) for name in _REQUIRED]
    present = [point for point in points if point is not None]
    result: dict[str, float] = {}
    if present:
        result["min_required_confidence"] = min(point.confidence for point in present)
    if len(present) != len(_REQUIRED):
        return result
    ls, rs, lh, rh = present
    shoulder_x = (ls.x + rs.x) / 2.0
    shoulder_y = (ls.y + rs.y) / 2.0
    hip_x = (lh.x + rh.x) / 2.0
    hip_y = (lh.y + rh.y) / 2.0
    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y
    torso = math.hypot(dx, dy)
    diagonal = math.hypot(pose.bbox.width, pose.bbox.height)
    if torso > 0.0:
        result["vertical_fraction"] = abs(dy) / torso
        result["horizontal_fraction"] = abs(dx) / torso
    if diagonal > 0.0:
        result["torso_fraction"] = torso / diagonal
    result["width_over_height"] = pose.bbox.width / pose.bbox.height
    result["height_over_width"] = pose.bbox.height / pose.bbox.width
    return result


class _PostureQualityAccumulator(_BaseAccumulator):
    """Extend the existing aggregate evidence with posture-failure diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        self.unknown_bases: Counter[str] = Counter()
        self.unknown_required_points: Counter[int] = Counter()
        self.metrics_by_posture: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {name: [] for name in _METRICS}
        )

    def add(
        self,
        *,
        selected: Any,
        decoded_count: int | None = None,
        associated: AssociatedReferencePose | None = None,
    ) -> None:
        super().add(selected=selected, decoded_count=decoded_count, associated=associated)
        if associated is None:
            return
        result = classify_posture(associated.candidate, _POSTURE_CONFIG)
        if result.posture == "unknown":
            self.unknown_bases[result.basis] += 1
            self.unknown_required_points[associated.required_points] += 1
        metrics = _pose_metrics(associated)
        for name, value in metrics.items():
            self.metrics_by_posture[result.posture][name].append(float(value))

    def freeze(self) -> dict[str, Any]:
        result = super().freeze()
        result["posture_quality"] = {
            "unknown_frames": sum(self.unknown_bases.values()),
            "unknown_basis_counts": dict(sorted(self.unknown_bases.items())),
            "unknown_required_points_histogram": {
                str(key): value for key, value in sorted(self.unknown_required_points.items())
            },
            "metrics_by_posture": {
                posture: {
                    name: association._summary(values)
                    for name, values in metrics.items()
                }
                for posture, metrics in sorted(self.metrics_by_posture.items())
            },
        }
        return result


def run_posture_quality_diagnostic(manifest: str | Path, candidate_root: str | Path) -> dict[str, Any]:
    """Run the existing association path with richer aggregate posture evidence."""
    original = association._Accumulator
    association._Accumulator = _PostureQualityAccumulator
    try:
        result = association.run_association_diagnostic(manifest, candidate_root)
    finally:
        association._Accumulator = original
    result["schema_version"] = 3
    result["diagnostic"] = "openpose_posture_quality"
    result["posture_diagnostic"] = {
        "required_keypoints": list(_REQUIRED),
        "required_keypoint_floor": _KEYPOINT_FLOOR,
        "metrics": list(_METRICS),
        "production_promoted": False,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(
            run_posture_quality_diagnostic(args.manifest, args.candidate_dir),
            allow_nan=False, sort_keys=True, separators=(",", ":"),
        ))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError):
        print("OpenPose posture-quality diagnostic rejected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
