"""Fail-closed admission metadata for real multi-object tracking evidence.

The tracking evaluator can only report false tracks, ID switches, fragmentation,
and continuity when the admitted labels are exhaustive for the measured object
class. A crowded video with only one target box is not multi-object ground
truth. This module encodes that boundary and cryptographically binds the exact
annotation and frame-manifest bytes without storing media in the repository.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXHAUSTIVE_MULTI_OBJECT = "exhaustive_multi_object"


def _text(value: str, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = value.strip()
    if not value or len(value) > maximum:
        raise ValueError(f"{name} must contain 1-{maximum} non-whitespace characters")
    return value


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class FrameDigest:
    frame_index: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))


def canonical_frame_manifest_sha256(frames: Iterable[FrameDigest], *, max_frames: int = 100_000) -> str:
    """Hash an exact, ordered list of admitted frame digests.

    The canonical bytes are ``<frame_index>\t<sha256>\n`` in strictly
    increasing frame order. Callers can keep image bytes ephemeral while the
    evidence record retains a stable cryptographic identity for the sequence.
    """
    if type(max_frames) is not int or max_frames < 1:
        raise ValueError("max_frames must be an integer >= 1")
    try:
        values = tuple(frames)
    except TypeError as exc:
        raise ValueError("frames must be iterable") from exc
    if not values:
        raise ValueError("at least one frame digest is required")
    if len(values) > max_frames:
        raise RuntimeError("frame manifest exceeds configured bound")
    if any(not isinstance(item, FrameDigest) for item in values):
        raise ValueError("frames must contain FrameDigest values")
    last = -1
    digest = hashlib.sha256()
    for item in values:
        if item.frame_index <= last:
            raise ValueError("frame_index values must be strictly increasing")
        last = item.frame_index
        digest.update(f"{item.frame_index}\t{item.sha256}\n".encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TrackingEvidenceManifest:
    """Metadata required before real multi-object tracking metrics are scored."""

    dataset: str
    provenance: str
    dataset_version: str
    license_expression: str
    attribution: str
    sequence_id: str
    annotation_scope: str
    annotation_sha256: str
    frame_manifest_sha256: str
    frame_count: int
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        for name, maximum in (
            ("dataset", 160),
            ("provenance", 512),
            ("dataset_version", 160),
            ("license_expression", 160),
            ("attribution", 512),
            ("sequence_id", 160),
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum))
        if self.annotation_scope != EXHAUSTIVE_MULTI_OBJECT:
            raise ValueError(
                "annotation_scope must be exhaustive_multi_object; single-target or partial labels cannot support multi-object metrics"
            )
        object.__setattr__(self, "annotation_sha256", _sha256(self.annotation_sha256, "annotation_sha256"))
        object.__setattr__(self, "frame_manifest_sha256", _sha256(self.frame_manifest_sha256, "frame_manifest_sha256"))
        for name in ("frame_count", "image_width", "image_height"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if self.frame_count > 100_000:
            raise RuntimeError("frame_count exceeds configured evidence bound")
        if self.image_width > 32_768 or self.image_height > 32_768:
            raise RuntimeError("image dimensions exceed configured evidence bound")


def require_multi_object_tracking_evidence(manifest: TrackingEvidenceManifest) -> None:
    """Fail closed unless a validated exhaustive multi-object manifest is supplied."""
    if not isinstance(manifest, TrackingEvidenceManifest):
        raise ValueError("manifest must be a TrackingEvidenceManifest")
