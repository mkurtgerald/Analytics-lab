"""Pinned commercial-use eligibility records for external validation/training data.

This is an engineering provenance gate, not a legal opinion and not a media
downloader. Records identify reviewed source terms; every local media asset still
requires its own exact checksum and an authorization reference before inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

_HTTPS = re.compile(r"https://[^\s]+\Z")
_ORIGINS = frozenset({"real_world", "synthetic"})
_PURPOSES = frozenset({"training", "evaluation"})


@dataclass(frozen=True)
class DataSourceRights:
    source_id: str
    title: str
    version: str
    media_origin: str
    license_id: str
    source_url: str
    license_url: str
    provenance_ref: str
    commercial_training: bool
    commercial_evaluation: bool
    attribution_required: bool
    exact_asset_identity_required: bool = True

    def __post_init__(self) -> None:
        for name in ("source_id", "title", "version", "license_id", "provenance_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError(f"{name} must be a bounded nonempty string")
            if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
                raise ValueError(f"{name} must not contain control characters")
        if self.media_origin not in _ORIGINS:
            raise ValueError("media_origin must be real_world or synthetic")
        for name in ("source_url", "license_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _HTTPS.fullmatch(value):
                raise ValueError(f"{name} must be https")
        for name in (
            "commercial_training", "commercial_evaluation",
            "attribution_required", "exact_asset_identity_required",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")

    def permits(self, purpose: str, *, require_real_world: bool = False) -> bool:
        if purpose not in _PURPOSES:
            raise ValueError("purpose must be training or evaluation")
        if require_real_world and self.media_origin != "real_world":
            return False
        return self.commercial_training if purpose == "training" else self.commercial_evaluation


UE4_FALL = DataSourceRights(
    source_id="ue4-fall-detection-dataset",
    title="UE4 Fall Detection Dataset",
    version="git:55041766dea68eaddc1df1c06aabd0a51931a22a",
    media_origin="synthetic",
    license_id="CC-BY-4.0",
    source_url="https://github.com/carolinehuang033/UE4_Fall_Detection_Dataset/tree/55041766dea68eaddc1df1c06aabd0a51931a22a",
    license_url="https://github.com/carolinehuang033/UE4_Fall_Detection_Dataset/blob/55041766dea68eaddc1df1c06aabd0a51931a22a/LICENSE.txt",
    provenance_ref="github:carolinehuang033/UE4_Fall_Detection_Dataset@55041766dea68eaddc1df1c06aabd0a51931a22a",
    commercial_training=True,
    commercial_evaluation=True,
    attribution_required=True,
)

FIGSHARE_FALL_2017 = DataSourceRights(
    source_id="figshare-fall-2017-activities",
    title="Video-Based Fall Detection Dataset with 2017 Activities from 29 Subjects",
    version="figshare:28596332:v2:2025-03-14",
    media_origin="real_world",
    license_id="CC-BY-4.0",
    source_url="https://figshare.com/articles/dataset/Sensor-Based_Fall_Detection_Dataset_with_2017_Activities_from_29_Subjects/28596332/2",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    provenance_ref="figshare:28596332:version-2",
    commercial_training=True,
    commercial_evaluation=True,
    attribution_required=True,
)

SOURCES = {item.source_id: item for item in (UE4_FALL, FIGSHARE_FALL_2017)}


def require_source(source_id: str, *, purpose: str, require_real_world: bool = False) -> DataSourceRights:
    """Return a pinned eligible source or fail closed.

    This does not authorize an arbitrary download. The caller must still bind an
    exact local asset checksum and rights reference through the validation suite.
    """
    if not isinstance(source_id, str) or source_id not in SOURCES:
        raise ValueError("data source is not in the reviewed registry")
    source = SOURCES[source_id]
    if not source.permits(purpose, require_real_world=require_real_world):
        raise ValueError("data source is not eligible for the requested commercial use")
    return source
