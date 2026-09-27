"""Fail-closed contract for project-owned Weapons training/evaluation manifests.

This module does not download media, run training, or admit a pretrained model.
It binds taxonomy, rights, exact asset identities, annotation provenance,
split isolation, and a bounded offline CPU proof budget before any future
project-owned training path is allowed to execute.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable

WEAPON_TAXONOMY = (
    "handgun",
    "rifle",
    "shotgun",
    "kitchen_knife",
    "knife",
    "sword",
    "weapon",
)
_SPLITS = frozenset({"train", "validation", "test"})
_HTTPS = re.compile(r"https://[^\s]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

RFDETR_REPOSITORY = "roboflow/rf-detr"
RFDETR_REVISION = "d96d6ff9303ec5af7bb8f73e83f0f04ecb6ef728"
RFDETR_LICENSE = "Apache-2.0"
RFDETR_VARIANT = "RFDETRNano"
PRETRAIN_WEIGHTS = None
NETWORK_ACCESS_ALLOWED = False
GPU_ALLOWED = False
MAX_PROOF_RUNTIME_MINUTES = 5
MAX_ASSETS = 4096


def _bounded_text(value: object, name: str, *, limit: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a bounded nonempty string")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class WeaponTrainingAsset:
    asset_id: str
    source_page_url: str
    asset_url: str
    license_id: str
    rights_basis: str
    source_class: str | None
    split: str
    expected_size: int
    sha256: str
    width: int
    height: int
    annotation_id: str
    annotation_sha256: str
    annotation_box_count: int
    annotation_classes: tuple[str, ...]
    commercial_training_allowed: bool
    commercial_evaluation_allowed: bool
    identifiable_people: bool
    third_party_rights_cleared: bool
    attribution_required: bool = False
    attribution_text: str = ""

    def __post_init__(self) -> None:
        for name in ("asset_id", "license_id", "rights_basis", "annotation_id"):
            _bounded_text(getattr(self, name), name)
        for name in ("source_page_url", "asset_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or _HTTPS.fullmatch(value) is None:
                raise ValueError(f"{name} must be https")
        if self.source_class is not None and self.source_class not in WEAPON_TAXONOMY:
            raise ValueError("source_class is outside the frozen Weapons taxonomy")
        if self.split not in _SPLITS:
            raise ValueError("split must be train, validation, or test")
        if type(self.expected_size) is not int or self.expected_size < 1:
            raise ValueError("expected_size must be a positive integer")
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("dimensions must be integers")
        if not 1 <= self.width <= 32768 or not 1 <= self.height <= 32768:
            raise ValueError("dimensions outside supported bounds")
        _sha256(self.sha256, "sha256")
        _sha256(self.annotation_sha256, "annotation_sha256")
        if type(self.annotation_box_count) is not int or not 0 <= self.annotation_box_count <= 4096:
            raise ValueError("annotation_box_count is outside the supported bound")
        if not isinstance(self.annotation_classes, tuple):
            raise ValueError("annotation_classes must be an immutable tuple")
        if any(item not in WEAPON_TAXONOMY for item in self.annotation_classes):
            raise ValueError("annotation_classes contains an unknown Weapons class")
        if len(set(self.annotation_classes)) != len(self.annotation_classes):
            raise ValueError("annotation_classes must not contain duplicates")
        if self.source_class is None:
            if self.annotation_box_count != 0 or self.annotation_classes != ():
                raise ValueError("negative asset must have zero boxes and no annotation classes")
        else:
            if self.annotation_box_count < 1:
                raise ValueError("positive asset must have at least one annotation box")
            if self.annotation_classes != (self.source_class,):
                raise ValueError("positive asset annotation class summary must match source_class")
        for name in (
            "commercial_training_allowed",
            "commercial_evaluation_allowed",
            "identifiable_people",
            "third_party_rights_cleared",
            "attribution_required",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.attribution_required:
            _bounded_text(self.attribution_text, "attribution_text", limit=1024)
        elif self.attribution_text:
            raise ValueError("attribution_text requires attribution_required=true")

    def eligible_for_declared_split(self) -> bool:
        if self.identifiable_people or not self.third_party_rights_cleared:
            return False
        if self.split == "train":
            return self.commercial_training_allowed
        return self.commercial_evaluation_allowed


@dataclass(frozen=True)
class WeaponTrainingPlan:
    architecture_repository: str = RFDETR_REPOSITORY
    architecture_revision: str = RFDETR_REVISION
    architecture_license: str = RFDETR_LICENSE
    model_variant: str = RFDETR_VARIANT
    pretrain_weights: None = PRETRAIN_WEIGHTS
    network_access_allowed: bool = NETWORK_ACCESS_ALLOWED
    gpu_allowed: bool = GPU_ALLOWED
    max_proof_runtime_minutes: int = MAX_PROOF_RUNTIME_MINUTES
    artifact_retention_allowed: bool = False

    def __post_init__(self) -> None:
        if self.architecture_repository != RFDETR_REPOSITORY:
            raise ValueError("unexpected Weapons training architecture repository")
        if self.architecture_revision != RFDETR_REVISION:
            raise ValueError("unexpected Weapons training architecture revision")
        if self.architecture_license != RFDETR_LICENSE:
            raise ValueError("unexpected Weapons training architecture license")
        if self.model_variant != RFDETR_VARIANT:
            raise ValueError("unexpected Weapons training model variant")
        if self.pretrain_weights is not None:
            raise ValueError("pretrained weights are prohibited for the fallback proof")
        if self.network_access_allowed is not False:
            raise ValueError("network access must remain disabled")
        if self.gpu_allowed is not False:
            raise ValueError("GPU use is not authorized for the first proof")
        if self.artifact_retention_allowed is not False:
            raise ValueError("artifact retention is not authorized for the first proof")
        if self.max_proof_runtime_minutes != MAX_PROOF_RUNTIME_MINUTES:
            raise ValueError("unexpected proof runtime ceiling")


def validate_training_manifest(
    assets: Iterable[WeaponTrainingAsset],
) -> tuple[WeaponTrainingAsset, ...]:
    try:
        normalized = tuple(assets)
    except TypeError as exc:
        raise ValueError("assets must be iterable") from exc
    if not 1 <= len(normalized) <= MAX_ASSETS:
        raise ValueError("asset count outside supported bound")
    if any(not isinstance(item, WeaponTrainingAsset) for item in normalized):
        raise ValueError("manifest contains unsupported asset record")

    asset_ids: set[str] = set()
    media_hashes: set[str] = set()
    annotation_ids: set[str] = set()
    annotation_hashes: set[str] = set()
    seen_splits: set[str] = set()

    for item in normalized:
        if not item.eligible_for_declared_split():
            raise ValueError("asset rights do not permit the declared split")
        if item.asset_id in asset_ids:
            raise ValueError("duplicate asset_id")
        if item.sha256 in media_hashes:
            raise ValueError("the same media asset cannot appear in multiple manifest rows")
        if item.annotation_id in annotation_ids or item.annotation_sha256 in annotation_hashes:
            raise ValueError("duplicate annotation identity")
        asset_ids.add(item.asset_id)
        media_hashes.add(item.sha256)
        annotation_ids.add(item.annotation_id)
        annotation_hashes.add(item.annotation_sha256)
        seen_splits.add(item.split)

    if "train" not in seen_splits or "validation" not in seen_splits:
        raise ValueError("manifest requires independent train and validation splits")
    return normalized


def manifest_digest(assets: Iterable[WeaponTrainingAsset]) -> str:
    normalized = validate_training_manifest(assets)
    payload = [
        {
            "asset_id": item.asset_id,
            "source_page_url": item.source_page_url,
            "asset_url": item.asset_url,
            "license_id": item.license_id,
            "rights_basis": item.rights_basis,
            "source_class": item.source_class,
            "split": item.split,
            "expected_size": item.expected_size,
            "sha256": item.sha256,
            "width": item.width,
            "height": item.height,
            "annotation_id": item.annotation_id,
            "annotation_sha256": item.annotation_sha256,
            "annotation_box_count": item.annotation_box_count,
            "annotation_classes": list(item.annotation_classes),
            "commercial_training_allowed": item.commercial_training_allowed,
            "commercial_evaluation_allowed": item.commercial_evaluation_allowed,
            "identifiable_people": item.identifiable_people,
            "third_party_rights_cleared": item.third_party_rights_cleared,
            "attribution_required": item.attribution_required,
            "attribution_text": item.attribution_text,
        }
        for item in normalized
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
