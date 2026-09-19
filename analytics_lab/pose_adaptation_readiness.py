"""Fail-closed readiness gate for any person-down pose adaptation work.

This module does not train, download, or execute a model. It records the
smallest currently justified trainer lineage and blocks training until every
artifact/right/split/export prerequisite is explicit and reviewable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DependencyPin:
    name: str
    version: str


@dataclass(frozen=True)
class PretrainedArtifact:
    name: str
    source_url: str
    sha256: str | None
    license_spdx: str | None
    rights_status: str


@dataclass(frozen=True)
class ResourceCeiling:
    cpu_only: bool
    max_cpu_threads: int
    max_wall_minutes: int
    max_ram_mib: int
    max_temp_mib: int
    max_trials: int
    paid_compute: bool


@dataclass(frozen=True)
class PoseAdaptationPlan:
    trainer_repository: str
    trainer_revision: str
    trainer_license_spdx: str
    python_version: str
    dependencies: Tuple[DependencyPin, ...]
    dependency_rights_status: str
    pretrained: PretrainedArtifact
    export_script: str
    export_onnx_name: str
    export_openvino_outputs: Tuple[str, ...]
    export_runtime_target: str
    export_smoke_tested: bool
    train_subjects: Tuple[str, ...]
    validation_subjects: Tuple[str, ...]
    holdout_subjects: Tuple[str, ...]
    resource_ceiling: ResourceCeiling


LIGHTWEIGHT_OPENPOSE_PLAN = PoseAdaptationPlan(
    trainer_repository="Daniil-Osokin/lightweight-human-pose-estimation.pytorch",
    trainer_revision="d23c284b09acf27a163e1febd511e7482cac25ed",
    trainer_license_spdx="Apache-2.0",
    python_version="3.6",
    dependencies=(
        DependencyPin("torch", "0.4.1"),
        DependencyPin("torchvision", "0.2.1"),
        DependencyPin("pycocotools", "2.0.0"),
        DependencyPin("opencv-python", "3.4.0.14"),
        DependencyPin("numpy", "1.14.0"),
    ),
    dependency_rights_status="unverified-transitive-rights",
    pretrained=PretrainedArtifact(
        name="checkpoint_iter_370000.pth",
        source_url=(
            "https://storage.openvinotoolkit.org/repositories/"
            "openvino_training_extensions/models/human_pose_estimation/"
            "checkpoint_iter_370000.pth"
        ),
        sha256=None,
        license_spdx=None,
        rights_status="hold-unresolved-artifact-rights",
    ),
    export_script="scripts/convert_to_onnx.py",
    export_onnx_name="human-pose-estimation.onnx",
    export_openvino_outputs=(
        "stage_1_output_0_pafs",
        "stage_1_output_1_heatmaps",
    ),
    export_runtime_target="OpenVINO Runtime 2026.3.1 CPU",
    export_smoke_tested=False,
    train_subjects=(),
    validation_subjects=(),
    holdout_subjects=(
        "GMDCSA24:Subject-2",
        "GMDCSA24:Subject-3",
        "GMDCSA24:Subject-4",
        "Figshare:SBJ_01",
        "Figshare:SBJ_10",
        "Figshare:SBJ_06",
        "Figshare:SBJ_03",
        "Figshare:SBJ_02",
        "Figshare:SBJ_09",
        "Figshare:SBJ_29",
        "Figshare:SBJ_07",
    ),
    resource_ceiling=ResourceCeiling(
        cpu_only=True,
        max_cpu_threads=2,
        max_wall_minutes=20,
        max_ram_mib=4096,
        max_temp_mib=2048,
        max_trials=1,
        paid_compute=False,
    ),
)


def training_blockers(plan: PoseAdaptationPlan = LIGHTWEIGHT_OPENPOSE_PLAN) -> Tuple[str, ...]:
    """Return deterministic reasons training is not yet authorized."""
    blockers: list[str] = []
    if plan.trainer_license_spdx not in {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause"}:
        blockers.append("trainer_license_not_approved")
    if plan.dependency_rights_status != "cleared-commercial-and-redistribution":
        blockers.append("transitive_dependency_rights_unverified")
    if not plan.pretrained.sha256:
        blockers.append("pretrained_weight_hash_missing")
    if not plan.pretrained.license_spdx or plan.pretrained.rights_status != "cleared-commercial-and-redistribution":
        blockers.append("pretrained_weight_license_unresolved")
    if not plan.train_subjects:
        blockers.append("training_split_empty")
    if not plan.validation_subjects:
        blockers.append("validation_split_empty")
    if set(plan.train_subjects) & set(plan.validation_subjects):
        blockers.append("train_validation_subject_overlap")
    if set(plan.train_subjects) & set(plan.holdout_subjects):
        blockers.append("train_holdout_subject_overlap")
    if set(plan.validation_subjects) & set(plan.holdout_subjects):
        blockers.append("validation_holdout_subject_overlap")
    if not plan.export_smoke_tested:
        blockers.append("openvino_export_not_smoke_tested")
    ceiling = plan.resource_ceiling
    if not ceiling.cpu_only or ceiling.paid_compute:
        blockers.append("resource_ceiling_violates_cpu_no-spend_policy")
    if min(
        ceiling.max_cpu_threads,
        ceiling.max_wall_minutes,
        ceiling.max_ram_mib,
        ceiling.max_temp_mib,
        ceiling.max_trials,
    ) <= 0:
        blockers.append("resource_ceiling_invalid")
    return tuple(blockers)


def training_ready(plan: PoseAdaptationPlan = LIGHTWEIGHT_OPENPOSE_PLAN) -> bool:
    return not training_blockers(plan)
