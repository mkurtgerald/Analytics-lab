"""Fail-closed verification for locally provisioned model artifacts.

This module never downloads model files. Callers provision artifacts outside the
repository and verify exact size + SHA-384 before a runtime adapter opens them.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Iterable

_HEX_384 = re.compile(r"[0-9a-f]{96}\Z")
_HTTPS = re.compile(r"https://[^\s]+\Z")


@dataclass(frozen=True)
class ArtifactSpec:
    component: str
    relative_path: str
    size_bytes: int
    sha384: str
    source_url: str
    license_id: str
    license_url: str

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component.strip():
            raise ValueError("artifact component is required")
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ValueError("artifact relative_path is required")
        path = PurePosixPath(self.relative_path)
        if (path.is_absolute() or path.as_posix() != self.relative_path or "\\" in self.relative_path
                or ".." in path.parts
                or any(not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in path.parts)):
            raise ValueError("artifact path must be a normalized safe relative path")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("artifact size_bytes must be a positive integer")
        if not isinstance(self.sha384, str) or not _HEX_384.fullmatch(self.sha384):
            raise ValueError("artifact sha384 must be lowercase 96-character hex")
        if not isinstance(self.source_url, str) or not _HTTPS.fullmatch(self.source_url):
            raise ValueError("artifact source_url must be https")
        if not isinstance(self.license_id, str) or not self.license_id.strip():
            raise ValueError("artifact license_id is required")
        if not isinstance(self.license_url, str) or not _HTTPS.fullmatch(self.license_url):
            raise ValueError("artifact license_url must be https")


@dataclass(frozen=True)
class VerifiedArtifact:
    spec: ArtifactSpec
    path: Path


def verify_local_artifact(root: str | Path, spec: ArtifactSpec, *, chunk_bytes: int = 1024 * 1024) -> VerifiedArtifact:
    """Verify one regular, non-symlink local artifact without following outside root."""
    if not isinstance(spec, ArtifactSpec):
        raise ValueError("spec must be an ArtifactSpec")
    if type(chunk_bytes) is not int or chunk_bytes < 1 or chunk_bytes > 16 * 1024 * 1024:
        raise ValueError("chunk_bytes must be in [1, 16777216]")
    base = Path(root)
    if not base.is_dir():
        raise ValueError("artifact root must be an existing directory")
    candidate = base.joinpath(*PurePosixPath(spec.relative_path).parts)
    if candidate.is_symlink():
        raise ValueError("artifact symlinks are not accepted")
    try:
        resolved_root = base.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ValueError("artifact path is unavailable or outside the artifact root") from exc
    if not resolved.is_file():
        raise ValueError("artifact must be a regular file")
    if resolved.stat().st_size != spec.size_bytes:
        raise ValueError("artifact size mismatch")
    digest = hashlib.sha384()
    with resolved.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    if digest.hexdigest() != spec.sha384:
        raise ValueError("artifact checksum mismatch")
    return VerifiedArtifact(spec, resolved)


def verify_artifact_set(root: str | Path, specs: Iterable[ArtifactSpec]) -> tuple[VerifiedArtifact, ...]:
    """Verify a finite set and reject duplicate logical paths."""
    verified: list[VerifiedArtifact] = []
    seen: set[str] = set()
    count = 0
    for spec in specs:
        count += 1
        if count > 64:
            raise ValueError("artifact set exceeds supported size")
        if not isinstance(spec, ArtifactSpec):
            raise ValueError("artifact set contains an unsupported item")
        if spec.relative_path in seen:
            raise ValueError("duplicate artifact path")
        seen.add(spec.relative_path)
        verified.append(verify_local_artifact(root, spec))
    if not verified:
        raise ValueError("artifact set must not be empty")
    return tuple(verified)


_OMZ_COMMIT = "6697dead54ed1cdd664b0313189c2cb52ee6335e"
_OMZ_LICENSE = f"https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/{_OMZ_COMMIT}/LICENSE"

OPENVINO_OMZ_2023_FP16 = (
    ArtifactSpec(
        component="open-model-zoo/person-detection-retail-0013/fp16",
        relative_path="person-detection-retail-0013/FP16/person-detection-retail-0013.xml",
        size_bytes=571233,
        sha384="99ad3d4580a0123bef05ff77b6f46ccec16de974d1f5699fb94cd842e3242c6aa641f4977f9a5bb2f0fab42fe51cbb63",
        source_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-retail-0013/FP16/person-detection-retail-0013.xml",
        license_id="Apache-2.0",
        license_url=_OMZ_LICENSE,
    ),
    ArtifactSpec(
        component="open-model-zoo/person-detection-retail-0013/fp16",
        relative_path="person-detection-retail-0013/FP16/person-detection-retail-0013.bin",
        size_bytes=1445734,
        sha384="a67422e3b5ec76057651d2a0237eab862de00e968c7eef1e5f333849ae64f91900bcd30a23e1b7dbaa07313e358759b9",
        source_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-retail-0013/FP16/person-detection-retail-0013.bin",
        license_id="Apache-2.0",
        license_url=_OMZ_LICENSE,
    ),
    ArtifactSpec(
        component="open-model-zoo/human-pose-estimation-0001/fp16",
        relative_path="human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml",
        size_bytes=218215,
        sha384="cffe8df7d053b9cbf858a21faa32e30cb8a645416e9ee4ce3fbcc3106094505477d66a06b7e46d6ddb9c2de4b0cee319",
        source_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml",
        license_id="Apache-2.0",
        license_url=_OMZ_LICENSE,
    ),
    ArtifactSpec(
        component="open-model-zoo/human-pose-estimation-0001/fp16",
        relative_path="human-pose-estimation-0001/FP16/human-pose-estimation-0001.bin",
        size_bytes=8197354,
        sha384="dabb7be42c5be008de354c6670aa8291c2d18e59f18a1c138df9b5200929a03c1e323930a1d21ecff69d3f06007de67c",
        source_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/human-pose-estimation-0001/FP16/human-pose-estimation-0001.bin",
        license_id="Apache-2.0",
        license_url=_OMZ_LICENSE,
    ),
)
