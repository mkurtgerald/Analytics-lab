"""Prepare a bounded, rights-reviewed GMDCSA-24 real-video validation subset.

The accepted Subject-1 seed remains pinned for reproducibility, while the active
evidence subset is a disjoint Subject-2 positive/hard-negative pair. This module
is deliberately opt-in and is never invoked by ordinary repository CI. It
fetches only the active two clips plus the four already-reviewed Open Model Zoo
artifacts, verifies exact upstream identities, computes local media SHA-256
values, and emits a validation manifest. It does not install OpenVINO, retain
decoded frames, or make any accuracy claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import urllib.request
from urllib.parse import quote
from typing import Any, Callable, BinaryIO

from .artifacts import OPENVINO_OMZ_2023_FP16, verify_artifact_set
from .source_rights import GMDCSA24, require_source

_GMD_REPOSITORY = "ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos"
_GMD_COMMIT = "5abac7693229900cf80f722e878fbb119211fc1c"
_GMD_RAW = f"https://raw.githubusercontent.com/{_GMD_REPOSITORY}/{_GMD_COMMIT}/"
_AUTHORIZATION_REF = "gmdcsa24:MIT:git:5abac7693229900cf80f722e878fbb119211fc1c:doi:10.5281/zenodo.13354453"
_MAX_ITEM_BYTES = 16 * 1024 * 1024
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class SeedMediaSpec:
    sample_id: str
    relative_path: str
    size_bytes: int
    git_blob_sha1: str
    end_timestamp_ms: int
    label_start_timestamp_ms: int | None = None
    label_end_timestamp_ms: int | None = None
    label_id: str | None = None

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (not self.sample_id or len(self.sample_id) > 128 or not self.relative_path
                or path.is_absolute() or ".." in path.parts or "\\" in self.relative_path):
            raise ValueError("invalid seed media identity")
        if type(self.size_bytes) is not int or not 1 <= self.size_bytes <= _MAX_ITEM_BYTES:
            raise ValueError("seed media size exceeds the bounded item budget")
        if not isinstance(self.git_blob_sha1, str) or not _SHA1.fullmatch(self.git_blob_sha1):
            raise ValueError("git_blob_sha1 must be lowercase SHA-1 hex")
        if type(self.end_timestamp_ms) is not int or self.end_timestamp_ms <= 0:
            raise ValueError("end_timestamp_ms must be positive")
        values = (self.label_start_timestamp_ms, self.label_end_timestamp_ms, self.label_id)
        if any(value is not None for value in values):
            if any(value is None for value in values):
                raise ValueError("positive seed label fields must be all present or all absent")
            if (type(self.label_start_timestamp_ms) is not int
                    or type(self.label_end_timestamp_ms) is not int
                    or not 0 <= self.label_start_timestamp_ms <= self.label_end_timestamp_ms <= self.end_timestamp_ms
                    or not isinstance(self.label_id, str) or not self.label_id or len(self.label_id) > 128):
                raise ValueError("invalid positive seed label")

    @property
    def source_url(self) -> str:
        return _GMD_RAW + quote(self.relative_path, safe="/")


# Accepted seed retained verbatim so the first end-to-end staged-real result can
# always be reproduced without relying on mutable external selection logic.
GMDCSA24_ACCEPTED_SEED = (
    SeedMediaSpec(
        sample_id="gmdcsa24-s1-fall-05",
        relative_path="Subject 1/Fall/05.mp4",
        size_bytes=5_872_655,
        git_blob_sha1="4e13ed24f6c2af7062b64b1920ef706c51efe94b",
        end_timestamp_ms=5_000,
        label_start_timestamp_ms=1_800,
        label_end_timestamp_ms=5_000,
        label_id="gmdcsa24-s1-fall-05-labelled-falling",
    ),
    SeedMediaSpec(
        sample_id="gmdcsa24-s1-adl-15",
        relative_path="Subject 1/ADL/15.mp4",
        size_bytes=7_233_079,
        git_blob_sha1="903da9245132cf70c10124edd0625c958f702cb8",
        end_timestamp_ms=7_000,
    ),
)


# First disjoint held-out rotation: a different subject, with a long side-fall
# positive and a prone sleeping normal-negative chosen specifically to stress
# false-alert behavior without increasing the two-clip resource envelope.
GMDCSA24_HELDOUT_S2 = (
    SeedMediaSpec(
        sample_id="gmdcsa24-s2-fall-01",
        relative_path="Subject 2/Fall/01.mp4",
        size_bytes=6_472_725,
        git_blob_sha1="6f308a3873d5768ea81b932d45a521e520368592",
        end_timestamp_ms=6_000,
        label_start_timestamp_ms=1_400,
        label_end_timestamp_ms=6_000,
        label_id="gmdcsa24-s2-fall-01-labelled-falling",
    ),
    SeedMediaSpec(
        sample_id="gmdcsa24-s2-adl-12",
        relative_path="Subject 2/ADL/12.mp4",
        size_bytes=7_791_870,
        git_blob_sha1="aa83cc15559687e49c40afad920949be7befa61d",
        end_timestamp_ms=7_000,
    ),
)


# Keep the active PR-only evidence lane bounded to two clips. The accepted seed
# above remains separately addressable for exact historical reproduction.
GMDCSA24_SEED = GMDCSA24_HELDOUT_S2


def _ensure_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("output root must be a non-symlink directory")
    return root.resolve(strict=True)


def _target(root: Path, relative_path: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative_path).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve(strict=True)
    resolved_parent.relative_to(root)
    if path.is_symlink():
        raise ValueError("seed paths may not be symlinks")
    return path


def _hash_media(path: Path, expected_size: int) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError("seed media size or file type mismatch")
    sha1 = hashlib.sha1(f"blob {expected_size}\0".encode("ascii"))
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            sha1.update(block)
            sha256.update(block)
    return sha1.hexdigest(), sha256.hexdigest()


def verify_seed_media(path: str | Path, spec: SeedMediaSpec) -> str:
    """Verify exact Git blob identity and return local SHA-256 evidence identity."""
    if not isinstance(spec, SeedMediaSpec):
        raise ValueError("spec must be SeedMediaSpec")
    git_sha1, sha256 = _hash_media(Path(path), spec.size_bytes)
    if git_sha1 != spec.git_blob_sha1:
        raise ValueError("seed media Git blob identity mismatch")
    return sha256


def _response_stream(response: Any) -> BinaryIO:
    if not hasattr(response, "read"):
        raise RuntimeError("download response is not readable")
    return response


def _download_exact(
    url: str,
    target: Path,
    *,
    expected_size: int,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
    """Download one bounded immutable asset without silently replacing local bytes."""
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.stat().st_size != expected_size:
            raise ValueError("existing target does not match the expected regular-file size")
        return
    if expected_size < 1 or expected_size > _MAX_ITEM_BYTES:
        raise ValueError("download exceeds bounded item budget")
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        raise ValueError("stale partial download exists; inspect it before retrying")
    request = urllib.request.Request(url, headers={"User-Agent": "Analytics-lab-validation-seed/1"})
    total = 0
    try:
        with opener(request, timeout=30) as response, partial.open("xb") as handle:
            stream = _response_stream(response)
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > expected_size:
                    raise RuntimeError("download exceeded the pinned byte count")
                handle.write(block)
        if total != expected_size:
            raise RuntimeError("download byte count did not match the pinned asset")
        os.replace(partial, target)
    except Exception:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _download_media(root: Path, *, opener: Callable[..., Any]) -> dict[str, str]:
    media_root = root / "media" / "gmdcsa24"
    media_root.mkdir(parents=True, exist_ok=True)
    identities: dict[str, str] = {}
    for spec in GMDCSA24_SEED:
        target = _target(media_root, spec.relative_path)
        _download_exact(spec.source_url, target, expected_size=spec.size_bytes, opener=opener)
        identities[spec.sample_id] = verify_seed_media(target, spec)
    return identities


def _download_models(root: Path, *, opener: Callable[..., Any]) -> Path:
    artifact_root = root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    for spec in OPENVINO_OMZ_2023_FP16:
        target = _target(artifact_root, spec.relative_path)
        _download_exact(spec.source_url, target, expected_size=spec.size_bytes, opener=opener)
    verify_artifact_set(artifact_root, OPENVINO_OMZ_2023_FP16)
    return artifact_root


def _manifest(identities: dict[str, str]) -> dict[str, Any]:
    require_source(GMDCSA24.source_id, purpose="evaluation", require_real_world=True)
    samples = []
    for spec in GMDCSA24_SEED:
        labels = []
        if spec.label_id is not None:
            labels.append({
                "start_timestamp_ms": spec.label_start_timestamp_ms,
                "end_timestamp_ms": spec.label_end_timestamp_ms,
                "label_id": spec.label_id,
            })
        # These are separate source clips, not overlapping windows from one
        # continuous camera timeline. Give each clip its own validation-stream
        # identity so aggregate camera-time accounting does not double-count an
        # invented shared clock while preserving the real source/site identity.
        samples.append({
            "sample_id": spec.sample_id,
            "site_id": "gmdcsa24-home-setup-1",
            "camera_id": f"gmdcsa24-clip-{spec.sample_id}",
            "authorization_ref": _AUTHORIZATION_REF,
            "video_path": (Path("media") / "gmdcsa24" / Path(*PurePosixPath(spec.relative_path).parts)).as_posix(),
            "media_sha256": identities[spec.sample_id],
            "start_timestamp_ms": 0,
            "end_timestamp_ms": spec.end_timestamp_ms,
            "labels": labels,
        })
    return {
        "schema_version": 1,
        "artifact_root": "artifacts",
        "samples": samples,
        "config": {
            "max_samples": len(samples),
            "required_device": "CPU",
            "max_total_video_bytes": sum(item.size_bytes for item in GMDCSA24_SEED),
        },
    }


def _attribution_text() -> str:
    source = require_source(GMDCSA24.source_id, purpose="evaluation", require_real_world=True)
    selected = "; ".join(spec.relative_path for spec in GMDCSA24_SEED)
    return (
        f"{source.title}\n"
        f"Version: {source.version}\n"
        f"License: {source.license_id}\n"
        f"Provenance: {source.provenance_ref}\n"
        f"Selected clips: {selected}\n"
        "This local evidence subset is for bounded evaluation; media is not committed to Analytics-lab.\n"
    )


def prepare_seed(output_dir: str | Path, *, opener: Callable[..., Any] = urllib.request.urlopen) -> Path:
    """Acquire and verify the active two-clip subset plus reviewed OMZ artifacts."""
    root = _ensure_root(Path(output_dir))
    identities = _download_media(root, opener=opener)
    _download_models(root, opener=opener)
    document = _manifest(identities)
    manifest = root / "validation-manifest.json"
    manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "ATTRIBUTION.txt").write_text(_attribution_text(), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_seed(args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Validation seed preparation rejected ({type(exc).__name__}).", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
