"""Dependency-admission probe for LiteRT Converter 0.4.0.

Downloads the stable Google LiteRT converter wheel closure for CPython 3.11
without installing or importing any package. Every wheel is bounded, hashed,
and its METADATA name/version/license/Requires-Dist fields are emitted so the
next head can pin the exact build-time conversion environment before any
FaceSSD conversion is attempted.
"""
from __future__ import annotations

import argparse
import email
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

_CONVERTER_VERSION = "0.4.0"
_CONVERTER_SOURCE_COMMIT = "7d683c3c1104c29a4777d7047cff2fbe92bacce3"
_CONVERTER_WHEEL_SHA256 = "77827e51886bca2fea2e56ce98e1fcc66f7985260c9b6bb6786ede1374638b51"
_MAX_TOTAL_BYTES = 180 * 1024 * 1024
_MAX_WHEELS = 32
_MAX_METADATA_BYTES = 512 * 1024


def _bounded_work_dir(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    root_text = os.environ.get("RUNNER_TEMP")
    if not root_text:
        raise RuntimeError("RUNNER_TEMP is required")
    root = Path(root_text).resolve()
    if candidate == root or root not in candidate.parents:
        raise RuntimeError("work directory must be below RUNNER_TEMP")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _metadata_bytes(wheel: Path) -> bytes:
    with zipfile.ZipFile(wheel) as archive:
        names = [
            name for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(names) != 1:
            raise RuntimeError("wheel must contain exactly one METADATA file")
        info = archive.getinfo(names[0])
        if info.file_size < 1 or info.file_size > _MAX_METADATA_BYTES:
            raise RuntimeError("wheel METADATA exceeds bound")
        payload = archive.read(info)
    if len(payload) != info.file_size:
        raise RuntimeError("wheel METADATA length mismatch")
    return payload


def _inspect_wheel(wheel: Path) -> dict[str, object]:
    if wheel.suffix != ".whl" or not wheel.is_file() or wheel.is_symlink():
        raise RuntimeError("dependency probe accepts regular wheel files only")
    message = email.message_from_bytes(_metadata_bytes(wheel))
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise RuntimeError("wheel METADATA missing package identity")
    licenses = message.get_all("License") or []
    classifiers = [
        value for value in (message.get_all("Classifier") or [])
        if value.startswith("License ::")
    ]
    requires = message.get_all("Requires-Dist") or []
    return {
        "filename": wheel.name,
        "bytes": wheel.stat().st_size,
        "sha256": _sha256(wheel),
        "name": name,
        "version": version,
        "license_fields": licenses,
        "license_classifiers": classifiers,
        "requires_dist": requires,
    }


def run(work_dir: str | Path) -> dict[str, object]:
    root = _bounded_work_dir(work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        command = [
            os.sys.executable, "-m", "pip", "download",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--dest", str(root),
            f"litert-converter=={_CONVERTER_VERSION}",
        ]
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError("LiteRT converter dependency download failed")

        wheels = sorted(root.glob("*.whl"))
        if not 1 <= len(wheels) <= _MAX_WHEELS:
            raise RuntimeError("unexpected LiteRT dependency wheel count")
        total = sum(path.stat().st_size for path in wheels)
        if total > _MAX_TOTAL_BYTES:
            raise RuntimeError("LiteRT dependency closure exceeds size bound")

        packages = [_inspect_wheel(path) for path in wheels]
        converters = [
            item for item in packages
            if str(item["name"]).lower().replace("_", "-") == "litert-converter"
        ]
        if len(converters) != 1:
            raise RuntimeError("LiteRT converter wheel was not uniquely resolved")
        converter = converters[0]
        if converter["version"] != _CONVERTER_VERSION:
            raise RuntimeError("LiteRT converter version mismatch")
        if converter["sha256"] != _CONVERTER_WHEEL_SHA256:
            raise RuntimeError("LiteRT converter wheel SHA-256 mismatch")

        return {
            "evidence": "face-donor-facessd-litert-dependency-admission-v1",
            "converter_version": _CONVERTER_VERSION,
            "converter_source_commit": _CONVERTER_SOURCE_COMMIT,
            "converter_wheel_sha256": _CONVERTER_WHEEL_SHA256,
            "wheel_count": len(packages),
            "total_wheel_bytes": total,
            "packages": packages,
            "installed": False,
            "imported": False,
            "conversion_run": False,
            "media_used": False,
            "claim": "build-time dependency identity only; no model conversion or inference",
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.work_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
