"""Run the rights-bound local validation suite from a bounded JSON manifest.

This command is intentionally local-only. It does not download models or media,
open URLs or devices, retain frames, or promote candidate alerts to injury/fall
claims. Output contains provenance identities and aggregate measurements, never
video paths or media bytes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

from .evaluation import LabeledPersonDown
from .validation import (
    ValidationSampleSpec,
    ValidationSuiteConfig,
    ValidationSuiteResult,
    run_validation_suite,
)

_MAX_MANIFEST_BYTES = 1024 * 1024
_SCHEMA_VERSION = 1


def _strict_object(value: Any, *, required: set[str], optional: set[str] | None = None, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    allowed = required | (optional or set())
    if set(value) != required | (set(value) & (optional or set())):
        missing = required - set(value)
        extra = set(value) - allowed
        if missing:
            raise ValueError(f"{name} is missing required fields")
        if extra:
            raise ValueError(f"{name} contains unsupported fields")
    return value


def _local_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError(f"{name} must be a bounded nonempty local path")
    lowered = value.lower()
    if "://" in value or lowered.startswith(("http:", "https:", "rtsp:", "rtsps:")) or value.startswith(("/dev/", "\\\\.\\")):
        raise ValueError(f"{name} must be an ordinary local path")
    return Path(value)


def load_manifest(path: str | Path) -> tuple[Path, tuple[ValidationSampleSpec, ...], ValidationSuiteConfig]:
    """Parse a bounded, strict validation manifest without touching media."""
    manifest_path = Path(path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("manifest must be an existing non-symlink regular file")
    if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise RuntimeError("validation manifest exceeds the byte limit")
    try:
        root = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("validation manifest must be UTF-8 JSON") from error
    root = _strict_object(
        root,
        required={"schema_version", "artifact_root", "samples"},
        optional={"config"},
        name="manifest",
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("unsupported validation manifest schema_version")
    artifact_root = _local_path(root["artifact_root"], "artifact_root")

    raw_config = root.get("config", {})
    raw_config = _strict_object(
        raw_config,
        required=set(),
        optional={"max_samples", "required_device", "max_total_video_bytes"},
        name="config",
    )
    config = ValidationSuiteConfig(**raw_config)

    raw_samples = root["samples"]
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("samples must be a nonempty array")
    if len(raw_samples) > config.max_samples:
        raise RuntimeError("validation sample limit exceeded")
    samples: list[ValidationSampleSpec] = []
    for raw_sample in raw_samples:
        sample = _strict_object(
            raw_sample,
            required={
                "sample_id", "site_id", "camera_id", "authorization_ref",
                "video_path", "media_sha256", "start_timestamp_ms",
                "end_timestamp_ms", "labels",
            },
            name="sample",
        )
        raw_labels = sample["labels"]
        if not isinstance(raw_labels, list):
            raise ValueError("sample labels must be an array")
        labels: list[LabeledPersonDown] = []
        for raw_label in raw_labels:
            label = _strict_object(
                raw_label,
                required={"start_timestamp_ms", "end_timestamp_ms", "label_id"},
                name="label",
            )
            labels.append(LabeledPersonDown(**label))
        samples.append(ValidationSampleSpec(
            sample_id=sample["sample_id"],
            site_id=sample["site_id"],
            camera_id=sample["camera_id"],
            authorization_ref=sample["authorization_ref"],
            video_path=_local_path(sample["video_path"], "video_path"),
            media_sha256=sample["media_sha256"],
            start_timestamp_ms=sample["start_timestamp_ms"],
            end_timestamp_ms=sample["end_timestamp_ms"],
            labels=tuple(labels),
        ))
    return artifact_root, tuple(samples), config


def result_document(result: ValidationSuiteResult) -> dict[str, Any]:
    """Return a JSON-safe evidence document containing no source file paths."""
    if not isinstance(result, ValidationSuiteResult):
        raise ValueError("result must be a ValidationSuiteResult")
    return {"schema_version": _SCHEMA_VERSION, "validation": asdict(result)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        artifact_root, samples, config = load_manifest(args.manifest)
        result = run_validation_suite(samples, artifact_root=artifact_root, config=config)
        print(json.dumps(result_document(result), allow_nan=False, sort_keys=True, separators=(",", ":")))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError, RecursionError) as error:
        # Never echo untrusted paths, URLs, sample values, or runtime details.
        print(f"Validation rejected ({type(error).__name__}).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
