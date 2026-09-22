"""Bounded OpenCV import-compatibility probe for the admitted Google FaceSSD donor.

This evidence-only probe verifies the already-pinned Google archive and regular
member identities, materializes only the admitted TensorFlow graph/config under
an explicitly supplied ephemeral work directory, and asks the already-reviewed
OpenCV 4.12 runtime whether it can import that graph. It performs no inference,
uses no media, retains no model artifact, and makes no detection-accuracy claim.
A negative import result is evidence that this zero-new-framework path is not
runtime-compatible; it is not a test failure and does not authorize a fallback
runtime or conversion dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from typing import Any, Callable

from . import face_donor_facessd_admission as admission

_MODEL_MEMBER = (
    "facessd_mobilenet_v2_quantized_320x320_open_image_v4/tflite_graph.pb"
)
_CONFIG_MEMBER = (
    "facessd_mobilenet_v2_quantized_320x320_open_image_v4/tflite_graph.pbtxt"
)
_EXPECTED_OPENCV_VERSION = "4.12.0"


def _expected_member(name: str) -> tuple[int, str]:
    for member_name, size, sha256 in admission._EXPECTED_ARCHIVE_MEMBERS:
        if member_name == name:
            return size, sha256
    raise RuntimeError("FaceSSD member is not admitted")


def _verified_member_bytes(payload: bytes, name: str) -> bytes:
    expected_size, expected_sha256 = _expected_member(name)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        try:
            member = archive.getmember(name)
        except KeyError as exc:
            raise RuntimeError("FaceSSD member is missing") from exc
        if not member.isfile() or member.issym() or member.islnk() or member.isdev():
            raise RuntimeError("FaceSSD member is not a regular file")
        if member.size != expected_size:
            raise RuntimeError("FaceSSD member size mismatch")
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("FaceSSD member is unreadable")
        data = stream.read(expected_size + 1)
    if len(data) != expected_size:
        raise RuntimeError("FaceSSD member length mismatch")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise RuntimeError("FaceSSD member SHA-256 mismatch")
    return data


def _attempt_loader(
    loader: Callable[..., Any],
    args: tuple[str, ...],
    handled_error: type[BaseException] | tuple[type[BaseException], ...],
) -> dict[str, object]:
    try:
        network = loader(*args)
    except handled_error:
        return {"loaded": False, "result": "opencv_error"}
    empty = getattr(network, "empty", None)
    if callable(empty) and bool(empty()):
        return {"loaded": False, "result": "empty_network"}
    layer_names = getattr(network, "getLayerNames", None)
    layer_count = len(layer_names()) if callable(layer_names) else None
    return {"loaded": True, "result": "loaded", "layer_count": layer_count}


def _verify_archive(payload: bytes) -> None:
    if len(payload) != admission._EXPECTED_ARCHIVE_SIZE:
        raise RuntimeError("FaceSSD archive size mismatch")
    if hashlib.sha256(payload).hexdigest() != admission._EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("FaceSSD archive SHA-256 mismatch")
    members = admission._safe_members(payload)
    if admission._member_identity(members) != admission._EXPECTED_ARCHIVE_MEMBERS:
        raise RuntimeError("FaceSSD archive member identity mismatch")


def run(work_dir: str | Path) -> dict[str, object]:
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("work directory must be a regular directory")

    payload = admission._download()
    _verify_archive(payload)
    model = _verified_member_bytes(payload, _MODEL_MEMBER)

    import cv2

    if cv2.__version__ != _EXPECTED_OPENCV_VERSION:
        raise RuntimeError("unexpected OpenCV runtime version")
    loader = getattr(getattr(cv2, "dnn", None), "readNetFromTensorflow", None)
    if not callable(loader):
        raise RuntimeError("OpenCV TensorFlow importer is unavailable")

    with tempfile.TemporaryDirectory(prefix="facessd-opencv-", dir=root) as temporary:
        temporary_path = Path(temporary)
        model_path = temporary_path / "tflite_graph.pb"
        model_path.write_bytes(model)
        binary_only = _attempt_loader(loader, (str(model_path),), cv2.error)

        with_config: dict[str, object] | None = None
        if not bool(binary_only["loaded"]):
            config = _verified_member_bytes(payload, _CONFIG_MEMBER)
            config_path = temporary_path / "tflite_graph.pbtxt"
            config_path.write_bytes(config)
            with_config = _attempt_loader(
                loader, (str(model_path), str(config_path)), cv2.error
            )

        compatible = bool(binary_only["loaded"]) or bool(
            with_config is not None and with_config["loaded"]
        )

    return {
        "evidence": "face-donor-facessd-opencv-import-v1",
        "archive_bytes": len(payload),
        "archive_sha256": admission._EXPECTED_ARCHIVE_SHA256,
        "model_member": _MODEL_MEMBER,
        "model_sha256": _expected_member(_MODEL_MEMBER)[1],
        "opencv_version": cv2.__version__,
        "binary_only": binary_only,
        "with_shipped_pbtxt": with_config,
        "direct_runtime_compatible": compatible,
        "inference_run": False,
        "media_used": False,
        "model_retained": False,
        "claim": "runtime import compatibility only; not face-detection accuracy",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.work_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
