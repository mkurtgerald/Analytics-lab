"""Bounded OpenVINO construction probe for the admitted standard OIDv4 SSD donor.

This evidence-only path re-verifies the admitted Google archive and exact frozen
GraphDef, materializes only that graph below ``RUNNER_TEMP``, and asks the
already-reviewed OpenVINO 2026.3.1 TensorFlow frontend to construct the model
and the CPU plugin to compile it. It performs no inference, supplies no input
tensor, uses no media, retains/uploads no model or conversion artifact, and
makes no detection-accuracy claim.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any

from . import face_donor_oid_ssd_admission as admission

_MODEL_MEMBER = "ssd_mobilenet_v2_oid_v4_2018_12_12/frozen_inference_graph.pb"
_EXPECTED_OPENVINO_VERSION = "2026.3.1"
_MAX_ERROR_CHARS = 800


def _bounded_work_dir(path: str | Path) -> Path:
    runner_temp_raw = os.environ.get("RUNNER_TEMP")
    if not runner_temp_raw:
        raise RuntimeError("RUNNER_TEMP is required")
    runner_temp = Path(runner_temp_raw).resolve()
    root = Path(path).resolve()
    if root == runner_temp or runner_temp not in root.parents:
        raise RuntimeError("work directory must be below RUNNER_TEMP")
    return root


def _shape_text(value: Any) -> str:
    try:
        return str(value.get_partial_shape())
    except Exception:
        return "unknown"


def _error_summary(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return text[:_MAX_ERROR_CHARS]


def _expected_model_identity() -> tuple[int, str]:
    for name, size, digest in admission._EXPECTED_ARCHIVE_MEMBERS:
        if name == _MODEL_MEMBER:
            return int(size), str(digest)
    raise RuntimeError("OID SSD frozen graph identity is not pinned")


def _verified_graph_bytes(payload: bytes) -> bytes:
    if len(payload) != admission._EXPECTED_ARCHIVE_SIZE:
        raise RuntimeError("OID SSD donor archive size mismatch")
    if hashlib.sha256(payload).hexdigest() != admission._EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("OID SSD donor archive SHA-256 mismatch")

    expected_size, expected_sha256 = _expected_model_identity()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        try:
            member = archive.getmember(_MODEL_MEMBER)
        except KeyError as exc:
            raise RuntimeError("OID SSD frozen graph missing") from exc
        if not member.isfile() or int(member.size) != expected_size:
            raise RuntimeError("OID SSD frozen graph metadata mismatch")
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("OID SSD frozen graph unreadable")
        graph = stream.read(expected_size + 1)
    if len(graph) != expected_size:
        raise RuntimeError("OID SSD frozen graph length mismatch")
    if hashlib.sha256(graph).hexdigest() != expected_sha256:
        raise RuntimeError("OID SSD frozen graph SHA-256 mismatch")
    return graph


def run(work_dir: str | Path) -> dict[str, object]:
    root = _bounded_work_dir(work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    payload = admission._download()
    graph = _verified_graph_bytes(payload)
    _, model_sha256 = _expected_model_identity()

    import openvino as ov

    version = str(getattr(ov, "__version__", ""))
    if not version.startswith(_EXPECTED_OPENVINO_VERSION):
        raise RuntimeError("unexpected OpenVINO runtime version")

    try:
        with tempfile.TemporaryDirectory(prefix="oid-ssd-openvino-", dir=root) as temporary:
            model_path = Path(temporary) / "frozen_inference_graph.pb"
            model_path.write_bytes(graph)

            convert_succeeded = False
            convert_result = "not_attempted"
            convert_error = ""
            converted = None
            try:
                converted = ov.convert_model(str(model_path))
                convert_succeeded = True
                convert_result = "loaded"
            except Exception as exc:
                convert_result = type(exc).__name__
                convert_error = _error_summary(exc)

            inputs: list[dict[str, str]] = []
            outputs: list[dict[str, str]] = []
            compile_succeeded = False
            compile_result = "not_attempted"
            compile_error = ""

            if converted is not None:
                inputs = [
                    {
                        "name": item.get_any_name() if item.get_names() else "",
                        "shape": _shape_text(item),
                        "element_type": str(item.get_element_type()),
                    }
                    for item in converted.inputs
                ]
                outputs = [
                    {
                        "name": item.get_any_name() if item.get_names() else "",
                        "shape": _shape_text(item),
                        "element_type": str(item.get_element_type()),
                    }
                    for item in converted.outputs
                ]
                try:
                    compiled = ov.Core().compile_model(converted, "CPU")
                    compile_succeeded = compiled is not None
                    compile_result = "compiled" if compile_succeeded else "empty_compiled_model"
                except Exception as exc:
                    compile_result = type(exc).__name__
                    compile_error = _error_summary(exc)

            return {
                "evidence": "face-donor-oid-ssd-openvino-construction-v1",
                "archive_sha256": admission._EXPECTED_ARCHIVE_SHA256,
                "model_member": _MODEL_MEMBER,
                "model_sha256": model_sha256,
                "openvino_version": version,
                "convert_succeeded": convert_succeeded,
                "convert_result": convert_result,
                "convert_error": convert_error,
                "compile_succeeded": compile_succeeded,
                "compile_result": compile_result,
                "compile_error": compile_error,
                "inputs": inputs,
                "outputs": outputs,
                "inference_run": False,
                "input_tensor_supplied": False,
                "media_used": False,
                "derived_artifact_retained": False,
                "claim": "runtime construction/compile compatibility only; not face-detection accuracy",
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
