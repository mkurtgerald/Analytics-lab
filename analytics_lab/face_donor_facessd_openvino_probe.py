"""Bounded OpenVINO TensorFlow-frontend classification for admitted FaceSSD.

This evidence-only path re-verifies the already admitted Google FaceSSD archive,
materializes only the pinned frozen GraphDef under RUNNER_TEMP, and asks the
already-reviewed OpenVINO 2026.3.1 TensorFlow frontend to convert it and the CPU
plugin to compile the converted model. It performs no inference, supplies no
input tensor, uses no media, retains/uploads no model or conversion artifact,
and makes no detection-accuracy claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from . import face_donor_facessd_admission as admission
from . import face_donor_facessd_opencv_probe as donor_probe

_MODEL_MEMBER = (
    "facessd_mobilenet_v2_quantized_320x320_open_image_v4/tflite_graph.pb"
)
_EXPECTED_OPENVINO_VERSION = "2026.3.1"


def _bounded_work_dir(path: str | Path) -> Path:
    import os

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


def run(work_dir: str | Path) -> dict[str, object]:
    root = _bounded_work_dir(work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    payload = admission._download()
    donor_probe._verify_archive(payload)
    graph = donor_probe._verified_member_bytes(payload, _MODEL_MEMBER)

    import openvino as ov

    version = str(getattr(ov, "__version__", ""))
    if not version.startswith(_EXPECTED_OPENVINO_VERSION):
        raise RuntimeError("unexpected OpenVINO runtime version")

    try:
        with tempfile.TemporaryDirectory(prefix="facessd-openvino-", dir=root) as temporary:
            tmp = Path(temporary)
            model_path = tmp / "tflite_graph.pb"
            model_path.write_bytes(graph)

            try:
                converted = ov.convert_model(str(model_path))
            except Exception as exc:
                return {
                    "evidence": "face-donor-facessd-openvino-convert-v1",
                    "archive_sha256": admission._EXPECTED_ARCHIVE_SHA256,
                    "model_sha256": donor_probe._expected_member(_MODEL_MEMBER)[1],
                    "openvino_version": version,
                    "convert_succeeded": False,
                    "compile_succeeded": False,
                    "conversion_result": type(exc).__name__,
                    "inputs": [],
                    "outputs": [],
                    "inference_run": False,
                    "media_used": False,
                    "derived_artifact_retained": False,
                    "claim": "runtime conversion/compile classification only; not face-detection accuracy",
                }

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

            compile_succeeded = False
            compile_result = "not_attempted"
            try:
                core = ov.Core()
                compiled = core.compile_model(converted, "CPU")
                compile_succeeded = compiled is not None
                compile_result = "compiled" if compile_succeeded else "empty_compiled_model"
            except Exception as exc:
                compile_result = type(exc).__name__

            return {
                "evidence": "face-donor-facessd-openvino-convert-v1",
                "archive_sha256": admission._EXPECTED_ARCHIVE_SHA256,
                "model_sha256": donor_probe._expected_member(_MODEL_MEMBER)[1],
                "openvino_version": version,
                "convert_succeeded": True,
                "compile_succeeded": compile_succeeded,
                "compile_result": compile_result,
                "inputs": inputs,
                "outputs": outputs,
                "inference_run": False,
                "media_used": False,
                "derived_artifact_retained": False,
                "claim": "runtime conversion/compile classification only; not face-detection accuracy",
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
