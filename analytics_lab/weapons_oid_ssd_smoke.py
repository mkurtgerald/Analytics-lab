"""Bounded synthetic exact-model execution smoke for the admitted OIDv4 security-object path.

This evidence lane re-verifies the already-admitted standard OIDv4 SSD archive
and frozen graph, constructs the reviewed OpenVINO 2026.3.1 CPU model exactly
once, and executes two sequential detections through the project-owned warm
runtime using one deterministic generated in-memory uint8 tensor. It uses no
image/video media, retains no model or pixel artifact, and makes no accuracy,
performance, or commercial-readiness claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

from . import face_donor_oid_ssd_admission as admission
from . import face_donor_oid_ssd_openvino_probe as construction
from .tracking import DetectionCandidate
from .weapons_oid_ssd import OID_V4_WEAPON_CLASSES, OpenVINOOIDSSDWeaponRuntime

_EXPECTED_OPENVINO_VERSION = "2026.3.1"
_SYNTHETIC_HEIGHT = 300
_SYNTHETIC_WIDTH = 300
_EXPECTED_INFERENCES = 2


def _validate_detections(
    detections: Iterable[DetectionCandidate],
) -> tuple[dict[str, object], ...]:
    summary: list[dict[str, object]] = []
    for detection in detections:
        class_id = detection.model_class_id
        if type(class_id) is not int or class_id not in OID_V4_WEAPON_CLASSES:
            raise RuntimeError("adapter emitted a non-allowlisted class")
        if detection.category != OID_V4_WEAPON_CLASSES[class_id]:
            raise RuntimeError("adapter class/category mapping mismatch")
        confidence = float(detection.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError("adapter emitted invalid confidence")
        box = detection.box
        coords = (float(box.x_min), float(box.y_min), float(box.x_max), float(box.y_max))
        if (
            any(value < 0.0 or value > 1.0 for value in coords)
            or coords[2] <= coords[0]
            or coords[3] <= coords[1]
        ):
            raise RuntimeError("adapter emitted invalid normalized box")
        summary.append(
            {
                "class_id": class_id,
                "category": detection.category,
                "confidence": confidence,
                "box": coords,
            }
        )
    return tuple(summary)


def run(work_dir: str | Path) -> dict[str, object]:
    root = construction._bounded_work_dir(work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    try:
        payload = admission._download()
        graph = construction._verified_graph_bytes(payload)
        _, model_sha256 = construction._expected_model_identity()

        import numpy as np
        import openvino as ov

        version = str(getattr(ov, "__version__", ""))
        if not version.startswith(_EXPECTED_OPENVINO_VERSION):
            raise RuntimeError("unexpected OpenVINO runtime version")

        with tempfile.TemporaryDirectory(prefix="oidv4-security-smoke-", dir=root) as temporary:
            model_path = Path(temporary) / "frozen_inference_graph.pb"
            model_path.write_bytes(graph)

            frame = np.zeros((_SYNTHETIC_HEIGHT, _SYNTHETIC_WIDTH, 3), dtype=np.uint8)
            original = frame.copy()
            runtime = OpenVINOOIDSSDWeaponRuntime.from_tensorflow_graph(model_path)

            first = _validate_detections(runtime.detect(frame))
            second = _validate_detections(runtime.detect(frame))

            if runtime.compile_count != 1:
                raise RuntimeError("runtime did not preserve compile-once lifecycle")
            if runtime.inference_count != _EXPECTED_INFERENCES:
                raise RuntimeError("runtime inference count mismatch")
            input_unchanged = bool(np.array_equal(frame, original))
            if not input_unchanged:
                raise RuntimeError("synthetic input tensor was mutated")

            return {
                "evidence": "oidv4-security-object-synthetic-execution-v1",
                "archive_sha256": admission._EXPECTED_ARCHIVE_SHA256,
                "model_sha256": model_sha256,
                "openvino_version": version,
                "input_shape": [_SYNTHETIC_HEIGHT, _SYNTHETIC_WIDTH, 3],
                "generated_input": True,
                "compile_count": runtime.compile_count,
                "inference_count": runtime.inference_count,
                "first_detection_count": len(first),
                "second_detection_count": len(second),
                "first_class_ids": [item["class_id"] for item in first],
                "second_class_ids": [item["class_id"] for item in second],
                "input_unchanged": input_unchanged,
                "media_used": False,
                "derived_artifact_retained": False,
                "claim": (
                    "synthetic exact-model execution compatibility only; not detection "
                    "accuracy, product performance, or commercial readiness"
                ),
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
