"""One-shot synthetic execution smoke for the admitted standard OIDv4 SSD donor.

The probe uses only a generated in-memory uint8 tensor. It re-verifies the
admitted Google archive and frozen graph, constructs and compiles the graph with
the reviewed OpenVINO runtime, executes exactly one inference through the
OIDv4 Human-face adapter, and passes the resulting normalized detections through
the existing default-on privacy-blur policy. It uses no image/video source,
retains no model/media artifact, and makes no accuracy claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from . import face_donor_oid_ssd_admission as admission
from . import face_donor_oid_ssd_openvino_probe as construction
from .face_oid_ssd import OID_V4_HUMAN_FACE_CLASS_ID, OpenVINOOIDSSDDetector
from .face_privacy import apply_face_privacy

_EXPECTED_OPENVINO_VERSION = "2026.3.1"
_SYNTHETIC_HEIGHT = 300
_SYNTHETIC_WIDTH = 300


def run(work_dir: str | Path) -> dict[str, object]:
    root = construction._bounded_work_dir(work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    payload = admission._download()
    graph = construction._verified_graph_bytes(payload)
    _, model_sha256 = construction._expected_model_identity()

    import numpy as np
    import openvino as ov

    version = str(getattr(ov, "__version__", ""))
    if not version.startswith(_EXPECTED_OPENVINO_VERSION):
        raise RuntimeError("unexpected OpenVINO runtime version")

    try:
        with tempfile.TemporaryDirectory(prefix="oid-ssd-smoke-", dir=root) as temporary:
            model_path = Path(temporary) / "frozen_inference_graph.pb"
            model_path.write_bytes(graph)
            converted = ov.convert_model(str(model_path))
            compiled = ov.Core().compile_model(converted, "CPU")

            frame = np.zeros(
                (_SYNTHETIC_HEIGHT, _SYNTHETIC_WIDTH, 3), dtype=np.uint8
            )
            original = frame.copy()
            detector = OpenVINOOIDSSDDetector(compiled)
            result = apply_face_privacy(frame, detector)

            if result.audit.action != "blur":
                raise RuntimeError("default-on privacy policy was not applied")
            input_unchanged = bool(np.array_equal(frame, original))
            if not input_unchanged:
                raise RuntimeError("synthetic input tensor was mutated")
            if any(
                not 0.0 <= detection.confidence <= 1.0
                for detection in result.detections
            ):
                raise RuntimeError("normalized face confidence is invalid")

            return {
                "evidence": "face-donor-oid-ssd-synthetic-smoke-v1",
                "archive_sha256": admission._EXPECTED_ARCHIVE_SHA256,
                "model_sha256": model_sha256,
                "openvino_version": version,
                "human_face_class_id": OID_V4_HUMAN_FACE_CLASS_ID,
                "inference_run": True,
                "inference_attempts": 1,
                "input_tensor_supplied": True,
                "input_shape": [1, _SYNTHETIC_HEIGHT, _SYNTHETIC_WIDTH, 3],
                "synthetic_input": True,
                "media_used": False,
                "face_count": len(result.detections),
                "privacy_action": result.audit.action,
                "input_unchanged": input_unchanged,
                "derived_artifact_retained": False,
                "claim": (
                    "synthetic execution/integration smoke only; "
                    "not face-detection accuracy"
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
