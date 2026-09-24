"""Bounded warm-runtime engineering evidence for the admitted OIDv4 face graph.

This lane re-verifies only the already-admitted standard OIDv4 graph, compiles
it exactly once, and runs three sequential detections against one deterministic
generated in-memory tensor through the same runtime instance. It is hosted-CI
engineering evidence only, not a product-performance or commercial-accuracy
claim. No image/video source, recognition, embeddings, ReID, identity matching,
threshold tuning, or retained model/pixel artifact is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time

from . import face_donor_oid_ssd_admission as model_admission
from . import face_donor_oid_ssd_openvino_probe as model_probe
from . import face_real_cc0_measurement as measurement
from .face_runtime import OpenVINOOIDSSDRuntime


_EXPECTED_OPENVINO_VERSION = "2026.3.1"
_EXPECTED_OPENCV_VERSION = "4.12.0"
_FIXED_THRESHOLD = 0.50
_INFERENCE_RUNS = 3


def _bounded_work_dir(path: str | Path) -> Path:
    runner_temp_raw = os.environ.get("RUNNER_TEMP")
    if not runner_temp_raw:
        raise RuntimeError("RUNNER_TEMP is required")
    runner_temp = Path(runner_temp_raw).resolve()
    root = Path(path).resolve()
    if root == runner_temp or runner_temp not in root.parents:
        raise RuntimeError("work directory must be below RUNNER_TEMP")
    return root


def _synthetic_frame():
    import numpy as np

    y, x = np.indices((300, 300), dtype=np.uint16)
    frame = np.empty((300, 300, 3), dtype=np.uint8)
    frame[:, :, 0] = (x % 256).astype(np.uint8)
    frame[:, :, 1] = (y % 256).astype(np.uint8)
    frame[:, :, 2] = ((x + y) % 256).astype(np.uint8)
    return frame


def run(work_dir: str | Path) -> dict[str, object]:
    root = _bounded_work_dir(work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    import cv2
    import openvino as ov

    openvino_version = str(getattr(ov, "__version__", ""))
    if not openvino_version.startswith(_EXPECTED_OPENVINO_VERSION):
        raise RuntimeError("unexpected OpenVINO runtime version")
    opencv_version = str(getattr(cv2, "__version__", ""))
    if opencv_version != _EXPECTED_OPENCV_VERSION:
        raise RuntimeError("unexpected OpenCV runtime version")

    try:
        started = time.perf_counter()
        graph = measurement._stream_verified_graph()
        graph_stream_seconds = time.perf_counter() - started

        frame = _synthetic_frame()
        before = hashlib.sha256(frame.tobytes()).hexdigest()
        rss_before_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        with tempfile.TemporaryDirectory(prefix="face-hardening-", dir=root) as temporary:
            model_path = Path(temporary) / "frozen_inference_graph.pb"
            model_path.write_bytes(graph)

            started = time.perf_counter()
            runtime = OpenVINOOIDSSDRuntime.from_tensorflow_graph(
                model_path,
                confidence_threshold=_FIXED_THRESHOLD,
            )
            compile_seconds = time.perf_counter() - started

            inference_seconds: list[float] = []
            detection_counts: list[int] = []
            for _ in range(_INFERENCE_RUNS):
                started = time.perf_counter()
                detections = runtime.detect(frame)
                inference_seconds.append(time.perf_counter() - started)
                detection_counts.append(len(detections))

        after = hashlib.sha256(frame.tobytes()).hexdigest()
        input_immutable = before == after
        if not input_immutable:
            raise RuntimeError("warm-runtime evidence mutated the synthetic input")
        if runtime.compile_count != 1:
            raise RuntimeError("warm runtime did not compile exactly once")
        if runtime.inference_count != _INFERENCE_RUNS:
            raise RuntimeError("warm runtime did not reuse the compiled detector")

        rss_after_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return {
            "evidence": "face-warm-runtime-hardening-v1",
            "model_archive_sha256": model_admission._EXPECTED_ARCHIVE_SHA256,
            "model_graph_sha256": model_probe._expected_model_identity()[1],
            "human_face_class_id": 502,
            "confidence_threshold": _FIXED_THRESHOLD,
            "openvino_version": openvino_version,
            "opencv_version": opencv_version,
            "synthetic_input": True,
            "synthetic_input_shape": [300, 300, 3],
            "compile_count": runtime.compile_count,
            "inference_runs": runtime.inference_count,
            "detection_count_each_run": detection_counts,
            "phase_seconds": {
                "model_graph_stream_verify": round(graph_stream_seconds, 6),
                "model_convert_compile_once": round(compile_seconds, 6),
                "warm_inference_each": [round(value, 6) for value in inference_seconds],
            },
            "peak_rss_kib_before_compile": rss_before_kib,
            "peak_rss_kib_after_repeated_inference": rss_after_kib,
            "input_immutable": input_immutable,
            "threshold_tuned_after_result": False,
            "recognition_used": False,
            "embedding_used": False,
            "reid_used": False,
            "identity_matching_used": False,
            "model_artifact_retained": False,
            "media_artifact_retained": False,
            "claim": (
                "bounded hosted-CI engineering observation only; not product "
                "performance, face-detection accuracy, or commercial readiness"
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
