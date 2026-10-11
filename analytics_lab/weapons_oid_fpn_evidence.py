"""Exactly three untouched engineering observations through one pinned FPN model.

One generated blank 512x512 source and the two already-admitted CC0 sources run
through one warm CPU runtime. Source descriptions are not detector annotations;
misses/false positives remain unscored. No model or pixel artifact is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Callable

from .face_real_cc0_measurement import _bounded_work_dir
from . import weapons_real_cc0_admission as sources
from . import weapons_real_cc0_measurement as measurement
from .weapons_oid_fpn import CONFIDENCE_THRESHOLD, OpenVINOOIDFPNWeaponRuntime
from .weapons_oid_ssd import OID_V4_WEAPON_CLASSES
from .weapons_oid_ssd_smoke import _validate_detections


def _progress(phase: str) -> None:
    print(json.dumps({"evidence": "weapons-oid-fpn-phase", "phase": phase}, sort_keys=True), flush=True)


def _check_versions() -> dict:
    import cv2
    import numpy as np
    import openvino as ov
    versions = {"openvino": str(ov.__version__), "opencv": str(cv2.__version__), "numpy": str(np.__version__)}
    if not versions["openvino"].startswith("2026.3.1"):
        raise RuntimeError("unexpected OpenVINO runtime version")
    if versions["opencv"] != "4.12.0" or versions["numpy"] != "2.2.6":
        raise RuntimeError("unexpected OpenCV/NumPy runtime version")
    return versions


def _generated(runtime: OpenVINOOIDFPNWeaponRuntime) -> dict:
    import numpy as np
    frame = np.zeros((512, 512, 3), dtype=np.uint8)
    before = hashlib.sha256(frame.tobytes()).hexdigest()
    detections = runtime.detect(frame)
    _validate_detections(detections)
    if hashlib.sha256(frame.tobytes()).hexdigest() != before:
        raise RuntimeError("FPN runtime mutated generated input pixels")
    return {
        "name": "generated-512-blank", "generated_input": True,
        "source_shape": [512, 512, 3], "input_sha256": before,
        "input_immutable": True, "detection_count": len(detections),
        "detections": measurement._serialize(detections),
        "ground_truth_available": False, "accuracy_claim": False,
    }


def run(work_dir: str | Path, *, opener: Callable | None = None) -> dict:
    if sys.platform != "linux":
        raise RuntimeError("FPN native evidence requires hosted Linux")
    from . import weapons_oid_fpn_admission as admission
    import resource

    _bounded_work_dir(work_dir)
    versions = _check_versions()
    identities = admission.identity()
    phase_seconds = {}
    observations = []
    rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Admission creates and removes only its own unique graph directory. Pass
    # the original path so its symlink checks precede filesystem creation.
    _progress("archive_verify")
    started = time.perf_counter()
    with admission.verified_graph(work_dir, opener=opener) as graph_path:
        phase_seconds["archive_verify"] = time.perf_counter() - started
        runtime = OpenVINOOIDFPNWeaponRuntime.from_tensorflow_graph(graph_path, progress=_progress)
        phase_seconds.update(runtime.phase_seconds)
        _progress("generated_inference")
        started = time.perf_counter()
        observations.append(_generated(runtime))
        phase_seconds["generated_inference"] = time.perf_counter() - started
        for source in sources._SOURCES:
            _progress(source.name + "_download_decode_inference")
            started = time.perf_counter()
            observation = (measurement._measure_source(runtime, source) if opener is None
                           else measurement._measure_source(runtime, source, opener=opener))
            for item in observation["detections"]:
                class_id = item["class_id"]
                if type(class_id) is not int or OID_V4_WEAPON_CLASSES.get(class_id) != item["category"]:
                    raise RuntimeError("FPN emitted a non-allowlisted class/category")
            observations.append(observation)
            phase_seconds[source.name + "_download_decode_inference"] = time.perf_counter() - started
        if runtime.compile_count != 1 or runtime.inference_count != 3 or len(observations) != 3:
            raise RuntimeError("FPN evidence must compile once and execute exactly three detections")
        if any(item.get("input_immutable") is not True for item in observations):
            raise RuntimeError("FPN observation input immutability was not established")
        converted_ports, compiled_ports = runtime.converted_ports, runtime.compiled_ports
    if Path(graph_path).exists():
        raise RuntimeError("FPN graph cleanup did not complete")
    _progress("cleanup_complete")
    return {
        "evidence": "weapons-oid-fpn-engineering-measurement-v1",
        "artifact_identity": identities, "runtime_versions": versions,
        "archive_identity_pinned": True, "archive_identity_reverified": True,
        "temporary_graph_removed": True,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "preprocessing": {
            "source": "original_size_bgr_uint8", "model_color_order": "RGB",
            "input_dtype": "uint8", "input_layout": "NHWC",
            "input_partial_shape": [1, None, None, 3],
            "external_resize": False, "external_normalization": False,
            "graph_image_resizer": "fixed_shape_resizer", "graph_height": 512,
            "graph_width": 512, "resize_and_normalization_owner": "pinned_graph",
        },
        "compile_count": 1, "inference_count": 3,
        "converted_ports": converted_ports, "compiled_ports": compiled_ports,
        "phase_seconds": {name: round(value, 6) for name, value in phase_seconds.items()},
        "peak_rss_kib_before_compile": rss_before,
        "peak_rss_kib_after_inference": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "observations": observations, "ground_truth_available": False,
        "threshold_tuned_after_result": False, "model_artifact_retained": False,
        "media_artifact_retained": False, "accuracy_claim": False,
        "performance_claim": False, "readiness_claim": False,
        "claim": "three untouched engineering observations only; independently reviewed labels are required before accuracy scoring",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.work_dir), sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
