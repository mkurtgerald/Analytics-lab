"""One bounded real-person face-detection -> automatic-blur measurement.

This evidence-only path re-verifies the admitted standard OIDv4 SSD donor and
the admitted Wikimedia Commons CC0 portrait, decodes the source only in memory,
runs exactly one OpenVINO CPU inference at the fixed 0.50 threshold through the
class-502-only adapter, and applies the existing default-on privacy blur.

It retains/uploads no model or media artifact, performs no recognition,
embedding, ReID, or identity matching, does no post-result tuning, and makes no
face-detection accuracy or commercial-readiness claim. Without independently
authored ground truth, misses and false positives remain explicitly unscored;
the raw bounded detections are preserved in the emitted engineering evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import face_donor_oid_ssd_admission as model_admission
from . import face_donor_oid_ssd_openvino_probe as model_probe
from . import face_real_cc0_admission as source_admission
from .face_oid_ssd import OpenVINOOIDSSDDetector
from .face_privacy import FaceDetection, apply_face_privacy

_FIXED_THRESHOLD = 0.50
_EXPECTED_OPENVINO_VERSION = "2026.3.1"
_EXPECTED_OPENCV_VERSION = "4.12.0"
_MAX_SOURCE_BYTES = 10_000_000
_SOURCE_CHUNK = 64 * 1024
_SOURCE_ALLOWED_HOST = "upload.wikimedia.org"
_USER_AGENT = "Analytics-lab bounded face measurement/1.0"


def _bounded_work_dir(path: str | Path) -> Path:
    runner_temp_raw = os.environ.get("RUNNER_TEMP")
    if not runner_temp_raw:
        raise RuntimeError("RUNNER_TEMP is required")
    runner_temp = Path(runner_temp_raw).resolve()
    root = Path(path).resolve()
    if root == runner_temp or runner_temp not in root.parents:
        raise RuntimeError("work directory must be below RUNNER_TEMP")
    return root


def _download_source() -> bytes:
    parsed = urlparse(source_admission._SOURCE_URL)
    if parsed.scheme != "https" or parsed.hostname != _SOURCE_ALLOWED_HOST:
        raise RuntimeError("unapproved face source URL")
    request = Request(
        source_admission._SOURCE_URL,
        headers={"User-Agent": _USER_AGENT},
    )
    payload = bytearray()
    with urlopen(request, timeout=30) as response:
        final = urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != _SOURCE_ALLOWED_HOST:
            raise RuntimeError("face source redirect escaped approved host")
        while True:
            chunk = response.read(_SOURCE_CHUNK)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > _MAX_SOURCE_BYTES:
                raise RuntimeError("face source exceeds bounded measurement limit")
    data = bytes(payload)
    if len(data) != source_admission._EXPECTED_SIZE:
        raise RuntimeError("face source byte length changed")
    if hashlib.sha1(data).hexdigest() != source_admission._EXPECTED_SHA1:
        raise RuntimeError("face source SHA-1 changed")
    if hashlib.sha256(data).hexdigest() != source_admission._SOURCE_SHA256:
        raise RuntimeError("face source SHA-256 changed")
    return data


def _serialize_detections(
    detections: tuple[FaceDetection, ...],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for item in detections:
        if not isinstance(item, FaceDetection):
            raise ValueError("unsupported face detection value")
        output.append(
            {
                "confidence": float(item.confidence),
                "box_xyxy_normalized": [
                    float(item.box.x_min),
                    float(item.box.y_min),
                    float(item.box.x_max),
                    float(item.box.y_max),
                ],
            }
        )
    return output


def _observation(
    *,
    detections: tuple[FaceDetection, ...],
    privacy_action: str,
    width: int,
    height: int,
    input_immutable: bool,
    output_changed: bool,
    openvino_version: str,
    opencv_version: str,
) -> dict[str, object]:
    return {
        "evidence": "face-real-cc0-measurement-v1",
        "source_url": source_admission._SOURCE_URL,
        "source_license": source_admission._SOURCE_LICENSE,
        "source_size": source_admission._EXPECTED_SIZE,
        "source_sha1": source_admission._EXPECTED_SHA1,
        "source_sha256": source_admission._SOURCE_SHA256,
        "model_archive_sha256": model_admission._EXPECTED_ARCHIVE_SHA256,
        "model_graph_sha256": model_probe._expected_model_identity()[1],
        "human_face_class_id": 502,
        "confidence_threshold": _FIXED_THRESHOLD,
        "openvino_version": openvino_version,
        "opencv_version": opencv_version,
        "image_width": width,
        "image_height": height,
        "inference_runs": 1,
        "detection_count": len(detections),
        "detections": _serialize_detections(detections),
        "privacy_action": privacy_action,
        "input_immutable": input_immutable,
        "output_changed": output_changed,
        "ground_truth_available": False,
        "misses": "unscored_without_independent_ground_truth",
        "false_positives": "unscored_without_independent_ground_truth",
        "threshold_tuned_after_result": False,
        "recognition_used": False,
        "embedding_used": False,
        "reid_used": False,
        "identity_matching_used": False,
        "model_artifact_retained": False,
        "media_artifact_retained": False,
        "claim": (
            "single-source engineering observation only; not face-detection "
            "accuracy or commercial readiness"
        ),
    }


def run(work_dir: str | Path) -> dict[str, object]:
    root = _bounded_work_dir(work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    model_payload = model_admission._download()
    graph = model_probe._verified_graph_bytes(model_payload)
    source_payload = _download_source()

    import cv2
    import numpy as np
    import openvino as ov

    openvino_version = str(getattr(ov, "__version__", ""))
    if not openvino_version.startswith(_EXPECTED_OPENVINO_VERSION):
        raise RuntimeError("unexpected OpenVINO runtime version")
    opencv_version = str(getattr(cv2, "__version__", ""))
    if opencv_version != _EXPECTED_OPENCV_VERSION:
        raise RuntimeError("unexpected OpenCV runtime version")

    encoded = np.frombuffer(source_payload, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if (
        frame is None
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or int(frame.shape[2]) != 3
    ):
        raise RuntimeError("admitted face source did not decode to HxWx3 uint8 BGR")

    height, width = int(frame.shape[0]), int(frame.shape[1])
    before = hashlib.sha256(frame.tobytes()).hexdigest()

    try:
        with tempfile.TemporaryDirectory(prefix="face-real-cc0-", dir=root) as temporary:
            model_path = Path(temporary) / "frozen_inference_graph.pb"
            model_path.write_bytes(graph)
            converted = ov.convert_model(str(model_path))
            compiled = ov.Core().compile_model(converted, "CPU")
            detector = OpenVINOOIDSSDDetector(
                compiled,
                confidence_threshold=_FIXED_THRESHOLD,
            )
            result = apply_face_privacy(frame, detector)

        after = hashlib.sha256(frame.tobytes()).hexdigest()
        input_immutable = before == after
        if not input_immutable:
            raise RuntimeError("face privacy path mutated the admitted input frame")
        output_changed = not bool(np.array_equal(frame, result.frame_bgr))
        return _observation(
            detections=result.detections,
            privacy_action=result.audit.action,
            width=width,
            height=height,
            input_immutable=input_immutable,
            output_changed=output_changed,
            openvino_version=openvino_version,
            opencv_version=opencv_version,
        )
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
