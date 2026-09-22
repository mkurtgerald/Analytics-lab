"""Classify OpenCV's exact SSD graph-rewrite path for admitted Google FaceSSD.

This evidence-only step preserves donor lineage and the current OpenCV runtime.
It re-verifies the admitted Google archive, downloads exactly two Apache-2.0
OpenCV 4.12.0 helper sources at the pinned commit, verifies their Git-blob
identities, rewrites only the pinned TensorFlow graph/config beneath RUNNER_TEMP,
and asks OpenCV to construct the rewritten graph. It performs no forward pass,
uses no media, uploads/retains no model or derived artifact, and makes no
detection-accuracy claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

from . import face_donor_facessd_admission as admission
from . import face_donor_facessd_opencv_probe as direct_probe

_OPENCV_COMMIT = "49486f61fb25722cbcf586b7f4320921d46fb38e"
_OPENCV_VERSION = "4.12.0"
_RAW_HOST = "raw.githubusercontent.com"
_USER_AGENT = "Analytics-Lab-face-opencv-rewrite/1"
_HELPERS = (
    (
        "tf_text_graph_ssd.py",
        18_314,
        "d27fd0d384f509789bf64f069203d3f9964db576",
    ),
    (
        "tf_text_graph_common.py",
        10_055,
        "c82053b4fb05aa2fd40c45b1cefa550431608847",
    ),
)
_MODEL_MEMBER = (
    "facessd_mobilenet_v2_quantized_320x320_open_image_v4/tflite_graph.pb"
)
_PIPELINE_MEMBER = (
    "facessd_mobilenet_v2_quantized_320x320_open_image_v4/pipeline.config"
)
_MAX_DERIVED_PBTXT_BYTES = 64 * 1024 * 1024


def _git_blob_sha1(payload: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _download_helper(name: str, expected_size: int, expected_blob_sha1: str) -> bytes:
    if "/" in name or "\\" in name or not name.endswith(".py"):
        raise RuntimeError("invalid helper name")
    url = (
        f"https://{_RAW_HOST}/opencv/opencv/{_OPENCV_COMMIT}/"
        f"samples/dnn/{name}"
    )
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != _RAW_HOST:
        raise RuntimeError("unapproved OpenCV helper URL")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        final = urllib.parse.urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != _RAW_HOST:
            raise RuntimeError("OpenCV helper redirect escaped approved host")
        payload = response.read(expected_size + 1)
    if len(payload) != expected_size:
        raise RuntimeError("OpenCV helper size mismatch")
    if _git_blob_sha1(payload) != expected_blob_sha1:
        raise RuntimeError("OpenCV helper Git blob mismatch")
    return payload


def _helper_sources() -> dict[str, bytes]:
    return {
        name: _download_helper(name, size, blob_sha1)
        for name, size, blob_sha1 in _HELPERS
    }


def _bounded_work_dir(path: str | Path) -> Path:
    root = Path(path).resolve()
    runner_temp = Path(str(__import__("os").environ.get("RUNNER_TEMP", ""))).resolve()
    if not str(runner_temp) or root == runner_temp or runner_temp not in root.parents:
        raise RuntimeError("work directory must be below RUNNER_TEMP")
    return root


def _network_result(loader, model_path: Path, config_path: Path, handled_error):
    try:
        network = loader(str(model_path), str(config_path))
    except handled_error:
        return {"loaded": False, "result": "opencv_error", "layer_count": None}
    empty = getattr(network, "empty", None)
    if callable(empty) and bool(empty()):
        return {"loaded": False, "result": "empty_network", "layer_count": None}
    layers = getattr(network, "getLayerNames", None)
    count = len(layers()) if callable(layers) else None
    return {"loaded": True, "result": "loaded", "layer_count": count}


def run(work_dir: str | Path) -> dict[str, object]:
    root = _bounded_work_dir(work_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    payload = admission._download()
    direct_probe._verify_archive(payload)
    model = direct_probe._verified_member_bytes(payload, _MODEL_MEMBER)
    pipeline = direct_probe._verified_member_bytes(payload, _PIPELINE_MEMBER)
    helpers = _helper_sources()

    import cv2

    if cv2.__version__ != _OPENCV_VERSION:
        raise RuntimeError("unexpected OpenCV runtime version")

    loader = getattr(getattr(cv2, "dnn", None), "readNetFromTensorflow", None)
    if not callable(loader):
        raise RuntimeError("OpenCV TensorFlow importer is unavailable")

    try:
        with tempfile.TemporaryDirectory(prefix="facessd-rewrite-", dir=root) as temporary:
            tmp = Path(temporary)
            helper_dir = tmp / "opencv-helper"
            helper_dir.mkdir()
            for name, source in helpers.items():
                (helper_dir / name).write_bytes(source)

            model_path = tmp / "tflite_graph.pb"
            pipeline_path = tmp / "pipeline.config"
            output_path = tmp / "generated.pbtxt"
            model_path.write_bytes(model)
            pipeline_path.write_bytes(pipeline)

            process = subprocess.run(
                [
                    sys.executable,
                    str(helper_dir / "tf_text_graph_ssd.py"),
                    "--input", str(model_path),
                    "--output", str(output_path),
                    "--config", str(pipeline_path),
                ],
                cwd=helper_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=90,
                check=False,
            )

            helper_ok = process.returncode == 0 and output_path.is_file()
            generated_size = None
            generated_sha256 = None
            network = {"loaded": False, "result": "rewrite_not_available", "layer_count": None}

            if helper_ok:
                generated_size = output_path.stat().st_size
                if not 1 <= generated_size <= _MAX_DERIVED_PBTXT_BYTES:
                    raise RuntimeError("derived OpenCV text graph exceeds bound")
                generated_bytes = output_path.read_bytes()
                if len(generated_bytes) != generated_size:
                    raise RuntimeError("derived OpenCV text graph length mismatch")
                generated_sha256 = hashlib.sha256(generated_bytes).hexdigest()
                network = _network_result(loader, model_path, output_path, cv2.error)

            return {
                "evidence": "face-donor-facessd-opencv-ssd-rewrite-v1",
                "archive_sha256": admission._EXPECTED_ARCHIVE_SHA256,
                "opencv_version": cv2.__version__,
                "opencv_commit": _OPENCV_COMMIT,
                "helper_blobs": {
                    name: blob for name, _size, blob in _HELPERS
                },
                "helper_returncode": int(process.returncode),
                "rewrite_succeeded": bool(helper_ok),
                "generated_pbtxt_bytes": generated_size,
                "generated_pbtxt_sha256": generated_sha256,
                "network": network,
                "rewritten_runtime_compatible": bool(network["loaded"]),
                "inference_run": False,
                "media_used": False,
                "derived_artifact_retained": False,
                "claim": "runtime rewrite/import classification only; not face-detection accuracy",
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
