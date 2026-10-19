"""One Windows-only retrospective Tesseract/CCT comparison; no promotion gate.

Two exposed, independently prebound CC0 plate crops and three generated blank
controls receive exactly five calls per arm. Source RGB is shared; Tesseract
uses its retained native-size gray/Otsu preprocessing and CCT uses raw RGB PPM.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any, Callable

from . import lpr_ocr_exact_evidence as exact
from . import lpr_ocr_sc_evidence as sc
from .lpr_ocr import LPROCRConfig, OCRText, TesseractPlateRecognizer
from .public_live_hls import LiveDeadline

CCT_MODEL_URL = "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global.onnx"
CCT_MODEL_SIZE = 5_262_230
CCT_MODEL_SHA256 = "384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b"
CCT_CONFIG_URL = "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global_plate_config.yaml"
CCT_CONFIG_SIZE = 1_725
CCT_CONFIG_SHA256 = "0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6"
_SOURCES = (("land_rover", exact), ("south_carolina", sc))
_BASELINE_PREPROCESS = "RGB24->gray->global-Otsu-binary->RGB24;native-size;PSM7"
_CCT_PREPROCESS = "raw-RGB24-PPM->published-CCT-resize-128x64-INTER_LINEAR;uint8-NHWC"


def _phase(name: str) -> None:
    print(json.dumps({"phase": name}), flush=True)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _require_hosted_windows() -> None:
    if (os.name != "nt" or os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
            or os.environ.get("RUNNER_OS") != "Windows"
            or os.environ.get("GITHUB_REPOSITORY") != "mkurtgerald/Analytics-lab"
            or os.environ.get("GITHUB_EVENT_NAME") != "pull_request"
            or not os.environ.get("GITHUB_HEAD_REF", "").startswith("evidence/lpr-cct-")):
        raise RuntimeError("CCT evidence requires its dedicated GitHub-hosted Windows PR lane")


def _work_directory(path: Path) -> Path:
    value = os.environ.get("RUNNER_TEMP")
    if not value or not isinstance(path, Path):
        raise RuntimeError("RUNNER_TEMP and a new work directory are required")
    root = Path(value).resolve(strict=True)
    candidate = path.resolve()
    if (not root.is_dir() or candidate.parent != root or path.absolute() != candidate or path.is_symlink()
            or candidate.exists()):
        raise RuntimeError("work directory must be a new direct child of canonical RUNNER_TEMP")
    return candidate


def _runtime_versions() -> dict[str, str]:
    import cv2
    import numpy
    import openvino
    telemetry = importlib.metadata.version("openvino-telemetry")
    if (cv2.__version__ != "4.12.0" or numpy.__version__ != "2.2.6"
            or not str(openvino.__version__).startswith("2026.3.1") or telemetry != "2025.2.0"):
        raise RuntimeError("CCT evidence runtime differs from the reviewed pins")
    return {"python": platform.python_version(), "opencv": cv2.__version__,
            "opencv_wheel": "4.12.0.88", "numpy": numpy.__version__,
            "openvino": str(openvino.__version__), "openvino_telemetry": telemetry}


def _source_identity(name: str, module: Any) -> dict[str, Any]:
    box = module._PLATE_BOX_PX if module is exact else module._PLATE_TEXT_BOX_PX
    return {"source_id": name, "source_url": module._SOURCE_URL, "rights_page": module._RIGHTS_PAGE,
            "license": "CC0-1.0", "bytes": module._SOURCE_SIZE,
            "sha1": module._SOURCE_SHA1, "sha256": module._SOURCE_SHA256,
            "source_dimensions": [module._SOURCE_WIDTH, module._SOURCE_HEIGHT],
            "crop_box_px_half_open": list(box), "crop_rgb24_sha256": module._CROP_RGB24_SHA256,
            "expected_text": module._EXPECTED_TEXT,
            "annotation_scope": "independent_pre_OCR_serial_and_crop;retrospectively_reused"}


def evidence_plan() -> dict[str, Any]:
    return {"schema": "lpr-cct-comparison-plan-v1",
            "sources": [_source_identity(name, module) for name, module in _SOURCES],
            "source_downloads": 2, "total_downloads": 6, "calls_per_arm": 5,
            "real_positive_cases": 2, "generated_blank_controls": 3, "real_negative_cases": 0,
            "baseline": {"engine": "Tesseract-5.5.3", "config": asdict(LPROCRConfig(tesseract_timeout_seconds=15.0)),
                         "psm": 7, "preprocess": _BASELINE_PREPROCESS},
            "candidate": {"engine": "fast-plate-ocr-CCT-S-v2-global", "min_confidence": .90,
                          "preprocess": _CCT_PREPROCESS, "model_url": CCT_MODEL_URL,
                          "model_bytes": CCT_MODEL_SIZE, "model_sha256": CCT_MODEL_SHA256,
                          "config_url": CCT_CONFIG_URL, "config_bytes": CCT_CONFIG_SIZE,
                          "config_sha256": CCT_CONFIG_SHA256, "config_handling": "hash_only;no_YAML_execution"},
            "installer": {"url": exact._TESSERACT_INSTALLER_URL, "bytes": exact._TESSERACT_INSTALLER_SIZE,
                          "sha256": exact._TESSERACT_INSTALLER_SHA256},
            "tessdata": {"url": exact._TESSDATA_URL, "bytes": exact._TESSDATA_SIZE,
                         "repository_commit": exact._TESSDATA_COMMIT, "git_blob_sha1": exact._TESSDATA_GIT_BLOB_SHA1},
            "cooperative_budget_seconds": 180, "cleanup_reserve_seconds": 5,
            "http_timeout_semantics": "existing helper;30s inactivity;elapsed checked around acquisitions",
            "hard_job_seconds": 300, "model_training_overlap": "unknown",
            "held_out": False, "default_promotion": False, "accuracy_gate_passed": False,
            "commercial_accuracy_claim": False, "model_release_approved": False}


def _decode_crop(payload: bytes, module: Any) -> tuple[bytes, int, int]:
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if (image is None or tuple(image.shape) != (module._SOURCE_HEIGHT, module._SOURCE_WIDTH, 3)
            or str(image.dtype) != "uint8"):
        raise RuntimeError("source decode dimensions/type changed")
    if module is exact:
        left, top, right, bottom = module._PLATE_BOX_PX
        width, height = right - left, bottom - top
        rgb = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB).tobytes(order="C")
    else:
        rgb, width, height, _ = module._crop_identity(image)
    if _hash(rgb) != module._CROP_RGB24_SHA256:
        raise RuntimeError("independently prebound crop RGB24 hash changed")
    if len(rgb) != width * height * 3 or len(rgb) > 16 * 1024 * 1024 - 128:
        raise RuntimeError("crop exceeds the OCR byte contract")
    return rgb, width, height


def _baseline_ppm(rgb: bytes, width: int, height: int) -> bytes:
    import cv2
    import numpy as np
    image = np.frombuffer(rgb, dtype=np.uint8).reshape((height, width, 3))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    transformed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB).tobytes(order="C")
    return exact._ppm(transformed, width=width, height=height)


def _make_cct(path: Path):
    from .lpr_cct import OpenVINOCCTPlateRecognizer
    return OpenVINOCCTPlateRecognizer(path, min_confidence=.90)


def _edit_distance(expected: str, observed: str | None) -> int:
    previous = list(range(len(expected) + 1))
    for index, character in enumerate(observed or "", 1):
        current = [index]
        for position, reference in enumerate(expected, 1):
            current.append(min(current[-1] + 1, previous[position] + 1,
                               previous[position - 1] + (character != reference)))
        previous = current
    return previous[-1]


def _summarize(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    positives = [row[arm] for row in rows if row["scope"] == "real_positive_retrospective"]
    blanks = [row[arm] for row in rows if row["scope"] == "generated_blank_control"]
    correct = sum(item["exact_match"] for item in positives)
    abstained = sum(item["abstained"] for item in positives)
    return {"real_positive_cases": len(positives), "exact_reads": correct,
            "wrong_accepted_reads": len(positives) - correct - abstained, "positive_abstentions": abstained,
            "character_errors": sum(item["character_errors"] for item in positives),
            "reference_characters": sum(len(row["expected_text"]) for row in rows if row["expected_text"] is not None),
            "generated_blank_controls": len(blanks),
            "generated_blank_false_accepts": sum(not item["abstained"] for item in blanks),
            "real_negative_cases": 0, "real_negative_false_positive_rate": None}


def run(work_dir: Path, *, runner: Callable = subprocess.run,
        clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    _require_hosted_windows()
    work_dir = _work_directory(work_dir)
    deadline = LiveDeadline(clock, total_seconds=175)
    plan, versions = evidence_plan(), _runtime_versions()
    counts = {"installations": 0, "version_probes": 0, "baseline_calls": 0, "candidate_calls": 0, "downloads": 0}
    work_dir.mkdir()
    try:
        def download(url: str, hosts: frozenset[str], cap: int) -> bytes:
            deadline.check(); counts["downloads"] += 1
            data = exact._download(url, allowed_hosts=hosts, max_bytes=cap)
            deadline.check()
            return data
        inputs = []
        for name, module in _SOURCES:
            _phase("acquire_" + name)
            payload = download(module._SOURCE_URL, frozenset({"upload.wikimedia.org"}), module._SOURCE_SIZE)
            module._verify_source(payload)
            rgb, width, height = _decode_crop(payload, module)
            inputs.append((name, "real_positive_retrospective", module._EXPECTED_TEXT, rgb, width, height))
            del payload
        for name, value in (("black", 0), ("gray", 127), ("white", 255)):
            inputs.append(("generated_blank_" + name, "generated_blank_control", None, bytes([value]) * (128 * 64 * 3), 128, 64))
        release_hosts = frozenset({"github.com", "release-assets.githubusercontent.com"})
        _phase("acquire_native_assets")
        installer = download(exact._TESSERACT_INSTALLER_URL, release_hosts, exact._TESSERACT_INSTALLER_SIZE)
        exact._verify_sha256(installer, size=exact._TESSERACT_INSTALLER_SIZE,
                             sha256=exact._TESSERACT_INSTALLER_SHA256, name="installer")
        installer_path = work_dir / "tesseract-installer.exe"; installer_path.write_bytes(installer); del installer
        data = download(exact._TESSDATA_URL, frozenset({"raw.githubusercontent.com"}), exact._TESSDATA_SIZE)
        tessdata_sha = exact._verify_tessdata_bytes(data)
        tessdata_dir = work_dir / "tessdata"; tessdata_dir.mkdir()
        traineddata = tessdata_dir / "eng.traineddata"; traineddata.write_bytes(data); del data
        model = download(CCT_MODEL_URL, release_hosts, CCT_MODEL_SIZE)
        exact._verify_sha256(model, size=CCT_MODEL_SIZE, sha256=CCT_MODEL_SHA256, name="CCT model")
        model_path = work_dir / "cct_s_v2_global.onnx"; model_path.write_bytes(model); del model
        config = download(CCT_CONFIG_URL, release_hosts, CCT_CONFIG_SIZE)
        exact._verify_sha256(config, size=CCT_CONFIG_SIZE, sha256=CCT_CONFIG_SHA256, name="CCT config")
        del config  # Verified metadata only; never parse or execute YAML.
        def bounded_runner(args, **kwargs):
            kwargs["timeout"] = deadline.timeout(float(kwargs["timeout"]))
            result = runner(args, **kwargs); deadline.check(); return result
        _phase("install_tesseract")
        install_dir = work_dir / "tesseract"; counts["installations"] += 1
        installed = bounded_runner([str(installer_path), "/S", f"/D={install_dir}"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120.0, check=False)
        if installed.returncode != 0: raise RuntimeError("Tesseract installation failed")
        binaries = tuple(install_dir.rglob("tesseract.exe"))
        if (len(binaries) != 1 or binaries[0].is_symlink() or not binaries[0].is_file()
                or install_dir.resolve() not in binaries[0].resolve().parents):
            raise RuntimeError("Tesseract executable is not uniquely bounded")
        binary_hash = _hash(binaries[0].read_bytes())
        def baseline_runner(args, **kwargs):
            counts["version_probes" if list(args) == [str(binaries[0]), "--version"] else "baseline_calls"] += 1
            return bounded_runner(args, **kwargs)
        package_runner = exact._OfficialWindowsPackageRunner(str(binaries[0]), runner=baseline_runner)
        baseline = TesseractPlateRecognizer(traineddata, binary=str(binaries[0]),
                                           config=LPROCRConfig(tesseract_timeout_seconds=15.0), runner=package_runner)
        _phase("compile_cct")
        deadline.check(); candidate = _make_cct(model_path); deadline.check()
        if candidate.compile_count != 1 or candidate.runtime_version != versions["openvino"]:
            raise RuntimeError("CCT compile/runtime conservation failed")
        rows = []
        for name, scope, expected, rgb, width, height in inputs:
            raw_ppm, baseline_ppm = exact._ppm(rgb, width=width, height=height), _baseline_ppm(rgb, width, height)
            row = {"case_id": name, "scope": scope, "expected_text": expected,
                   "shared_rgb24_sha256": _hash(rgb), "dimensions": [width, height],
                   "baseline_input_ppm_sha256": _hash(baseline_ppm), "candidate_input_ppm_sha256": _hash(raw_ppm)}
            for arm, recognizer, ppm in (("baseline", baseline, baseline_ppm), ("candidate", candidate, raw_ppm)):
                _phase(name + "_" + arm); deadline.check()
                before, cpu = clock(), time.process_time()
                observed = recognizer(ppm); deadline.check()
                if arm == "candidate": counts["candidate_calls"] += 1
                if observed is not None and not isinstance(observed, OCRText):
                    raise RuntimeError("recognizer returned an unsupported OCR value")
                text = None if observed is None else observed.text
                row[arm] = {"observed_text": text, "confidence": None if observed is None else observed.confidence,
                            "abstained": observed is None, "exact_match": None if expected is None else text == expected,
                            "character_errors": None if expected is None else _edit_distance(expected, text),
                            "elapsed_seconds": clock() - before, "parent_python_cpu_seconds": time.process_time() - cpu}
            rows.append(row)
        if (counts != {"installations": 1, "version_probes": 1, "baseline_calls": 5, "candidate_calls": 5, "downloads": 6}
                or candidate.inference_count != 5 or candidate.compile_count != 1
                or package_runner.reported_version != exact._TESSERACT_WINDOWS_VERSION_LINE):
            raise RuntimeError("CCT comparison execution count conservation failed")
        baseline_summary, candidate_summary = _summarize(rows, "baseline"), _summarize(rows, "candidate")
        files = ("lpr_cct_evidence.py", "lpr_cct.py", "lpr_ocr.py", "lpr_ocr_exact_evidence.py", "lpr_ocr_sc_evidence.py")
        report = {"evidence": "lpr-cct-retrospective-comparison-v1", "plan": plan, "plan_sha256": _hash(_canonical(plan)),
                  "runtime": {**versions, "tesseract_reported_version": package_runner.reported_version,
                              "tesseract_executable_sha256": binary_hash, "tessdata_sha256": tessdata_sha},
                  "code_sha256": {name: _hash(Path(__file__).with_name(name).read_bytes()) for name in files},
                  "cpu": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor() or "unknown",
                          "logical_cpus": os.cpu_count(), "subprocess_cpu_measured": False},
                  "github_checkout_sha": os.environ.get("GITHUB_SHA", "unknown"), "observations": rows,
                  "baseline": baseline_summary, "candidate": candidate_summary,
                  "delta_candidate_minus_baseline": {key: candidate_summary[key] - baseline_summary[key]
                                                      for key in ("exact_reads", "wrong_accepted_reads", "positive_abstentions",
                                                                  "character_errors", "generated_blank_false_accepts")},
                  **counts, "candidate_compile_count": candidate.compile_count, "candidate_inference_count": candidate.inference_count,
                  "held_out": False, "accuracy_gate_passed": False, "default_promotion": False,
                  "commercial_accuracy_claim": False, "real_negative_false_positive_rate": None}
        deadline.check()
    finally:
        try:
            _phase("cleanup")
        finally:
            shutil.rmtree(work_dir)
    if work_dir.exists(): raise RuntimeError("CCT temporary assets remain")
    report["temporary_assets_removed"] = True
    report["evidence_elapsed_seconds"] = clock() - deadline.started
    if report["evidence_elapsed_seconds"] > 180: raise TimeoutError("CCT evidence elapsed limit exceeded")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().work_dir.resolve()), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
