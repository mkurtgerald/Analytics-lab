"""One retrospective, same-pixel PSM 7/13 comparison on three admitted crops.

This Windows-hosted evidence harness acquires each pinned asset once, installs
the existing Tesseract package once and performs exactly six OCR calls. It does
not measure detector accuracy, held-out generalization or negative false reads.
Historical gray/Otsu observations are not the paired control.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import lpr_ocr_exact_evidence as exact
from . import lpr_ocr_sc_evidence as sc
from . import lpr_ocr_wa_evidence as wa
from .lpr_ocr import LPROCRConfig, OCRText, TesseractPlateRecognizer

_SOURCES = (("land_rover", exact), ("south_carolina", sc), ("washington", wa))
_PAYLOAD_BYTES = 39_945_587
_EVIDENCE_SECONDS = 210.0
_CLEANUP_SECONDS = 5.0
_PREPROCESS = "raw-rgb24;native-size;no-resize;no-threshold;no-invert"
_BUDGET_SEMANTICS = {
    "elapsed_budget": "cooperative elapsed rejection budget; not a wall-clock interrupt",
    "http_timeout": "open/header inactivity capped at 15s; each body read capped by remaining budget",
    "hard_hosted_job_limit_seconds": 300,
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _file_hash(path: Path, deadline: _Deadline | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if deadline is not None:
                deadline.timeout(15.0)
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _source_identity(name: str, module: Any) -> dict[str, Any]:
    box = getattr(module, "_PLATE_BOX_PX", None) or module._PLATE_TEXT_BOX_PX
    width, height = box[2] - box[0], box[3] - box[1]
    return {
        "source_id": name, "license_id": "CC0-1.0", "rights_page": module._RIGHTS_PAGE,
        "source_url": module._SOURCE_URL, "source_bytes": module._SOURCE_SIZE,
        "source_sha1": module._SOURCE_SHA1, "source_sha256": module._SOURCE_SHA256,
        "source_dimensions": [module._SOURCE_WIDTH, module._SOURCE_HEIGHT],
        "crop_box_px_half_open": list(box), "crop_dimensions": [width, height],
        "crop_rgb24_sha256": module._CROP_RGB24_SHA256,
        "crop_rgb24_bytes": width * height * 3,
        "expected_text": module._EXPECTED_TEXT,
        "annotation_scope": "existing_manual_pre_ocr_serial;retrospectively_reused",
    }


def comparison_plan() -> dict[str, Any]:
    sources = [_source_identity(name, module) for name, module in _SOURCES]
    if len(sources) != 3 or [item["source_id"] for item in sources] != [
        "land_rover", "south_carolina", "washington"
    ]:
        raise RuntimeError("comparison source scope changed")
    total = sum(item["source_bytes"] for item in sources) + exact._TESSERACT_INSTALLER_SIZE + exact._TESSDATA_SIZE
    if total != _PAYLOAD_BYTES or any(
        not isinstance(item["source_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", item["source_sha256"]) is None
        or not isinstance(item["crop_rgb24_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", item["crop_rgb24_sha256"]) is None
        for item in sources
    ):
        raise RuntimeError("comparison requires the complete pinned payload identities")
    files = ("lpr_serial_comparison.py", "lpr_ocr.py", "lpr_ocr_exact_evidence.py",
             "lpr_ocr_sc_evidence.py", "lpr_ocr_wa_evidence.py")
    return {
        "schema": "lpr-serial-comparison-plan-v1", "sources": sources,
        "profiles": [asdict(LPROCRConfig(tesseract_psm=psm, tesseract_timeout_seconds=15.0))
                     for psm in (7, 13)],
        "preprocess": _PREPROCESS, "source_acquisitions": 3, "ocr_calls": 6,
        "version_probes": 2, "installations": 1, "payload_bytes": total,
        "payload_budget_excludes": "Python runtime/decoder wheels and HTTP overhead",
        "evidence_elapsed_limit_seconds": _EVIDENCE_SECONDS,
        "bound_semantics": dict(_BUDGET_SEMANTICS),
        "installer": {"url": exact._TESSERACT_INSTALLER_URL,
                      "bytes": exact._TESSERACT_INSTALLER_SIZE,
                      "sha256": exact._TESSERACT_INSTALLER_SHA256},
        "ocr_model": {"url": exact._TESSDATA_URL, "bytes": exact._TESSDATA_SIZE,
                      "repository_commit": exact._TESSDATA_COMMIT,
                      "git_blob_sha1": exact._TESSDATA_GIT_BLOB_SHA1,
                      "license_id": "Apache-2.0"},
        "tesseract_source_commit": "db0ec62f81b0737fbbe184d8fea40af5738f8eef",
        "project_source_sha256": {name: _file_hash(Path(__file__).with_name(name)) for name in files},
        "paired_control": "fresh_psm7_raw_rgb_same_run",
        "historical_gray_otsu_is_paired_control": False,
        "held_out": False, "positive_cases": 3, "negative_cases": 0,
        "accuracy_gate_passed": False, "default_promotion": False,
        "commercial_accuracy_claim": False, "performance_claim": False,
    }


class _Deadline:
    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.started = clock()

    def timeout(self, cap: float) -> float:
        remaining = _EVIDENCE_SECONDS - _CLEANUP_SECONDS - (self.clock() - self.started)
        if remaining <= 0:
            raise TimeoutError("LPR comparison evidence deadline exhausted")
        return min(cap, remaining)


def _download(url: str, *, size: int, hosts: frozenset[str], deadline: _Deadline,
              opener: Callable[..., Any]) -> bytes:
    request = Request(url, headers={"User-Agent": exact._USER_AGENT})
    with opener(request, timeout=deadline.timeout(15.0)) as response:
        final = urlparse(response.geturl())
        if final.scheme != "https" or final.hostname not in hosts:
            raise RuntimeError("comparison redirect escaped reviewed hosts")
        if getattr(response, "status", None) != 200:
            raise RuntimeError("comparison acquisition requires HTTP 200")
        declared = response.headers.get("Content-Length")
        # Only fixed-length bodies are admitted. HTTPResponse's chunk/trailer
        # parsing can perform multiple receives inside read1, beyond our
        # per-chunk deadline checks. These five immutable assets have fixed
        # published lengths; do not silently switch transfer framing.
        if (declared is None or int(declared) != size
                or response.headers.get("Transfer-Encoding")):
            raise RuntimeError("comparison payload requires its exact Content-Length without transfer encoding")
        # urllib's HTTPResponse has no public per-read timeout setter. Fail
        # closed if its socket cannot be bounded, rather than inheriting a stale
        # 15-second timeout when the shared deadline is nearly exhausted.
        sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        if not callable(getattr(sock, "settimeout", None)) or not callable(getattr(response, "read1", None)):
            raise RuntimeError("comparison bounded HTTP stream unavailable")
        blocks: list[bytes] = []
        count = 0
        while True:
            sock.settimeout(deadline.timeout(15.0))
            block = response.read1(min(64 * 1024, size + 1 - count))
            deadline.timeout(15.0)
            if not block:
                break
            count += len(block)
            if count > size:
                raise RuntimeError("comparison payload exceeds exact byte cap")
            blocks.append(block)
            # HTTPResponse.read1 closes its socket when the declared body is
            # exhausted. Do not set a timeout on that closed socket merely to
            # obtain a redundant EOF.
            if count == size:
                break
    if count != size:
        raise RuntimeError("comparison payload size mismatch")
    return b"".join(blocks)


def _require_hosted_windows() -> None:
    if (os.name != "nt" or os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
            or os.environ.get("RUNNER_OS") != "Windows"
            or os.environ.get("GITHUB_REPOSITORY") != "mkurtgerald/Analytics-lab"
            or os.environ.get("GITHUB_EVENT_NAME") != "pull_request"
            or not os.environ.get("GITHUB_HEAD_REF", "").startswith("evidence/lpr-serial-compare-")):
        raise RuntimeError("LPR comparison requires its GitHub-hosted Windows PR lane")


def _work_directory(path: Path) -> Path:
    value = os.environ.get("RUNNER_TEMP")
    if not value:
        raise RuntimeError("RUNNER_TEMP is required")
    root = Path(value).absolute()
    candidate = path.absolute()
    if (not root.is_dir() or root.is_symlink() or root.resolve() != root
            or candidate.parent != root or candidate.resolve() != candidate
            or candidate.is_symlink() or candidate.exists()):
        raise RuntimeError("comparison requires a new dedicated directory directly below RUNNER_TEMP")
    return candidate


def _runtime_versions() -> dict[str, str]:
    import cv2
    import numpy
    if cv2.__version__ != "4.12.0" or numpy.__version__ != "2.2.6":
        raise RuntimeError("comparison decoder runtime differs from reviewed versions")
    return {"python": platform.python_version(), "opencv": cv2.__version__,
            "opencv_wheel": "4.12.0.88", "numpy": numpy.__version__}


def _decode_crop(payload: bytes, module: Any) -> bytes:
    import cv2
    import numpy
    image = cv2.imdecode(numpy.frombuffer(payload, dtype=numpy.uint8), cv2.IMREAD_COLOR)
    if (image is None or tuple(image.shape) != (module._SOURCE_HEIGHT, module._SOURCE_WIDTH, 3)
            or str(image.dtype) != "uint8"):
        raise RuntimeError("comparison source decode dimensions/type mismatch")
    if module is exact:
        left, top, right, bottom = module._PLATE_BOX_PX
        rgb = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB).tobytes(order="C")
        width, height = right - left, bottom - top
    else:
        rgb, width, height, _ = module._crop_identity(image)
    if hashlib.sha256(rgb).hexdigest() != module._CROP_RGB24_SHA256:
        raise RuntimeError("comparison crop RGB24 identity mismatch")
    ppm = exact._ppm(rgb, width=width, height=height)
    if len(ppm) > 16 * 1024 * 1024:
        raise RuntimeError("comparison crop exceeds OCR byte bound")
    return ppm


def _character_errors(expected: str, observed: str | None) -> int:
    previous = list(range(len(expected) + 1))
    for index, character in enumerate(observed or "", 1):
        current = [index]
        for position, reference in enumerate(expected, 1):
            current.append(min(current[-1] + 1, previous[position] + 1,
                               previous[position - 1] + (character != reference)))
        previous = current
    return previous[-1]


def _aggregate(rows: list[dict[str, Any]], psm: int) -> dict[str, Any]:
    observations = [row[f"psm{psm}"] for row in rows]
    exact_reads = sum(item["exact_match"] for item in observations)
    abstentions = sum(item["abstained"] for item in observations)
    errors = sum(item["character_errors"] for item in observations)
    characters = sum(len(row["expected_text"]) for row in rows)
    return {"cases": len(rows), "exact_reads": exact_reads, "abstentions": abstentions,
            "incorrect_reads": len(rows) - exact_reads - abstentions,
            "character_errors": errors, "reference_characters": characters,
            "character_error_rate": errors / characters,
            "false_positive_rate": None}


def _phase(name: str, **metadata: Any) -> None:
    print(json.dumps({"phase": name, **metadata}, sort_keys=True), flush=True)


def run(work_dir: Path, *, opener: Callable[..., Any] = urlopen,
        runner: Callable[..., Any] = subprocess.run,
        clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    _require_hosted_windows()
    work_dir = _work_directory(work_dir)
    deadline = _Deadline(clock)
    plan = comparison_plan()
    versions = _runtime_versions()
    counts = {"installations": 0, "version_probes": 0, "ocr_calls": 0}
    work_dir.mkdir()
    try:
        crops: list[bytes] = []
        for name, module in _SOURCES:
            _phase("source_acquisition", source_id=name)
            payload = _download(module._SOURCE_URL, size=module._SOURCE_SIZE,
                                hosts=frozenset({"upload.wikimedia.org"}), deadline=deadline, opener=opener)
            module._verify_source(payload)
            crops.append(_decode_crop(payload, module))
            del payload
            deadline.timeout(15.0)

        _phase("runtime_acquisition")
        installer = _download(exact._TESSERACT_INSTALLER_URL, size=exact._TESSERACT_INSTALLER_SIZE,
                              hosts=frozenset({"github.com", "release-assets.githubusercontent.com"}),
                              deadline=deadline, opener=opener)
        exact._verify_sha256(installer, size=exact._TESSERACT_INSTALLER_SIZE,
                             sha256=exact._TESSERACT_INSTALLER_SHA256, name="Tesseract installer")
        installer_path = work_dir / "tesseract-ocr-w64-setup-5.5.3.20260724.exe"
        installer_path.write_bytes(installer)
        del installer
        tessdata = _download(exact._TESSDATA_URL, size=exact._TESSDATA_SIZE,
                             hosts=frozenset({"raw.githubusercontent.com"}), deadline=deadline, opener=opener)
        tessdata_sha256 = exact._verify_tessdata_bytes(tessdata)
        data_dir = work_dir / "tessdata"
        data_dir.mkdir()
        traineddata = data_dir / "eng.traineddata"
        traineddata.write_bytes(tessdata)
        del tessdata

        def bounded_runner(args: Any, **kwargs: Any) -> Any:
            kwargs["timeout"] = deadline.timeout(float(kwargs["timeout"]))
            result = runner(args, **kwargs)
            deadline.timeout(15.0)
            return result

        _phase("runtime_installation")
        install_dir = work_dir / "tesseract"
        counts["installations"] += 1
        installed = bounded_runner([str(installer_path), "/S", f"/D={install_dir}"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120.0, check=False)
        if installed.returncode != 0:
            raise RuntimeError("comparison Tesseract installation failed")
        binaries = tuple(install_dir.rglob("tesseract.exe"))
        if (len(binaries) != 1 or binaries[0].is_symlink() or not binaries[0].is_file()
                or install_dir.resolve() not in binaries[0].resolve().parents):
            raise RuntimeError("comparison Tesseract executable is not uniquely bounded")
        binary_hash = _file_hash(binaries[0], deadline)

        def ocr_runner(args: Any, **kwargs: Any) -> Any:
            key = "version_probes" if list(args) == [str(binaries[0]), "--version"] else "ocr_calls"
            counts[key] += 1
            return bounded_runner(args, **kwargs)

        package_runner = exact._OfficialWindowsPackageRunner(str(binaries[0]), runner=ocr_runner)
        recognizers = {
            psm: TesseractPlateRecognizer(traineddata, binary=str(binaries[0]),
                                         config=LPROCRConfig(tesseract_psm=psm, tesseract_timeout_seconds=15.0),
                                         runner=package_runner)
            for psm in (7, 13)
        }
        if any(item.artifact_identity.sha256 != tessdata_sha256 for item in recognizers.values()):
            raise RuntimeError("comparison traineddata verification disagreement")
        rows: list[dict[str, Any]] = []
        for source, ppm in zip(plan["sources"], crops, strict=True):
            row = {"source_id": source["source_id"], "expected_text": source["expected_text"],
                   "input_ppm_sha256": hashlib.sha256(ppm).hexdigest(), "input_ppm_bytes": len(ppm)}
            for psm, recognizer in recognizers.items():
                _phase("ocr", source_id=source["source_id"], psm=psm)
                before = clock()
                parent_cpu = time.process_time()
                observed: OCRText | None = recognizer(ppm)
                deadline.timeout(15.0)
                text = None if observed is None else observed.text
                row[f"psm{psm}"] = {
                    "observed_text": text, "observed_confidence": None if observed is None else observed.confidence,
                    "abstained": observed is None, "exact_match": text == source["expected_text"],
                    "character_errors": _character_errors(source["expected_text"], text),
                    "elapsed_seconds": round(clock() - before, 6),
                    "parent_python_cpu_seconds": round(time.process_time() - parent_cpu, 6),
                }
            rows.append(row)
        if counts != {"installations": 1, "version_probes": 2, "ocr_calls": 6}:
            raise RuntimeError("comparison native call conservation failed")
        if package_runner.reported_version != exact._TESSERACT_WINDOWS_VERSION_LINE:
            raise RuntimeError("comparison package version was not verified")
        baseline, candidate = _aggregate(rows, 7), _aggregate(rows, 13)
        report = {
            "evidence": "lpr-serial-same-rgb-psm7-vs13-v1", "plan": plan,
            "plan_sha256": hashlib.sha256(_canonical(plan)).hexdigest(),
            "runtime": {**versions, "tesseract_reported_version": package_runner.reported_version,
                        "executable_sha256": binary_hash, "tessdata_sha256": tessdata_sha256},
            "cpu": {"platform": platform.platform(), "machine": platform.machine(),
                    "processor": platform.processor() or "unknown", "logical_cpus": os.cpu_count(),
                    "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
                    "subprocess_cpu_time_measured": False},
            "github_checkout_sha": os.environ.get("GITHUB_SHA", "unknown"),
            "observations": rows, "baseline_psm7": baseline, "candidate_psm13": candidate,
            "delta_candidate_minus_baseline": {key: candidate[key] - baseline[key] for key in
                                               ("exact_reads", "abstentions", "incorrect_reads", "character_errors")},
            **counts, "source_acquisitions": len(crops), "assets_acquired": 5,
            "payload_bytes": _PAYLOAD_BYTES, "execution_complete": True,
            "bound_semantics": dict(_BUDGET_SEMANTICS),
            "accuracy_gate_passed": False, "default_promotion": False,
            "held_out": False, "commercial_accuracy_claim": False, "performance_claim": False,
        }
        deadline.timeout(15.0)
    finally:
        try:
            _phase("cleanup")
        finally:
            shutil.rmtree(work_dir)
    if work_dir.exists():
        raise RuntimeError("comparison temporary assets remain")
    report["temporary_assets_removed"] = True
    report["evidence_elapsed_seconds"] = round(clock() - deadline.started, 6)
    if report["evidence_elapsed_seconds"] > _EVIDENCE_SECONDS:
        raise TimeoutError("LPR comparison evidence elapsed limit exceeded")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.work_dir), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
