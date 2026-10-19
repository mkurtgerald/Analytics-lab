"""Windows-only paired CCT CPU precision experiment; no accuracy promotion.

Four pinned acquisitions, two compiles, and five identical RGB inputs per arm.
The two exposed real positives and three generated blanks are retrospective
diagnostics, not held-out accuracy or representative real-negative exposure.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Callable

from . import lpr_cct_evidence as shared
from .lpr_ocr import OCRText
from .public_live_hls import LiveDeadline


def _require_hosted_windows() -> None:
    shared._require_hosted_windows()
    if not os.environ.get("GITHUB_HEAD_REF", "").startswith("evidence/lpr-cct-fp32-"):
        raise RuntimeError("precision evidence requires its dedicated Windows PR lane")


def evidence_plan() -> dict[str, Any]:
    previous = shared.evidence_plan()
    donor = dict(previous["candidate"])
    return {"schema": "lpr-cct-precision-plan-v1", "sources": previous["sources"],
            "donor": donor, "baseline_precision_requested": "default", "candidate_precision_requested": "f32",
            "shared_preprocess": shared._CCT_PREPROCESS, "min_confidence": .90,
            "source_downloads": 2, "total_downloads": 4, "calls_per_arm": 5, "total_compiles": 2,
            "real_positive_cases": 2, "generated_blank_controls": 3, "real_negative_cases": 0,
            "cooperative_budget_seconds": 180, "cleanup_reserve_seconds": 5, "hard_job_seconds": 300,
            "http_timeout_semantics": previous["http_timeout_semantics"],
            "model_training_overlap": "unknown", "held_out": False, "accuracy_gate_passed": False,
            "default_promotion": False, "commercial_accuracy_claim": False, "model_release_approved": False}


def _make_cct(path: Path, precision: str):
    from .lpr_cct import OpenVINOCCTPlateRecognizer
    return OpenVINOCCTPlateRecognizer(path, min_confidence=.90, inference_precision=precision)


def run(work_dir: Path, *, clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    _require_hosted_windows()
    work_dir = shared._work_directory(work_dir)
    deadline = LiveDeadline(clock, total_seconds=175)
    plan, versions = evidence_plan(), shared._runtime_versions()
    counts = {"downloads": 0, "baseline_calls": 0, "candidate_calls": 0}
    work_dir.mkdir()
    try:
        def download(url: str, hosts: frozenset[str], size: int) -> bytes:
            deadline.check(); counts["downloads"] += 1
            payload = shared.exact._download(url, allowed_hosts=hosts, max_bytes=size)
            deadline.check()
            return payload

        inputs = []
        for name, module in shared._SOURCES:
            shared._phase("acquire_" + name)
            payload = download(module._SOURCE_URL, frozenset({"upload.wikimedia.org"}), module._SOURCE_SIZE)
            module._verify_source(payload)
            rgb, width, height = shared._decode_crop(payload, module)
            inputs.append((name, "real_positive_retrospective", module._EXPECTED_TEXT, rgb, width, height))
            del payload
        for name, value in (("black", 0), ("gray", 127), ("white", 255)):
            inputs.append(("generated_blank_" + name, "generated_blank_control", None,
                           bytes([value]) * (128 * 64 * 3), 128, 64))
        hosts = frozenset({"github.com", "release-assets.githubusercontent.com"})
        shared._phase("acquire_cct_assets")
        model = download(shared.CCT_MODEL_URL, hosts, shared.CCT_MODEL_SIZE)
        shared.exact._verify_sha256(model, size=shared.CCT_MODEL_SIZE,
                                   sha256=shared.CCT_MODEL_SHA256, name="CCT model")
        model_path = work_dir / "cct_s_v2_global.onnx"
        model_path.write_bytes(model); del model
        config = download(shared.CCT_CONFIG_URL, hosts, shared.CCT_CONFIG_SIZE)
        shared.exact._verify_sha256(config, size=shared.CCT_CONFIG_SIZE,
                                   sha256=shared.CCT_CONFIG_SHA256, name="CCT config")
        del config  # Identity only: no YAML parsing or execution.
        recognizers, precision_metadata = {}, {}
        for arm, precision in (("baseline", "default"), ("candidate", "f32")):
            shared._phase("compile_" + arm)
            deadline.check(); recognizer = _make_cct(model_path, precision); deadline.check()
            if (recognizer.compile_count != 1 or recognizer.inference_count != 0
                    or recognizer.runtime_version != versions["openvino"]
                    or recognizer.requested_inference_precision != precision
                    or recognizer.effective_inference_precision not in ("f32", "bf16", "f16")
                    or (precision == "f32" and recognizer.effective_inference_precision != "f32")):
                raise RuntimeError("CCT precision/compile/runtime conservation failed")
            recognizers[arm] = recognizer
            precision_metadata[arm] = {"requested": precision, "effective": recognizer.effective_inference_precision}
        rows = []
        for name, scope, expected, rgb, width, height in inputs:
            ppm = shared.exact._ppm(rgb, width=width, height=height)
            row = {"case_id": name, "scope": scope, "expected_text": expected,
                   "shared_rgb24_sha256": shared._hash(rgb), "shared_input_ppm_sha256": shared._hash(ppm),
                   "dimensions": [width, height]}
            for arm, recognizer in recognizers.items():
                shared._phase(name + "_" + arm); deadline.check()
                before, cpu = clock(), time.process_time()
                observed = recognizer(ppm); counts[arm + "_calls"] += 1; deadline.check()
                if observed is not None and not isinstance(observed, OCRText):
                    raise RuntimeError("recognizer returned an unsupported OCR value")
                diagnostics = recognizer.last_read_diagnostics
                if diagnostics is None:
                    raise RuntimeError("CCT inference diagnostics missing")
                diagnostic_values = asdict(diagnostics)
                shared._canonical(diagnostic_values)  # Reject nonfinite metadata before report output.
                text = None if observed is None else observed.text
                row[arm] = {"observed_text": text, "confidence": None if observed is None else observed.confidence,
                            "abstained": observed is None, "exact_match": None if expected is None else text == expected,
                            "character_errors": None if expected is None else shared._edit_distance(expected, text),
                            "diagnostics": diagnostic_values,
                            "diagnostics_are_accepted_ocr": False,
                            "elapsed_seconds": clock() - before, "parent_python_cpu_seconds": time.process_time() - cpu}
            rows.append(row)
        if (counts != {"downloads": 4, "baseline_calls": 5, "candidate_calls": 5}
                or any(item.compile_count != 1 or item.inference_count != 5 for item in recognizers.values())):
            raise RuntimeError("CCT precision execution count conservation failed")
        baseline, candidate = shared._summarize(rows, "baseline"), shared._summarize(rows, "candidate")
        files = ("lpr_cct_precision_evidence.py", "lpr_cct_evidence.py", "lpr_cct.py", "lpr_ocr.py",
                 "lpr_ocr_exact_evidence.py", "lpr_ocr_sc_evidence.py")
        report = {"evidence": "lpr-cct-precision-retrospective-v1", "plan": plan,
                  "plan_sha256": shared._hash(shared._canonical(plan)), "runtime": versions,
                  "precision": precision_metadata, "observations": rows, "baseline": baseline, "candidate": candidate,
                  "delta_candidate_minus_baseline": {key: candidate[key] - baseline[key] for key in
                        ("exact_reads", "wrong_accepted_reads", "positive_abstentions", "character_errors", "generated_blank_false_accepts")},
                  "code_sha256": {name: shared._hash(Path(__file__).with_name(name).read_bytes()) for name in files},
                  "github_event_sha": os.environ.get("GITHUB_SHA", "unknown"),
                  "github_head_ref": os.environ.get("GITHUB_HEAD_REF", "unknown"),
                  "github_event_sha_is_checkout_identity": False,
                  "cpu": {"platform": platform.platform(), "machine": platform.machine(),
                          "processor": platform.processor() or "unknown", "logical_cpus": os.cpu_count()},
                  **counts, "total_compile_count": sum(item.compile_count for item in recognizers.values()),
                  "total_inference_count": sum(item.inference_count for item in recognizers.values()),
                  "held_out": False, "accuracy_gate_passed": False, "default_promotion": False,
                  "commercial_accuracy_claim": False, "real_negative_cases": 0,
                  "real_negative_false_positive_rate": None}
        deadline.check()
    finally:
        try:
            shared._phase("cleanup")
        finally:
            shutil.rmtree(work_dir)
    if work_dir.exists():
        raise RuntimeError("CCT temporary assets remain")
    report["temporary_assets_removed"] = True
    report["evidence_elapsed_seconds"] = clock() - deadline.started
    if report["evidence_elapsed_seconds"] > 180:
        raise TimeoutError("CCT evidence elapsed limit exceeded")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().work_dir.resolve()), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
