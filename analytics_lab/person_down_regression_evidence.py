"""One hosted, frozen replay of the reviewed Subject-4 fall/ADL pair.

Publisher activity labels are a proxy for candidate evaluation, not clinical
person-down truth. All four model files and both clips are ephemeral; only
aggregate evidence is emitted. This is a retrospective regression, not a new
held-out benchmark or an automatic promotion gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
from urllib.parse import quote
from typing import Any, Callable

from .artifacts import OPENVINO_OMZ_2023_FP16
from .evaluation import EvaluationConfig
from .openvino_pipeline import OpenVINOOMZPipelineConfig
from .source_rights import GMDCSA24, require_source
from .validation import ValidationSuiteResult, run_validation_suite
from .validation_cli import load_manifest, _safe_error_code
from .validation_seed import GMDCSA24_HELDOUT_S4, _AUTHORIZATION_REF, prepare_seed, verify_seed_media
from .video import VideoRunConfig

_REPOSITORY = "ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos"
_COMMIT = "5abac7693229900cf80f722e878fbb119211fc1c"
_HEADER = "File Name,Length (seconds),Time of Recording,Attire,Description, Classes"
_TABLES = (
    {
        "path": "Subject 4/Fall.csv", "size_bytes": 2399,
        "git_blob_sha1": "31c1625bbf51eff2e5bd767455c400f50f5786e4",
        "sha256": "d8cbc7f404dcf0cf674f18fa4c963a382aa64e7c93f2053194915bd0bdb5b4e5",
        "sample_id": "gmdcsa24-s4-fall-03",
        "row": "03.mp4,06,Day (Lights Off),Full Shirt & Pant,Walking then falling (forward) view 1, Falling (FW)[1.7 to 6]; Walking[0 to 1.7]",
        "fall_activity_interval_ms": [1700, 6000],
        "normal_activity_interval_ms": [0, 1700],
    },
    {
        "path": "Subject 4/ADL.csv", "size_bytes": 3092,
        "git_blob_sha1": "4d97c1ba3b3a43290529ae5c7be236322706816d",
        "sha256": "9b9a4aef43725126d8f8d9cf04a393ba98dfea9f30554e6391cc0584ea661c2d",
        "sample_id": "gmdcsa24-s4-adl-07",
        "row": "07.mp4,04,Day (Lights Off),Full Shirt & Pant,Doing exercise (push-up); view 3, Exercising[0 to 4]",
        "fall_activity_interval_ms": None,
        "normal_activity_interval_ms": [0, 4000],
    },
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def evidence_plan() -> dict[str, Any]:
    """Bind publisher labels, immutable assets and untouched scoring settings."""
    require_source(GMDCSA24.source_id, purpose="evaluation", require_real_world=True)
    pipeline = OpenVINOOMZPipelineConfig(video=VideoRunConfig(max_frames=1000))
    return {
        "source_repository": _REPOSITORY,
        "source_commit": _COMMIT,
        "source_rights": asdict(GMDCSA24),
        "annotations": json.loads(_canonical(_TABLES)),
        "media": [asdict(item) for item in GMDCSA24_HELDOUT_S4],
        "models": [asdict(item) for item in OPENVINO_OMZ_2023_FP16],
        "pipeline": "retail-0013_detector/cropped-pose-0001_four-keypoint/IoU-tracker/default-temporal",
        "pipeline_config": asdict(pipeline),
        "evaluation_config": asdict(EvaluationConfig()),
        "scoring_scope": "whole-clip activity proxy; annotated negative counts/rates unavailable if decoded tail is unannotated; no acceptance gate",
        "decoder": "opencv-python-headless==4.12.0.88",
        "max_media_bytes": sum(item.size_bytes for item in GMDCSA24_HELDOUT_S4),
        "max_model_bytes": sum(item.size_bytes for item in OPENVINO_OMZ_2023_FP16),
        "subset": "subject4-regression",
    }


def verify_annotations(*, opener: Callable[..., Any] = urllib.request.urlopen) -> None:
    """Verify whole publisher tables and selected rows before model/media work."""
    for table in _TABLES:
        url = f"https://raw.githubusercontent.com/{_REPOSITORY}/{_COMMIT}/" + quote(table["path"], safe="/")
        request = urllib.request.Request(url, headers={"User-Agent": "Analytics-lab-regression/1"})
        with opener(request, timeout=30) as response:
            payload = response.read(table["size_bytes"] + 1)
        if not isinstance(payload, bytes) or len(payload) != table["size_bytes"]:
            raise RuntimeError("publisher annotation byte count changed")
        git_blob = hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
        if git_blob != table["git_blob_sha1"] or hashlib.sha256(payload).hexdigest() != table["sha256"]:
            raise RuntimeError("publisher annotation identity changed")
        lines = payload.decode("utf-8").splitlines()
        filename = table["row"].split(",", 1)[0]
        selected = [line for line in lines[1:] if line.split(",", 1)[0] == filename]
        if not lines or lines[0] != _HEADER or selected != [table["row"]]:
            raise RuntimeError("publisher annotation row changed")


def run_regression(work_dir: str | Path, *, opener: Callable[..., Any] = urllib.request.urlopen) -> dict[str, Any]:
    plan = evidence_plan()
    verify_annotations(opener=opener)
    root = Path(work_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("work directory must be an existing non-symlink directory")
    # Cleanup includes download/decoder/inference failures; no artifact is uploaded.
    with tempfile.TemporaryDirectory(prefix="person-down-regression-", dir=root) as directory:
        manifest = prepare_seed(directory, opener=opener, subset="subject4-regression")
        artifact_root, samples, config = load_manifest(manifest)
        prepared_root = Path(directory)
        if (artifact_root != prepared_root / "artifacts" or asdict(config) != {
                "max_samples": 2, "required_device": "CPU",
                "max_total_video_bytes": plan["max_media_bytes"]}):
            raise RuntimeError("regression manifest budget or artifact root changed")
        if [item.sample_id for item in samples] != [item.sample_id for item in GMDCSA24_HELDOUT_S4]:
            raise RuntimeError("regression sample selection changed")
        for sample, spec, table in zip(samples, GMDCSA24_HELDOUT_S4, _TABLES):
            labels = [(item.start_timestamp_ms, item.end_timestamp_ms, item.label_id) for item in sample.labels]
            expected = [(spec.label_start_timestamp_ms, spec.label_end_timestamp_ms, spec.label_id)] if spec.label_id else []
            if (sample.start_timestamp_ms != 0 or sample.end_timestamp_ms != spec.end_timestamp_ms
                    or labels != expected
                    or (list(expected[0][:2]) if expected else None) != table["fall_activity_interval_ms"]):
                raise RuntimeError("regression activity-label binding changed")
            if (sample.site_id != "gmdcsa24-home-setup-1"
                    or sample.camera_id != f"gmdcsa24-clip-{spec.sample_id}"
                    or sample.authorization_ref != _AUTHORIZATION_REF
                    or sample.video_path != prepared_root / "media" / "gmdcsa24" / spec.relative_path
                    or verify_seed_media(sample.video_path, spec) != sample.media_sha256):
                raise RuntimeError("regression immutable source binding changed")
        result = run_validation_suite(
            samples, artifact_root=artifact_root, config=config,
            pipeline_config=OpenVINOOMZPipelineConfig(video=VideoRunConfig(max_frames=1000)),
            evaluation_config=EvaluationConfig(),
        )
        if (not isinstance(result, ValidationSuiteResult)
                or [item.sample_id for item in result.sample_runs] != [item.sample_id for item in samples]):
            raise RuntimeError("regression suite result contract changed")
        negative = result.sample_runs[1].evaluation
        positive = result.sample_runs[0].evaluation
        # A decoded tail beyond the publisher's rounded nominal endpoint is
        # usable engineering exposure, but has no additional activity annotation.
        nominal_normal_ms = _TABLES[1]["normal_activity_interval_ms"][1]
        return {
            "schema_version": 1,
            "mode": "retrospective_fall_activity_regression",
            "plan": plan,
            "plan_sha256": hashlib.sha256(_canonical(plan)).hexdigest(),
            "runtime_versions": _runtime_versions(),
            "activity_proxy_validation": asdict(result),
            "fall_activity_candidate_matches": positive.matched_events,
            "fall_activity_candidate_misses": positive.missed_episodes,
            "normal_clip_candidate_events": negative.candidate_events,
            "normal_clip_decoded_exposure_ms": negative.duration_ms,
            "normal_clip_decoded_camera_hours": negative.duration_ms / 3600000,
            "normal_clip_nominal_annotated_ms": nominal_normal_ms,
            "normal_clip_unannotated_tail_ms": max(0, negative.duration_ms - nominal_normal_ms),
            "normal_clip_observed_annotated_activity_ms": min(negative.duration_ms, nominal_normal_ms),
            "normal_clip_scored_activity_false_alerts": negative.false_alerts if negative.duration_ms <= nominal_normal_ms else None,
            "normal_clip_scored_activity_camera_hours": negative.duration_ms / 3600000 if negative.duration_ms <= nominal_normal_ms else None,
            "false_alert_gate_eligible": False,
            "zero_candidates_is_success": False,
            "new_heldout_evidence": False,
            "clinical_person_down_ground_truth": False,
            "commercial_accuracy_claim": False,
            "production_promoted": False,
            "baseline_delta": None,
            "baseline_delta_reason": "No retained comparable baseline result bound to this exact plan.",
            "interpretation": "Whole-clip activity proxy, including any unannotated decoded tail. If a tail exists, annotated negative event counts/rates are unscored; no false-alert acceptance gate is qualified. Rounded endpoints are preserved; uncovered positive labels reject scoring.",
        }


def _runtime_versions() -> dict[str, str]:
    import cv2
    import numpy
    import openvino
    if cv2.__version__ != "4.12.0" or not openvino.__version__.startswith("2026.3.1"):
        raise RuntimeError("regression runtime version changed")
    return {"opencv": cv2.__version__, "numpy": numpy.__version__, "openvino": openvino.__version__}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if (os.environ.get("GITHUB_ACTIONS") != "true"
                or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
                or not os.environ.get("RUNNER_TEMP")):
            raise RuntimeError("regression evidence requires the hosted ephemeral lane")
        runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
        root = args.work_dir.resolve()
        root.relative_to(runner_temp)
        if root == runner_temp or args.work_dir.is_symlink():
            raise ValueError("work directory must be a dedicated ephemeral child")
        root.mkdir(parents=True, exist_ok=True)
        _runtime_versions()
        report = run_regression(root)
        print(json.dumps(report, allow_nan=False, sort_keys=True, separators=(",", ":")))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as error:
        print(f"Person-down regression rejected ({type(error).__name__}; code={_safe_error_code(error)}).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
