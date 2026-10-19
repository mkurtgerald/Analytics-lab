"""Generated-byte offline paired precision orchestration, without native work."""
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from analytics_lab import lpr_cct_precision_evidence as evidence
from analytics_lab.lpr_ocr import OCRText

# Reuse generated image/source fakes only; do not inherit another test suite.
_spec = importlib.util.spec_from_file_location("cct_generated_fixtures", Path(__file__).with_name("test_lpr_cct_evidence.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


@dataclass(frozen=True)
class Diagnostics:
    raw_text: str
    slot_maxima: tuple[float, ...]
    min_probability: float
    rejection_reason: str | None


class Fixture(_fixtures.Fixture):
    def __init__(self):
        super().__init__()
        self.arms, self.inputs = [], {"default": [], "f32": []}
        self.failure, self.bad_effective, self.bad_counts = None, False, False
        self.outputs = {"default": (OCRText("AB9", .99), None, None, OCRText("X1", .96), None),
                        "f32": (OCRText("AB1", .99), OCRText("CD2", .98), None, None, None)}

    def make(self, path, precision):
        self.assert_model = hashlib.sha256(path.read_bytes()).hexdigest() == evidence.shared.CCT_MODEL_SHA256
        self.arms.append(precision)
        fixture = self
        class Recognizer:
            compile_count = 1
            inference_count = 0
            runtime_version = fixture.versions["openvino"]
            requested_inference_precision = precision
            effective_inference_precision = "bf16" if precision == "default" or fixture.bad_effective else "f32"
            last_read_diagnostics = None
            def __call__(self, ppm):
                fixture.inputs[precision].append(ppm)
                self.inference_count += 1
                if fixture.failure == precision: raise RuntimeError("generated inference failure")
                result = fixture.outputs[precision][self.inference_count - 1]
                raw = (result.text if result else "").ljust(10, "_")
                confidence = .99 if result else .89
                self.last_read_diagnostics = Diagnostics(raw, (confidence,) * 10, confidence,
                                                        None if result else "low_confidence")
                if fixture.bad_counts and self.inference_count == 5: self.inference_count = 6
                return result
        return Recognizer()

    def run(self, root):
        with patch.object(evidence, "_require_hosted_windows"), \
                patch.object(evidence, "_make_cct", side_effect=self.make), redirect_stdout(io.StringIO()):
            return evidence.run(root / "precision", clock=self.clock)


class CCTPrecisionEvidenceTests(unittest.TestCase):
    def test_plan_pins_unchanged_rgb_cutoff_and_nonqualification(self):
        plan = evidence.evidence_plan()
        self.assertEqual([item["expected_text"] for item in plan["sources"]], ["MPR318", "354AVV"])
        self.assertEqual((plan["total_downloads"], plan["calls_per_arm"], plan["total_compiles"]), (4, 5, 2))
        self.assertEqual(plan["donor"]["model_bytes"], 5262230)
        self.assertEqual(plan["donor"]["config_bytes"], 1725)
        self.assertEqual(plan["min_confidence"], .90)
        self.assertNotIn("installer", plan)
        self.assertEqual(plan["model_training_overlap"], "unknown")
        for key in ("held_out", "accuracy_gate_passed", "default_promotion", "commercial_accuracy_claim"):
            self.assertFalse(plan[key])

    def test_four_gets_two_compiles_ten_identical_inputs_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, fixture = Path(temporary).resolve(), Fixture()
            (root / "keep").write_text("unrelated")
            with fixture.environment(root), patch.dict(os.environ, {"GITHUB_SHA": "event-sha", "GITHUB_HEAD_REF": "evidence/lpr-cct-fp32-generated"}):
                report = fixture.run(root)
            self.assertEqual(len(fixture.downloads), 4)
            self.assertEqual(len({url for url, _ in fixture.downloads}), 4)
            self.assertEqual({url for url, _ in fixture.downloads}, {evidence.shared.exact._SOURCE_URL, evidence.shared.sc._SOURCE_URL,
                             evidence.shared.CCT_MODEL_URL, evidence.shared.CCT_CONFIG_URL})
            self.assertFalse(fixture.commands)
            self.assertEqual(fixture.arms, ["default", "f32"])
            self.assertTrue(fixture.assert_model)
            self.assertEqual(fixture.inputs["default"], fixture.inputs["f32"])
            self.assertEqual((report["baseline_calls"], report["candidate_calls"], report["total_inference_count"], report["total_compile_count"]), (5, 5, 10, 2))
            self.assertEqual(report["precision"], {"baseline": {"requested": "default", "effective": "bf16"},
                                                  "candidate": {"requested": "f32", "effective": "f32"}})
            self.assertEqual((report["baseline"]["exact_reads"], report["candidate"]["exact_reads"]), (0, 2))
            self.assertEqual((report["baseline"]["generated_blank_false_accepts"], report["candidate"]["generated_blank_false_accepts"]), (1, 0))
            self.assertEqual(report["candidate"]["reference_characters"], 6)
            self.assertEqual(report["github_event_sha"], "event-sha")
            self.assertFalse(report["github_event_sha_is_checkout_identity"])
            for index, row in enumerate(report["observations"]):
                ppm = fixture.inputs["default"][index]
                self.assertEqual(row["shared_input_ppm_sha256"], hashlib.sha256(ppm).hexdigest())
                self.assertEqual(row["shared_rgb24_sha256"], hashlib.sha256(ppm.split(b"\n", 3)[3]).hexdigest())
                for arm in ("baseline", "candidate"):
                    self.assertEqual(len(row[arm]["diagnostics"]["slot_maxima"]), 10)
                    self.assertEqual(len(row[arm]["diagnostics"]["raw_text"]), 10)
                    self.assertFalse(row[arm]["diagnostics_are_accepted_ocr"])
            self.assertIsNone(report["real_negative_false_positive_rate"])
            self.assertFalse(report["accuracy_gate_passed"])
            self.assertFalse((root / "precision").exists())
            self.assertEqual((root / "keep").read_text(), "unrelated")
            self.assertNotIn(str(root), json.dumps(report))

    def test_regression_and_silent_arms_are_reported_without_promotion_or_execution_failure(self):
        for silent in (False, True):
            with self.subTest(silent=silent), tempfile.TemporaryDirectory() as temporary:
                root, fixture = Path(temporary).resolve(), Fixture()
                if silent:
                    fixture.outputs = {key: (None,) * 5 for key in fixture.outputs}
                else:
                    fixture.outputs["default"], fixture.outputs["f32"] = fixture.outputs["f32"], fixture.outputs["default"]
                with fixture.environment(root): report = fixture.run(root)
                self.assertFalse(report["default_promotion"])
                self.assertFalse(report["accuracy_gate_passed"])
                if silent:
                    self.assertEqual(report["candidate"]["positive_abstentions"], 2)
                else:
                    self.assertEqual(report["delta_candidate_minus_baseline"]["exact_reads"], -2)
                    self.assertEqual(report["delta_candidate_minus_baseline"]["generated_blank_false_accepts"], 1)

    def test_all_identity_failures_stop_before_compile_and_cleanup(self):
        for problem in ("source", "crop", "model", "config"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temporary:
                root, fixture = Path(temporary).resolve(), Fixture()
                with fixture.environment(root):
                    if problem == "crop": fixture.images[fixture.payloads[evidence.shared.exact._SOURCE_URL]].pixel = b"\0\0\0"
                    else:
                        url = {"source": evidence.shared.exact._SOURCE_URL, "model": evidence.shared.CCT_MODEL_URL,
                               "config": evidence.shared.CCT_CONFIG_URL}[problem]
                        fixture.payloads[url] += b"overflow"
                    with self.assertRaises(RuntimeError): fixture.run(root)
                self.assertFalse(fixture.arms)
                self.assertFalse((root / "precision").exists())

    def test_inference_precision_count_and_cleanup_log_failures_remove_assets(self):
        for problem in ("default", "f32", "effective", "counts", "cleanup_log"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temporary:
                root, fixture = Path(temporary).resolve(), Fixture()
                fixture.failure, fixture.bad_effective, fixture.bad_counts = problem, problem == "effective", problem == "counts"
                with fixture.environment(root):
                    def phase(name):
                        if problem == "cleanup_log" and name == "cleanup": raise BrokenPipeError("generated closed stdout")
                    with patch.object(evidence.shared, "_phase", side_effect=phase), self.assertRaises((RuntimeError, BrokenPipeError)):
                        fixture.run(root)
                self.assertFalse((root / "precision").exists())

    def test_429_and_late_download_stop_without_retry_or_native(self):
        for late in (False, True):
            with self.subTest(late=late), tempfile.TemporaryDirectory() as temporary:
                root, fixture = Path(temporary).resolve(), Fixture()
                with fixture.environment(root):
                    def fail(*args, **kwargs):
                        if not late: raise HTTPError(args[0], 429, "generated rate limit", {}, None)
                        fixture.clock.value = 176
                        return fixture.payloads[args[0]]
                    with patch.object(evidence.shared.exact, "_download", side_effect=fail) as download:
                        with self.assertRaises((HTTPError, TimeoutError)): fixture.run(root)
                    self.assertEqual(download.call_count, 1)
                self.assertFalse(fixture.arms)
                self.assertFalse((root / "precision").exists())

    def test_effective_legacy_precision_is_measured_not_assumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, fixture = Path(temporary).resolve(), Fixture()
            original = fixture.make
            def already_f32(*args):
                value = original(*args)
                value.effective_inference_precision = "f32"
                return value
            with fixture.environment(root), patch.object(fixture, "make", side_effect=already_f32): report = fixture.run(root)
            self.assertEqual(report["precision"]["baseline"]["effective"], "f32")

    def test_dedicated_hosted_guard_and_cli_canonical_path(self):
        with patch.object(evidence.shared, "_require_hosted_windows"):
            for branch in ("evidence/lpr-cct-old", "main", ""):
                with patch.dict(os.environ, {"GITHUB_HEAD_REF": branch}), self.assertRaises(RuntimeError): evidence._require_hosted_windows()
            with patch.dict(os.environ, {"GITHUB_HEAD_REF": "evidence/lpr-cct-fp32-generated"}): evidence._require_hosted_windows()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "new"
            with patch.object(sys, "argv", ["precision", "--work-dir", str(path)]), \
                    patch.object(evidence, "run", return_value={}) as run, redirect_stdout(io.StringIO()):
                self.assertEqual(evidence.main(), 0)
            run.assert_called_once_with(path.resolve())


if __name__ == "__main__":
    unittest.main()
