"""Offline orchestration with generated assets, decoder and native fakes."""
from contextlib import ExitStack, contextmanager, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from analytics_lab import lpr_cct_evidence as evidence
from analytics_lab import lpr_ocr
from analytics_lab.lpr_ocr import OCRText


class Clock:
    value = 0.0
    def __call__(self): return self.value


class Image:
    dtype = "uint8"
    def __init__(self, pixel, width, height):
        self.pixel, self.shape = pixel, (height, width, 3)
    def __getitem__(self, slices):
        y, x = slices
        return Image(self.pixel, x.stop - x.start, y.stop - y.start)
    def tobytes(self, **kwargs): return self.pixel * (self.shape[0] * self.shape[1])


class Buffer:
    def __init__(self, data): self.data = data
    def reshape(self, shape): return Image(self.data[:3], shape[1], shape[0])


class Fixture:
    def __init__(self):
        self.clock = Clock()
        self.downloads, self.commands, self.candidate_inputs, self.decodes = [], [], [], []
        self.payloads, self.images = {}, {}
        self.fail_url, self.install_error, self.candidate_error = None, False, False
        self.candidate_constructed, self.baseline_call = 0, 0
        self.candidate_count_override = None
        self.version = evidence.exact._TESSERACT_WINDOWS_VERSION_LINE
        self.versions = {"python": "generated", "opencv": "4.12.0", "opencv_wheel": "4.12.0.88",
                         "numpy": "2.2.6", "openvino": "2026.3.1-generated", "openvino_telemetry": "2025.2.0"}

    def download(self, url, **kwargs):
        self.downloads.append((url, kwargs))
        if url == self.fail_url: raise HTTPError(url, 429, "generated rate limit", {}, None)
        return self.payloads[url]

    def runner(self, args, **kwargs):
        self.commands.append((list(args), kwargs))
        self.clock.value += 1
        if args[1] == "/S":
            directory = Path(args[2][3:]); directory.mkdir()
            (directory / "tesseract.exe").write_bytes(b"generated-tesseract")
            return subprocess.CompletedProcess(args, int(self.install_error), stdout=b"", stderr=b"")
        if args[-1] == "--version":
            return subprocess.CompletedProcess(args, 0, stdout=(self.version + "\n").encode())
        self.baseline_call += 1
        text = ("AB9", "CD2", None, None, None)[self.baseline_call - 1]
        tsv = "level\tconf\ttext\n" + ("1\t-1\t\n" if text is None else f"5\t99\t{text}\n")
        return subprocess.CompletedProcess(args, 0, stdout=tsv.encode(), stderr=b"")

    def candidate(self, path):
        self.candidate_constructed += 1
        assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence.CCT_MODEL_SHA256
        fixture = self
        class Recognizer:
            compile_count = 1
            inference_count = 0
            runtime_version = fixture.versions["openvino"]
            def __call__(self, ppm):
                fixture.candidate_inputs.append(ppm)
                self.inference_count += 1
                if fixture.candidate_error: raise RuntimeError("generated candidate failure")
                outputs = (OCRText("AB1", .99), None, None, OCRText("X1", .96), None)
                result = outputs[self.inference_count - 1]
                if fixture.candidate_count_override is not None and self.inference_count == 5:
                    self.inference_count = fixture.candidate_count_override
                return result
        return Recognizer()

    @contextmanager
    def environment(self, root):
        with ExitStack() as stack:
            stack.enter_context(patch.object(evidence, "_require_hosted_windows"))
            stack.enter_context(patch.dict(os.environ, {"RUNNER_TEMP": str(root)}))
            stack.enter_context(patch.object(evidence, "_runtime_versions", return_value=self.versions))
            stack.enter_context(patch.object(evidence.exact, "_download", side_effect=self.download))
            stack.enter_context(patch.object(evidence, "_make_cct", side_effect=self.candidate))
            for index, (_, module) in enumerate(evidence._SOURCES):
                payload = f"source-{index}".encode(); rgb = bytes((index + 3, index + 2, index + 1)) * 2
                values = {"_SOURCE_SIZE": len(payload), "_SOURCE_SHA1": hashlib.sha1(payload).hexdigest(),
                          "_SOURCE_SHA256": hashlib.sha256(payload).hexdigest(), "_SOURCE_WIDTH": 3,
                          "_SOURCE_HEIGHT": 2, "_CROP_RGB24_SHA256": hashlib.sha256(rgb).hexdigest(),
                          "_EXPECTED_TEXT": ("AB1", "CD2")[index],
                          "_PLATE_BOX_PX" if module is evidence.exact else "_PLATE_TEXT_BOX_PX": (0, 0, 2, 1)}
                for key, value in values.items(): stack.enter_context(patch.object(module, key, value))
                self.payloads[module._SOURCE_URL] = payload
                self.images[payload] = Image(bytes((index + 1, index + 2, index + 3)), 3, 2)
            installer, model, config, trained = b"installer", b"cct-model", b"yaml-metadata", b"traineddata"
            for key, value in {"CCT_MODEL_SIZE": len(model), "CCT_MODEL_SHA256": hashlib.sha256(model).hexdigest(),
                               "CCT_CONFIG_SIZE": len(config), "CCT_CONFIG_SHA256": hashlib.sha256(config).hexdigest()}.items():
                stack.enter_context(patch.object(evidence, key, value))
            blob = evidence.exact._git_blob_sha1(trained)
            for key, value in {"_TESSERACT_INSTALLER_SIZE": len(installer),
                               "_TESSERACT_INSTALLER_SHA256": hashlib.sha256(installer).hexdigest(),
                               "_TESSDATA_SIZE": len(trained), "_TESSDATA_GIT_BLOB_SHA1": blob}.items():
                stack.enter_context(patch.object(evidence.exact, key, value))
            stack.enter_context(patch.object(lpr_ocr, "_TESSDATA_ENG_SIZE", len(trained)))
            stack.enter_context(patch.object(lpr_ocr, "_TESSDATA_ENG_GIT_BLOB_SHA1", blob))
            self.payloads.update({evidence.exact._TESSERACT_INSTALLER_URL: installer,
                                  evidence.exact._TESSDATA_URL: trained,
                                  evidence.CCT_MODEL_URL: model, evidence.CCT_CONFIG_URL: config})
            def decode(buffer, flags):
                self.decodes.append(buffer.data)
                return self.images[buffer.data]
            def convert(image, mode):
                if mode == 1: return Image(image.pixel[::-1], image.shape[1], image.shape[0])
                return image
            cv2 = SimpleNamespace(IMREAD_COLOR=1, COLOR_BGR2RGB=1, COLOR_RGB2GRAY=2, COLOR_GRAY2RGB=3,
                                  THRESH_BINARY=1, THRESH_OTSU=2, imdecode=decode, cvtColor=convert,
                                  threshold=lambda image, *args: (0, Image(b"\xff\xff\xff", image.shape[1], image.shape[0])))
            numpy = SimpleNamespace(uint8="uint8", frombuffer=lambda data, **kwargs: Buffer(data))
            stack.enter_context(patch.dict(sys.modules, {"cv2": cv2, "numpy": numpy}))
            yield self

    def run(self, root):
        with redirect_stdout(io.StringIO()):
            return evidence.run(root / "cct", runner=self.runner, clock=self.clock)


class CCTEvidenceTests(unittest.TestCase):
    def test_real_plan_pins_and_scope_are_explicit(self):
        plan = evidence.evidence_plan()
        self.assertEqual([item["expected_text"] for item in plan["sources"]], ["MPR318", "354AVV"])
        self.assertEqual(plan["candidate"]["model_bytes"], 5262230)
        self.assertEqual(plan["candidate"]["config_bytes"], 1725)
        self.assertEqual(plan["total_downloads"], 6)
        self.assertEqual(plan["baseline"]["psm"], 7)
        self.assertEqual(plan["baseline"]["config"]["min_ocr_confidence"], 0)
        self.assertEqual(plan["model_training_overlap"], "unknown")
        for key in ("held_out", "default_promotion", "accuracy_gate_passed", "commercial_accuracy_claim"):
            self.assertFalse(plan[key])

    def test_six_acquisitions_shared_pixels_ten_calls_and_separate_blank_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, fixture = Path(temporary).resolve(), Fixture()
            sentinel = root / "keep"; sentinel.write_text("unrelated")
            with fixture.environment(root): report = fixture.run(root)
            self.assertEqual(len(fixture.downloads), 6)
            self.assertEqual(len(set(url for url, _ in fixture.downloads)), 6)
            self.assertEqual(len(fixture.decodes), 2)
            self.assertEqual(len(fixture.commands), 7)
            self.assertEqual(fixture.candidate_constructed, 1)
            self.assertEqual((report["baseline_calls"], report["candidate_calls"], report["candidate_inference_count"]), (5, 5, 5))
            self.assertEqual(report["candidate_compile_count"], 1)
            self.assertEqual((report["installations"], report["version_probes"]), (1, 1))
            self.assertEqual((report["baseline"]["exact_reads"], report["baseline"]["wrong_accepted_reads"], report["baseline"]["positive_abstentions"]), (1, 1, 0))
            self.assertEqual((report["candidate"]["exact_reads"], report["candidate"]["wrong_accepted_reads"], report["candidate"]["positive_abstentions"]), (1, 0, 1))
            self.assertEqual(report["candidate"]["generated_blank_false_accepts"], 1)
            self.assertEqual(report["baseline"]["generated_blank_false_accepts"], 0)
            self.assertIsNone(report["candidate"]["real_negative_false_positive_rate"])
            self.assertEqual(report["candidate"]["character_errors"], 3)
            self.assertEqual(report["baseline"]["character_errors"], 1)
            for index, row in enumerate(report["observations"]):
                ppm = fixture.candidate_inputs[index]
                self.assertEqual(row["candidate_input_ppm_sha256"], hashlib.sha256(ppm).hexdigest())
                self.assertEqual(row["shared_rgb24_sha256"], hashlib.sha256(ppm.split(b"\n", 3)[3]).hexdigest())
                baseline_ppm = [kwargs["input"] for args, kwargs in fixture.commands if args[1] == "stdin"][index]
                self.assertEqual(row["baseline_input_ppm_sha256"], hashlib.sha256(baseline_ppm).hexdigest())
            self.assertEqual([row["dimensions"] for row in report["observations"][2:]], [[128, 64]] * 3)
            self.assertEqual({args[args.index("--psm") + 1] for args, _ in fixture.commands if args[1] == "stdin"}, {"7"})
            self.assertFalse((root / "cct").exists())
            self.assertEqual(sentinel.read_text(), "unrelated")
            self.assertNotIn(str(root), json.dumps(report))
            self.assertTrue(report["temporary_assets_removed"])

    def test_identity_overflow_or_native_failures_clean_before_complete_report(self):
        for problem in ("source", "crop", "installer", "trained", "model", "config", "install", "candidate", "count"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temporary:
                root, fixture = Path(temporary).resolve(), Fixture()
                with fixture.environment(root):
                    if problem in ("source", "installer", "trained", "model", "config"):
                        url = {"source": evidence.exact._SOURCE_URL, "installer": evidence.exact._TESSERACT_INSTALLER_URL,
                               "trained": evidence.exact._TESSDATA_URL, "model": evidence.CCT_MODEL_URL,
                               "config": evidence.CCT_CONFIG_URL}[problem]
                        fixture.payloads[url] += b"overflow"
                    elif problem == "crop": fixture.images[fixture.payloads[evidence.exact._SOURCE_URL]].pixel = b"\0\0\0"
                    elif problem == "install": fixture.install_error = True
                    elif problem == "candidate": fixture.candidate_error = True
                    else: fixture.candidate_count_override = 6
                    with self.assertRaises((RuntimeError, ValueError)): fixture.run(root)
                    if problem in ("source", "crop", "installer", "trained", "model", "config"):
                        self.assertEqual(fixture.candidate_constructed, 0)
                        self.assertFalse(fixture.commands)
                self.assertFalse((root / "cct").exists())

    def test_http_429_stops_once_without_install_or_mirror(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, fixture = Path(temporary).resolve(), Fixture()
            with fixture.environment(root):
                fixture.fail_url = evidence.exact._SOURCE_URL
                with self.assertRaises(HTTPError): fixture.run(root)
            self.assertEqual(len(fixture.downloads), 1)
            self.assertFalse(fixture.commands)
            self.assertFalse((root / "cct").exists())

    def test_version_mismatch_and_broken_cleanup_logging_still_remove_assets(self):
        for problem in ("version", "cleanup_log"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temporary:
                root, fixture = Path(temporary).resolve(), Fixture()
                with fixture.environment(root):
                    if problem == "version": fixture.version = "tesseract v5.5.4"
                    def phase(name):
                        if problem == "cleanup_log" and name == "cleanup": raise BrokenPipeError("generated closed log")
                    with patch.object(evidence, "_phase", side_effect=phase), self.assertRaises((RuntimeError, BrokenPipeError)):
                        fixture.run(root)
                self.assertFalse((root / "cct").exists())

    def test_cli_passes_canonical_resolved_windows_compatible_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "child"
            with patch.object(sys, "argv", ["lpr_cct_evidence", "--work-dir", str(path)]), \
                    patch.object(evidence, "run", return_value={}) as run, redirect_stdout(io.StringIO()):
                self.assertEqual(evidence.main(), 0)
            run.assert_called_once_with(path.resolve())

    def test_cooperative_late_acquisition_stops_followup_and_cleans(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, fixture = Path(temporary).resolve(), Fixture()
            with fixture.environment(root):
                def late(*args, **kwargs):
                    fixture.clock.value = 176
                    return fixture.payloads[evidence.exact._SOURCE_URL]
                with patch.object(evidence.exact, "_download", side_effect=late), self.assertRaises(TimeoutError): fixture.run(root)
            self.assertFalse(fixture.commands)
            self.assertFalse((root / "cct").exists())

    def test_hosted_guard_and_canonical_direct_child_contract(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError): evidence._require_hosted_windows()
        valid = {"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_OS": "Windows",
                 "GITHUB_REPOSITORY": "mkurtgerald/Analytics-lab", "GITHUB_EVENT_NAME": "pull_request",
                 "GITHUB_HEAD_REF": "evidence/lpr-cct-generated"}
        with patch.object(evidence.os, "name", "nt"), patch.dict(os.environ, valid, clear=True): evidence._require_hosted_windows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict(os.environ, {"RUNNER_TEMP": str(root)}):
                self.assertEqual(evidence._work_directory(root / "new"), (root / "new").resolve())
                (root / "existing").mkdir()
                for path in (root, root.parent / "outside", root / "missing" / "nested", root / "existing"):
                    with self.subTest(path=path), self.assertRaises(RuntimeError): evidence._work_directory(path)


if __name__ == "__main__":
    unittest.main()
