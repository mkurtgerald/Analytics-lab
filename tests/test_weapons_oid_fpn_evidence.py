from contextlib import contextmanager, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analytics_lab import weapons_oid_fpn_admission as admission
from analytics_lab import weapons_oid_fpn_evidence as evidence


class Runtime:
    compile_count = 1
    converted_ports = {"semantic_output_order": ["boxes", "scores", "classes", "count"]}
    compiled_ports = {"outputs": []}
    phase_seconds = {"conversion": .01, "cpu_compile": .02}

    def __init__(self):
        self.inference_count = 0
        self.calls = []

    def detect(self, name):
        self.calls.append(name)
        self.inference_count += 1
        return ()


class FPNEvidenceTests(unittest.TestCase):
    def exercise(self, *, fail_at=None, factory_failure=False, bad_count=False, bad_immutability=False):
        runtime = Runtime()
        paths = []
        @contextmanager
        def verified(work_dir, **kwargs):
            path = Path(work_dir) / "synthetic.pb"
            path.parent.mkdir()
            path.write_bytes(b"generated fixture, not a model")
            paths.append(path)
            try:
                yield path
            finally:
                path.unlink(missing_ok=True)
                path.parent.rmdir()

        def generated(bound):
            bound.detect("generated")
            if fail_at == "generated":
                raise RuntimeError("fixture generated failure")
            return {"name": "generated", "input_immutable": True, "detections": []}

        def source(bound, asset, **kwargs):
            bound.detect(asset.name)
            if fail_at == asset.name:
                raise RuntimeError("fixture source failure")
            if bad_count and asset is evidence.sources._SOURCES[-1]:
                bound.inference_count += 1
            return {"name": asset.name, "input_immutable": not bad_immutability, "detections": []}

        resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=123))
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with patch.dict(os.environ, {"RUNNER_TEMP": str(root)}), \
                 patch.dict("sys.modules", {"resource": resource}), \
                 patch.object(evidence.sys, "platform", "linux"), \
                 patch.object(evidence, "_check_versions", return_value={"openvino": "2026.3.1", "opencv": "4.12.0", "numpy": "2.2.6"}), \
                 patch.object(admission, "identity", return_value={"fixture": True}), \
                 patch.object(admission, "verified_graph", side_effect=verified), \
                 patch.object(evidence.OpenVINOOIDFPNWeaponRuntime, "from_tensorflow_graph", return_value=runtime,
                              side_effect=RuntimeError("fixture conversion failure") if factory_failure else None) as factory, \
                 patch.object(evidence, "_generated", side_effect=generated), \
                 patch.object(evidence.measurement, "_measure_source", side_effect=source), \
                 redirect_stdout(io.StringIO()) as progress:
                try:
                    result = evidence.run(root / "work")
                finally:
                    self.assertTrue(paths)
                    self.assertTrue(all(not path.exists() for path in paths))
                    self.assertFalse((root / "work").exists())
            factory.assert_called_once()
            self.assertIn("archive_verify", progress.getvalue())
            return result, runtime

    def test_one_runtime_three_calls_cleanup_and_nonclaims(self):
        result, runtime = self.exercise()
        self.assertEqual(runtime.calls, ["generated", "winchester-model-1895", "chair"])
        self.assertEqual((result["compile_count"], result["inference_count"]), (1, 3))
        for name in ("archive_identity_pinned", "archive_identity_reverified", "temporary_graph_removed"):
            self.assertIs(result[name], True)
        self.assertFalse(result["preprocessing"]["external_resize"])
        self.assertEqual(result["preprocessing"]["input_partial_shape"], [1, None, None, 3])
        for name in ("accuracy_claim", "performance_claim", "readiness_claim",
                     "model_artifact_retained", "media_artifact_retained", "ground_truth_available"):
            self.assertIs(result[name], False)

    def test_conversion_and_each_inference_failure_clean_up_without_success(self):
        with self.assertRaisesRegex(RuntimeError, "conversion failure"):
            self.exercise(factory_failure=True)
        for name in ("generated", "winchester-model-1895", "chair"):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "fixture .* failure"):
                self.exercise(fail_at=name)

    def test_extra_inference_or_missing_immutability_cannot_emit_success(self):
        with self.assertRaisesRegex(RuntimeError, "exactly three"):
            self.exercise(bad_count=True)
        with self.assertRaisesRegex(RuntimeError, "immutability"):
            self.exercise(bad_immutability=True)

    def test_runtime_versions_are_checked_without_native_execution(self):
        for ov_version, cv_version, np_version in (
            ("2026.3.1-build", "4.12.0", "2.2.6"),
            ("2026.4.0", "4.12.0", "2.2.6"),
            ("2026.3.1", "4.11.0", "2.2.6"),
            ("2026.3.1", "4.12.0", "2.2.5"),
        ):
            modules = {"openvino": SimpleNamespace(__version__=ov_version),
                       "cv2": SimpleNamespace(__version__=cv_version),
                       "numpy": SimpleNamespace(__version__=np_version)}
            with self.subTest(versions=(ov_version, cv_version, np_version)), patch.dict("sys.modules", modules):
                if ov_version == "2026.3.1-build":
                    self.assertEqual(evidence._check_versions()["numpy"], "2.2.6")
                else:
                    with self.assertRaises(RuntimeError):
                        evidence._check_versions()

    def test_generated_mutation_fails_closed_without_numpy(self):
        class Frame:
            data = b"generated pixels"
            def tobytes(self):
                return self.data
        frame = Frame()
        numpy = SimpleNamespace(uint8="uint8", zeros=lambda shape, dtype: frame)
        class Mutating:
            def detect(self, value):
                value.data = b"changed pixels"
                return ()
        with patch.dict("sys.modules", {"numpy": numpy}):
            with self.assertRaisesRegex(RuntimeError, "mutated generated"):
                evidence._generated(Mutating())

    def test_work_directory_outside_runner_temp_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            runner = Path(root) / "runner"
            runner.mkdir()
            with patch.dict(os.environ, {"RUNNER_TEMP": str(runner)}):
                with self.assertRaisesRegex(RuntimeError, "below RUNNER_TEMP"):
                    evidence._bounded_work_dir(Path(root) / "outside")


if __name__ == "__main__":
    unittest.main()
