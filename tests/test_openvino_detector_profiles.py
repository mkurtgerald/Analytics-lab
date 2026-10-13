"""Detector profile contracts exercised with generated rows and native metadata mocks."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from analytics_lab.artifacts import (
    OPENVINO_OMZ_2023_FP16, OPENVINO_OMZ_PERSON_0200_FP16,
    VerifiedArtifact, omz_artifacts_for_detector,
)
from analytics_lab.detector_shootout import DETECTOR_CANDIDATES
from analytics_lab.openvino_omz import (
    OpenVINOOMZConfig, OpenVINOOMZPoseBackend, _OpenVINORuntime,
    parse_person_detections,
)
from analytics_lab.openvino_pipeline import OpenVINOOMZPipelineConfig, OpenVINOOMZRunResult
from analytics_lab.validation import ValidationSampleSpec, run_validation_suite
from analytics_lab.video import VideoRunResult


_RETAIL = "person-detection-retail-0013"
_0200 = "person-detection-0200"


def _verified(model):
    return tuple(
        VerifiedArtifact(spec, Path("generated-artifacts") / spec.relative_path)
        for spec in omz_artifacts_for_detector(model)
    )


class _RowRuntime:
    runtime_version = "2026.3.1-generated"

    def __init__(self, person_label):
        self.rows = (
            (0, 0, .50, .1, .1, .3, .4),
            (0, 1, .50, .6, .2, .8, .7),
            (0, person_label, .49, .2, .3, .4, .5),
        )
        self.pose_boxes = []

    def detect_rows(self, image):
        return self.rows

    def pose_heatmaps(self, image, bbox):
        self.pose_boxes.append(bbox)
        maps = [[[0.0] * 4 for _ in range(4)] for _ in range(19)]
        for channel, y, x in ((2, 1, 1), (5, 1, 2), (8, 3, 1), (11, 3, 2)):
            maps[channel][y][x] = .9
        return maps


class _Port:
    def __init__(self, shape):
        self.shape = shape


class _DetectorRows:
    size = 7

    def __init__(self, person_label):
        self.row = (0, person_label, .9, .1, .1, .8, .8)

    def reshape(self, shape):
        if shape != (-1, 7):
            raise AssertionError("unexpected detector reshape")
        return (self.row,)


class _Blob:
    """Tensor-layout metadata only; contains no pixel or native array data."""

    def __init__(self, source, size):
        self.source = source
        self.size = size
        self.layout = None
        self.batched = False

    def transpose(self, order):
        self.layout = order
        return self

    def __getitem__(self, index):
        if index is not None:
            raise AssertionError("unexpected batch insertion")
        self.batched = True
        return self


class _NativeMetadata:
    """Mock import modules implementing only the compiled-model metadata contract."""

    def __init__(self, *, detector_shape=None):
        self.detector_shape = detector_shape
        self.models_read = []
        self.compiles = []
        self.detector_blobs = []
        cv2 = ModuleType("cv2")
        cv2.INTER_LINEAR = 1
        cv2.resize = Mock(side_effect=lambda image, size, **kwargs: _Blob(image, size))
        numpy = ModuleType("numpy")
        numpy.float32 = "generated-float32"
        numpy.asarray = lambda value: value
        numpy.ascontiguousarray = Mock(side_effect=lambda value, **kwargs: value)
        openvino = ModuleType("openvino")
        openvino.__version__ = "2026.3.1-generated"
        openvino.Core = Mock(return_value=self)
        self.modules = {"cv2": cv2, "numpy": numpy, "openvino": openvino}

    def read_model(self, *, model, weights):
        self.models_read.append((model, weights))
        return Path(model).name

    def compile_model(self, model, device):
        self.compiles.append((model, device))
        if model == "human-pose-estimation-0001.xml":
            return SimpleNamespace(
                inputs=(_Port((1, 3, 256, 456)),),
                outputs=(_Port((1, 19, 32, 57)), _Port((1, 38, 32, 57))),
            )
        model_is_0200 = model == "person-detection-0200.xml"
        shape = self.detector_shape or ((1, 3, 256, 256) if model_is_0200 else (1, 3, 320, 544))
        port = _Port((1, 1, 100, 7))
        metadata = self

        class Detector:
            inputs = (_Port(shape),)
            outputs = (port,)

            def output(self, index):
                return self.outputs[index]

            def __call__(self, blobs):
                metadata.detector_blobs.extend(blobs)
                return {port: _DetectorRows(0 if model_is_0200 else 1)}

        return Detector()


class OpenVINODetectorProfileTests(unittest.TestCase):
    def test_0200_manifest_reuses_exact_admitted_catalogue_and_shared_pose(self):
        donor = DETECTOR_CANDIDATES[0]
        self.assertEqual(donor.name, _0200)
        self.assertEqual((donor.input_width, donor.input_height, donor.person_label), (256, 256, 0))
        self.assertEqual(OPENVINO_OMZ_PERSON_0200_FP16[:2], donor.specs)
        self.assertEqual(sum(spec.size_bytes for spec in donor.specs), 3_889_273)
        self.assertEqual(OPENVINO_OMZ_PERSON_0200_FP16[2:], OPENVINO_OMZ_2023_FP16[2:])
        self.assertIs(omz_artifacts_for_detector(_RETAIL), OPENVINO_OMZ_2023_FP16)
        self.assertIs(omz_artifacts_for_detector(_0200), OPENVINO_OMZ_PERSON_0200_FP16)
        for model in (_RETAIL, _0200):
            specs = omz_artifacts_for_detector(model)
            self.assertEqual(len(specs), 4)
            self.assertEqual(len({spec.relative_path for spec in specs}), 4)
            self.assertTrue(all(spec.license_id == "Apache-2.0" for spec in specs))

    def test_config_accepts_only_two_named_profiles_without_changing_cutoff(self):
        self.assertEqual(OpenVINOOMZConfig().detector_model, _RETAIL)
        for model in (_RETAIL, _0200):
            config = OpenVINOOMZConfig(detector_model=model)
            self.assertEqual(config.detection_threshold, .50)
            self.assertEqual(config.device, "CPU")
        for model in ("person-detection-0202", "unknown", "../person-detection-0200", "", None, 0, True):
            with self.subTest(model=model), self.assertRaises(ValueError):
                OpenVINOOMZConfig(detector_model=model)

    def test_parser_preserves_default_class_and_exact_inclusive_cutoff(self):
        rows = ((0, 0, .50, .1, .1, .3, .4), (0, 1, .50, .6, .2, .8, .7),
                (0, 0, .49, .2, .3, .4, .5), (0, 1, .49, .2, .3, .4, .5))
        args = dict(image_width=100, image_height=100, threshold=.50, max_people=32)
        default = parse_person_detections(rows, **args)
        self.assertEqual(default, parse_person_detections(rows, person_label=1, **args))
        for label, expected_left in ((0, 10), (1, 60)):
            with self.subTest(label=label):
                detections = parse_person_detections(rows, person_label=label, **args)
                self.assertEqual(len(detections), 1)
                self.assertEqual(detections[0].confidence, .50)
                self.assertEqual(detections[0].bbox.x1, expected_left)
        for label in (True, False, .0, 2, -1, "0", None):
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_person_detections(rows, person_label=label, **args)

    def test_backend_verifies_selected_four_artifacts_before_factory_and_classifies_rows(self):
        for model, label, expected_left in ((_RETAIL, 1, 240), (_0200, 0, 40)):
            with self.subTest(model=model):
                verified = _verified(model)
                runtime = _RowRuntime(label)
                order = []

                def verify(root, specs):
                    order.append("verify")
                    self.assertEqual(specs, omz_artifacts_for_detector(model))
                    return verified

                def factory(artifacts, device):
                    self.assertEqual(order, ["verify"])
                    order.append("factory")
                    self.assertIs(artifacts, verified)
                    self.assertEqual(device, "CPU")
                    return runtime

                with patch("analytics_lab.openvino_omz.verify_artifact_set", side_effect=verify):
                    backend = OpenVINOOMZPoseBackend(
                        "generated-artifacts", config=OpenVINOOMZConfig(detector_model=model), runtime_factory=factory,
                    )
                    poses = backend(SimpleNamespace(shape=(200, 400, 3)), 0, 0)
                self.assertEqual(order, ["verify", "factory"])
                self.assertEqual(len(poses), 1)
                self.assertEqual(poses[0].confidence, .50)
                self.assertEqual(poses[0].bbox.x1, expected_left)
                self.assertEqual(runtime.pose_boxes, [poses[0].bbox])

    def test_unknown_profile_and_failed_verification_cannot_reach_factory(self):
        factory = Mock()
        config = OpenVINOOMZConfig()
        object.__setattr__(config, "detector_model", "unknown")
        with patch("analytics_lab.openvino_omz.verify_artifact_set") as verify:
            with self.assertRaises(ValueError):
                OpenVINOOMZPoseBackend("unused", config=config, runtime_factory=factory)
            verify.assert_not_called()
        with patch("analytics_lab.openvino_omz.verify_artifact_set", side_effect=ValueError("generated mismatch")):
            with self.assertRaises(ValueError):
                OpenVINOOMZPoseBackend("unused", config=OpenVINOOMZConfig(detector_model=_0200), runtime_factory=factory)
        factory.assert_not_called()

    def test_native_metadata_and_bgr_resize_follow_selected_profile(self):
        for model, size, label in ((_RETAIL, (544, 320), 1), (_0200, (256, 256), 0)):
            with self.subTest(model=model):
                metadata = _NativeMetadata()
                image = SimpleNamespace(shape=(200, 400, 3), bgr_marker=(11, 22, 33))
                with patch.dict(sys.modules, metadata.modules):
                    runtime = _OpenVINORuntime(_verified(model), "CPU")
                    rows = runtime.detect_rows(image)
                self.assertEqual(runtime._detector_size, size)
                self.assertEqual(len(metadata.compiles), 2)
                self.assertEqual([Path(model_path).name for model_path, _ in metadata.models_read],
                                 [f"{model}.xml", "human-pose-estimation-0001.xml"])
                metadata.modules["cv2"].resize.assert_called_once_with(image, size, interpolation=1)
                blob = metadata.detector_blobs[0]
                self.assertIs(blob.source, image)
                self.assertEqual(blob.layout, (2, 0, 1))
                self.assertTrue(blob.batched)
                self.assertEqual(metadata.modules["numpy"].ascontiguousarray.call_args.kwargs, {"dtype": "generated-float32"})
                self.assertEqual(rows[0][1], label)
                self.assertEqual(image.bgr_marker, (11, 22, 33))

    def test_native_metadata_rejects_wrong_shape_for_both_profiles(self):
        for model, wrong in ((_RETAIL, (1, 3, 256, 256)), (_0200, (1, 3, 320, 544))):
            with self.subTest(model=model):
                metadata = _NativeMetadata(detector_shape=wrong)
                with patch.dict(sys.modules, metadata.modules), self.assertRaisesRegex(RuntimeError, "detector model input shape"):
                    _OpenVINORuntime(_verified(model), "CPU")

    def test_native_metadata_rejects_ambiguous_or_incomplete_profile_before_core(self):
        for artifacts in (_verified(_RETAIL) + _verified(_0200)[:2], _verified(_0200)[:1] + _verified(_0200)[2:]):
            with self.subTest(paths=[item.spec.relative_path for item in artifacts]):
                metadata = _NativeMetadata()
                with patch.dict(sys.modules, metadata.modules), self.assertRaises(ValueError):
                    _OpenVINORuntime(artifacts, "CPU")
                metadata.modules["openvino"].Core.assert_not_called()

    def test_validation_report_preserves_selected_detector_and_shared_pose_identities(self):
        sample = ValidationSampleSpec("generated", "generated-site", "generated-camera", "synthetic-fixture",
                                      Path("generated-metadata.bin"), "a" * 64, 0, 1000)
        for model in (_RETAIL, _0200):
            with self.subTest(model=model):
                pipeline = OpenVINOOMZPipelineConfig(openvino=OpenVINOOMZConfig(detector_model=model))
                runner = Mock(return_value=OpenVINOOMZRunResult(
                    "2026.3.1-generated", "CPU", VideoRunResult(2, 0, (), 0, 1000),
                ))
                with patch("analytics_lab.validation._preflight_samples"), patch("analytics_lab.validation._verify_media_unchanged"):
                    result = run_validation_suite((sample,), artifact_root="generated-artifacts", pipeline_config=pipeline, runner=runner)
                selected = omz_artifacts_for_detector(model)
                self.assertEqual([item.component for item in result.model_artifacts], [item.component for item in selected])
                self.assertEqual([item.relative_path for item in result.model_artifacts], [item.relative_path for item in selected])
                self.assertEqual([item.sha384 for item in result.model_artifacts], [item.sha384 for item in selected])
                self.assertEqual([item.size_bytes for item in result.model_artifacts], [item.size_bytes for item in selected])
                self.assertEqual(runner.call_args.kwargs["config"].openvino.detector_model, model)


if __name__ == "__main__":
    unittest.main()
