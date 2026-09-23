import os
from pathlib import Path
import tempfile
import unittest

from analytics_lab import face_donor_facessd_openvino_probe as probe


class FakePort:
    def __init__(self, name="image_tensor", shape="[1,320,320,3]", element="u8"):
        self._name = name
        self._shape = shape
        self._element = element

    def get_names(self):
        return {self._name} if self._name else set()

    def get_any_name(self):
        return self._name

    def get_partial_shape(self):
        return self._shape

    def get_element_type(self):
        return self._element


class FaceSSDOpenVINOProbeTests(unittest.TestCase):
    def test_expected_runtime_and_model_identity_are_pinned(self):
        self.assertEqual(probe._EXPECTED_OPENVINO_VERSION, "2026.3.1")
        self.assertEqual(
            probe._MODEL_MEMBER,
            "facessd_mobilenet_v2_quantized_320x320_open_image_v4/tflite_graph.pb",
        )
        self.assertEqual(
            probe._CUT_OUTPUTS,
            (
                "raw_outputs/box_encodings",
                "raw_outputs/class_predictions",
                "anchors",
            ),
        )

    def test_work_dir_must_be_below_runner_temp(self):
        with tempfile.TemporaryDirectory() as root:
            old = os.environ.get("RUNNER_TEMP")
            os.environ["RUNNER_TEMP"] = root
            try:
                with self.assertRaisesRegex(RuntimeError, "below RUNNER_TEMP"):
                    probe._bounded_work_dir(Path(root))
            finally:
                if old is None:
                    os.environ.pop("RUNNER_TEMP", None)
                else:
                    os.environ["RUNNER_TEMP"] = old

    def test_shape_text_is_bounded_to_string(self):
        self.assertEqual(probe._shape_text(FakePort()), "[1,320,320,3]")

    def test_shape_text_fails_closed_to_unknown(self):
        class Bad:
            def get_partial_shape(self):
                raise RuntimeError("boom")
        self.assertEqual(probe._shape_text(Bad()), "unknown")


if __name__ == "__main__":
    unittest.main()
