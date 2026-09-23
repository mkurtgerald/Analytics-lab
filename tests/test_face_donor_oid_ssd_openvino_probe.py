import os
from pathlib import Path
import tempfile
import unittest

from analytics_lab import face_donor_oid_ssd_openvino_probe as probe


class FakePort:
    def __init__(self, name="image_tensor", shape="[1,?, ?,3]", element="u8"):
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


class OIDSSDOpenVINOProbeTests(unittest.TestCase):
    def test_expected_runtime_and_model_identity_are_pinned(self):
        self.assertEqual(probe._EXPECTED_OPENVINO_VERSION, "2026.3.1")
        self.assertEqual(
            probe._MODEL_MEMBER,
            "ssd_mobilenet_v2_oid_v4_2018_12_12/frozen_inference_graph.pb",
        )
        self.assertEqual(
            probe._expected_model_identity(),
            (
                66_606_111,
                "150f3eb77b741ed24e1a19559589205ccd616b059fc976c8b8d5cefc126bf86b",
            ),
        )

    def test_work_dir_must_be_below_runner_temp(self):
        with tempfile.TemporaryDirectory() as root:
            old = os.environ.get("RUNNER_TEMP")
            os.environ["RUNNER_TEMP"] = root
            try:
                with self.assertRaisesRegex(RuntimeError, "below RUNNER_TEMP"):
                    probe._bounded_work_dir(Path(root))
                child = probe._bounded_work_dir(Path(root) / "oid-runtime")
                self.assertEqual(child, (Path(root) / "oid-runtime").resolve())
            finally:
                if old is None:
                    os.environ.pop("RUNNER_TEMP", None)
                else:
                    os.environ["RUNNER_TEMP"] = old

    def test_shape_text_is_bounded_to_string(self):
        self.assertEqual(probe._shape_text(FakePort()), "[1,?, ?,3]")

    def test_shape_text_fails_closed_to_unknown(self):
        class Bad:
            def get_partial_shape(self):
                raise RuntimeError("boom")

        self.assertEqual(probe._shape_text(Bad()), "unknown")

    def test_error_summary_is_single_line_and_bounded(self):
        summary = probe._error_summary(RuntimeError("a\n" + "b" * 1000))
        self.assertNotIn("\n", summary)
        self.assertLessEqual(len(summary), probe._MAX_ERROR_CHARS)


if __name__ == "__main__":
    unittest.main()
