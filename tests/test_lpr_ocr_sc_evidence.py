import unittest
from unittest import mock

from analytics_lab import lpr_ocr_sc_evidence as evidence


class SouthCarolinaOCREvidenceTests(unittest.TestCase):
    def test_source_and_expected_text_are_predeclared(self):
        self.assertEqual(evidence._SOURCE_SIZE, 1_046_799)
        self.assertEqual(evidence._SOURCE_SHA1, "b3d301e18457b16e0729f89be1abd438f935aa55")
        self.assertEqual(evidence._SOURCE_SHA256, "ace70508c959f5c0e8f915d6cd07aa81ad6f7f92e12d8a72e01cc50a2d40750c")
        self.assertEqual(evidence._EXPECTED_TEXT, "354AVV")
        self.assertEqual(evidence._PLATE_TEXT_BOX_PX, (45, 145, 1220, 455))\n        self.assertEqual(evidence._CROP_RGB24_SHA256, "c13f5cb796cbe2d6a6505e41bf75d69d3dc1c9877bd4c8ef0a55a3389b3ced4b")

    def test_source_admission_fails_closed_on_wrong_size(self):
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            evidence._verify_source(b"x")

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_work_dir_requires_runner_temp(self):
        with self.assertRaises(RuntimeError):
            evidence._bounded_work_dir(evidence.Path("child"))


if __name__ == "__main__":
    unittest.main()
