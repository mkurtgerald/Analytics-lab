import unittest
from unittest import mock

from analytics_lab import lpr_ocr_sc_evidence as evidence


class SouthCarolinaOCREvidenceTests(unittest.TestCase):
    def test_source_and_expected_text_are_predeclared(self):
        self.assertEqual(evidence._SOURCE_SIZE, 1_046_799)
        self.assertEqual(evidence._SOURCE_SHA1, "b3d301e18457b16e0729f89be1abd438f935aa55")
        self.assertIsNone(evidence._SOURCE_SHA256)
        self.assertEqual(evidence._EXPECTED_TEXT, "354AVV")
        self.assertEqual(evidence._PLATE_TEXT_BOX_PX, (45, 145, 1220, 455))

    def test_source_admission_fails_closed_on_wrong_size(self):
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            evidence._verify_source(b"x")

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_work_dir_requires_runner_temp(self):
        with self.assertRaises(RuntimeError):
            evidence._bounded_work_dir(evidence.Path("child"))


if __name__ == "__main__":
    unittest.main()
