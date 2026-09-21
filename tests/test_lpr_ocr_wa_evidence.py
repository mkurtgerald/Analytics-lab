import unittest
from unittest import mock

from analytics_lab import lpr_ocr_wa_evidence as evidence


class WashingtonOCREvidenceTests(unittest.TestCase):
    def test_source_and_expected_text_are_predeclared(self):
        self.assertEqual(evidence._SOURCE_SIZE, 5_532_415)
        self.assertEqual(evidence._SOURCE_SHA1, "39a9d530a1d40d960043d51f7f5c67c906f0450f")
        self.assertEqual(evidence._SOURCE_SHA256, "265046b1e374d1a457658308e75f713aa3796c44092f3f8173930d3a7aa17a09")
        self.assertEqual(evidence._SOURCE_WIDTH, 3_549)
        self.assertEqual(evidence._SOURCE_HEIGHT, 1_779)
        self.assertEqual(evidence._EXPECTED_TEXT, "CPU4704")
        self.assertEqual(evidence._PLATE_TEXT_BOX_PX, (180, 600, 3380, 1510))
        self.assertEqual(evidence._CROP_RGB24_SHA256, "0e4655be2fe57d47c890855cf22371ff2b7ab764cfc9c1084a692a9ee67d8bca")

    def test_source_admission_fails_closed_on_wrong_size(self):
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            evidence._verify_source(b"x")

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_work_dir_requires_runner_temp(self):
        with self.assertRaises(RuntimeError):
            evidence._bounded_work_dir(evidence.Path("child"))


if __name__ == "__main__":
    unittest.main()
