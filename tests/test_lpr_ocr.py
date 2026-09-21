import hashlib
from pathlib import Path
import tempfile
import unittest

from analytics_lab.lpr_ocr import (
    LPRObservation,
    OCRText,
    PlateDetection,
    encode_plate_crop_ppm,
    normalize_plate_text,
    parse_omz_plate_rows,
    parse_tesseract_tsv,
    run_lpr_ocr,
    verify_tessdata_artifact,
)


class LPROCRTests(unittest.TestCase):
    def test_omz_parser_admits_only_plate_label_and_sorts(self) -> None:
        rows = (
            (0, 1, 0.99, 0.1, 0.1, 0.8, 0.8),
            (0, 2, 0.70, 0.2, 0.3, 0.4, 0.5),
            (0, 2, 0.95, -0.1, 0.1, 1.2, 0.4),
            (0, 2, 0.49, 0.2, 0.2, 0.3, 0.3),
            (-1, 0, 0, 0, 0, 0, 0),
        )
        result = parse_omz_plate_rows(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], PlateDetection(0.95, 0.0, 0.1, 1.0, 0.4))
        self.assertEqual(result[1].confidence, 0.70)

    def test_omz_parser_rejects_unbounded_or_malformed_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "seven numeric"):
            parse_omz_plate_rows(((0, 2, 0.9),))
        with self.assertRaisesRegex(ValueError, "threshold"):
            parse_omz_plate_rows((), threshold=1.1)

    def test_tessdata_verifier_binds_size_git_blob_and_sha256(self) -> None:
        payload = b"commercial-clean-test-artifact"
        git_sha = hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "eng.traineddata")
            path.write_bytes(payload)
            identity = verify_tessdata_artifact(path, expected_size=len(payload), expected_git_blob_sha1=git_sha)
        self.assertEqual(identity.git_blob_sha1, git_sha)
        self.assertEqual(identity.sha256, hashlib.sha256(payload).hexdigest())

    def test_tessdata_verifier_fails_closed_on_wrong_blob(self) -> None:
        payload = b"x"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "eng.traineddata")
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify_tessdata_artifact(path, expected_size=1, expected_git_blob_sha1="0" * 40)

    def test_crop_encoding_is_rgb_ppm_without_external_dependency(self) -> None:
        image = [
            [(1, 2, 3), (4, 5, 6)],
            [(7, 8, 9), (10, 11, 12)],
        ]
        encoded = encode_plate_crop_ppm(image, PlateDetection(0.9, 0.0, 0.0, 1.0, 1.0))
        self.assertTrue(encoded.startswith(b"P6\n2 2\n255\n"))
        self.assertTrue(encoded.endswith(bytes((3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10))))

    def test_tesseract_tsv_normalizes_plate_text_and_confidence(self) -> None:
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\tabc-\n"
            "5\t1\t1\t1\t1\t2\t10\t0\t10\t10\t80\t123\n"
        )
        result = parse_tesseract_tsv(tsv)
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "ABC123")
        self.assertAlmostEqual(result.confidence, 0.85)
        self.assertEqual(normalize_plate_text(" a b-c 123 "), "ABC123")

    def test_tesseract_tsv_accepts_bounded_preamble_before_header(self) -> None:
        tsv = (
            "Estimating resolution as 300\n"
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t91\tMPR318\n"
        )
        result = parse_tesseract_tsv(tsv)
        self.assertEqual(result, OCRText("MPR318", 0.91))

    def test_tesseract_tsv_still_fails_closed_without_real_header(self) -> None:
        with self.assertRaisesRegex(ValueError, "header is incomplete"):
            parse_tesseract_tsv("Estimating resolution as 300\nMPR318\n")

    def test_pipeline_runs_detector_then_ocr_without_retaining_image(self) -> None:
        image = [[(0, 0, 0), (0, 0, 0)]]
        plate = PlateDetection(0.8, 0.0, 0.0, 1.0, 1.0)
        seen = []

        def detector(value):
            self.assertIs(value, image)
            return (plate,)

        def recognizer(ppm):
            seen.append(ppm)
            return OCRText("K5LAB", 0.9)

        result = run_lpr_ocr(image, detector, recognizer)
        self.assertEqual(result, (LPRObservation(plate, OCRText("K5LAB", 0.9)),))
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].startswith(b"P6\n"))


if __name__ == "__main__":
    unittest.main()
