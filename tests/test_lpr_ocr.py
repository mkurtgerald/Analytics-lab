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
    def test_page_segmentation_profile_is_strict_and_defaults_to_seven(self) -> None:
        from analytics_lab.lpr_ocr import LPROCRConfig
        self.assertEqual(LPROCRConfig().tesseract_psm, 7)
        self.assertEqual(LPROCRConfig(tesseract_psm=13).tesseract_psm, 13)
        for invalid in (True, False, 7.0, 13.0, "13", None, 6, 8, 14):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                LPROCRConfig(tesseract_psm=invalid)

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

    def test_crop_pixel_budget_rejects_before_touching_oversize_frame(self) -> None:
        class LargeImage:
            shape = (4096, 4096, 3)
            reads = 0

            def __getitem__(self, index):
                self.reads += 1
                raise AssertionError("pixel access before PPM size admission")

        image = LargeImage()
        with self.assertRaisesRegex(ValueError, "plate crop exceeds bounded OCR input"):
            encode_plate_crop_ppm(image, PlateDetection(0.9, 0.0, 0.0, 1.0, 1.0))
        self.assertEqual(image.reads, 0)

    def test_tesseract_tsv_normalizes_plate_text_and_confidence(self) -> None:
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\tabc-\n"
            "5\t1\t1\t1\t1\t2\t10\t0\t10\t10\t80\t123\n"
        )
        result = parse_tesseract_tsv(tsv)
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "ABC123")
        self.assertAlmostEqual(result.confidence, 0.80)
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

    def test_normalization_rejects_unknown_glyphs_unicode_and_overflow(self) -> None:
        for text in ("ABC?123", "ABC/123", "ABC_123", "ABC\t123", "ABC\n123",
                     "ABCß123", "ABCé123", "ＡＢＣ123", "ABCDEFGHIJKLMNOPQ"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                normalize_plate_text(text)
        self.assertEqual(normalize_plate_text(" abc- 123 "), "ABC123")
        self.assertEqual(normalize_plate_text("abcdefghijklmnop"), "ABCDEFGHIJKLMNOP")

    def test_tsv_invalid_confidence_abstains_on_whole_read(self) -> None:
        header = "level\tconf\ttext\n"
        for confidence in ("1000", "100.01", "-1", "nan", "inf", "bad"):
            for words in ((f"5\t{confidence}\t123\n", "5\t99\tABC\n"),
                          ("5\t99\tABC\n", f"5\t{confidence}\t123\n")):
                with self.subTest(confidence=confidence, words=words):
                    self.assertIsNone(parse_tesseract_tsv(header + "".join(words), min_confidence=.95))

    def test_tsv_non_ascii_or_non_decimal_numeric_fields_abstain(self) -> None:
        header = "level\tconf\ttext\n"
        fields = [(level, "99") for level in ("５", " 5", "5 ", "05", "+5", "5\u2028")]
        fields += [("5", confidence) for confidence in (
            "9_9", "９９", "99\u2028", " 99", "99 ", "9.9e1", "+99", "99.", "99\u00a0",
        )]
        for level, confidence in fields:
            uncertain = f"{level}\t{confidence}\t123\n"
            for rows in (uncertain + "5\t99\tABC\n", "5\t99\tABC\n" + uncertain):
                with self.subTest(level=level, confidence=confidence, rows=rows):
                    self.assertIsNone(parse_tesseract_tsv(header + rows, min_confidence=.95))

    def test_tsv_native_decimal_confidences_and_structural_sentinel_are_supported(self) -> None:
        tsv = "level\tconf\ttext\n1\t-1.000000\t\n5\t90.125000\tabc-\n5\t80.500000\t123\n"
        self.assertEqual(parse_tesseract_tsv(tsv), OCRText("ABC123", .805))
        for confidence, cutoff in (("0.000000", 0), ("100.000000", 1), ("95.125000", .95125)):
            with self.subTest(confidence=confidence):
                self.assertEqual(parse_tesseract_tsv(f"level\tconf\ttext\n5\t{confidence}\tABC123\n",
                                                     min_confidence=cutoff), OCRText("ABC123", cutoff))

    def test_tsv_low_confidence_word_or_separator_never_becomes_partial_plate(self) -> None:
        header = "level\tconf\ttext\n"
        for uncertain in ("123", "-"):
            for words in ((f"5\t5\t{uncertain}\n", "5\t99\tABC\n"),
                          ("5\t99\tABC\n", f"5\t5\t{uncertain}\n")):
                with self.subTest(uncertain=uncertain, words=words):
                    self.assertIsNone(parse_tesseract_tsv(header + "".join(words), min_confidence=.95))

    def test_tsv_unsupported_glyph_and_joint_token_overflow_abstain(self) -> None:
        header = "level\tconf\ttext\n"
        rows = ("5\t99\tABC?123\n", "5\t99\tABCß123\n",
                "5\t99\tABCDEFGHIJKLMNOPQ\n", "5\t99\tABCDEFGHIJ\n5\t99\tKLMNOPQ\n")
        for row in rows:
            with self.subTest(row=row):
                self.assertIsNone(parse_tesseract_tsv(header + row))

    def test_tsv_duplicate_or_empty_header_columns_abstain(self) -> None:
        for tsv in ("level\tconf\ttext\tconf\n5\t99\tABC123\t5\n",
                    "level\tconf\ttext\tlevel\n5\t99\tABC123\t1\n",
                    "level\tconf\ttext\t\n5\t99\tABC123\t\n"):
            with self.subTest(tsv=tsv):
                self.assertIsNone(parse_tesseract_tsv(tsv))

    def test_tsv_malformed_nonempty_rows_never_leave_confident_prefix(self) -> None:
        header = "level\tconf\ttext\n"
        for bad in ("5\t99\n", "5\t99\t123\textra\n", "bad\t99\t123\n",
                    "6\t99\t123\n", "2\t99\t123\n", "5\t99\t\n", "garbage\n"):
            for rows in ("5\t99\tABC\n" + bad, bad + "5\t99\tABC\n"):
                with self.subTest(bad=bad, rows=rows):
                    self.assertIsNone(parse_tesseract_tsv(header + rows))

    def test_tsv_structural_empty_rows_and_blank_lines_are_preserved(self) -> None:
        self.assertIsNone(parse_tesseract_tsv(""))
        self.assertIsNone(parse_tesseract_tsv("level\tconf\ttext\n1\t-1\t\n"))
        tsv = "level\tconf\ttext\n\n1\t-1\t\n2\t-1\t\n3\t-1\t\n4\t-1\t\n5\t90\tabc-\n5\t80\t123\n"
        self.assertEqual(parse_tesseract_tsv(tsv), OCRText("ABC123", .80))
        self.assertEqual(parse_tesseract_tsv(tsv.replace("\n", "\r\n")), OCRText("ABC123", .80))

    def test_tsv_unicode_line_break_cannot_manufacture_a_second_word(self) -> None:
        tsv = "level\tconf\ttext\n5\t99\tABC\u20285\t99\t123\n"
        self.assertIsNone(parse_tesseract_tsv(tsv))

    def test_tsv_confidence_boundaries_cutoff_equality_and_weakest_word(self) -> None:
        for confidence, cutoff in ((0, 0), (100, 1), (95, .95)):
            tsv = f"level\tconf\ttext\n5\t{confidence}\tABC123\n"
            with self.subTest(confidence=confidence, cutoff=cutoff):
                self.assertEqual(parse_tesseract_tsv(tsv, min_confidence=cutoff), OCRText("ABC123", confidence / 100))
        self.assertEqual(parse_tesseract_tsv("level\tconf\ttext\n5\t99\tABC\n5\t5\t123\n"),
                         OCRText("ABC123", .05))

    def test_lpr_pipeline_emits_no_observation_when_complete_read_abstains(self) -> None:
        image = [[(0, 0, 0), (0, 0, 0)]]
        plate = PlateDetection(.99, 0., 0., 1., 1.)
        tsv = "level\tconf\ttext\n5\t99\tABC\n5\t5\t123\n"
        result = run_lpr_ocr(image, lambda image: (plate,),
                             lambda ppm: parse_tesseract_tsv(tsv, min_confidence=.95))
        self.assertEqual(result, ())

    def test_tesseract_recognizer_requests_tsv_without_external_config_file(self) -> None:
        from analytics_lab.lpr_ocr import LPROCRConfig, TesseractPlateRecognizer
        import subprocess

        seen = []

        def fake_runner(args, **kwargs):
            seen.append(list(args))
            if args[-1] == "--version":
                return subprocess.CompletedProcess(args, 0, stdout=b"tesseract 5.5.3\n", stderr=b"")
            tsv = (
                "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
                "5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t90\tMPR318\n"
            ).encode("utf-8")
            return subprocess.CompletedProcess(args, 0, stdout=tsv, stderr=b"")

        recognizer = object.__new__(TesseractPlateRecognizer)
        recognizer.config = LPROCRConfig()
        recognizer._tessdata_dir = Path("tessdata")
        recognizer._binary = "tesseract"
        recognizer._runner = fake_runner
        recognizer._version_checked = False

        result = recognizer(b"P6\n1 1\n255\n\x00\x00\x00")
        self.assertEqual(result, OCRText("MPR318", 0.9))
        self.assertEqual(
            seen[1][-2:],
            ["-c", "tessedit_create_tsv=1"],
        )
        self.assertNotIn("tsv", seen[1])
        self.assertEqual(seen[1][seen[1].index("--psm") + 1], "7")
        recognizer.config = LPROCRConfig(tesseract_psm=13)
        self.assertEqual(recognizer(b"P6\n1 1\n255\n\x00\x00\x00"), result)
        expected = list(seen[1])
        expected[expected.index("--psm") + 1] = "13"
        self.assertEqual(seen[2], expected)

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
