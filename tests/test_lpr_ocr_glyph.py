import unittest

from analytics_lab.lpr_ocr_glyph import GlyphOCRResult, recognize_plate_glyphs


class GlyphOCRTests(unittest.TestCase):
    def test_result_contract_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            GlyphOCRResult("bad-plate", 0.5, 1, 1)
        with self.assertRaises(ValueError):
            GlyphOCRResult("ABC", 1.5, 1, 1)
        with self.assertRaises(ValueError):
            GlyphOCRResult("ABC", 0.5, 2, 1)

    def test_threshold_validation_is_fail_closed_without_opencv(self):
        class Dummy:
            shape = (32, 64, 3)

        with self.assertRaisesRegex(ValueError, "min_match_score"):
            recognize_plate_glyphs(Dummy(), min_match_score=-0.1)

    def test_synthetic_plate_strings_across_two_fonts(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/NumPy optional outside glyph evidence lane")

        for text, font in (
            ("CPU4704", cv2.FONT_HERSHEY_SIMPLEX),
            ("354AVV", cv2.FONT_HERSHEY_DUPLEX),
            ("MPR318", cv2.FONT_HERSHEY_SIMPLEX),
        ):
            with self.subTest(text=text, font=font):
                bgr = np.full((180, 800, 3), 255, dtype=np.uint8)
                cv2.putText(
                    bgr, text, (35, 125), font, 2.8, (0, 0, 0), 6, cv2.LINE_AA
                )
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = recognize_plate_glyphs(rgb)
                self.assertEqual(result.text, text)
                self.assertEqual(result.accepted_components, len(text))
                self.assertGreater(result.mean_score, 0.75)


if __name__ == "__main__":
    unittest.main()
