import hashlib
import unittest

from analytics_lab import lpr_ocr_fixed_crop_evidence as evidence


class LPROCRFixedCropEvidenceTests(unittest.TestCase):
    def test_plate_box_is_fixed_and_inside_source(self) -> None:
        box = evidence.validate_plate_box()
        self.assertEqual(box, (960, 2020, 2740, 2470))
        left, top, right, bottom = box
        self.assertGreaterEqual(right - left, 96)
        self.assertGreaterEqual(bottom - top, 24)

    def test_plate_box_rejects_malformed_and_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            evidence.validate_plate_box((0, 0, 5000, 100))
        with self.assertRaises(ValueError):
            evidence.validate_plate_box((0, 0, 10, 10))
        with self.assertRaises(ValueError):
            evidence.validate_plate_box([960, 2020, 2740, 2470])  # type: ignore[arg-type]

    def test_source_identity_fails_closed(self) -> None:
        payload = b"not-the-reviewed-source"
        with self.assertRaisesRegex(RuntimeError, "source size mismatch"):
            evidence.validate_source(payload)

    def test_expected_text_and_source_are_predeclared(self) -> None:
        self.assertEqual(evidence._EXPECTED_TEXT_NORMALIZED, "MPR318")
        self.assertEqual(
            evidence._SOURCE_SHA256,
            "32e5637e39b54c26192c011c1cc5516bd6d35573582b8e09d7bc9aae90ef1db4",
        )
        self.assertEqual(len(bytes.fromhex(evidence._SOURCE_SHA256)), 32)

    def test_crop_identity_uses_canonical_rgb24_bytes(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/NumPy optional for local fixed-crop identity test")
        image = np.zeros((evidence._HEIGHT, evidence._WIDTH, 3), dtype=np.uint8)
        left, top, right, bottom = evidence.validate_plate_box()
        image[top:bottom, left:right, 0] = 1
        image[top:bottom, left:right, 1] = 2
        image[top:bottom, left:right, 2] = 3
        result = evidence.crop_identity(image)
        expected_pixel = bytes((3, 2, 1))
        expected = expected_pixel * ((right - left) * (bottom - top))
        self.assertEqual(result["crop_rgb24_bytes"], len(expected))
        self.assertEqual(result["crop_rgb24_sha256"], hashlib.sha256(expected).hexdigest())


if __name__ == "__main__":
    unittest.main()
