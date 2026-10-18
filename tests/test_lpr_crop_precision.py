"""Pixel-exact generated regressions; no OpenCV, models or native OCR required."""
import math
from pathlib import Path
import subprocess
import unittest

from analytics_lab.lpr_ocr import (
    LPROCRConfig, OCRText, PlateDetection, TesseractPlateRecognizer,
    encode_plate_crop_ppm, run_lpr_ocr,
)
from analytics_lab.lpr_plate_proposals import OpenCVPlateProposalDetector


class GeneratedImage:
    def __init__(self, width, height, boxes):
        self.shape = (height, width, 3)
        self.boxes = boxes

    def __getitem__(self, y):
        boxes = self.boxes

        class Row:
            def __getitem__(self, x):
                for left, top, right, bottom, bgr in boxes:
                    if left <= x < right and top <= y < bottom:
                        return bgr
                return (0, 0, 255)  # conspicuous outside red border

        return Row()


class ContourCV:
    __version__ = "4.12.0"
    COLOR_BGR2GRAY = CV_32F = THRESH_BINARY = THRESH_OTSU = 0
    MORPH_RECT = MORPH_CLOSE = RETR_EXTERNAL = CHAIN_APPROX_SIMPLE = 0

    def __init__(self, rectangles):
        self.rectangles = rectangles

    def cvtColor(self, image, code):
        return image

    def GaussianBlur(self, image, *args):
        return image

    def Sobel(self, image, *args, **kwargs):
        return image

    def convertScaleAbs(self, image):
        return image

    def threshold(self, image, *args):
        return (0, image)

    def getStructuringElement(self, *args):
        return None

    def morphologyEx(self, image, *args, **kwargs):
        return image

    def findContours(self, *args):
        return (self.rectangles, None)

    def boundingRect(self, contour):
        return contour

    def contourArea(self, contour):
        return contour[2] * contour[3] * .8


def recognizer_fixture():
    crops = []

    def runner(args, **kwargs):
        if args[-1] == "--version":
            return subprocess.CompletedProcess(args, 0, stdout=b"tesseract 5.5.3\n", stderr=b"")
        crops.append(kwargs["input"])
        text = ("ABC123", "XYZ789")[len(crops) - 1]
        return subprocess.CompletedProcess(args, 0,
                                         stdout=f"level\tconf\ttext\n5\t95\t{text}\n".encode(), stderr=b"")

    # Bypass provisioning only; exercise the actual adapter/version gate/parser.
    recognizer = object.__new__(TesseractPlateRecognizer)
    recognizer.config = LPROCRConfig()
    recognizer._tessdata_dir = Path("generated-only")
    recognizer._binary = "tesseract"
    recognizer._runner = runner
    recognizer._version_checked = False
    return recognizer, crops


class LPRCropPrecisionTests(unittest.TestCase):
    def assert_exact_crop(self, ppm, width, height, bgr=(0, 0, 0)):
        self.assertEqual(ppm, f"P6\n{width} {height}\n255\n".encode() + bytes(bgr[::-1]) * (width * height))

    def test_actual_proposal_to_ocr_preserves_integer_crop_and_excludes_outside_column(self):
        image = GeneratedImage(960, 200, [(245, 50, 405, 90, (0, 0, 0))])
        detector = OpenCVPlateProposalDetector(cv2_module=ContourCV([(245, 50, 160, 40)]))
        recognizer, crops = recognizer_fixture()
        observations = run_lpr_ocr(image, detector, recognizer)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].text, OCRText("ABC123", .95))
        self.assertEqual(len(crops), 1)
        self.assert_exact_crop(crops[0], 160, 40)

    def test_both_lower_and_upper_integer_edges_recover_exact_pixel_extents(self):
        for width, height, box in (
            (100, 100, (57, 29, 80, 50)),  # floor must not add top/left pixels
            (100, 100, (10, 10, 28, 55)),  # ceil must not add right/bottom pixels
            (960, 960, (245, 245, 405, 405)),
        ):
            with self.subTest(width=width, height=height, box=box):
                left, top, right, bottom = box
                image = GeneratedImage(width, height, [(*box, (0, 0, 0))])
                plate = PlateDetection(.9, left / width, top / height, right / width, bottom / height)
                self.assert_exact_crop(encode_plate_crop_ppm(image, plate), right - left, bottom - top)

    def test_genuinely_fractional_boundaries_keep_floor_and_ceil(self):
        image = GeneratedImage(100, 100, [(57, 29, 81, 51, (0, 0, 0))])
        plate = PlateDetection(.9, 57.25 / 100, 29.75 / 100, 80.25 / 100, 50.01 / 100)
        self.assert_exact_crop(encode_plate_crop_ppm(image, plate), 24, 22)

    def test_fractional_neighbors_one_ulp_from_integer_are_not_snapped(self):
        # These are different normalized floats from 7/100 and 29/100, even
        # though their products are within one ULP of the integer boundaries.
        left = math.nextafter(7 / 100, -math.inf)
        right = math.nextafter(29 / 100, math.inf)
        self.assertEqual(math.floor(left * 100), 6)
        self.assertEqual(math.ceil(right * 100), 30)
        image = GeneratedImage(100, 4, [(6, 0, 30, 1, (0, 0, 0))])
        plate = PlateDetection(.9, left, 0., right, .25)
        self.assert_exact_crop(encode_plate_crop_ppm(image, plate), 24, 1)

    def test_single_pixel_and_frame_edges_remain_nonempty_and_exact(self):
        for width, height, box in ((100, 100, (57, 57, 58, 58)), (1, 1, (0, 0, 1, 1)),
                                   (100, 100, (0, 0, 100, 100))):
            with self.subTest(width=width, height=height, box=box):
                left, top, right, bottom = box
                image = GeneratedImage(width, height, [(*box, (1, 2, 3))])
                plate = PlateDetection(.9, left / width, top / height, right / width, bottom / height)
                self.assert_exact_crop(encode_plate_crop_ppm(image, plate), right - left, bottom - top, (1, 2, 3))
        image = GeneratedImage(10, 10, [(0, 0, 1, 1, (1, 2, 3))])
        self.assert_exact_crop(encode_plate_crop_ppm(image, PlateDetection(.9, .001, .001, .002, .002)),
                               1, 1, (1, 2, 3))

    def test_separated_proposals_keep_their_own_pixels_and_ocr_observations(self):
        image = GeneratedImage(960, 200, [(245, 50, 405, 90, (1, 2, 3)), (600, 50, 760, 90, (4, 5, 6))])
        detector = OpenCVPlateProposalDetector(cv2_module=ContourCV([(245, 50, 160, 40), (600, 50, 160, 40)]))
        recognizer, crops = recognizer_fixture()
        observations = run_lpr_ocr(image, detector, recognizer)
        self.assertEqual(tuple(item.text.text for item in observations), ("ABC123", "XYZ789"))
        self.assertEqual(len(crops), 2)
        self.assert_exact_crop(crops[0], 160, 40, (1, 2, 3))
        self.assert_exact_crop(crops[1], 160, 40, (4, 5, 6))


if __name__ == "__main__":
    unittest.main()
