import hashlib
import importlib.util
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analytics_lab import lpr_cct
from analytics_lab.lpr_cct import OpenVINOCCTPlateRecognizer, parse_cct_plate_probabilities
from analytics_lab.lpr_ocr import OCRText


def probabilities(text="MPR318", confidence=.99):
    result = []
    for character in text.ljust(10, "_"):
        row = [(1.0 - confidence) / 36] * 37
        row[lpr_cct.CCT_ALPHABET.index(character)] = confidence
        result.append(row)
    return result


class CCTProbabilityTests(unittest.TestCase):
    def test_decodes_slots_and_retains_repeated_characters(self):
        self.assertEqual(parse_cct_plate_probabilities(probabilities("354AVV")), OCRText("354AVV", .99))
        self.assertEqual(parse_cct_plate_probabilities(probabilities("CPU4704")), OCRText("CPU4704", .99))
        self.assertEqual(parse_cct_plate_probabilities(probabilities("0123456789")), OCRText("0123456789", .99))

    def test_blank_interior_padding_and_weak_padding_abstain(self):
        for rows in (probabilities(""), probabilities("AB_C"),
                     probabilities("MPR318", .89)):
            with self.subTest(rows=rows):
                self.assertIsNone(parse_cct_plate_probabilities(rows))
        rows = probabilities()
        rows[-1] = probabilities("", .89)[0]
        self.assertIsNone(parse_cct_plate_probabilities(rows))

    def test_cutoff_equality_and_weakest_slot_confidence(self):
        rows = probabilities()
        rows[-1] = probabilities("", .90)[0]
        self.assertEqual(parse_cct_plate_probabilities(rows), OCRText("MPR318", .90))
        rows[0] = probabilities("M", .95)[0]
        self.assertEqual(parse_cct_plate_probabilities(rows), OCRText("MPR318", .90))

    def test_invalid_probability_shape_type_range_and_normalization_fail(self):
        invalid = [[], probabilities()[:9], probabilities() + probabilities()[:1], "x"]
        for bad_value in (math.nan, math.inf, -0.1, 1.1, True, "1"):
            rows = probabilities()
            rows[0][0] = bad_value
            invalid.append(rows)
        invalid += [[[0.0] * 37] * 10, [[1.0] * 37] * 10,
                    [[.5, .5]] * 10, ["0" * 37] * 10]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                parse_cct_plate_probabilities(rows)
        for cutoff in (-1, 1.1, math.nan, True, "0.9"):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                parse_cct_plate_probabilities(probabilities(), min_confidence=cutoff)

    def test_ppm_is_canonical_exact_and_bounded(self):
        valid = b"P6\n2 1\n255\n\xff\x02\x03\x04\x05\x06"
        width, height, pixels = lpr_cct._ppm_pixels(valid)
        self.assertEqual((width, height, bytes(pixels)), (2, 1, valid[-6:]))
        for payload in (b"P6\n0 1\n255\n", b"P6\n01 1\n255\nabc", b"P6\r\n1 1\n255\nabc",
                        b"P6\n1 1\n254\nabc", b"P6\n1 1\n255\nab", b"P6\n1 1\n255\nabcd",
                        bytearray(b"P6\n1 1\n255\nabc")):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                lpr_cct._ppm_pixels(payload)
        with patch.object(lpr_cct, "CCT_MAX_INPUT_BYTES", 12), self.assertRaises(ValueError):
            lpr_cct._ppm_pixels(valid)


class Dimension:
    def __init__(self, value):
        self.value = value
        self.is_dynamic = value is None

    def get_length(self):
        return self.value


class Shape(list):
    rank = SimpleNamespace(is_dynamic=False)

    def __init__(self, values):
        super().__init__(Dimension(value) for value in values)


class Port:
    def __init__(self, name, shape, dtype):
        self.name, self.shape, self.dtype = name, shape, dtype

    def get_names(self):
        return {self.name}

    def get_partial_shape(self):
        return Shape(self.shape)

    def get_element_type(self):
        return SimpleNamespace(get_type_name=lambda: self.dtype)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required for injected native array tests")
class CCTRuntimeTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve() / "generated.onnx"
        self.payload = b"generated fake model identity"
        self.path.write_bytes(self.payload)
        self.events, self.inputs = [], []
        self.input_port = Port("input", (None, 64, 128, 3), "u8")
        self.plate = Port("plate", (None, 370), "f32")
        self.region = Port("region", (None, 66), "f32")
        self.model = SimpleNamespace(inputs=[self.input_port], outputs=[self.region, self.plate])
        self.output = np.array(probabilities(), dtype=np.float32).reshape((1, 370))
        owner = self

        class Compiled:
            inputs = [owner.input_port]

            def output(self, index):
                owner.events.append(("output", index))
                return owner.model.outputs[index]

            def __call__(self, values):
                owner.events.append("infer")
                owner.inputs.append(values[0].copy())
                return {owner.plate: owner.output, owner.region: np.zeros((1, 66), dtype=np.float32)}

        class Core:
            def __init__(self):
                owner.events.append("core")

            def read_model(self, path):
                owner.events.append(("read", path))
                return owner.model

            def compile_model(self, model, device):
                owner.events.append(("compile", device))
                return Compiled()

        def resize(rgb, size, *, interpolation):
            self.events.append(("resize", size, interpolation))
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertEqual(rgb.tolist(), [[[255, 2, 3], [4, 5, 6]]])
            return np.tile(rgb[0, 0], (64, 128, 1))

        modules = {"openvino": SimpleNamespace(__version__="2026.3.1-test", Core=Core),
                   "cv2": SimpleNamespace(INTER_LINEAR=1, resize=resize)}
        self.addCleanup(patch.stopall)
        patch.dict("sys.modules", modules).start()
        patch.object(lpr_cct, "CCT_MODEL_SIZE", len(self.payload)).start()
        patch.object(lpr_cct, "CCT_MODEL_SHA256", hashlib.sha256(self.payload).hexdigest()).start()

    def test_rgb_uint8_no_rescale_named_plate_compile_once_and_warm_reuse(self):
        self.plate.get_names = lambda: {"another_plate_alias", "plate"}
        recognizer = OpenVINOCCTPlateRecognizer(self.path)
        ppm = b"P6\n2 1\n255\n\xff\x02\x03\x04\x05\x06"
        for _ in range(2):
            result = recognizer(ppm)
            self.assertEqual(result.text, "MPR318")
            self.assertAlmostEqual(result.confidence, .99, places=6)
        self.assertEqual(recognizer.runtime_version, "2026.3.1-test")
        self.assertEqual((recognizer.compile_count, recognizer.inference_count), (1, 2))
        self.assertEqual(self.events.count(("compile", "CPU")), 1)
        self.assertIn(("output", 1), self.events)
        for blob in self.inputs:
            self.assertEqual((blob.shape, blob.dtype), ((1, 64, 128, 3), self.np.uint8))
            self.assertEqual(blob[0, 0, 0].tolist(), [255, 2, 3])
            self.assertTrue(blob.flags.owndata)
        self.assertEqual(ppm[-6:], b"\xff\x02\x03\x04\x05\x06")

    def test_unchanged_model_access_time_and_fixed_batch_are_valid(self):
        os.utime(self.path, (1, 1))
        self.input_port.shape = (1, 64, 128, 3)
        self.plate.shape = (1, 10, 37)
        recognizer = OpenVINOCCTPlateRecognizer(self.path)
        self.assertEqual(recognizer.compile_count, 1)

    def test_unnamed_plate_shape_selects_unique_head_in_either_order(self):
        self.plate.name, self.region.name = "Identity", "Identity_1"
        ppm = b"P6\n2 1\n255\n\xff\x02\x03\x04\x05\x06"
        for outputs, expected_index in (([self.plate, self.region], 0), ([self.region, self.plate], 1)):
            self.model.outputs = outputs
            recognizer = OpenVINOCCTPlateRecognizer(self.path)
            self.assertEqual(recognizer(ppm).text, "MPR318")
            self.assertIn(("output", expected_index), self.events)

    def test_unnamed_ambiguous_heads_and_invalid_named_plate_fail_before_compile(self):
        for outputs in (
            [Port("Identity", (None, 370), "f32"), Port("Identity_1", (None, 370), "f32")],
            [Port("plate", (None, 66), "f32"), Port("Identity", (None, 370), "f32")],
        ):
            self.model.outputs = outputs
            with self.subTest(outputs=outputs), self.assertRaises(RuntimeError):
                OpenVINOCCTPlateRecognizer(self.path)
        self.assertNotIn(("compile", "CPU"), self.events)

    def test_bad_size_hash_directory_and_symlink_fail_before_native_load(self):
        self.path.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            OpenVINOCCTPlateRecognizer(self.path)
        self.path.write_bytes(b"x" * len(self.payload))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            OpenVINOCCTPlateRecognizer(self.path)
        with self.assertRaisesRegex(ValueError, "regular"):
            OpenVINOCCTPlateRecognizer(self.path.parent)
        link = self.path.with_name("alias.onnx")
        try:
            link.symlink_to(self.path)
        except (OSError, NotImplementedError):
            pass  # Windows may deny symlink creation; other identity checks still run.
        else:
            with self.assertRaisesRegex(ValueError, "regular"):
                OpenVINOCCTPlateRecognizer(link)
        self.assertEqual(self.events, [])

    def test_input_and_named_plate_contracts_reject_before_compile(self):
        for port in (Port("wrong", (1, 64, 128, 3), "u8"),
                     Port("input", (1, 3, 64, 128), "u8"),
                     Port("input", (2, 64, 128, 3), "u8"),
                     Port("input", (1, 64, 128, 3), "f32")):
            self.model.inputs = [port]
            with self.subTest(port=port), self.assertRaises(RuntimeError):
                OpenVINOCCTPlateRecognizer(self.path)
        self.model.inputs = [self.input_port]
        for outputs in ([self.region], [Port("plate", (1, 369), "f32")],
                        [Port("plate", (1, 370), "u8")], [self.plate, self.plate]):
            self.model.outputs = outputs
            with self.subTest(outputs=outputs), self.assertRaises(RuntimeError):
                OpenVINOCCTPlateRecognizer(self.path)
        self.assertNotIn(("compile", "CPU"), self.events)

    def test_bad_ppm_never_infers_and_bad_output_cannot_become_read(self):
        recognizer = OpenVINOCCTPlateRecognizer(self.path)
        with self.assertRaises(ValueError):
            recognizer(b"not ppm")
        self.assertEqual(recognizer.inference_count, 0)
        ppm = b"P6\n2 1\n255\n\xff\x02\x03\x04\x05\x06"
        for output in (self.np.zeros((2, 370), dtype=self.np.float32),
                       self.np.zeros((1, 370), dtype=self.np.float64)):
            self.output = output
            with self.assertRaises(RuntimeError):
                recognizer(ppm)
        self.output = self.np.full((1, 370), self.np.nan, dtype=self.np.float32)
        with self.assertRaises(ValueError):
            recognizer(ppm)


if __name__ == "__main__":
    unittest.main()
