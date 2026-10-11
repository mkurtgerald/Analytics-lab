from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from analytics_lab import weapons_oid_fpn as fpn


class Dimension:
    def __init__(self, value):
        self.value, self.is_static = value, value is not None

    def get_length(self):
        return self.value


class Shape:
    def __init__(self, values):
        self.values = values
        self.rank = SimpleNamespace(is_static=True)

    def __iter__(self):
        return iter(Dimension(value) for value in self.values)


class Port:
    def __init__(self, shape, dtype, name):
        self.shape, self.dtype, self.name = shape, dtype, name

    def get_partial_shape(self):
        return Shape(self.shape)

    def get_element_type(self):
        return SimpleNamespace(get_type_name=lambda: self.dtype)

    def get_names(self):
        return {self.name}


class Model:
    def __init__(self):
        self.inputs = [Port([1, None, None, 3], "u8", "image_tensor:0")]
        self.outputs = [
            Port([1, 100, 4], "f32", "internal_boxes:0"),
            Port([1, 100], "f32", "internal_scores:0"),
            Port([1, 100], "f32", "internal_classes:0"),
            Port([1], "f32", "internal_count:0"),
        ]
        self.result_names = fpn.OUTPUT_NAMES

    def get_results(self):
        return [SimpleNamespace(get_friendly_name=lambda name=name: name) for name in self.result_names]


class Compiled(Model):
    def __init__(self, infer=None):
        super().__init__()
        self.infer = infer
        self.calls = 0

    def input(self, index):
        return self.inputs[index]

    def output(self, index):
        return self.outputs[index]

    def __call__(self, inputs):
        self.calls += 1
        return self.infer(inputs)


class Source:
    def __init__(self):
        self.missing = set()
        self.order, self.shape = None, None

    def get_place_by_tensor_name(self, name):
        return None if name in self.missing else name

    def override_all_outputs(self, places):
        self.order = tuple(places)

    def set_partial_shape(self, image, shape):
        self.image, self.shape = image, shape


class Factory:
    def __init__(self):
        self.source, self.model, self.compiled = Source(), Model(), Compiled()
        self.conversions, self.compilations = 0, 0

    def load(self, path):
        self.loaded = path
        return self.source

    def convert(self, source):
        if source is not self.source:
            raise AssertionError("wrong source")
        self.conversions += 1
        return self.model

    def compile_model(self, model, device):
        if model is not self.model or device != "CPU":
            raise AssertionError("wrong model/device")
        self.compilations += 1
        return self.compiled


def construct(factory):
    with tempfile.TemporaryDirectory() as root:
        graph = Path(root) / "frozen.pb"
        graph.write_bytes(b"generated graph fixture, not weights")
        return fpn.OpenVINOOIDFPNWeaponRuntime.from_tensorflow_graph(
            graph, frontend=factory, core=factory, shape_factory=tuple,
        )


def outputs(count):
    return ([[[.1, .2, .3, .4]] * count], [[.9] * count], [[351.0] * count], [float(count)])


class FPNParserTests(unittest.TestCase):
    def test_all_100_admitted_proposals_survive(self):
        result = fpn.parse_fpn_weapon_detections(*outputs(100))
        self.assertEqual(len(result), 100)
        self.assertTrue(all(item.category == "rifle" for item in result))

    def test_101_proposals_and_mismatched_tensor_lengths_fail(self):
        for values in (outputs(101), ([[[.1, .2, .3, .4]]], [[.9, .8]], [[351.0]], [1.0])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                fpn.parse_fpn_weapon_detections(*values)

    def test_active_classes_and_counts_fail_closed(self):
        for model_class in (0, 602, 351.5, True):
            with self.subTest(model_class=model_class), self.assertRaises(ValueError):
                fpn.parse_fpn_weapon_detections([[[.1, .2, .3, .4]]], [[.9]], [[model_class]], [1.0])
        for count in (-1, 1.5, 101, float("nan"), True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                fpn.parse_fpn_weapon_detections(*outputs(1)[:3], [count])

    def test_zero_and_class_threshold_filters_preserve_contract(self):
        self.assertEqual(fpn.parse_fpn_weapon_detections(*outputs(0)), ())
        detections = fpn.parse_fpn_weapon_detections(
            [[[.1, .2, .3, .4]] * 3], [[.50, .49, .99]], [[351, 533, 502]], [3.0],
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].confidence, .50)
        self.assertEqual(detections[0].box.x_min, .2)


class FPNFrontendTests(unittest.TestCase):
    def test_explicit_endpoint_order_and_dynamic_source_shape_compile_once(self):
        factory = Factory()
        runtime = construct(factory)
        self.assertEqual(factory.source.order, fpn.OUTPUT_NAMES)
        self.assertEqual(factory.source.image, "image_tensor:0")
        self.assertEqual(factory.source.shape, (1, -1, -1, 3))
        self.assertEqual((factory.conversions, factory.compilations), (1, 1))
        self.assertEqual(runtime.compile_count, 1)
        self.assertEqual(runtime.inference_count, 0)
        self.assertNotIn("detection_scores:0", runtime.converted_ports["outputs"][1]["names"])

    def test_semantic_identity_precedes_cpu_compile(self):
        for names in (tuple(reversed(fpn.OUTPUT_NAMES)), fpn.OUTPUT_NAMES[:-1],
                      (fpn.OUTPUT_NAMES[0], fpn.OUTPUT_NAMES[2], fpn.OUTPUT_NAMES[1], fpn.OUTPUT_NAMES[3])):
            factory = Factory()
            factory.model.result_names = names
            with self.subTest(names=names), self.assertRaisesRegex(RuntimeError, "Result identities"):
                construct(factory)
            self.assertEqual(factory.compilations, 0)

    def test_missing_input_or_endpoint_fails_before_conversion(self):
        for name in (fpn.INPUT_NAME, *fpn.OUTPUT_NAMES):
            factory = Factory()
            factory.source.missing.add(name)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                construct(factory)
            self.assertEqual((factory.conversions, factory.compilations), (0, 0))

    def test_type_geometry_and_proposal_failures_precede_compile(self):
        cases = (("input", "dtype", "f32"), ("input", "shape", [1, 512, 512, 3]),
                 (0, "shape", [1, 100, 5]), (1, "shape", [1, 101]),
                 (2, "dtype", "i32"), (2, "shape", [1, 99]), (3, "shape", [1, 1]))
        for index, attribute, value in cases:
            factory = Factory()
            port = factory.model.inputs[0] if index == "input" else factory.model.outputs[index]
            setattr(port, attribute, value)
            with self.subTest(case=(index, attribute, value)), self.assertRaises(RuntimeError):
                construct(factory)
            self.assertEqual(factory.compilations, 0)

    def test_compiled_port_changes_fail_after_exactly_one_compile(self):
        factory = Factory()
        factory.compiled.outputs[2].dtype = "i32"
        with self.assertRaises(RuntimeError):
            construct(factory)
        self.assertEqual(factory.compilations, 1)
        self.assertEqual(factory.compiled.calls, 0)


class FPNNumpyBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ModuleNotFoundError:
            raise unittest.SkipTest("NumPy required in the hosted FPN qualification step")
        cls.np = np

    def make_runtime(self, callback=None, count=1):
        np = self.np
        compiled = Compiled()
        def infer(inputs):
            batch = inputs[compiled.input(0)]
            if callback:
                callback(batch)
            values = [np.asarray(value, dtype=np.float32) for value in outputs(count)]
            if count == 0:
                values[0] = np.empty((1, 0, 4), dtype=np.float32)
            return dict(zip(compiled.outputs, values))
        compiled.infer = infer
        return fpn.OpenVINOOIDFPNWeaponRuntime(compiled, ports={}, phase_seconds={}), compiled

    def test_original_rectangular_geometry_rgb_order_and_three_warm_calls(self):
        np = self.np
        frame = np.empty((7, 11, 3), dtype=np.uint8)
        frame[:] = (17, 31, 89)
        original = frame.copy()
        def mutate(batch):
            self.assertEqual(batch.shape, (1, 7, 11, 3))
            self.assertTrue(batch.flags.c_contiguous)
            self.assertEqual(batch[0, 0, 0].tolist(), [89, 31, 17])
            batch[:] = 255
        runtime, compiled = self.make_runtime(mutate)
        for _ in range(3):
            self.assertEqual(len(runtime.detect(frame)), 1)
        self.assertTrue(np.array_equal(frame, original))
        self.assertEqual((runtime.compile_count, runtime.inference_count, compiled.calls), (1, 3, 3))

    def test_negative_channel_stride_cannot_alias_mutating_inference(self):
        np = self.np
        base = np.zeros((5, 9, 3), dtype=np.uint8)
        frame = base[:, :, ::-1]
        runtime, _ = self.make_runtime(lambda batch: batch.fill(211))
        runtime.detect(frame)
        self.assertFalse(np.any(base))

    def test_output_failure_and_overflow_never_count_success(self):
        np = self.np
        for count in (101,):
            runtime, _ = self.make_runtime(count=count)
            with self.assertRaises(RuntimeError):
                runtime.detect(np.zeros((3, 5, 3), dtype=np.uint8))
            self.assertEqual(runtime.inference_count, 0)
        runtime, compiled = self.make_runtime()
        compiled.infer = lambda inputs: {}
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            runtime.detect(np.zeros((3, 5, 3), dtype=np.uint8))
        self.assertEqual(runtime.inference_count, 0)


if __name__ == "__main__":
    unittest.main()
