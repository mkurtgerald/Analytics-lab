import unittest

from analytics_lab import face_donor_facessd_opencv_probe as probe


class _Network:
    def __init__(self, *, empty=False, layers=()):
        self._empty = empty
        self._layers = tuple(layers)

    def empty(self):
        return self._empty

    def getLayerNames(self):
        return self._layers


class _HandledError(Exception):
    pass


class FaceSSDOpenCVProbeTests(unittest.TestCase):
    def test_loader_records_success_without_inference(self):
        calls = []

        def loader(*args):
            calls.append(args)
            return _Network(layers=("a", "b"))

        result = probe._attempt_loader(loader, ("model.pb",), _HandledError)
        self.assertEqual(calls, [("model.pb",)])
        self.assertEqual(result, {
            "loaded": True,
            "result": "loaded",
            "layer_count": 2,
        })

    def test_loader_classifies_only_reviewed_runtime_error(self):
        def loader(*_args):
            raise _HandledError("unsupported graph")

        result = probe._attempt_loader(loader, ("model.pb",), _HandledError)
        self.assertEqual(result, {"loaded": False, "result": "opencv_error"})

    def test_unexpected_error_fails_closed(self):
        def loader(*_args):
            raise ValueError("unexpected")

        with self.assertRaisesRegex(ValueError, "unexpected"):
            probe._attempt_loader(loader, ("model.pb",), _HandledError)

    def test_empty_network_is_not_compatible(self):
        result = probe._attempt_loader(
            lambda *_args: _Network(empty=True), ("model.pb",), _HandledError
        )
        self.assertEqual(result, {"loaded": False, "result": "empty_network"})

    def test_admitted_members_are_reused_exactly(self):
        model_size, model_sha = probe._expected_member(probe._MODEL_MEMBER)
        config_size, config_sha = probe._expected_member(probe._CONFIG_MEMBER)
        self.assertEqual(model_size, 22_222_216)
        self.assertEqual(
            model_sha,
            "dc8e2c9e21407b2f6d35f1eb655ba8a0c9c73094e5987231a1a8de2edae74978",
        )
        self.assertEqual(config_size, 62_525_550)
        self.assertEqual(
            config_sha,
            "e1232ff66eedd5676bfa31e78aba28ee6244b0322a5646d1a1f89efa02cb781b",
        )


if __name__ == "__main__":
    unittest.main()
