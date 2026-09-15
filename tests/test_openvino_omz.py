import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

from analytics_lab.openvino_omz import (
    OpenVINOOMZConfig, OpenVINOOMZPoseBackend, PersonDetection,
    parse_person_detections, pose_candidate_from_heatmaps,
)
from analytics_lab.perception import BBox


class Image:
    shape = (200, 400, 3)


class FakeRuntime:
    runtime_version = "2026.3.1-test"

    def __init__(self):
        self.boxes = [[0, 1, .91, .1, .2, .6, .9], [0, 2, .99, 0, 0, 1, 1], [-1, 0, 0, 0, 0, 0, 0]]

    def detect_rows(self, image):
        return self.boxes

    def pose_heatmaps(self, image, bbox):
        maps = [[[0.0 for _ in range(4)] for _ in range(4)] for _ in range(19)]
        for channel, (y, x, score) in {2: (1, 1, .8), 5: (1, 2, .7), 8: (3, 1, .9), 11: (3, 2, .85)}.items():
            maps[channel][y][x] = score
        return maps


class Tests(unittest.TestCase):
    def test_parse_filters_and_scales(self):
        rows = [
            [0, 1, .7, .5, .5, 1.2, 1.2],
            [0, 2, .99, 0, 0, 1, 1],
            [0, 1, .9, .1, .2, .4, .8],
            [-1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 1],
        ]
        out = parse_person_detections(rows, image_width=100, image_height=200, threshold=.5, max_people=4)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0].confidence, .9)
        self.assertEqual((out[0].bbox.x1, out[0].bbox.y1, out[0].bbox.x2, out[0].bbox.y2), (10, 40, 40, 160))
        self.assertEqual((out[1].bbox.x1, out[1].bbox.y1, out[1].bbox.x2, out[1].bbox.y2), (50, 100, 100, 200))

    def test_parse_caps_highest_confidence(self):
        rows = [[0, 1, confidence, 0, 0, .5, .5] for confidence in (.51, .9, .7)]
        out = parse_person_detections(rows, image_width=10, image_height=10, threshold=.5, max_people=2)
        self.assertEqual([item.confidence for item in out], [.9, .7])

    def test_parse_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            parse_person_detections([[0, 1, math.nan, 0, 0, 1, 1]], image_width=10, image_height=10, threshold=.5, max_people=1)
        with self.assertRaises(ValueError):
            parse_person_detections([[0, 1, .9, 0, 0, 1]], image_width=10, image_height=10, threshold=.5, max_people=1)

    def test_heatmaps_produce_required_keypoints(self):
        maps = [[[0.0] * 4 for _ in range(4)] for _ in range(19)]
        peaks = {2: (1, 1, .8), 5: (1, 2, .7), 8: (3, 1, .9), 11: (3, 2, .85)}
        for channel, (y, x, score) in peaks.items():
            maps[channel][y][x] = score
        candidate = pose_candidate_from_heatmaps(maps, PersonDetection(BBox(100, 50, 300, 150), .95))
        self.assertEqual([keypoint.name for keypoint in candidate.keypoints], ["right_shoulder", "left_shoulder", "right_hip", "left_hip"])
        self.assertEqual([keypoint.confidence for keypoint in candidate.keypoints], [.8, .7, .9, .85])
        self.assertTrue(all(100 <= keypoint.x <= 300 and 50 <= keypoint.y <= 150 for keypoint in candidate.keypoints))
        self.assertEqual(candidate.confidence, .95)

    def test_heatmaps_require_expected_channels_and_rectangles(self):
        detection = PersonDetection(BBox(0, 0, 10, 10), .9)
        with self.assertRaises(ValueError):
            pose_candidate_from_heatmaps([[[0]]] * 18, detection)
        bad = [[[0.0] * 2 for _ in range(2)] for _ in range(19)]
        bad[2] = [[0], [0, 1]]
        with self.assertRaises(ValueError):
            pose_candidate_from_heatmaps(bad, detection)

    def test_config_is_bounded(self):
        OpenVINOOMZConfig()
        for kwargs in (
            {"detection_threshold": 0}, {"max_people": 0}, {"max_people": 257},
            {"device": "AUTO GPU"}, {"device": "x://y"}, {"runtime_version_prefix": "latest"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OpenVINOOMZConfig(**kwargs)

    @mock.patch("analytics_lab.openvino_omz.verify_artifact_set")
    def test_backend_verifies_before_runtime_and_emits_pose(self, verify):
        verify.return_value = (SimpleNamespace(spec=SimpleNamespace(relative_path="x"), path=Path("/tmp/x")),)
        created = []

        def factory(artifacts, device):
            created.append((artifacts, device))
            return FakeRuntime()

        backend = OpenVINOOMZPoseBackend("/models", runtime_factory=factory)
        poses = backend(Image(), 0, 1000)
        self.assertEqual(len(poses), 1)
        verify.assert_called_once()
        self.assertEqual(created[0][1], "CPU")
        self.assertEqual(backend.runtime_version, "2026.3.1-test")

    @mock.patch("analytics_lab.openvino_omz.verify_artifact_set", side_effect=ValueError("bad artifact"))
    def test_artifact_failure_prevents_runtime_factory(self, verify):
        factory = mock.Mock()
        with self.assertRaises(ValueError):
            OpenVINOOMZPoseBackend("/models", runtime_factory=factory)
        factory.assert_not_called()

    @mock.patch("analytics_lab.openvino_omz.verify_artifact_set")
    def test_runtime_version_required(self, verify):
        verify.return_value = ()

        class Bad:
            runtime_version = ""

        with self.assertRaises(RuntimeError):
            OpenVINOOMZPoseBackend("/models", runtime_factory=lambda artifacts, device: Bad())

        class Wrong:
            runtime_version = "2025.4.1"

        with self.assertRaises(RuntimeError):
            OpenVINOOMZPoseBackend("/models", runtime_factory=lambda artifacts, device: Wrong())

    @mock.patch("analytics_lab.openvino_omz.verify_artifact_set")
    def test_backend_validates_frame_metadata_and_image(self, verify):
        verify.return_value = ()
        backend = OpenVINOOMZPoseBackend("/models", runtime_factory=lambda artifacts, device: FakeRuntime())
        with self.assertRaises(ValueError):
            backend(Image(), -1, 0)
        with self.assertRaises(ValueError):
            backend(Image(), 0, -1)
        with self.assertRaises(ValueError):
            backend(object(), 0, 0)


if __name__ == "__main__":
    unittest.main()
