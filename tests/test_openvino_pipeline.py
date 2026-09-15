from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

from analytics_lab.openvino_pipeline import (
    OpenVINOOMZPipelineConfig,
    OpenVINOOMZPerceptionAdapter,
    run_frame_source_openvino_omz,
    run_local_video_openvino_omz,
)
from analytics_lab.video import Frame, VideoRunResult


class Image:
    shape = (200, 400, 3)


class DownRuntime:
    runtime_version = "2026.3.1-test"

    def detect_rows(self, image):
        return [[0, 1, .95, .1, .4, .9, .6], [-1, 0, 0, 0, 0, 0, 0]]

    def pose_heatmaps(self, image, bbox):
        maps = [[[0.0 for _ in range(4)] for _ in range(4)] for _ in range(19)]
        for channel, (y, x) in {2: (1, 0), 5: (1, 1), 8: (1, 2), 11: (1, 3)}.items():
            maps[channel][y][x] = .9
        return maps


class Tests(unittest.TestCase):
    @mock.patch("analytics_lab.openvino_omz.verify_artifact_set")
    def test_complete_frame_path_emits_evidence_linked_candidate(self, verify):
        verify.return_value = (SimpleNamespace(spec=SimpleNamespace(relative_path="x"), path=Path("/tmp/x")),)
        frames = [Frame(index, index * 1000, Image()) for index in range(4)]
        result = run_frame_source_openvino_omz(
            frames,
            artifact_root="/models",
            source_id="camera-1",
            session_id="session-1",
            runtime_factory=lambda artifacts, device: DownRuntime(),
        )
        self.assertEqual(result.runtime_version, "2026.3.1-test")
        self.assertEqual(result.device, "CPU")
        self.assertEqual(result.video.frames_processed, 4)
        self.assertEqual(result.video.observations_processed, 4)
        self.assertEqual(len(result.video.events), 1)
        event = result.video.events[0]
        self.assertEqual(event["event_type"], "person_down_candidate")
        self.assertEqual(event["objects"][0]["track_id"], "track-000001")
        self.assertEqual(event["evidence"][0]["start_timestamp_ms"], 0)
        self.assertEqual(event["evidence"][0]["end_timestamp_ms"], 3000)

    @mock.patch("analytics_lab.openvino_omz.verify_artifact_set")
    def test_perception_adapter_exposes_reviewed_runtime_identity(self, verify):
        verify.return_value = ()
        adapter = OpenVINOOMZPerceptionAdapter(
            "/models", runtime_factory=lambda artifacts, device: DownRuntime()
        )
        self.assertEqual(adapter.runtime_version, "2026.3.1-test")
        self.assertEqual(adapter.device, "CPU")
        observations = adapter(Image(), 0, 0)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].posture, "down")

    @mock.patch("analytics_lab.openvino_pipeline.run_local_video")
    @mock.patch("analytics_lab.openvino_omz.verify_artifact_set")
    def test_local_video_wrapper_uses_existing_bounded_video_bridge(self, verify, run_video):
        verify.return_value = ()
        run_video.return_value = VideoRunResult(8, 8, ())
        result = run_local_video_openvino_omz(
            "authorized.mp4",
            artifact_root="/models",
            start_timestamp_ms=1000,
            source_id="camera-1",
            session_id="session-1",
            runtime_factory=lambda artifacts, device: DownRuntime(),
        )
        self.assertEqual(result.video.frames_processed, 8)
        kwargs = run_video.call_args.kwargs
        self.assertEqual(kwargs["start_timestamp_ms"], 1000)
        self.assertEqual(kwargs["source_id"], "camera-1")
        self.assertEqual(kwargs["session_id"], "session-1")
        self.assertIsInstance(kwargs["perception"], OpenVINOOMZPerceptionAdapter)

    def test_pipeline_config_rejects_wrong_nested_type(self):
        with self.assertRaises(ValueError):
            OpenVINOOMZPipelineConfig(openvino=object())


if __name__ == "__main__":
    unittest.main()
