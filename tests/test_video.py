import tempfile
from pathlib import Path
import unittest

from analytics_lab.temporal import PersonDownEngine
from analytics_lab.video import (
    Frame,
    OpenCVVideoFileSource,
    PerceptionObservation,
    VideoRunConfig,
    run_frame_source,
    run_local_video,
)


class VideoBridgeTests(unittest.TestCase):
    def frames(self, count=16, start=1_000_000, step=250):
        return [Frame(i, start + i * step, object()) for i in range(count)]

    @staticmethod
    def down(_image, _index, _timestamp):
        return [PerceptionObservation("track-1", "down", 0.9)]

    def test_frame_source_closes_video_to_temporal_boundary(self):
        engine = PersonDownEngine("camera-a", "session-a")
        result = run_frame_source(self.frames(), self.down, engine)
        self.assertEqual(result.frames_processed, 16)
        self.assertEqual(result.observations_processed, 16)
        self.assertEqual(result.first_frame_timestamp_ms, 1_000_000)
        self.assertEqual(result.last_frame_timestamp_ms, 1_003_750)
        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertEqual(event["event_type"], "person_down_candidate")
        self.assertEqual(event["evidence"][0]["start_timestamp_ms"], 1_000_000)
        self.assertEqual(event["evidence"][0]["end_timestamp_ms"], 1_003_000)

    def test_unknown_output_interrupts_continuity(self):
        def perception(_image, index, _timestamp):
            posture = "unknown" if index == 8 else "down"
            return [PerceptionObservation("track-1", posture, 0.9)]
        result = run_frame_source(self.frames(), perception, PersonDownEngine("camera-a", "session-a"))
        self.assertEqual(result.events, ())

    def test_multiple_tracks_same_frame_are_supported(self):
        def perception(_image, _index, _timestamp):
            return [PerceptionObservation("a", "down", 0.9), PerceptionObservation("b", "upright", 0.95)]
        result = run_frame_source(self.frames(), perception, PersonDownEngine("camera-a", "session-a"))
        self.assertEqual(result.observations_processed, 32)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0]["objects"][0]["track_id"], "a")

    def test_duplicate_track_in_one_frame_is_rejected(self):
        def perception(_image, _index, _timestamp):
            return [PerceptionObservation("a", "down", 0.9), PerceptionObservation("a", "down", 0.9)]
        with self.assertRaises(ValueError):
            run_frame_source(self.frames(1), perception, PersonDownEngine("camera-a", "session-a"))

    def test_frame_order_is_strict(self):
        frames = [Frame(0, 1000, object()), Frame(1, 1000, object())]
        with self.assertRaises(ValueError):
            run_frame_source(frames, self.down, PersonDownEngine("camera-a", "session-a"))

    def test_frame_limit_is_fail_closed(self):
        with self.assertRaises(RuntimeError):
            run_frame_source(self.frames(3), self.down, PersonDownEngine("camera-a", "session-a"), VideoRunConfig(max_frames=2))

    def test_per_frame_observation_limit_is_fail_closed(self):
        def perception(_image, _index, _timestamp):
            return [PerceptionObservation("a", "down", 0.9), PerceptionObservation("b", "down", 0.9)]
        with self.assertRaises(RuntimeError):
            run_frame_source(self.frames(1), perception, PersonDownEngine("camera-a", "session-a"), VideoRunConfig(max_observations_per_frame=1))

    def test_event_limit_is_fail_closed(self):
        def perception(_image, _index, _timestamp):
            return [PerceptionObservation("a", "down", 0.9), PerceptionObservation("b", "down", 0.9)]
        with self.assertRaises(RuntimeError):
            run_frame_source(self.frames(), perception, PersonDownEngine("camera-a", "session-a"), VideoRunConfig(max_events=1))

    def test_local_source_rejects_urls_and_devices_before_opencv(self):
        for path in ("https://example.com/video.mp4", "rtsp://example/camera", "/dev/video0", r"\\.\PhysicalDrive0"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                OpenCVVideoFileSource(path, start_timestamp_ms=0)

    def test_local_source_requires_real_file(self):
        source = OpenCVVideoFileSource("definitely-missing.mp4", start_timestamp_ms=0)
        with self.assertRaises(ValueError):
            source.__enter__()

    def test_perception_result_must_use_typed_observations(self):
        with self.assertRaises(ValueError):
            run_frame_source(self.frames(1), lambda *_: [{"track_id": "a"}], PersonDownEngine("camera-a", "session-a"))

    def test_actual_opencv_file_decode_and_bridge_when_available(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV is an optional local-video dependency")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (64, 48))
            if not writer.isOpened():
                self.skipTest("local OpenCV build cannot create MJPG AVI")
            for index in range(16):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:, :, 0] = index
                writer.write(frame)
            writer.release()
            seen = []
            def perception(image, index, timestamp_ms):
                seen.append((index, timestamp_ms, image.shape))
                return [PerceptionObservation("track-1", "down", 0.9)]
            result = run_local_video(
                path,
                start_timestamp_ms=1_000_000,
                source_id="camera-a",
                session_id="session-a",
                perception=perception,
            )
            self.assertEqual(result.frames_processed, 16)
            self.assertEqual(len(result.events), 1)
            self.assertEqual(result.first_frame_timestamp_ms, 1_000_000)
            self.assertEqual(result.last_frame_timestamp_ms, 1_003_750)
            self.assertEqual(seen[0], (0, 1_000_000, (48, 64, 3)))
            self.assertEqual(seen[12][1], 1_003_000)
            self.assertEqual(result.events[0]["evidence"][0]["end_timestamp_ms"], 1_003_000)


if __name__ == "__main__":
    unittest.main()
