import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analytics_lab.video import OpenCVVideoFileSource


class _FakeCapture:
    def __init__(self, positions_ms, fps=10.0):
        self.positions_ms = list(positions_ms)
        self.fps = fps
        self.read_index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == 1:
            return self.fps
        if prop == 2:
            index = max(0, self.read_index - 1)
            return self.positions_ms[index]
        return 0.0

    def read(self):
        if self.read_index >= len(self.positions_ms):
            return False, None
        self.read_index += 1
        return True, object()

    def release(self):
        self.released = True


class LocalVideoTimestampTests(unittest.TestCase):
    def _decode(self, positions_ms):
        capture = _FakeCapture(positions_ms)
        fake_cv2 = SimpleNamespace(
            CAP_PROP_FPS=1,
            CAP_PROP_POS_MSEC=2,
            VideoCapture=lambda _path: capture,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.mp4"
            path.write_bytes(b"fixture")
            with patch.dict(sys.modules, {"cv2": fake_cv2}):
                with OpenCVVideoFileSource(path, start_timestamp_ms=1_000_000) as source:
                    frames = list(source)
        return capture, [frame.timestamp_ms for frame in frames]

    def test_media_clock_preserves_variable_frame_timing(self):
        capture, timestamps = self._decode([5_000.0, 5_100.0, 5_350.0, 5_700.0])
        self.assertEqual(timestamps, [1_000_000, 1_000_100, 1_000_350, 1_000_700])
        self.assertTrue(capture.released)

    def test_stalled_media_clock_falls_back_to_nominal_fps(self):
        _capture, timestamps = self._decode([0.0, 0.0, 0.0, 300.0])
        self.assertEqual(timestamps, [1_000_000, 1_000_100, 1_000_200, 1_000_300])


if __name__ == "__main__":
    unittest.main()
