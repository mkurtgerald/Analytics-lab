import unittest

from analytics_lab.tracking import (
    DetectionCandidate, NormalizedBox, TrackedDetection,
    TrackingSession, TrackingSessionConfig,
)


def det(category="person", confidence=.9):
    return DetectionCandidate(category, confidence, NormalizedBox(.1, .1, .4, .8), 0)


class EchoBackend:
    def update(self, frame_index, timestamp_ms, detections):
        return tuple(
            TrackedDetection(f"track-{index+1}", item.category, item.confidence, item.box, item.model_class_id)
            for index, item in enumerate(detections)
        )


class TrackingContractTests(unittest.TestCase):
    def test_normalized_contract_accepts_multiple_categories(self):
        session = TrackingSession(EchoBackend())
        out = session.update(0, 1000, (det("person"), det("vehicle", .8)))
        self.assertEqual([item.category for item in out], ["person", "vehicle"])
        self.assertEqual([item.track_id for item in out], ["track-1", "track-2"])

    def test_track_ids_are_session_local_and_unique_per_frame(self):
        class DuplicateBackend:
            def update(self, *_):
                box = NormalizedBox(.1, .1, .2, .2)
                return (
                    TrackedDetection("same", "person", .9, box),
                    TrackedDetection("same", "person", .8, box),
                )
        with self.assertRaises(ValueError):
            TrackingSession(DuplicateBackend()).update(0, 1, (det(),))

    def test_frames_and_timestamps_must_increase(self):
        session = TrackingSession(EchoBackend())
        session.update(0, 1000, (det(),))
        with self.assertRaises(ValueError):
            session.update(0, 1001, (det(),))
        with self.assertRaises(ValueError):
            session.update(1, 1000, (det(),))

    def test_bounds_fail_closed_without_partial_state_change(self):
        session = TrackingSession(EchoBackend(), TrackingSessionConfig(max_detections_per_frame=1, max_tracks_per_frame=1))
        with self.assertRaises(RuntimeError):
            session.update(0, 1000, (det(), det("vehicle")))
        out = session.update(0, 1000, (det(),))
        self.assertEqual(len(out), 1)

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            NormalizedBox(-.1, .1, .2, .2)
        with self.assertRaises(ValueError):
            NormalizedBox(.2, .2, .2, .4)
        with self.assertRaises(ValueError):
            DetectionCandidate("", .9, NormalizedBox(.1, .1, .2, .2))
        with self.assertRaises(ValueError):
            DetectionCandidate("person", 1.1, NormalizedBox(.1, .1, .2, .2))
        with self.assertRaises(ValueError):
            TrackedDetection("", "person", .9, NormalizedBox(.1, .1, .2, .2))


if __name__ == "__main__":
    unittest.main()
