import unittest

from analytics_lab.iou_tracker import SimpleIoUAssociationBackend, SimpleIoUConfig
from analytics_lab.tracking import DetectionCandidate, NormalizedBox, TrackingSession


def det(x_min=.1, x_max=.3, confidence=.9, category="person"):
    return DetectionCandidate(
        category,
        confidence,
        NormalizedBox(x_min, .1, x_max, .8),
        0,
    )


class SimpleIoUTrackerTests(unittest.TestCase):
    def test_defaults_are_frozen_pre_measurement(self):
        config = SimpleIoUConfig()
        self.assertEqual(
            (config.track_threshold, config.new_track_threshold, config.min_iou),
            (.5, .6, .3),
        )

    def test_track_continues_on_raw_iou_with_existing_threshold(self):
        session = TrackingSession(SimpleIoUAssociationBackend())
        first = session.update(0, 1000, (det(),))
        second = session.update(1, 1040, (det(.11, .31, .55),))
        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_low_confidence_gap_is_not_recovered(self):
        session = TrackingSession(SimpleIoUAssociationBackend())
        first = session.update(0, 1000, (det(),))
        self.assertEqual(session.update(1, 1040, (det(.11, .31, .4),)), ())
        third = session.update(2, 1080, (det(.12, .32, .9),))
        self.assertNotEqual(first[0].track_id, third[0].track_id)

    def test_sub_new_threshold_detection_does_not_start_track(self):
        session = TrackingSession(SimpleIoUAssociationBackend())
        self.assertEqual(session.update(0, 1000, (det(confidence=.55),)), ())

    def test_category_mismatch_never_associates(self):
        session = TrackingSession(SimpleIoUAssociationBackend())
        first = session.update(0, 1000, (det(category="person"),))
        second = session.update(1, 1040, (det(category="vehicle"),))
        self.assertNotEqual(first[0].track_id, second[0].track_id)

    def test_two_object_assignment_is_deterministic(self):
        session = TrackingSession(SimpleIoUAssociationBackend())
        first = session.update(0, 1000, (det(.1, .3), det(.6, .8)))
        second = session.update(1, 1040, (det(.61, .81), det(.11, .31)))
        self.assertEqual(
            [item.track_id for item in second],
            [first[0].track_id, first[1].track_id],
        )

    def test_capacity_and_config_fail_closed(self):
        with self.assertRaises(ValueError):
            SimpleIoUConfig(track_threshold=.7, new_track_threshold=.6)
        backend = SimpleIoUAssociationBackend(SimpleIoUConfig(max_tracks=1))
        with self.assertRaises(RuntimeError):
            backend.update(0, 0, (det(), det(.6, .8)))


if __name__ == "__main__":
    unittest.main()
