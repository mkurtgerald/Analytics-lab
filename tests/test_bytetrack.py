import unittest

from analytics_lab.bytetrack import (
    BYTE_TRACK_REVISION,
    ByteTrackAssociationBackend,
    ByteTrackAssociationConfig,
)
from analytics_lab.tracking import DetectionCandidate, NormalizedBox


def det(confidence=.9, *, category="person", x=.10):
    return DetectionCandidate(
        category,
        confidence,
        NormalizedBox(x, .10, x + .20, .60),
        0,
    )


class ByteTrackAssociationTests(unittest.TestCase):
    def test_pinned_donor_revision(self):
        self.assertEqual(
            BYTE_TRACK_REVISION,
            "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99",
        )

    def test_low_confidence_detection_preserves_existing_track(self):
        backend = ByteTrackAssociationBackend()
        first = tuple(backend.update(0, 0, (det(.90),)))
        low = tuple(backend.update(1, 33, (det(.20),)))
        high = tuple(backend.update(2, 66, (det(.90),)))
        self.assertEqual([item.track_id for item in first], ["bt-000001"])
        self.assertEqual([item.track_id for item in low], ["bt-000001"])
        self.assertEqual([item.track_id for item in high], ["bt-000001"])

    def test_low_confidence_detection_never_starts_track(self):
        backend = ByteTrackAssociationBackend()
        self.assertEqual(tuple(backend.update(0, 0, (det(.20),))), ())

    def test_new_track_after_first_frame_requires_confirmation(self):
        backend = ByteTrackAssociationBackend()
        tuple(backend.update(0, 0, (det(.9, x=.05),)))
        frame1 = tuple(backend.update(1, 33, (det(.9, x=.05), det(.9, x=.65))))
        self.assertEqual([item.track_id for item in frame1], ["bt-000001"])
        frame2 = tuple(backend.update(2, 66, (det(.9, x=.05), det(.9, x=.65))))
        self.assertEqual(
            [item.track_id for item in frame2],
            ["bt-000001", "bt-000002"],
        )

    def test_categories_do_not_cross_associate(self):
        backend = ByteTrackAssociationBackend()
        first = tuple(backend.update(0, 0, (det(.9, category="person"),)))
        second = tuple(backend.update(1, 33, (det(.9, category="vehicle"),)))
        self.assertEqual(first[0].track_id, "bt-000001")
        self.assertEqual(second, ())
        third = tuple(backend.update(2, 66, (det(.9, category="vehicle"),)))
        self.assertEqual(third[0].track_id, "bt-000002")
        self.assertEqual(third[0].category, "vehicle")

    def test_lost_track_expires_at_bounded_buffer(self):
        backend = ByteTrackAssociationBackend(
            ByteTrackAssociationConfig(track_buffer_frames=1)
        )
        first = tuple(backend.update(0, 0, (det(.9),)))
        self.assertEqual(first[0].track_id, "bt-000001")
        self.assertEqual(tuple(backend.update(1, 33, ())), ())
        recovered = tuple(backend.update(2, 66, (det(.9),)))
        self.assertEqual(recovered[0].track_id, "bt-000001")
        tuple(backend.update(3, 99, ()))
        tuple(backend.update(5, 165, ()))
        probation = tuple(backend.update(6, 198, (det(.9),)))
        self.assertEqual(probation, ())
        confirmed = tuple(backend.update(7, 231, (det(.9),)))
        self.assertEqual(confirmed[0].track_id, "bt-000002")

    def test_config_and_frame_order_fail_closed(self):
        with self.assertRaises(ValueError):
            ByteTrackAssociationConfig(low_threshold=.6, track_threshold=.5)
        backend = ByteTrackAssociationBackend()
        tuple(backend.update(0, 0, ()))
        with self.assertRaises(ValueError):
            tuple(backend.update(0, 1, ()))


if __name__ == "__main__":
    unittest.main()
