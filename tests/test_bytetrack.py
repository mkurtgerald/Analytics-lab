from dataclasses import asdict, replace
import unittest
from unittest.mock import patch

from analytics_lab.bytetrack import (
    BYTE_TRACK_REVISION,
    ByteTrackAssociationBackend,
    ByteTrackAssociationConfig,
)
from analytics_lab.tracking import DetectionCandidate, NormalizedBox, TrackingSession


def det(confidence=.9, *, category="person", x=.10):
    return DetectionCandidate(
        category,
        confidence,
        NormalizedBox(x, .10, x + .20, .60),
        0,
    )


def state(backend):
    """Snapshot all mutable association state without sharing track objects."""
    return (
        {key: asdict(track) for key, track in backend._tracks.items()},
        backend._next_id,
        backend._last_frame,
        backend._first_frame,
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

    def test_capacity_rejection_preserves_low_confidence_same_frame_retry(self):
        backend = ByteTrackAssociationBackend(ByteTrackAssociationConfig(max_tracks=1))
        session = TrackingSession(backend)
        first = session.update(0, 0, (det(),))
        before = state(backend)
        with self.assertRaisesRegex(RuntimeError, "^track capacity exceeded$"):
            session.update(1, 33, (det(x=.65),))
        self.assertEqual(state(backend), before)
        retried = session.update(1, 33, (det(.2),))
        self.assertEqual([item.track_id for item in retried], [first[0].track_id])
        self.assertEqual(retried[0].confidence, .2)

    def test_rejection_rolls_back_matches_new_ids_and_unconfirmed_removal(self):
        config = ByteTrackAssociationConfig(max_tracks=3)
        backend, control = (ByteTrackAssociationBackend(config) for _ in range(2))
        sessions = (TrackingSession(backend), TrackingSession(control))
        for session in sessions:
            session.update(0, 0, (det(x=.05), det(x=.40)))
            session.update(1, 33, (det(x=.05), det(x=.40), det(x=.70)))
        before = state(backend)
        # Update A's geometry/confidence/model class, lose B, remove unconfirmed
        # C, stage D, then reject E. None of those changes may survive rejection.
        changed_a = replace(det(.8, x=.06), model_class_id=77)
        overflow = (changed_a, det(category="vehicle"), det(category="bicycle"))
        with self.assertRaisesRegex(RuntimeError, "^track capacity exceeded$"):
            sessions[0].update(2, 66, overflow)
        self.assertEqual(state(backend), before)
        valid = (det(x=.05), det(x=.40), det(x=.70))
        self.assertEqual(sessions[0].update(2, 66, valid), sessions[1].update(2, 66, valid))
        self.assertEqual(state(backend), state(control))
        for session in sessions:
            session.update(40, 1320, ())
            self.assertEqual(session.update(41, 1353, (det(category="vehicle"),)), ())
        actual = sessions[0].update(42, 1386, (det(category="vehicle"),))
        expected = sessions[1].update(42, 1386, (det(category="vehicle"),))
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0].track_id, "bt-000004")

    def test_capacity_rejection_rolls_back_low_confidence_association(self):
        backend = ByteTrackAssociationBackend(ByteTrackAssociationConfig(max_tracks=2))
        session = TrackingSession(backend)
        session.update(0, 0, (det(x=.05), det(x=.40)))
        before = state(backend)
        with self.assertRaisesRegex(RuntimeError, "^track capacity exceeded$"):
            session.update(1, 33, (det(.2, x=.06), det(x=.70)))
        self.assertEqual(state(backend), before)
        retried = session.update(1, 33, (det(.2, x=.05), det(.2, x=.40)))
        self.assertEqual([item.track_id for item in retried], ["bt-000001", "bt-000002"])

    def test_expired_lost_tracks_free_capacity_after_recovery_opportunity(self):
        for buffer, replacement_frame in ((0, 1), (1, 2), (30, 40)):
            with self.subTest(buffer=buffer):
                backend = ByteTrackAssociationBackend(ByteTrackAssociationConfig(
                    max_tracks=1, track_buffer_frames=buffer,
                ))
                session = TrackingSession(backend)
                session.update(0, 0, (det(),))
                self.assertEqual(session.update(replacement_frame, replacement_frame * 33,
                                                (det(x=.65),)), ())
                self.assertEqual(backend.retained_tracks, 1)
                confirmed = session.update(replacement_frame + 1, (replacement_frame + 1) * 33,
                                           (det(x=.65),))
                self.assertEqual([item.track_id for item in confirmed], ["bt-000002"])

    def test_unexpired_and_recovered_tracks_still_count_toward_capacity(self):
        backend = ByteTrackAssociationBackend(ByteTrackAssociationConfig(
            max_tracks=1, track_buffer_frames=1,
        ))
        session = TrackingSession(backend)
        first = session.update(0, 0, (det(),))
        before = state(backend)
        with self.assertRaisesRegex(RuntimeError, "^track capacity exceeded$"):
            session.update(1, 33, (det(x=.65),))
        self.assertEqual(state(backend), before)
        self.assertEqual(session.update(1, 33, ()), ())
        # The over-age lost track gets its existing high-confidence recovery
        # opportunity, even though it would expire if still unmatched.
        recovered = session.update(2, 66, (det(),))
        self.assertEqual([item.track_id for item in recovered], [first[0].track_id])

    def test_output_failure_does_not_consume_first_frame_or_ids(self):
        backend = ByteTrackAssociationBackend()
        before = state(backend)
        with patch("analytics_lab.bytetrack.TrackedDetection", side_effect=ValueError("output rejected")):
            with self.assertRaisesRegex(ValueError, "output rejected"):
                tuple(backend.update(7, 231, (det(),)))
        self.assertEqual(state(backend), before)
        accepted = tuple(backend.update(7, 231, (det(),)))
        self.assertEqual([item.track_id for item in accepted], ["bt-000001"])


if __name__ == "__main__":
    unittest.main()
