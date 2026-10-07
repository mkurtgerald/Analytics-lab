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


class StubBackend:
    def __init__(self, result=()):
        self.result = result
        self.calls = []

    def update(self, frame_index, timestamp_ms, detections):
        self.calls.append((frame_index, timestamp_ms, detections))
        return self.result


class OverflowSentinel:
    """Fail promptly if a collector reads past the first excess item."""

    def __init__(self, value, limit):
        self.value = value
        self.limit = limit
        self.consumed = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.consumed == self.limit + 1:
            raise AssertionError("iterator consumed beyond the overflow witness")
        self.consumed += 1
        return self.value


class LengthHintIterator:
    def __init__(self, values, hint):
        self.values = iter(values)
        self.hint = hint
        self.hint_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.values)

    def __length_hint__(self):
        self.hint_calls += 1
        if isinstance(self.hint, Exception):
            raise self.hint
        return self.hint


class LengthIterator(LengthHintIterator):
    def __len__(self):
        return self.__length_hint__()


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

    def test_detection_iterator_stops_at_first_excess_item(self):
        for limit in (1, 2, 512):
            with self.subTest(limit=limit):
                backend = StubBackend()
                session = TrackingSession(backend, TrackingSessionConfig(limit, 1))
                session.update(0, 1000, ())
                values = OverflowSentinel(det(), limit)
                with self.assertRaisesRegex(RuntimeError, "^detection count exceeds configured bound$"):
                    session.update(1, 1001, values)
                self.assertEqual(values.consumed, limit + 1)
                self.assertEqual(backend.calls, [(0, 1000, ())])
                self.assertEqual(session.update(1, 1001, (det(),)), ())
                self.assertEqual(len(backend.calls), 2)

    def test_backend_iterator_stops_at_first_excess_item(self):
        track = EchoBackend().update(0, 1000, (det(),))[0]
        for limit in (1, 2, 512):
            with self.subTest(limit=limit):
                backend = StubBackend()
                session = TrackingSession(backend, TrackingSessionConfig(1, limit))
                session.update(0, 1000, ())
                backend.result = values = OverflowSentinel(track, limit)
                with self.assertRaisesRegex(RuntimeError, "^track count exceeds configured bound$"):
                    session.update(1, 1001, (det(),))
                self.assertEqual(values.consumed, limit + 1)
                self.assertEqual(len(backend.calls), 2)
                backend.result = (track,)
                self.assertEqual(session.update(1, 1001, (det(),)), (track,))

    def test_detection_collection_ignores_untrusted_length_hints(self):
        for iterator in (LengthHintIterator, LengthIterator):
            for hint in (0, -1, 1 << 100, "invalid", AssertionError("hint must not run")):
                with self.subTest(iterator=iterator.__name__, hint=hint):
                    backend = StubBackend()
                    expected = (det(), det("vehicle"))
                    values = iterator(expected, hint)
                    session = TrackingSession(backend, TrackingSessionConfig(2, 1))
                    self.assertEqual(session.update(0, 1000, values), ())
                    self.assertEqual(values.hint_calls, 0)
                    self.assertEqual(backend.calls, [(0, 1000, expected)])
                    self.assertIsInstance(backend.calls[0][2], tuple)

    def test_backend_collection_ignores_untrusted_length_hints(self):
        expected = EchoBackend().update(0, 1000, (det(), det("vehicle")))
        for iterator in (LengthHintIterator, LengthIterator):
            for hint in (0, -1, 1 << 100, "invalid", AssertionError("hint must not run")):
                with self.subTest(iterator=iterator.__name__, hint=hint):
                    values = iterator(expected, hint)
                    session = TrackingSession(StubBackend(values), TrackingSessionConfig(1, 2))
                    self.assertEqual(session.update(0, 1000, ()), expected)
                    self.assertEqual(values.hint_calls, 0)

    def test_empty_below_and_exact_bounds_preserve_order_and_tuple_contract(self):
        for count in (0, 1, 2):
            with self.subTest(count=count):
                detections = (det(), det("vehicle"))[:count]
                tracks = EchoBackend().update(0, 1000, detections)
                backend = StubBackend(item for item in tracks)
                session = TrackingSession(backend, TrackingSessionConfig(2, 2))
                result = session.update(0, 1000, (item for item in detections))
                self.assertIsInstance(result, tuple)
                self.assertEqual(result, tracks)
                self.assertEqual(backend.calls, [(0, 1000, detections)])
                self.assertIsInstance(backend.calls[0][2], tuple)

    def test_large_positive_configured_bounds_remain_valid(self):
        detections = (det(), det("vehicle"))
        tracks = EchoBackend().update(0, 1000, detections)
        backend = StubBackend(item for item in tracks)
        session = TrackingSession(backend, TrackingSessionConfig(1 << 100, 1 << 100))
        self.assertEqual(session.update(0, 1000, (item for item in detections)), tracks)
        self.assertEqual(backend.calls, [(0, 1000, detections)])

    def test_overflow_precedes_item_validation(self):
        backend = StubBackend()
        session = TrackingSession(backend, TrackingSessionConfig(1, 1))
        with self.assertRaisesRegex(RuntimeError, "^detection count exceeds configured bound$"):
            session.update(0, 1000, iter((None, det())))
        self.assertEqual(backend.calls, [])
        backend.result = iter((None, None))
        with self.assertRaisesRegex(RuntimeError, "^track count exceeds configured bound$"):
            session.update(0, 1000, ())
        backend.result = ()
        self.assertEqual(session.update(0, 1000, ()), ())

    def test_invalid_items_preserve_validation_errors_and_frame_retry(self):
        backend = StubBackend()
        session = TrackingSession(backend)
        with self.assertRaisesRegex(ValueError, "^detections must contain DetectionCandidate values$"):
            session.update(0, 1000, iter((None,)))
        self.assertEqual(backend.calls, [])
        backend.result = iter((None,))
        with self.assertRaisesRegex(ValueError, "^tracking backend returned an unsupported value$"):
            session.update(0, 1000, ())
        backend.result = ()
        self.assertEqual(session.update(0, 1000, ()), ())

    def test_noniterables_preserve_errors_and_frame_retry(self):
        for value in (None, 3):
            with self.subTest(value=value):
                backend = StubBackend()
                session = TrackingSession(backend)
                with self.assertRaisesRegex(ValueError, "^detections must be iterable$") as caught:
                    session.update(0, 1000, value)
                self.assertIsInstance(caught.exception.__cause__, TypeError)
                self.assertEqual(backend.calls, [])
                backend.result = value
                expected = "tracking backend must return an iterable"
                if value is None:
                    expected += ", not None"
                with self.assertRaisesRegex(ValueError, "^" + expected + "$"):
                    session.update(0, 1000, ())
                backend.result = ()
                self.assertEqual(session.update(0, 1000, ()), ())

    def test_iteration_errors_preserve_conversion_and_frame_retry(self):
        def broken(value, error):
            yield value
            raise error

        track = EchoBackend().update(0, 1000, (det(),))[0]
        for side in ("detections", "backend"):
            for error in (TypeError("iteration failed"), LookupError("iteration failed")):
                with self.subTest(side=side, error=type(error).__name__):
                    backend = StubBackend()
                    session = TrackingSession(backend, TrackingSessionConfig(1, 1))
                    detections = ()
                    if side == "detections":
                        detections = broken(det(), error)
                    else:
                        backend.result = broken(track, error)
                    expected = ValueError if isinstance(error, TypeError) else LookupError
                    with self.assertRaises(expected) as caught:
                        session.update(0, 1000, detections)
                    if isinstance(error, TypeError):
                        self.assertIs(caught.exception.__cause__, error)
                        message = ("detections must be iterable" if side == "detections"
                                   else "tracking backend must return an iterable")
                        self.assertEqual(str(caught.exception), message)
                    else:
                        self.assertIs(caught.exception, error)
                    self.assertEqual(len(backend.calls), 0 if side == "detections" else 1)
                    backend.result = ()
                    self.assertEqual(session.update(0, 1000, ()), ())

    def test_duplicate_ids_from_bounded_iterator_preserve_frame_retry(self):
        track = EchoBackend().update(0, 1000, (det(),))[0]
        backend = StubBackend(iter((track, track)))
        session = TrackingSession(backend, TrackingSessionConfig(1, 2))
        with self.assertRaisesRegex(ValueError, "^duplicate track_id in one frame$"):
            session.update(0, 1000, ())
        backend.result = iter((track,))
        self.assertEqual(session.update(0, 1000, ()), (track,))

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
