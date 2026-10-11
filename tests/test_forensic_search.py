import unittest

from analytics_lab.forensic_search import (
    AttributeFilter,
    ForensicAttribute,
    ForensicEvidenceLink,
    ForensicIndex,
    ForensicQuery,
    ForensicRecord,
    record_from_lpr,
    record_from_track,
)
from analytics_lab.lpr_ocr import LPRObservation, OCRText, PlateDetection
from analytics_lab.tracking import NormalizedBox, TrackedDetection


def _evidence(event_id: str) -> ForensicEvidenceLink:
    return ForensicEvidenceLink(
        event_id=event_id,
        producer="analytics-test",
        producer_version="1.0.0",
        config_sha256="1" * 64,
        model_sha256="2" * 64,
        source_revision="source-rev-1",
    )


class ForensicSearchTests(unittest.TestCase):
    def test_track_adapter_and_multifilter_search(self):
        track = TrackedDetection(
            track_id="session-a:17",
            category="person",
            confidence=0.91,
            box=NormalizedBox(0.1, 0.2, 0.4, 0.8),
        )
        attrs = (
            ForensicAttribute("upper_color", "blue", 0.88, "appearance-v1"),
            ForensicAttribute("direction", "east", 0.93, "motion-v1"),
        )
        evidence = _evidence("event-track-1")
        record = record_from_track(
            observation_id="obs-1",
            source_id="camera-1",
            timestamp_ms=1_000,
            tracked=track,
            evidence=evidence,
            attributes=attrs,
        )
        index = ForensicIndex()
        index.add(record)

        query = ForensicQuery(
            start_ms=900,
            end_ms=1_100,
            source_ids=("camera-1",),
            categories=("person",),
            track_ids=("session-a:17",),
            attributes=(AttributeFilter("upper_color", "blue"),),
            min_confidence=0.9,
        )
        self.assertEqual(index.search(query), (record,))
        self.assertEqual(record.evidence, evidence)

    def test_lpr_adapter_supports_exact_and_prefix_search(self):
        observation = LPRObservation(
            PlateDetection(0.95, 0.2, 0.3, 0.7, 0.5),
            OCRText("ABC123", 0.8),
        )
        record = record_from_lpr(
            observation_id="plate-1",
            source_id="gate-2",
            timestamp_ms=2_000,
            observation=observation,
            evidence=_evidence("event-lpr-1"),
        )
        index = ForensicIndex()
        index.add(record)

        self.assertEqual(
            index.search(ForensicQuery(plate_text="ABC123")),
            (record,),
        )
        self.assertEqual(
            index.search(ForensicQuery(plate_prefix="ABC")),
            (record,),
        )
        self.assertEqual(
            index.search(ForensicQuery(plate_prefix="XYZ")),
            (),
        )
        self.assertEqual(record.confidence, 0.8)
        self.assertEqual(record.category, "license_plate")

    def test_search_order_and_limit_are_deterministic(self):
        index = ForensicIndex()
        records = (
            ForensicRecord("c", "camera-2", 20, "vehicle", 0.8, evidence=_evidence("event-c")),
            ForensicRecord("b", "camera-1", 10, "vehicle", 0.8, evidence=_evidence("event-b")),
            ForensicRecord("a", "camera-1", 10, "vehicle", 0.8, evidence=_evidence("event-a")),
        )
        index.extend(records)
        result = index.search(ForensicQuery(categories=("vehicle",), limit=2))
        self.assertEqual([item.observation_id for item in result], ["a", "b"])

    def test_duplicate_id_is_idempotent_but_conflict_fails_closed(self):
        index = ForensicIndex()
        first = ForensicRecord(
            "obs-1",
            "camera-1",
            10,
            "person",
            0.8,
            evidence=_evidence("event-one"),
        )
        index.add(first)
        index.add(first)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            index.add(
                ForensicRecord(
                    "obs-1",
                    "camera-1",
                    11,
                    "person",
                    0.8,
                    evidence=_evidence("event-two"),
                )
            )

    def test_capacity_is_bounded(self):
        index = ForensicIndex(max_records=1)
        index.add(
            ForensicRecord(
                "one",
                "camera-1",
                1,
                "person",
                0.8,
                evidence=_evidence("event-one"),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            index.add(
                ForensicRecord(
                    "two",
                    "camera-1",
                    2,
                    "person",
                    0.8,
                    evidence=_evidence("event-two"),
                )
            )

    def test_attribute_filters_require_all_requested_facets(self):
        record = ForensicRecord(
            "obs-1",
            "camera-1",
            10,
            "vehicle",
            0.9,
            attributes=(
                ForensicAttribute("color", "red", 0.9, "vehicle-attrs-v1"),
                ForensicAttribute("type", "pickup", 0.8, "vehicle-attrs-v1"),
            ),
            evidence=_evidence("event-attrs"),
        )
        index = ForensicIndex()
        index.add(record)
        self.assertEqual(
            index.search(
                ForensicQuery(
                    attributes=(
                        AttributeFilter("color", "red"),
                        AttributeFilter("type", "pickup"),
                    )
                )
            ),
            (record,),
        )
        self.assertEqual(
            index.search(
                ForensicQuery(
                    attributes=(
                        AttributeFilter("color", "red"),
                        AttributeFilter("type", "sedan"),
                    )
                )
            ),
            (),
        )

    def test_searchable_record_requires_evidence_provenance(self):
        index = ForensicIndex()
        with self.assertRaisesRegex(ValueError, "requires immutable evidence"):
            index.add(ForensicRecord("obs", "camera-1", 10, "person", 0.9))

    def test_evidence_link_is_hash_bound(self):
        evidence = _evidence("event-1")
        self.assertEqual(evidence.producer, "analytics-test")
        self.assertEqual(evidence.config_sha256, "1" * 64)
        self.assertEqual(evidence.model_sha256, "2" * 64)
        with self.assertRaisesRegex(ValueError, "config_sha256"):
            ForensicEvidenceLink(
                event_id="bad",
                producer="analytics-test",
                producer_version="1.0.0",
                config_sha256="not-a-hash",
            )

    def test_plate_text_is_restricted_to_plate_records(self):
        with self.assertRaisesRegex(ValueError, "only valid"):
            ForensicRecord(
                "obs",
                "camera-1",
                10,
                "person",
                0.9,
                plate_text="ABC123",
                evidence=_evidence("event-plate-invalid"),
            )

    def test_query_rejects_ambiguous_plate_modes_and_bad_time_window(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            ForensicQuery(plate_text="ABC123", plate_prefix="ABC")
        with self.assertRaisesRegex(ValueError, "end_ms"):
            ForensicQuery(start_ms=20, end_ms=10)


class ForensicBatchBoundTests(unittest.TestCase):
    @staticmethod
    def record(name="one"):
        return ForensicRecord(name, "camera-1", 1, "person", 0.8,
                              evidence=_evidence("event-" + name))

    @staticmethod
    def producer(values, *, stop_after=None):
        class FiniteProducer:
            def __init__(self):
                self.calls = 0
                self.values = iter(values)

            def __iter__(self):
                return self

            def __next__(self):
                self.calls += 1
                if stop_after is not None and self.calls > stop_after:
                    raise AssertionError("finite producer read beyond overflow boundary")
                return next(self.values)

        return FiniteProducer()

    def test_overflow_reads_only_one_excess_item_before_any_add(self):
        for capacity in (1, 2, 4):
            with self.subTest(capacity=capacity):
                index = ForensicIndex(max_records=capacity)
                existing = self.record()
                index.add(existing)
                values = self.producer((existing,) * (capacity + 4))
                with self.assertRaisesRegex(RuntimeError, "input batch exceeds forensic index capacity"):
                    index.extend(values)
                self.assertEqual(values.calls, capacity + 1)
                self.assertEqual(index.search(ForensicQuery()), (existing,))

    def test_finite_overflow_does_not_reach_a_later_producer_failure(self):
        index = ForensicIndex(max_records=2)
        values = self.producer((self.record(),) * 4, stop_after=3)
        with self.assertRaisesRegex(RuntimeError, "input batch exceeds forensic index capacity"):
            index.extend(values)
        self.assertEqual(values.calls, 3)
        self.assertEqual(index.search(ForensicQuery()), ())

    def test_length_hints_are_not_consulted(self):
        for method in ("__len__", "__length_hint__"):
            with self.subTest(method=method):
                values = self.producer((self.record(),))

                def forbidden_hint(self):
                    raise AssertionError("producer length hint was consulted")

                setattr(type(values), method, forbidden_hint)
                index = ForensicIndex(max_records=2)
                index.extend(values)
                self.assertEqual(values.calls, 2)
                self.assertEqual(index.search(ForensicQuery()), (self.record(),))

    def test_empty_under_and_exact_capacity_batches(self):
        for capacity in (1, 2, 4):
            for count in range(capacity + 1):
                with self.subTest(capacity=capacity, count=count):
                    records = tuple(self.record(str(i)) for i in range(count))
                    values = self.producer(records)
                    index = ForensicIndex(max_records=capacity)
                    index.extend(values)
                    self.assertEqual(values.calls, count + 1)
                    self.assertEqual(index.search(ForensicQuery()), records)

    def test_configured_capacity_bounds_without_large_allocation(self):
        for capacity in (0, -1, True, 1.0, 1_000_001):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                ForensicIndex(max_records=capacity)
        index = ForensicIndex(max_records=1_000_000)
        values = self.producer((self.record(),))
        index.extend(values)
        self.assertEqual(values.calls, 2)
        self.assertEqual(index.search(ForensicQuery()), (self.record(),))

    def test_duplicate_replay_at_capacity_and_conflict_semantics(self):
        index = ForensicIndex(max_records=1)
        first = self.record()
        index.add(first)
        index.extend(self.producer((first,)))
        self.assertEqual(index.search(ForensicQuery()), (first,))
        conflict = ForensicRecord("one", "camera-2", 1, "person", 0.8,
                                  evidence=_evidence("event-one"))
        with self.assertRaisesRegex(ValueError, "observation_id conflicts with existing record"):
            index.extend(self.producer((conflict,)))
        self.assertEqual(index.search(ForensicQuery()), (first,))

    def test_malformed_and_capacity_add_failures_keep_existing_partial_add_behavior(self):
        first, second = self.record("one"), self.record("two")
        no_evidence = ForensicRecord("missing", "camera-1", 1, "person", 0.8)
        for invalid, message in ((object(), "record must be ForensicRecord"),
                                 (no_evidence, "requires immutable evidence provenance")):
            with self.subTest(message=message):
                index = ForensicIndex(max_records=2)
                with self.assertRaisesRegex(ValueError, message):
                    index.extend(self.producer((first, invalid)))
                self.assertEqual(index.search(ForensicQuery()), (first,))
        index = ForensicIndex(max_records=2)
        index.add(first)
        with self.assertRaisesRegex(RuntimeError, "forensic index capacity exceeded"):
            index.extend(self.producer((second, self.record("three"))))
        self.assertEqual(index.search(ForensicQuery()), (first, second))

    def test_overflow_still_precedes_item_validation(self):
        index = ForensicIndex(max_records=1)
        values = self.producer((object(), object(), object()))
        with self.assertRaisesRegex(RuntimeError, "input batch exceeds forensic index capacity"):
            index.extend(values)
        self.assertEqual(values.calls, 2)
        self.assertEqual(index.search(ForensicQuery()), ())

    def test_raising_producers_preserve_state_and_public_error_behavior(self):
        for error in (ValueError("producer fault"), RuntimeError("producer fault"),
                      TypeError("producer type fault")):
            with self.subTest(error=type(error).__name__):
                index = ForensicIndex(max_records=4)
                existing = self.record()
                index.add(existing)

                def finite_raising():
                    yield self.record("two")
                    raise error

                expected = ValueError if isinstance(error, TypeError) else type(error)
                message = "records must be iterable" if isinstance(error, TypeError) else "producer fault"
                with self.assertRaisesRegex(expected, message) as raised:
                    index.extend(finite_raising())
                if isinstance(error, TypeError):
                    self.assertIs(raised.exception.__cause__, error)
                else:
                    self.assertIs(raised.exception, error)
                self.assertEqual(index.search(ForensicQuery()), (existing,))

    def test_noniterable_and_raising_iter_keep_public_error_behavior(self):
        class BadIterable:
            def __iter__(self):
                raise TypeError("iteration unavailable")

        for values in (None, 3, BadIterable()):
            with self.subTest(values=type(values).__name__):
                index = ForensicIndex(max_records=1)
                with self.assertRaisesRegex(ValueError, "records must be iterable") as raised:
                    index.extend(values)
                self.assertIsInstance(raised.exception.__cause__, TypeError)
                self.assertEqual(index.search(ForensicQuery()), ())


if __name__ == "__main__":
    unittest.main()
