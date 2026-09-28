import tempfile
import unittest
from pathlib import Path

from analytics_lab.forensic_search import (
    AttributeFilter,
    ForensicAttribute,
    ForensicEvidenceLink,
    ForensicIndex,
    ForensicQuery,
    ForensicRecord,
)
from analytics_lab.forensic_sqlite import SQLiteForensicIndex
from analytics_lab.forensic_temporal import (
    TemporalAssociationConfig,
    attribute_temporal_trails,
    exact_plate_temporal_trails,
)


def _evidence(suffix: str) -> ForensicEvidenceLink:
    return ForensicEvidenceLink(
        event_id=f"event-{suffix}",
        producer="lpr-test",
        producer_version="1.0.0",
        config_sha256="1" * 64,
        model_sha256="2" * 64,
        source_revision="source-rev-1",
    )


def _plate(
    observation_id: str,
    source_id: str,
    timestamp_ms: int,
    plate_text: str,
) -> ForensicRecord:
    return ForensicRecord(
        observation_id=observation_id,
        source_id=source_id,
        timestamp_ms=timestamp_ms,
        category="license_plate",
        confidence=0.9,
        plate_text=plate_text,
        evidence=_evidence(observation_id),
    )


def _object(
    observation_id: str,
    source_id: str,
    timestamp_ms: int,
    category: str,
    *,
    attributes: tuple[ForensicAttribute, ...],
) -> ForensicRecord:
    return ForensicRecord(
        observation_id=observation_id,
        source_id=source_id,
        timestamp_ms=timestamp_ms,
        category=category,
        confidence=0.9,
        attributes=attributes,
        evidence=_evidence(observation_id),
    )


class ForensicTemporalTests(unittest.TestCase):
    def test_exact_plate_links_distinct_sources_without_identity_claim(self):
        index = ForensicIndex()
        first = _plate("a", "camera-1", 1000, "ABC123")
        second = _plate("b", "camera-2", 1200, "ABC123")
        index.extend((first, second))

        trails = exact_plate_temporal_trails(index, ForensicQuery())
        self.assertEqual(len(trails), 1)
        trail = trails[0]
        self.assertEqual(trail.association_kind, "exact_plate_text")
        self.assertEqual(trail.association_value, "ABC123")
        self.assertEqual(trail.records, (first, second))
        self.assertFalse(trail.identity_claim)
        self.assertEqual(trail.records[0].evidence, first.evidence)
        self.assertEqual(trail.records[1].evidence, second.evidence)

    def test_prefix_is_retrieval_only_and_never_cross_links_different_exact_plates(self):
        index = ForensicIndex()
        records = (
            _plate("a1", "camera-1", 1000, "ABC123"),
            _plate("a2", "camera-2", 1100, "ABC123"),
            _plate("b1", "camera-3", 1200, "ABC999"),
            _plate("b2", "camera-4", 1300, "ABC999"),
        )
        index.extend(records)
        trails = exact_plate_temporal_trails(
            index,
            ForensicQuery(plate_prefix="ABC"),
        )
        self.assertEqual(
            [(trail.association_value, [r.observation_id for r in trail.records]) for trail in trails],
            [
                ("ABC123", ["a1", "a2"]),
                ("ABC999", ["b1", "b2"]),
            ],
        )

    def test_same_source_only_records_do_not_create_cross_camera_trail(self):
        index = ForensicIndex()
        index.extend(
            (
                _plate("a", "camera-1", 1000, "ABC123"),
                _plate("b", "camera-1", 1100, "ABC123"),
            )
        )
        self.assertEqual(exact_plate_temporal_trails(index, ForensicQuery()), ())

    def test_link_gap_splits_trails(self):
        index = ForensicIndex()
        records = (
            _plate("a", "camera-1", 1000, "ABC123"),
            _plate("b", "camera-2", 1100, "ABC123"),
            _plate("c", "camera-3", 5000, "ABC123"),
            _plate("d", "camera-4", 5100, "ABC123"),
        )
        index.extend(records)
        trails = exact_plate_temporal_trails(
            index,
            ForensicQuery(),
            config=TemporalAssociationConfig(
                max_link_gap_ms=500,
                max_total_span_ms=2000,
            ),
        )
        self.assertEqual(
            [[r.observation_id for r in trail.records] for trail in trails],
            [["a", "b"], ["c", "d"]],
        )

    def test_total_span_blocks_unbounded_transitive_chain(self):
        index = ForensicIndex()
        records = (
            _plate("a", "camera-1", 0, "ABC123"),
            _plate("b", "camera-2", 400, "ABC123"),
            _plate("c", "camera-3", 800, "ABC123"),
            _plate("d", "camera-4", 1200, "ABC123"),
        )
        index.extend(records)
        trails = exact_plate_temporal_trails(
            index,
            ForensicQuery(),
            config=TemporalAssociationConfig(
                max_link_gap_ms=500,
                max_total_span_ms=900,
            ),
        )
        self.assertEqual(
            [[r.observation_id for r in trail.records] for trail in trails],
            [["a", "b", "c"]],
        )

    def test_record_and_source_caps_bound_trails(self):
        index = ForensicIndex()
        records = (
            _plate("a", "camera-1", 100, "ABC123"),
            _plate("b", "camera-2", 200, "ABC123"),
            _plate("c", "camera-3", 300, "ABC123"),
            _plate("d", "camera-4", 400, "ABC123"),
        )
        index.extend(records)
        trails = exact_plate_temporal_trails(
            index,
            ForensicQuery(),
            config=TemporalAssociationConfig(
                max_link_gap_ms=1000,
                max_total_span_ms=5000,
                max_records_per_trail=2,
                max_sources_per_trail=2,
            ),
        )
        self.assertEqual(
            [[r.observation_id for r in trail.records] for trail in trails],
            [["a", "b"], ["c", "d"]],
        )

    def test_non_plate_records_are_ignored(self):
        index = ForensicIndex()
        index.extend(
            (
                ForensicRecord(
                    "person-1",
                    "camera-1",
                    100,
                    "person",
                    0.9,
                    evidence=_evidence("person-1"),
                ),
                _plate("plate-1", "camera-1", 200, "ABC123"),
            )
        )
        self.assertEqual(exact_plate_temporal_trails(index, ForensicQuery()), ())

    def test_sqlite_backend_produces_same_temporal_result(self):
        records = (
            _plate("a", "camera-1", 1000, "ABC123"),
            _plate("b", "camera-2", 1200, "ABC123"),
            _plate("c", "camera-3", 1300, "XYZ999"),
        )
        memory = ForensicIndex()
        memory.extend(records)
        expected = exact_plate_temporal_trails(memory, ForensicQuery())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forensics.sqlite"
            with SQLiteForensicIndex.create(path) as store:
                for record in records:
                    store.add(record)
                actual = exact_plate_temporal_trails(store, ForensicQuery())
        self.assertEqual(actual, expected)

    def test_query_limit_bounds_candidate_generation(self):
        index = ForensicIndex()
        index.extend(
            (
                _plate("a", "camera-1", 100, "ABC123"),
                _plate("b", "camera-2", 200, "ABC123"),
                _plate("c", "camera-3", 300, "ABC123"),
            )
        )
        trails = exact_plate_temporal_trails(index, ForensicQuery(limit=1))
        self.assertEqual(trails, ())



class ForensicAttributeTemporalTests(unittest.TestCase):
    def test_exact_attributes_create_nonidentity_cross_camera_candidate_trail(self):
        attrs = (
            ForensicAttribute("upper_color", "blue", 0.91, "appearance-v1"),
            ForensicAttribute("direction", "east", 0.88, "motion-v1"),
        )
        first = _object("a", "camera-1", 1000, "person", attributes=attrs)
        second = _object("b", "camera-2", 1200, "person", attributes=attrs)
        index = ForensicIndex()
        index.extend((first, second))

        trails = attribute_temporal_trails(
            index,
            ForensicQuery(
                categories=("person",),
                attributes=(
                    AttributeFilter("upper_color", "blue"),
                    AttributeFilter("direction", "east"),
                ),
            ),
        )
        self.assertEqual(len(trails), 1)
        trail = trails[0]
        self.assertEqual(trail.category, "person")
        self.assertEqual(
            [(item.name, item.value) for item in trail.required_attributes],
            [("direction", "east"), ("upper_color", "blue")],
        )
        self.assertEqual(trail.records, (first, second))
        self.assertFalse(trail.identity_claim)
        self.assertEqual(trail.records[0].evidence, first.evidence)
        self.assertEqual(trail.records[1].evidence, second.evidence)

    def test_duplicate_filters_are_canonical_and_set_equivalent(self):
        attrs = (ForensicAttribute("color", "red", 0.9, "vehicle-v1"),)
        index = ForensicIndex()
        index.extend(
            (
                _object("a", "camera-1", 1000, "vehicle", attributes=attrs),
                _object("b", "camera-2", 1200, "vehicle", attributes=attrs),
            )
        )
        trails = attribute_temporal_trails(
            index,
            ForensicQuery(
                attributes=(
                    AttributeFilter("color", "red"),
                    AttributeFilter("color", "red"),
                )
            ),
        )
        self.assertEqual(len(trails), 1)
        self.assertEqual(len(trails[0].required_attributes), 1)

    def test_categories_never_mix_in_one_candidate_trail(self):
        attrs = (ForensicAttribute("color", "red", 0.9, "appearance-v1"),)
        index = ForensicIndex()
        index.extend(
            (
                _object("p1", "camera-1", 1000, "person", attributes=attrs),
                _object("p2", "camera-2", 1100, "person", attributes=attrs),
                _object("v1", "camera-3", 1200, "vehicle", attributes=attrs),
                _object("v2", "camera-4", 1300, "vehicle", attributes=attrs),
            )
        )
        trails = attribute_temporal_trails(
            index,
            ForensicQuery(attributes=(AttributeFilter("color", "red"),)),
        )
        self.assertEqual([trail.category for trail in trails], ["person", "vehicle"])
        self.assertTrue(all(len({r.category for r in trail.records}) == 1 for trail in trails))

    def test_same_source_only_does_not_emit_cross_camera_candidate(self):
        attrs = (ForensicAttribute("upper_color", "blue", 0.9, "appearance-v1"),)
        index = ForensicIndex()
        index.extend(
            (
                _object("a", "camera-1", 1000, "person", attributes=attrs),
                _object("b", "camera-1", 1200, "person", attributes=attrs),
            )
        )
        self.assertEqual(
            attribute_temporal_trails(
                index,
                ForensicQuery(attributes=(AttributeFilter("upper_color", "blue"),)),
            ),
            (),
        )

    def test_attribute_trails_require_exact_attribute_query(self):
        index = ForensicIndex()
        with self.assertRaisesRegex(ValueError, "requires exact attribute"):
            attribute_temporal_trails(index, ForensicQuery())

    def test_time_bounds_split_candidate_trails(self):
        attrs = (ForensicAttribute("upper_color", "blue", 0.9, "appearance-v1"),)
        index = ForensicIndex()
        index.extend(
            (
                _object("a", "camera-1", 0, "person", attributes=attrs),
                _object("b", "camera-2", 100, "person", attributes=attrs),
                _object("c", "camera-3", 2000, "person", attributes=attrs),
                _object("d", "camera-4", 2100, "person", attributes=attrs),
            )
        )
        trails = attribute_temporal_trails(
            index,
            ForensicQuery(attributes=(AttributeFilter("upper_color", "blue"),)),
            config=TemporalAssociationConfig(
                max_link_gap_ms=500,
                max_total_span_ms=1000,
            ),
        )
        self.assertEqual(
            [[record.observation_id for record in trail.records] for trail in trails],
            [["a", "b"], ["c", "d"]],
        )

    def test_license_plate_records_are_not_attribute_candidate_trails(self):
        plate = ForensicRecord(
            "plate",
            "camera-1",
            1000,
            "license_plate",
            0.9,
            plate_text="ABC123",
            attributes=(ForensicAttribute("color", "blue", 0.9, "appearance-v1"),),
            evidence=_evidence("plate"),
        )
        index = ForensicIndex()
        index.extend((plate,))
        self.assertEqual(
            attribute_temporal_trails(
                index,
                ForensicQuery(attributes=(AttributeFilter("color", "blue"),)),
            ),
            (),
        )

    def test_sqlite_backend_matches_in_memory_attribute_trails(self):
        attrs = (
            ForensicAttribute("upper_color", "blue", 0.91, "appearance-v1"),
            ForensicAttribute("direction", "east", 0.88, "motion-v1"),
        )
        records = (
            _object("a", "camera-1", 1000, "person", attributes=attrs),
            _object("b", "camera-2", 1200, "person", attributes=attrs),
            _object("c", "camera-3", 1400, "vehicle", attributes=attrs),
        )
        query = ForensicQuery(
            attributes=(
                AttributeFilter("upper_color", "blue"),
                AttributeFilter("direction", "east"),
            )
        )
        memory = ForensicIndex()
        memory.extend(records)
        expected = attribute_temporal_trails(memory, query)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forensics.sqlite"
            with SQLiteForensicIndex.create(path) as store:
                for record in records:
                    store.add(record)
                actual = attribute_temporal_trails(store, query)
        self.assertEqual(actual, expected)

    def test_query_limit_bounds_attribute_candidate_generation(self):
        attrs = (ForensicAttribute("color", "red", 0.9, "appearance-v1"),)
        index = ForensicIndex()
        index.extend(
            (
                _object("a", "camera-1", 100, "vehicle", attributes=attrs),
                _object("b", "camera-2", 200, "vehicle", attributes=attrs),
                _object("c", "camera-3", 300, "vehicle", attributes=attrs),
            )
        )
        trails = attribute_temporal_trails(
            index,
            ForensicQuery(
                attributes=(AttributeFilter("color", "red"),),
                limit=1,
            ),
        )
        self.assertEqual(trails, ())


if __name__ == "__main__":
    unittest.main()
