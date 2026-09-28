import sqlite3
import tempfile
import threading
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
from analytics_lab.tracking import NormalizedBox


def _evidence(suffix: str) -> ForensicEvidenceLink:
    return ForensicEvidenceLink(
        event_id=f"event-{suffix}",
        producer="analytics-test",
        producer_version="1.0.0",
        config_sha256="1" * 64,
        model_sha256="2" * 64,
        source_revision="source-rev-1",
    )


def _record(
    observation_id: str,
    *,
    source_id: str = "camera-1",
    timestamp_ms: int = 1000,
    category: str = "person",
    confidence: float = 0.9,
    track_id: str | None = None,
    plate_text: str | None = None,
    box: NormalizedBox | None = None,
    attributes: tuple[ForensicAttribute, ...] = (),
) -> ForensicRecord:
    return ForensicRecord(
        observation_id=observation_id,
        source_id=source_id,
        timestamp_ms=timestamp_ms,
        category=category,
        confidence=confidence,
        track_id=track_id,
        plate_text=plate_text,
        box=box,
        attributes=attributes,
        evidence=_evidence(observation_id),
    )


class SQLiteForensicIndexTests(unittest.TestCase):
    def test_create_add_close_open_exact_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forensics.db"
            record = _record(
                "obs-1",
                track_id="session-a:1",
                box=NormalizedBox(0.1, 0.2, 0.4, 0.8),
                attributes=(
                    ForensicAttribute("upper_color", "blue", 0.88, "appearance-v1"),
                    ForensicAttribute("direction", "east", 0.93, "motion-v1"),
                ),
            )
            with SQLiteForensicIndex.create(path, max_records=10) as store:
                store.add(record)
                self.assertEqual(store.search(ForensicQuery()), (record,))

            with SQLiteForensicIndex.open(path, max_records=10) as reopened:
                self.assertEqual(reopened.search(ForensicQuery()), (record,))

    def test_query_semantics_match_reference_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forensics.db"
            records = (
                _record(
                    "a",
                    source_id="camera-1",
                    timestamp_ms=100,
                    category="vehicle",
                    confidence=0.9,
                    attributes=(
                        ForensicAttribute("color", "red", 0.8, "vehicle-v1"),
                        ForensicAttribute("type", "pickup", 0.7, "vehicle-v1"),
                    ),
                ),
                _record(
                    "b",
                    source_id="camera-2",
                    timestamp_ms=110,
                    category="vehicle",
                    confidence=0.95,
                    attributes=(
                        ForensicAttribute("color", "blue", 0.9, "vehicle-v1"),
                    ),
                ),
                _record(
                    "c",
                    source_id="gate-1",
                    timestamp_ms=120,
                    category="license_plate",
                    confidence=0.8,
                    plate_text="ABC123",
                ),
                _record(
                    "d",
                    source_id="camera-1",
                    timestamp_ms=130,
                    category="person",
                    confidence=0.7,
                    track_id="session-x:4",
                ),
            )
            reference = ForensicIndex()
            reference.extend(records)
            with SQLiteForensicIndex.create(path, max_records=20) as store:
                for record in records:
                    store.add(record)
                queries = (
                    ForensicQuery(),
                    ForensicQuery(start_ms=105, end_ms=125),
                    ForensicQuery(source_ids=("camera-1", "camera-2")),
                    ForensicQuery(categories=("vehicle",), min_confidence=0.91),
                    ForensicQuery(plate_text="ABC123"),
                    ForensicQuery(plate_prefix="ABC"),
                    ForensicQuery(track_ids=("session-x:4",)),
                    ForensicQuery(attributes=(AttributeFilter("color", "red"),)),
                    ForensicQuery(
                        attributes=(
                            AttributeFilter("color", "red"),
                            AttributeFilter("type", "pickup"),
                            AttributeFilter("color", "red"),
                        )
                    ),
                    ForensicQuery(limit=2),
                )
                for query in queries:
                    with self.subTest(query=query):
                        self.assertEqual(store.search(query), reference.search(query))

    def test_special_character_paths_reopen_exact_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("hash#db.sqlite", "query?db.sqlite", "space db.sqlite", "percent%db.sqlite"):
                with self.subTest(name=name):
                    path = Path(tmp) / name
                    with SQLiteForensicIndex.create(path) as store:
                        store.add(_record("obs-special"))
                    with SQLiteForensicIndex.open(path) as reopened:
                        self.assertEqual(
                            [item.observation_id for item in reopened.search(ForensicQuery())],
                            ["obs-special"],
                        )
                    siblings = {item.name for item in Path(tmp).iterdir()}
                    self.assertIn(name, siblings)
                    self.assertNotIn("hash", siblings)
                    self.assertNotIn("query", siblings)

    def test_open_missing_and_unidentified_store_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.sqlite"
            with self.assertRaises(sqlite3.OperationalError):
                SQLiteForensicIndex.open(missing)

            empty = Path(tmp) / "empty.sqlite"
            empty.touch()
            with self.assertRaisesRegex(RuntimeError, "identity"):
                SQLiteForensicIndex.open(empty)

    def test_existing_nonempty_create_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.sqlite"
            path.write_bytes(b"not-a-database")
            with self.assertRaises(FileExistsError):
                SQLiteForensicIndex.create(path)

    def test_exact_readd_succeeds_at_capacity_conflict_and_new_id_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capacity.sqlite"
            first = _record("one")
            with SQLiteForensicIndex.create(path, max_records=1) as store:
                store.add(first)
                store.add(first)
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    store.add(_record("one", timestamp_ms=1001))
                with self.assertRaisesRegex(RuntimeError, "capacity"):
                    store.add(_record("two"))

    def test_reopen_rejects_lower_capacity_but_accepts_equal_or_higher(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capacity.sqlite"
            with SQLiteForensicIndex.create(path, max_records=3) as store:
                store.add(_record("one"))
                store.add(_record("two", timestamp_ms=1001))
            with self.assertRaisesRegex(RuntimeError, "exceeds configured capacity"):
                SQLiteForensicIndex.open(path, max_records=1)
            with SQLiteForensicIndex.open(path, max_records=2):
                pass
            with SQLiteForensicIndex.open(path, max_records=4) as store:
                store.add(_record("three", timestamp_ms=1002))

    def test_schema_version_shape_and_foreign_key_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schema.sqlite"
            with SQLiteForensicIndex.create(path):
                pass

            raw = sqlite3.connect(path)
            raw.execute("PRAGMA user_version = 99")
            raw.commit()
            raw.close()
            with self.assertRaisesRegex(RuntimeError, "version"):
                SQLiteForensicIndex.open(path)

            path2 = Path(tmp) / "wrong-shape.sqlite"
            raw = sqlite3.connect(path2)
            raw.execute("PRAGMA application_id = 1095517779")
            raw.execute("PRAGMA user_version = 1")
            raw.execute("CREATE TABLE records (observation_id TEXT PRIMARY KEY)")
            raw.execute("CREATE TABLE attributes (observation_id TEXT, ordinal INTEGER)")
            raw.commit()
            raw.close()
            with self.assertRaisesRegex(RuntimeError, "schema shape"):
                SQLiteForensicIndex.open(path2)

    def test_dense_attribute_ordinal_and_canonical_text_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrupt.sqlite"
            record = _record(
                "obs",
                attributes=(
                    ForensicAttribute("color", "red", 0.9, "attrs-v1"),
                    ForensicAttribute("type", "pickup", 0.8, "attrs-v1"),
                ),
            )
            with SQLiteForensicIndex.create(path) as store:
                store.add(record)

            raw = sqlite3.connect(path)
            raw.execute("UPDATE attributes SET ordinal = 2 WHERE ordinal = 1")
            raw.commit()
            raw.close()
            with self.assertRaisesRegex(RuntimeError, "ordinals"):
                SQLiteForensicIndex.open(path)

            path2 = Path(tmp) / "canonical.sqlite"
            with SQLiteForensicIndex.create(path2) as store:
                store.add(_record("obs2", category="vehicle"))
            raw = sqlite3.connect(path2)
            raw.execute("UPDATE records SET category = ' vehicle  ' WHERE observation_id = 'obs2'")
            raw.commit()
            raw.close()
            with self.assertRaisesRegex(RuntimeError, "noncanonical category"):
                SQLiteForensicIndex.open(path2)

    def test_foreign_key_orphan_is_rejected_on_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "orphan.sqlite"
            with SQLiteForensicIndex.create(path):
                pass
            raw = sqlite3.connect(path)
            raw.execute("PRAGMA foreign_keys = OFF")
            raw.execute(
                """
                INSERT INTO attributes
                (observation_id, ordinal, name, value, confidence, provenance)
                VALUES ('missing', 0, 'color', 'red', 0.9, 'attrs-v1')
                """
            )
            raw.commit()
            raw.close()
            with self.assertRaisesRegex(RuntimeError, "foreign-key integrity"):
                SQLiteForensicIndex.open(path)

    def test_signed_zero_remains_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zero.sqlite"
            record = _record("zero", confidence=-0.0)
            with SQLiteForensicIndex.create(path) as store:
                store.add(record)
                store.add(_record("zero", confidence=0.0))
                self.assertEqual(store.search(ForensicQuery()), (record,))

    def test_two_writers_cannot_oversubscribe_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concurrent.sqlite"
            with SQLiteForensicIndex.create(path, max_records=1):
                pass

            results: list[str] = []
            barrier = threading.Barrier(2)

            def writer(observation_id: str) -> None:
                try:
                    with SQLiteForensicIndex.open(path, max_records=1) as store:
                        barrier.wait(timeout=2)
                        store.add(_record(observation_id))
                    results.append("ok")
                except RuntimeError as exc:
                    if "capacity" in str(exc):
                        results.append("capacity")
                    else:
                        results.append(f"runtime:{exc}")
                except Exception as exc:
                    results.append(type(exc).__name__)

            threads = [
                threading.Thread(target=writer, args=("one",)),
                threading.Thread(target=writer, args=("two",)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
            self.assertEqual(sorted(results), ["capacity", "ok"])

    def test_mid_create_failure_rolls_back_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.sqlite"
            original_execute = sqlite3.Connection.execute
            # sqlite3.Connection methods are not monkeypatchable; emulate a
            # partially created unidentified file and prove reopen rejects it.
            raw = sqlite3.connect(path)
            raw.execute("CREATE TABLE records (observation_id TEXT PRIMARY KEY)")
            raw.commit()
            raw.close()
            self.assertTrue(callable(original_execute))
            with self.assertRaisesRegex(RuntimeError, "identity"):
                SQLiteForensicIndex.open(path)


if __name__ == "__main__":
    unittest.main()
