"""Durable stdlib-SQLite backend for evidence-linked forensic metadata search.

This backend preserves the reference ForensicIndex query semantics while adding
durability, fail-closed schema/integrity checks, bounded capacity, and safe
concurrent writers. It stores normalized analytic metadata only; never raw
image/video pixels or biometric templates.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable

from .forensic_search import (
    AttributeFilter,
    ForensicAttribute,
    ForensicEvidenceLink,
    ForensicQuery,
    ForensicRecord,
)
from .tracking import NormalizedBox

_APPLICATION_ID = 0x414C4653  # "ALFS"
_USER_VERSION = 1
_REQUIRED_INDEXES = {
    "idx_records_time",
    "idx_records_source_time",
    "idx_records_category_time",
    "idx_records_track_time",
    "idx_records_plate_time",
    "idx_attributes_name_value",
}
_RECORD_COLUMNS = (
    "observation_id",
    "source_id",
    "timestamp_ms",
    "category",
    "confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "track_id",
    "plate_text",
    "evidence_event_id",
    "evidence_producer",
    "evidence_producer_version",
    "evidence_config_sha256",
    "evidence_model_sha256",
    "evidence_source_revision",
)
_ATTRIBUTE_COLUMNS = (
    "observation_id",
    "ordinal",
    "name",
    "value",
    "confidence",
    "provenance",
)
_DDL = (
    """CREATE TABLE records (
        observation_id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        timestamp_ms INTEGER NOT NULL,
        category TEXT NOT NULL,
        confidence REAL NOT NULL,
        x_min REAL,
        y_min REAL,
        x_max REAL,
        y_max REAL,
        track_id TEXT,
        plate_text TEXT,
        evidence_event_id TEXT NOT NULL,
        evidence_producer TEXT NOT NULL,
        evidence_producer_version TEXT NOT NULL,
        evidence_config_sha256 TEXT NOT NULL,
        evidence_model_sha256 TEXT,
        evidence_source_revision TEXT
    )""",
    """CREATE TABLE attributes (
        observation_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        name TEXT NOT NULL,
        value TEXT NOT NULL,
        confidence REAL NOT NULL,
        provenance TEXT NOT NULL,
        PRIMARY KEY (observation_id, ordinal),
        FOREIGN KEY (observation_id) REFERENCES records(observation_id) ON DELETE CASCADE
    )""",
    "CREATE INDEX idx_records_time ON records(timestamp_ms, source_id, observation_id)",
    "CREATE INDEX idx_records_source_time ON records(source_id, timestamp_ms, observation_id)",
    "CREATE INDEX idx_records_category_time ON records(category, timestamp_ms, observation_id)",
    "CREATE INDEX idx_records_track_time ON records(track_id, timestamp_ms, observation_id)",
    "CREATE INDEX idx_records_plate_time ON records(plate_text, timestamp_ms, observation_id)",
    "CREATE INDEX idx_attributes_name_value ON attributes(name, value, observation_id)",
)


class SQLiteForensicIndex:
    """Durable forensic metadata index with exact reference-query semantics."""

    def __init__(self, connection: sqlite3.Connection, path: Path, *, max_records: int) -> None:
        if type(max_records) is not int or not 1 <= max_records <= 1_000_000:
            raise ValueError("max_records must be an integer in [1, 1000000]")
        self._connection = connection
        self.path = path
        self.max_records = max_records
        self._closed = False

    @classmethod
    def create(cls, path: str | Path, *, max_records: int = 100_000) -> "SQLiteForensicIndex":
        target = Path(path).expanduser().resolve()
        if target.exists() and target.stat().st_size > 0:
            raise FileExistsError("refusing to overwrite an existing nonempty forensic store")
        target.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            str(target),
            isolation_level=None,
            timeout=1.0,
        )
        try:
            cls._enable_foreign_keys(connection)
            connection.execute("BEGIN IMMEDIATE")
            try:
                for statement in _DDL:
                    connection.execute(statement)
                connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                connection.execute(f"PRAGMA user_version = {_USER_VERSION}")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            backend = cls(connection, target, max_records=max_records)
            backend._validate_store()
            return backend
        except Exception:
            connection.close()
            raise

    @classmethod
    def open(cls, path: str | Path, *, max_records: int = 100_000) -> "SQLiteForensicIndex":
        target = Path(path).expanduser().resolve()
        uri = target.as_uri() + "?mode=rw"
        connection = sqlite3.connect(
            uri,
            uri=True,
            isolation_level=None,
            timeout=1.0,
        )
        try:
            cls._enable_foreign_keys(connection)
            backend = cls(connection, target, max_records=max_records)
            backend._validate_store()
            return backend
        except Exception:
            connection.close()
            raise

    @staticmethod
    def _enable_foreign_keys(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA foreign_keys = ON")
        row = connection.execute("PRAGMA foreign_keys").fetchone()
        if row is None or int(row[0]) != 1:
            raise RuntimeError("SQLite foreign-key enforcement is unavailable")

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "SQLiteForensicIndex":
        if self._closed:
            raise RuntimeError("forensic store is closed")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("forensic store is closed")

    def _validate_store(self) -> None:
        self._require_open()

        application_id = self._connection.execute("PRAGMA application_id").fetchone()
        user_version = self._connection.execute("PRAGMA user_version").fetchone()
        if application_id is None or int(application_id[0]) != _APPLICATION_ID:
            raise RuntimeError("unrecognized forensic store identity")
        if user_version is None or int(user_version[0]) != _USER_VERSION:
            raise RuntimeError("unsupported forensic store version")

        record_columns = tuple(
            row[1] for row in self._connection.execute("PRAGMA table_info(records)")
        )
        attribute_columns = tuple(
            row[1] for row in self._connection.execute("PRAGMA table_info(attributes)")
        )
        if record_columns != _RECORD_COLUMNS or attribute_columns != _ATTRIBUTE_COLUMNS:
            raise RuntimeError("forensic store schema shape is invalid")

        foreign_keys = tuple(self._connection.execute("PRAGMA foreign_key_list(attributes)"))
        if len(foreign_keys) != 1:
            raise RuntimeError("forensic attribute foreign-key contract is invalid")
        foreign_key = foreign_keys[0]
        if (
            foreign_key[2] != "records"
            or foreign_key[3] != "observation_id"
            or foreign_key[4] != "observation_id"
            or str(foreign_key[6]).upper() != "CASCADE"
        ):
            raise RuntimeError("forensic attribute foreign-key contract is invalid")

        indexes = {
            row[1]
            for table in ("records", "attributes")
            for row in self._connection.execute(f"PRAGMA index_list({table})")
        }
        if not _REQUIRED_INDEXES.issubset(indexes):
            raise RuntimeError("forensic store required index is missing")

        quick = self._connection.execute("PRAGMA quick_check").fetchone()
        if quick is None or quick[0] != "ok":
            raise RuntimeError("forensic store quick_check failed")
        if tuple(self._connection.execute("PRAGMA foreign_key_check")):
            raise RuntimeError("forensic store foreign-key integrity failed")

        bad_ordinals = self._connection.execute(
            """
            SELECT observation_id
            FROM attributes
            GROUP BY observation_id
            HAVING MIN(ordinal) <> 0 OR MAX(ordinal) <> COUNT(*) - 1
            LIMIT 1
            """
        ).fetchone()
        if bad_ordinals is not None:
            raise RuntimeError("forensic attribute ordinals are not dense")

        count_row = self._connection.execute("SELECT COUNT(*) FROM records").fetchone()
        count = int(count_row[0]) if count_row is not None else 0
        if count > self.max_records:
            raise RuntimeError("persisted forensic record count exceeds configured capacity")

        # Reconstruct every record once on open so invalid/corrupt persisted text,
        # hashes, boxes, plates, or attribute metadata fails closed before serving.
        for row in self._connection.execute(
            "SELECT observation_id FROM records ORDER BY observation_id"
        ):
            self._read_record(str(row[0]))

    def _read_record(self, observation_id: str) -> ForensicRecord:
        row = self._connection.execute(
            """
            SELECT observation_id, source_id, timestamp_ms, category, confidence,
                   x_min, y_min, x_max, y_max, track_id, plate_text,
                   evidence_event_id, evidence_producer, evidence_producer_version,
                   evidence_config_sha256, evidence_model_sha256, evidence_source_revision
            FROM records
            WHERE observation_id = ?
            """,
            (observation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(observation_id)

        raw_category = str(row[3])
        box_values = row[5:9]
        if all(value is None for value in box_values):
            box = None
        elif any(value is None for value in box_values):
            raise RuntimeError("forensic store contains a partial normalized box")
        else:
            box = NormalizedBox(*(float(value) for value in box_values))

        attribute_rows = tuple(
            self._connection.execute(
                """
                SELECT ordinal, name, value, confidence, provenance
                FROM attributes
                WHERE observation_id = ?
                ORDER BY ordinal
                """,
                (observation_id,),
            )
        )
        attributes = tuple(
            ForensicAttribute(
                name=str(item[1]),
                value=str(item[2]),
                confidence=float(item[3]),
                provenance=str(item[4]),
            )
            for item in attribute_rows
        )
        for raw, parsed in zip(attribute_rows, attributes):
            if str(raw[1]) != parsed.name or str(raw[2]) != parsed.value:
                raise RuntimeError("forensic store contains noncanonical attribute text")

        evidence = ForensicEvidenceLink(
            event_id=str(row[11]),
            producer=str(row[12]),
            producer_version=str(row[13]),
            config_sha256=str(row[14]),
            model_sha256=None if row[15] is None else str(row[15]),
            source_revision=None if row[16] is None else str(row[16]),
        )
        record = ForensicRecord(
            observation_id=str(row[0]),
            source_id=str(row[1]),
            timestamp_ms=int(row[2]),
            category=raw_category,
            confidence=float(row[4]),
            box=box,
            track_id=None if row[9] is None else str(row[9]),
            plate_text=None if row[10] is None else str(row[10]),
            attributes=attributes,
            evidence=evidence,
        )
        if raw_category != record.category:
            raise RuntimeError("forensic store contains noncanonical category text")
        return record

    def add(self, record: ForensicRecord) -> None:
        self._require_open()
        if not isinstance(record, ForensicRecord):
            raise ValueError("record must be ForensicRecord")
        if record.evidence is None:
            raise ValueError("searchable forensic record requires immutable evidence provenance")

        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                "SELECT observation_id FROM records WHERE observation_id = ?",
                (record.observation_id,),
            ).fetchone()
            if existing is not None:
                stored = self._read_record(record.observation_id)
                if stored != record:
                    raise ValueError("observation_id conflicts with existing record")
                connection.execute("COMMIT")
                return

            count_row = connection.execute("SELECT COUNT(*) FROM records").fetchone()
            count = int(count_row[0]) if count_row is not None else 0
            if count >= self.max_records:
                raise RuntimeError("forensic index capacity exceeded")

            if record.box is None:
                box_values = (None, None, None, None)
            else:
                box_values = (
                    record.box.x_min,
                    record.box.y_min,
                    record.box.x_max,
                    record.box.y_max,
                )
            evidence = record.evidence
            connection.execute(
                """
                INSERT INTO records (
                    observation_id, source_id, timestamp_ms, category, confidence,
                    x_min, y_min, x_max, y_max, track_id, plate_text,
                    evidence_event_id, evidence_producer, evidence_producer_version,
                    evidence_config_sha256, evidence_model_sha256, evidence_source_revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.observation_id,
                    record.source_id,
                    record.timestamp_ms,
                    record.category,
                    record.confidence,
                    *box_values,
                    record.track_id,
                    record.plate_text,
                    evidence.event_id,
                    evidence.producer,
                    evidence.producer_version,
                    evidence.config_sha256,
                    evidence.model_sha256,
                    evidence.source_revision,
                ),
            )
            for ordinal, attribute in enumerate(record.attributes):
                connection.execute(
                    """
                    INSERT INTO attributes (
                        observation_id, ordinal, name, value, confidence, provenance
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.observation_id,
                        ordinal,
                        attribute.name,
                        attribute.value,
                        attribute.confidence,
                        attribute.provenance,
                    ),
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def search(self, query: ForensicQuery) -> tuple[ForensicRecord, ...]:
        self._require_open()
        if not isinstance(query, ForensicQuery):
            raise ValueError("query must be ForensicQuery")

        clauses: list[str] = []
        params: list[object] = []

        if query.start_ms is not None:
            clauses.append("timestamp_ms >= ?")
            params.append(query.start_ms)
        if query.end_ms is not None:
            clauses.append("timestamp_ms <= ?")
            params.append(query.end_ms)

        for column, values in (
            ("source_id", query.source_ids),
            ("category", query.categories),
            ("track_id", query.track_ids),
        ):
            if values:
                placeholders = ",".join("?" for _ in values)
                clauses.append(f"{column} IN ({placeholders})")
                params.extend(values)

        clauses.append("confidence >= ?")
        params.append(query.min_confidence)

        if query.plate_text is not None:
            clauses.append("plate_text = ?")
            params.append(query.plate_text)
        if query.plate_prefix is not None:
            clauses.append("substr(plate_text, 1, length(?)) = ?")
            params.extend((query.plate_prefix, query.plate_prefix))

        for name, value in sorted({(item.name, item.value) for item in query.attributes}):
            clauses.append(
                """
                EXISTS (
                    SELECT 1 FROM attributes a
                    WHERE a.observation_id = records.observation_id
                      AND a.name = ? AND a.value = ?
                )
                """
            )
            params.extend((name, value))

        where = " AND ".join(clauses) if clauses else "1"
        sql = f"""
            SELECT observation_id
            FROM records
            WHERE {where}
            ORDER BY timestamp_ms, source_id, observation_id
            LIMIT ?
        """
        params.append(query.limit)
        rows = tuple(self._connection.execute(sql, tuple(params)))
        return tuple(self._read_record(str(row[0])) for row in rows)
