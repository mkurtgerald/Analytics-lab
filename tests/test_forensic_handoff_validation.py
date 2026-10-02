"""Generated metadata-only mutations at the serialized consumer boundary."""
import copy
import json
import unittest

from analytics_lab.forensic_appearance import AppearanceMatch
from analytics_lab.forensic_handoff import (
    ForensicResultEnvelope,
    envelope_for_appearance_matches,
    envelope_for_hit,
    envelope_for_trail,
    parse_envelope_json,
)
from analytics_lab.forensic_results import (
    operator_hit_from_record,
    operator_result_from_attribute_trail,
    operator_result_from_plate_trail,
)
from analytics_lab.forensic_search import (
    AttributeFilter, ForensicAttribute, ForensicEvidenceLink, ForensicRecord,
)
from analytics_lab.forensic_temporal import AttributeCandidateTrail, TemporalTrail


def generated_envelopes():
    def record(index, plate=False):
        return ForensicRecord(
            observation_id=f"synthetic-observation-{index}",
            source_id=f"synthetic-camera-{index}", timestamp_ms=1000 + index,
            category="license_plate" if plate else "person", confidence=0.875,
            track_id=f"session-local-{index}", plate_text="SYN123" if plate else None,
            attributes=() if plate else (ForensicAttribute("upper_color", "blue", .9, "synthetic-v1"),),
            evidence=ForensicEvidenceLink(f"synthetic-event-{index}", "synthetic-producer", "1.0", "a" * 64),
        )

    a, b = record(1), record(2)
    yield envelope_for_hit(operator_hit_from_record(a))
    yield envelope_for_trail(operator_result_from_attribute_trail(AttributeCandidateTrail(
        "person", (AttributeFilter("upper_color", "blue"),), (a, b))))
    yield envelope_for_trail(operator_result_from_plate_trail(TemporalTrail(
        "exact_plate_text", "SYN123", (record(1, True), record(2, True)))))
    yield envelope_for_appearance_matches((AppearanceMatch("synthetic-probe", a, .95),
                                           AppearanceMatch("synthetic-probe", b, .90)))


def objects(value, path=()):
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from objects(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from objects(item, (*path, index))


def at_path(value, path):
    for key in path:
        value = value[key]
    return value


class ForensicHandoffValidationTests(unittest.TestCase):
    def assert_rejected(self, value):
        with self.assertRaises(ValueError):
            parse_envelope_json(json.dumps(value))

    def test_generated_producer_payloads_round_trip_without_change(self):
        for envelope in generated_envelopes():
            with self.subTest(kind=envelope.payload["kind"]):
                raw = envelope.to_json()
                self.assertEqual(parse_envelope_json(raw).to_json(), raw)

    def test_every_nested_authority_and_identity_flag_fails_closed(self):
        for envelope in generated_envelopes():
            raw = json.loads(envelope.to_json())
            for path, obj in objects(raw):
                for flag in ("identity_claim", "authorizes_action"):
                    if flag not in obj:
                        continue
                    with self.subTest(kind=envelope.payload["kind"], path=path, flag=flag):
                        changed = copy.deepcopy(raw)
                        at_path(changed, path)[flag] = True
                        self.assert_rejected(changed)

    def test_unversioned_fields_fail_closed_at_every_object(self):
        for envelope in generated_envelopes():
            raw = json.loads(envelope.to_json())
            for path, _ in objects(raw):
                with self.subTest(kind=envelope.payload["kind"], path=path):
                    changed = copy.deepcopy(raw)
                    at_path(changed, path)["unversioned_extension"] = "synthetic-value"
                    self.assert_rejected(changed)

    def test_playback_must_bind_same_source_timestamp_and_evidence(self):
        raw = json.loads(next(generated_envelopes()).to_json())
        for field, value in (("source_id", "different-camera"), ("timestamp_ms", 9000),
                             ("evidence", {**raw["payload"]["evidence"], "event_id": "different-event"})):
            with self.subTest(field=field):
                changed = copy.deepcopy(raw)
                changed["payload"]["playback"][field] = value
                self.assert_rejected(changed)

    def test_typed_hit_constraints_survive_serialization(self):
        raw = json.loads(next(generated_envelopes()).to_json())
        for field, value in (("timestamp_ms", True), ("timestamp_ms", -1),
                             ("confidence", float("nan")), ("confidence", 1.1),
                             ("confidence", 10 ** 400), ("source_id", ""),
                             ("category", " person "), ("attributes", {})):
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(raw)
                changed["payload"][field] = value
                self.assert_rejected(changed)

    def test_kind_is_required_and_closed(self):
        for kind in (None, "future-result", [], {}):
            raw = json.loads(next(generated_envelopes()).to_json())
            raw["payload"]["kind"] = kind
            self.assert_rejected(raw)
        self.assert_rejected({"schema": "analytics.forensic-result.v1",
                              "payload": {"identity_claim": False, "authorizes_action": False}})

    def test_trail_filter_category_and_order_survive_serialization(self):
        raw = json.loads(list(generated_envelopes())[1].to_json())
        for path, value in (
            (("payload", "hits", 0, "attributes", 0, "value"), "red"),
            (("payload", "hits", 0, "category"), "vehicle"),
            (("payload", "hits"), list(reversed(raw["payload"]["hits"]))),
        ):
            with self.subTest(path=path):
                changed = copy.deepcopy(raw)
                at_path(changed, path[:-1])[path[-1]] = value
                self.assert_rejected(changed)

    def test_appearance_binding_order_and_bounds_survive_serialization(self):
        raw = json.loads(list(generated_envelopes())[-1].to_json())
        changes = (
            ("descriptor_schema", "future"), ("descriptor_kind", "unreviewed"),
            ("probe_observation_id", ""), ("matches", []),
            ("matches", list(reversed(raw["payload"]["matches"]))),
            ("matches", [raw["payload"]["matches"][0]] * 2),
            ("matches", raw["payload"]["matches"] * 501),
        )
        for field, value in changes:
            with self.subTest(field=field):
                changed = copy.deepcopy(raw)
                changed["payload"][field] = value
                self.assert_rejected(changed)

    def test_serialization_revalidates_mutated_payload(self):
        envelope = next(generated_envelopes())
        envelope.payload["playback"]["authorizes_action"] = True
        with self.assertRaises(ValueError):
            envelope.to_json()

    def test_direct_construction_uses_same_validation(self):
        payload = copy.deepcopy(next(generated_envelopes()).payload)
        payload["playback"]["source_id"] = "different-camera"
        with self.assertRaises(ValueError):
            ForensicResultEnvelope("analytics.forensic-result.v1", payload)

    def test_duplicate_json_fields_fail_closed(self):
        raw = next(generated_envelopes()).to_json()
        for original, replacement in (
            ('"schema":', '"schema":"future","schema":'),
            ('"authorizes_action":false', '"authorizes_action":true,"authorizes_action":false'),
        ):
            with self.subTest(original=original):
                with self.assertRaises(ValueError):
                    parse_envelope_json(raw.replace(original, replacement, 1))

    def test_malformed_depth_and_unicode_raise_validation_error(self):
        for raw in ("[" * 1500 + "]" * 1500, "\ud800"):
            with self.subTest(raw_length=len(raw)):
                with self.assertRaises(ValueError):
                    parse_envelope_json(raw)

    def test_serializer_uses_same_byte_bound_as_parser(self):
        evidence = ForensicEvidenceLink("event-" + "a" * 120, "p" * 128, "v" * 128,
                                        "a" * 64, "b" * 64, "r" * 128)
        matches = tuple(AppearanceMatch("probe", ForensicRecord(
            f"observation-{i:04}", "synthetic-camera", i, "person", .9,
            evidence=evidence), .9) for i in range(1000))
        envelope = envelope_for_appearance_matches(matches)
        with self.assertRaisesRegex(ValueError, "too large"):
            envelope.to_json()
        raw = envelope_for_appearance_matches(matches[:100]).to_json()
        self.assertEqual(parse_envelope_json(raw).to_json(), raw)


if __name__ == "__main__":
    unittest.main()
