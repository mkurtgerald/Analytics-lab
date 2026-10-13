"""Synthetic producer/consumer contract tests; never accuracy evidence."""
import hashlib
import json
import unittest
from dataclasses import replace

from analytics_lab.observation_handoff import (
    SCHEMA, TrackingFrameContext, TrackingFrameEnvelope, parse_tracking_frame_json,
)
from analytics_lab.tracking import (
    DetectionCandidate, NormalizedBox, TrackedDetection, TrackingSession,
)


class EchoTracker:
    def update(self, frame_index, timestamp_ms, candidates):
        return (TrackedDetection(f"id-{i}", d.category, d.confidence, d.box, d.model_class_id)
                for i, d in enumerate(candidates))


def fixture():
    return TrackingFrameContext(
        event_id="event-1", source_ref="source-1", stream_ref="stream-1", session_id="session-1",
        frame_index=13, utc_timestamp_ms=1_800_000_000_000,
        producer="analytics-lab", producer_revision="rev-abc", config_sha256="a" * 64,
        model_sha256=None, source_revision="fixture-r1", media_pts=300, timebase_num=1, timebase_den=30,
    )


def real_session_frame():
    session = TrackingSession(EchoTracker())
    source = DetectionCandidate("person", .72, NormalizedBox(.1, .2, .4, .8), 7)
    return TrackingFrameEnvelope(fixture(), session.update(13, fixture().utc_timestamp_ms, (source,)))


class HandoffTests(unittest.TestCase):
    def test_actual_session_output_roundtrip_and_golden_semantics(self):
        obj = real_session_frame()
        raw = obj.to_json()
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(),
                         "a9d4af5858b6fcd0c6f643f287f4b645089021543f4854cfbd0aa805d84f8b71")
        result = parse_tracking_frame_json(raw)
        self.assertEqual(result.to_json(), raw)
        self.assertEqual(result.tracks[0].track_id, "id-0")
        self.assertEqual(result.tracks[0].box, NormalizedBox(.1, .2, .4, .8))
        payload = json.loads(raw)
        self.assertEqual(payload["schema"], SCHEMA)
        self.assertEqual(payload["frame"]["media_pts"], 300)
        self.assertEqual((payload["frame"]["timebase_num"], payload["frame"]["timebase_den"]), (1, 30))
        self.assertEqual(payload["producer"]["config_sha256"], "a" * 64)
        self.assertIsNone(payload["producer"]["model_revision"])
        self.assertEqual(payload["semantics"]["coordinates"], "normalized_xyxy")
        self.assertEqual(payload["semantics"]["confidence"], "uncalibrated_model_score")
        self.assertEqual(payload["semantics"]["track_scope"], "source_stream_session_local")
        for flag in ("identity_claim", "authorizes_action", "source_scope_authorized"):
            self.assertIs(payload["semantics"][flag], False)
        self.assertEqual(set(payload["tracks"][0]), {"track_id", "category", "confidence", "box", "model_class_id"})

    def test_empty_ok_is_distinct_from_unavailable(self):
        empty = TrackingFrameEnvelope(fixture()).to_json()
        outage = TrackingFrameEnvelope(fixture(), (), "unavailable", "detector_unavailable").to_json()
        self.assertEqual(json.loads(empty)["tracks"], [])
        self.assertEqual(json.loads(empty)["status"], "ok")
        self.assertEqual(json.loads(outage)["status"], "unavailable")
        self.assertEqual(hashlib.sha256(outage.encode()).hexdigest(),
                         "78d6dd7b5d1954230357af83ae9c06fc01990a2ff7d4ee89b3042ecbcb32bae8")
        self.assertEqual(parse_tracking_frame_json(outage).to_json(), outage)
        self.assertNotEqual(empty, outage)

    def test_unavailable_must_not_carry_tracks_or_arbitrary_error_text(self):
        track = real_session_frame().tracks[0]
        with self.assertRaises(ValueError):
            TrackingFrameEnvelope(fixture(), (track,), "unavailable", "detector_unavailable")
        with self.assertRaises(ValueError):
            TrackingFrameEnvelope(fixture(), (), "unavailable", "private camera URI")
        with self.assertRaises(ValueError):
            TrackingFrameEnvelope(fixture(), (), "unavailable")
        with self.assertRaises(ValueError):
            TrackingFrameEnvelope(fixture(), (), "ok", "detector_unavailable")

    def test_no_unknown_or_authorizing_fields(self):
        payload = json.loads(real_session_frame().to_json())
        for key, value in (("tenant_id", "self-declared"), ("identity_claim", True), ("image", "pixels")):
            with self.subTest(key=key):
                modified = dict(payload, **{key: value})
                with self.assertRaises(ValueError):
                    parse_tracking_frame_json(json.dumps(modified))
        payload["semantics"]["source_scope_authorized"] = True
        with self.assertRaises(ValueError):
            parse_tracking_frame_json(json.dumps(payload))

    def test_semantics_flags_reject_numeric_boolean_impersonation(self):
        original = json.loads(real_session_frame().to_json())
        for field in ("identity_claim", "authorizes_action", "source_scope_authorized"):
            for value in (0, 0.0, 1, 1.0):
                with self.subTest(field=field, value=value):
                    payload = json.loads(json.dumps(original))
                    payload["semantics"][field] = value
                    with self.assertRaisesRegex(ValueError, "unsupported or authorizing"):
                        parse_tracking_frame_json(json.dumps(payload))

    def test_model_revision_requires_pinned_model_identity(self):
        with self.assertRaises(ValueError):
            replace(fixture(), model_revision="oidv4-r1")
        tagged = replace(fixture(), model_revision="oidv4-r1", model_sha256="b" * 64)
        roundtrip = parse_tracking_frame_json(TrackingFrameEnvelope(tagged).to_json())
        self.assertEqual(roundtrip.context.model_revision, "oidv4-r1")
        self.assertEqual(roundtrip.context.model_sha256, "b" * 64)

    def test_unsupported_schema_is_rejected(self):
        payload = json.loads(real_session_frame().to_json())
        payload["schema"] = "analytics.tracked-frame.v2"
        with self.assertRaisesRegex(ValueError, "unsupported tracking frame schema"):
            parse_tracking_frame_json(json.dumps(payload))

    def test_cross_session_same_track_is_explicitly_separated(self):
        frame_a = real_session_frame()
        frame_b = TrackingFrameEnvelope(replace(frame_a.context, session_id="session-2", event_id="event-2"), frame_a.tracks)
        payloads = [json.loads(x.to_json()) for x in (frame_a, frame_b)]
        self.assertEqual(payloads[0]["tracks"][0]["track_id"], payloads[1]["tracks"][0]["track_id"])
        self.assertNotEqual(payloads[0]["frame"]["session_id"], payloads[1]["frame"]["session_id"])

    def test_reject_corrupt_nonfinite_or_out_of_bounds_track(self):
        for field, bad in (("confidence", float("nan")), ("confidence", 1.2), ("track_id", "")):
            with self.subTest(field=field):
                payload = json.loads(real_session_frame().to_json())
                payload["tracks"][0][field] = bad
                with self.assertRaises(ValueError):
                    parse_tracking_frame_json(json.dumps(payload))
        payload = json.loads(real_session_frame().to_json())
        payload["tracks"][0]["box"] = [-1, .2, .4, .8]
        with self.assertRaises(ValueError):
            parse_tracking_frame_json(json.dumps(payload))

    def test_frozen_dataclass_corruption_is_detected_and_results_detached(self):
        track = real_session_frame().tracks[0]
        envelope = TrackingFrameEnvelope(fixture(), (track,))
        object.__setattr__(track.box, "x_min", -.1)
        self.assertEqual(envelope.tracks[0].box.x_min, .1)
        object.__setattr__(envelope.tracks[0], "confidence", float("nan"))
        with self.assertRaises(ValueError):
            envelope.to_json()

    def test_bounded_iterator_does_not_read_length_hint_or_more_than_one_excess(self):
        class BadIterable:
            def __init__(self, track): self.track, self.reads, self.hints = track, 0, 0
            def __iter__(self): return self
            def __length_hint__(self):
                self.hints += 1
                raise AssertionError("must not trust length hint")
            def __next__(self):
                self.reads += 1
                if self.reads > 513:
                    raise AssertionError("overconsumption")
                return self.track
        stream = BadIterable(real_session_frame().tracks[0])
        with self.assertRaises(ValueError):
            TrackingFrameEnvelope(fixture(), stream)
        self.assertEqual(stream.reads, 2)  # duplicate track rejected before overflow
        self.assertEqual(stream.hints, 0)

    def test_overflow_iterator_reads_exactly_one_excess_item_without_length_hint(self):
        box = NormalizedBox(.1, .1, .2, .2)
        class UntrustedTracks:
            reads = 0
            hints = 0
            def __iter__(self): return self
            def __length_hint__(self):
                self.hints += 1
                raise AssertionError("untrusted length hint called")
            def __next__(self):
                self.reads += 1
                if self.reads > 513:
                    raise AssertionError("iterated past excess witness")
                return TrackedDetection(f"unique-{self.reads}", "person", .5, box)
        source = UntrustedTracks()
        with self.assertRaisesRegex(ValueError, "track bound"):
            TrackingFrameEnvelope(fixture(), source)
        self.assertEqual(source.reads, 513)
        self.assertEqual(source.hints, 0)

    def test_exact_upper_bound_and_overflow(self):
        base = real_session_frame().tracks[0]
        data = tuple(TrackedDetection(f"t{i}", "person", .8, base.box) for i in range(512))
        self.assertEqual(len(parse_tracking_frame_json(TrackingFrameEnvelope(fixture(), data).to_json()).tracks), 512)
        with self.assertRaisesRegex(ValueError, "track bound"):
            TrackingFrameEnvelope(fixture(), (TrackedDetection(f"x{i}", "person", .8, base.box) for i in range(513)))

    def test_serialized_output_size_limit_enforced(self):
        big_category = "🧪" * 64
        box = NormalizedBox(.1, .1, .2, .2)
        batch = tuple(TrackedDetection(f"t{i}", big_category, .5, box) for i in range(512))
        with self.assertRaisesRegex(ValueError, "size bound"):
            TrackingFrameEnvelope(fixture(), batch).to_json()

    def test_duplicate_track_ids_fail(self):
        t = real_session_frame().tracks[0]
        with self.assertRaisesRegex(ValueError, "duplicate track_id"):
            TrackingFrameEnvelope(fixture(), (t, t))

    def test_duplicate_json_keys_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
            parse_tracking_frame_json(real_session_frame().to_json().replace('"schema":', '"schema":"bad","schema":'))

    def test_timestamp_and_pts_types_fail_closed(self):
        for values in ({"frame_index": True}, {"utc_timestamp_ms": -1},
                       {"media_pts": 1, "timebase_num": None},
                       {"media_pts": None, "timebase_num": 1},
                       {"timebase_den": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrackingFrameContext(**{**vars(fixture()), **values})

    def test_source_ref_cannot_carry_uri_credentials(self):
        with self.assertRaises(ValueError):
            replace(fixture(), source_ref="rtsp://camera.invalid/video")
        with self.assertRaises(ValueError):
            replace(fixture(), stream_ref="my private path")

    def test_reject_json_nan_infinity_and_oversized_message(self):
        base = real_session_frame().to_json()
        with self.assertRaises(ValueError):
            parse_tracking_frame_json(base.replace('"confidence":0.72', '"confidence":NaN'))
        with self.assertRaises(ValueError):
            parse_tracking_frame_json(base + " " * 262_144)

    def test_replay_same_event_id_stable_canonical_serialization(self):
        frame = real_session_frame()
        self.assertEqual(frame.to_json(), frame.to_json())
        self.assertEqual(frame.to_json(), parse_tracking_frame_json(frame.to_json()).to_json())
        self.assertEqual(frame.context.event_id, "event-1")


if __name__ == "__main__":
    unittest.main()
