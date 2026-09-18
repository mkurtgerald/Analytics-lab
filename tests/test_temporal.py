import json
import unittest

from analytics_lab.temporal import Config, Observation, PersonDownEngine


class TemporalTests(unittest.TestCase):
    def engine(self, **kwargs):
        return PersonDownEngine("synthetic-camera", "session-a", Config(**kwargs))

    def feed(self, engine, times, posture="down", confidence=0.9, track="track-1"):
        events = []
        for timestamp in times:
            events.extend(engine.observe(Observation(timestamp, track, posture, confidence)))
        return events

    def test_persistent_down_is_candidate_not_injury(self):
        events = self.feed(self.engine(), range(0, 3500, 500))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "candidate")
        self.assertEqual(events[0]["event_type"], "person_down_candidate")
        self.assertIn("injury", events[0]["metadata"]["inferences_not_supported"])

    def test_short_run_does_not_trigger(self):
        self.assertEqual(self.feed(self.engine(), range(0, 3000, 500)), [])

    def test_upright_does_not_trigger(self):
        self.assertEqual(self.feed(self.engine(), range(0, 10000, 500), "upright"), [])

    def test_other_posture_does_not_trigger(self):
        self.assertEqual(self.feed(self.engine(), range(0, 10000, 500), "other"), [])

    def test_unknown_interrupts_continuity(self):
        engine = self.engine()
        self.feed(engine, range(0, 3000, 500))
        self.feed(engine, [3000], "unknown")
        self.assertEqual(self.feed(engine, range(3500, 6500, 500)), [])
        self.assertEqual(len(self.feed(engine, [6500])), 1)

    def test_bounded_unknown_gap_can_bridge_when_explicitly_enabled(self):
        engine = self.engine(max_unknown_gap_ms=750)
        self.feed(engine, [0, 500, 1000, 1500])
        self.feed(engine, [2000], "unknown")
        self.assertEqual(self.feed(engine, [2250]), [])
        event = self.feed(engine, [3000])
        self.assertEqual(len(event), 1)
        self.assertEqual(event[0]["metadata"]["thresholds"]["max_unknown_gap_ms"], 750)
        self.assertEqual(event[0]["observations"][0]["sample_count"], 6)

    def test_unknown_gap_beyond_bound_resets_even_with_intermediate_unknowns(self):
        engine = self.engine(down_duration_ms=1000, max_gap_ms=750, max_unknown_gap_ms=750)
        self.feed(engine, [0, 250, 500])
        self.feed(engine, [750], "unknown")
        self.feed(engine, [1000], "unknown")
        self.feed(engine, [1250], "unknown")
        self.assertEqual(self.feed(engine, [1500]), [])
        self.assertEqual(self.feed(engine, [1750, 2000]), [])
        self.assertEqual(len(self.feed(engine, [2500])), 1)

    def test_upright_still_breaks_unknown_tolerant_run(self):
        engine = self.engine(down_duration_ms=1000, max_unknown_gap_ms=750)
        self.feed(engine, [0, 250, 500])
        self.feed(engine, [750], "unknown")
        self.feed(engine, [1000], "upright")
        self.assertEqual(self.feed(engine, [1250, 1500, 1750, 2000]), [])
        self.assertEqual(len(self.feed(engine, [2250])), 1)

    def test_low_confidence_interrupts_continuity(self):
        engine = self.engine()
        self.feed(engine, range(0, 3000, 500))
        self.feed(engine, [3000], confidence=0.2)
        self.assertEqual(self.feed(engine, range(3500, 6500, 500)), [])

    def test_gap_resets_run(self):
        engine = self.engine()
        self.feed(engine, range(0, 3000, 500))
        self.assertEqual(self.feed(engine, [4000]), [])

    def test_max_gap_boundary_is_inclusive(self):
        self.assertEqual(len(self.feed(self.engine(), [0, 750, 1500, 2250, 3000])), 1)

    def test_gap_above_boundary_resets(self):
        self.assertEqual(self.feed(self.engine(), [0, 751, 1502, 2253, 3004]), [])

    def test_duration_and_sample_count_are_both_required(self):
        self.assertEqual(self.feed(self.engine(min_samples=10), range(0, 3500, 500)), [])

    def test_one_event_per_continuous_episode(self):
        self.assertEqual(len(self.feed(self.engine(), range(0, 20000, 500))), 1)

    def test_new_episode_can_trigger_new_id(self):
        engine = self.engine()
        first = self.feed(engine, range(0, 3500, 500))[0]
        self.feed(engine, [3500], "upright")
        second = self.feed(engine, range(4000, 7500, 500))[0]
        self.assertNotEqual(first["event_id"], second["event_id"])

    def test_tracks_do_not_share_duration(self):
        engine = self.engine()
        self.feed(engine, range(0, 3000, 500))
        self.assertEqual(self.feed(engine, [3000], track="track-2"), [])

    def test_same_frame_different_tracks_is_allowed(self):
        engine = self.engine()
        self.feed(engine, [0], track="a")
        self.feed(engine, [0], track="b")
        self.assertEqual(engine.active_tracks, 2)

    def test_duplicate_track_timestamp_is_rejected(self):
        engine = self.engine()
        self.feed(engine, [0])
        with self.assertRaises(ValueError):
            self.feed(engine, [0])

    def test_out_of_order_source_is_rejected(self):
        engine = self.engine()
        self.feed(engine, [1000])
        with self.assertRaises(ValueError):
            self.feed(engine, [500], track="other")

    def test_capacity_is_bounded(self):
        engine = self.engine(max_tracks=1)
        self.feed(engine, [0])
        with self.assertRaises(RuntimeError):
            self.feed(engine, [0], track="another")
        self.assertEqual(engine.active_tracks, 1)

    def test_expiry_allows_new_track_but_not_carried_duration(self):
        engine = self.engine(max_tracks=1)
        self.feed(engine, [0])
        self.assertEqual(self.feed(engine, [31000], track="another"), [])
        self.assertEqual(engine.active_tracks, 1)

    def test_deterministic_replay(self):
        a = self.feed(self.engine(), range(0, 3500, 500))
        b = self.feed(self.engine(), range(0, 3500, 500))
        self.assertEqual(a, b)

    def test_sessions_do_not_collide(self):
        a = self.feed(self.engine(), range(0, 3500, 500))[0]
        b = self.feed(PersonDownEngine("synthetic-camera", "session-b"), range(0, 3500, 500))[0]
        self.assertNotEqual(a["event_id"], b["event_id"])

    def test_confidence_is_minimum_not_fake_probability(self):
        engine = self.engine()
        self.feed(engine, range(0, 3000, 500))
        result = self.feed(engine, [3000], confidence=0.75)[0]
        self.assertEqual(result["confidence"], 0.75)
        self.assertIn("not_calibrated", result["metadata"]["confidence_basis"])

    def test_event_is_json_and_evidence_is_reference_only(self):
        result = self.feed(self.engine(), range(0, 3500, 500))[0]
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertEqual(result["evidence"][0]["type"], "observation_window")
        self.assertEqual(result["start_time"], "1970-01-01T00:00:00.000Z")
        self.assertEqual(result["end_time"], "1970-01-01T00:00:03.000Z")

    def test_invalid_confidences_are_rejected(self):
        for value in (float("nan"), float("inf"), -1, 1.1, True, "0.9", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Observation(0, "t", "down", value)

    def test_invalid_timestamps_are_rejected(self):
        for value in (-1, True, 1.5, 253402300800000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Observation(value, "t", "down", 0.9)

    def test_invalid_config_is_rejected(self):
        for kwargs in ({"min_samples": 1}, {"max_tracks": 0}, {"down_duration_ms": 0},
                       {"max_gap_ms": -1}, {"max_unknown_gap_ms": -1},
                       {"max_unknown_gap_ms": 751}, {"track_ttl_ms": 750},
                       {"min_confidence": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)

    def test_identifiers_cannot_be_urls_or_credentials(self):
        for source in ("rtsp://user:password@camera", "", "a" * 129, "a\nb"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                PersonDownEngine(source, "session")

    def test_unknown_posture_value_is_rejected(self):
        with self.assertRaises(ValueError):
            Observation(0, "t", "injured", 0.9)


if __name__ == "__main__":
    unittest.main()
