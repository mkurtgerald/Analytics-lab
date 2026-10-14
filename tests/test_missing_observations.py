"""Generated regressions for missing-frame evidence in the real core pipeline."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analytics_lab.openvino_omz import OpenVINOOMZConfig
from analytics_lab.openvino_pipeline import OpenVINOOMZPipelineConfig, run_frame_source_openvino_omz
from analytics_lab.temporal import Config, Observation, PersonDownEngine
from analytics_lab.video import Frame, PerceptionObservation, run_frame_source


class _GeneratedRuntime:
    runtime_version = "2026.3.1-generated"

    def __init__(self, weak_score, weak_times, person_label):
        self.weak_score = weak_score
        self.weak_times = weak_times
        self.person_label = person_label
        self.maps = [[[0.0] * 4 for _ in range(4)] for _ in range(19)]
        for channel, y, x in ((2, 1, 1), (5, 2, 1), (8, 1, 3), (11, 2, 3)):
            self.maps[channel][y][x] = .95

    def detect_rows(self, image):
        score = self.weak_score if image.timestamp in self.weak_times else .9
        return ((0, self.person_label, score, .1, .1, .9, .5),)

    def pose_heatmaps(self, image, bbox):
        return self.maps


class _MissingRecorder(PersonDownEngine):
    def __init__(self, config=None):
        super().__init__("generated-camera", "generated-session", config)
        self.nonempty_missing = []

    def mark_missing(self, timestamp_ms, track_ids):
        ids = tuple(track_ids)
        if ids:
            self.nonempty_missing.append((timestamp_ms, ids))
        return super().mark_missing(timestamp_ms, ids)


class MissingObservationTests(unittest.TestCase):
    @staticmethod
    def _frames(count=13, step=250):
        return (
            Frame(index, index * step, SimpleNamespace(shape=(200, 400, 3), timestamp=index * step))
            for index in range(count)
        )

    def _omz(self, weak_score=.49, *, weak_times=(1250,), temporal=None, model="person-detection-retail-0013"):
        pipeline = OpenVINOOMZPipelineConfig(
            openvino=OpenVINOOMZConfig(detector_model=model), temporal=temporal or Config(),
        )
        runtime = _GeneratedRuntime(weak_score, weak_times, 0 if model == "person-detection-0200" else 1)
        with patch("analytics_lab.openvino_omz.verify_artifact_set", return_value=()):
            return run_frame_source_openvino_omz(
                self._frames(), artifact_root="generated-artifacts", source_id="generated-camera",
                session_id="generated-session", config=pipeline, runtime_factory=lambda *_: runtime,
            ).video

    def test_filtering_weaker_detection_cannot_promote_rejected_episode(self):
        for model in ("person-detection-retail-0013", "person-detection-0200"):
            with self.subTest(model=model):
                accepted_weak = self._omz(.60, model=model)
                filtered_weaker = self._omz(.49, model=model)
                self.assertEqual(accepted_weak.events, ())
                self.assertEqual(filtered_weaker.events, ())
                self.assertEqual(accepted_weak.observations_processed, 13)
                self.assertEqual(filtered_weaker.observations_processed, 12)

    def test_uninterrupted_confident_positive_is_preserved(self):
        result = self._omz(weak_times=())
        self.assertEqual(result.observations_processed, 13)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0]["confidence"], .9)
        window = result.events[0]["evidence"][0]
        self.assertEqual((window["start_timestamp_ms"], window["end_timestamp_ms"]), (0, 3000))

    def test_first_absent_frame_of_next_chunk_cannot_bypass_unknown_policy(self):
        frames = tuple(self._frames())
        engine = PersonDownEngine("generated-camera", "generated-session")

        def perception(image, index, timestamp):
            return () if timestamp == 1250 else (PerceptionObservation("person-a", "down", .9),)

        first = run_frame_source(frames[:5], perception, engine)
        self.assertEqual(engine.last_observed_track_ids, ("person-a",))
        missing = run_frame_source(frames[5:6], perception, engine)
        self.assertEqual(engine.last_observed_track_ids, ())
        last = run_frame_source(frames[6:], perception, engine)
        self.assertEqual(first.events + missing.events + last.events, ())
        self.assertEqual(first.observations_processed + missing.observations_processed + last.observations_processed, 12)

    def test_explicit_unknown_bridge_still_allows_short_detection_absence(self):
        result = self._omz(temporal=Config(max_unknown_gap_ms=750))
        self.assertEqual(result.observations_processed, 12)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0]["evidence"][0]["track_id"], "track-000001")

    def test_unknown_bridge_uses_last_down_inclusive_boundary(self):
        # Two absent frames retain the IoU handle. Last down=1000, next=1750.
        for bound, expected in ((750, 1), (500, 0)):
            with self.subTest(bound=bound):
                result = self._omz(weak_times=(1250, 1500), temporal=Config(max_unknown_gap_ms=bound))
                self.assertEqual(result.observations_processed, 11)
                self.assertEqual(len(result.events), expected)

    def test_missing_one_person_does_not_interrupt_other_person(self):
        engine = _MissingRecorder()

        def perception(image, index, timestamp):
            return tuple(
                PerceptionObservation(track, "down", .9)
                for track in ("person-a", "person-b")
                if track != "person-a" or timestamp != 1250
            )

        result = run_frame_source(self._frames(), perception, engine)
        self.assertEqual(result.observations_processed, 25)
        self.assertEqual([event["evidence"][0]["track_id"] for event in result.events], ["person-b"])
        self.assertEqual(engine.nonempty_missing, [(1250, ("person-a",))])

    def test_absence_marked_once_until_track_is_actually_seen_again(self):
        engine = _MissingRecorder()

        def perception(image, index, timestamp):
            return () if timestamp in (500, 750, 1000, 1500, 1750) else (PerceptionObservation("person-a", "down", .9),)

        result = run_frame_source(self._frames(count=8), perception, engine)
        self.assertEqual(result.observations_processed, 3)
        self.assertEqual(engine.nonempty_missing, [(500, ("person-a",)), (1500, ("person-a",))])

    def test_expired_previous_id_cannot_create_ghost_or_block_new_track_at_capacity(self):
        engine = PersonDownEngine("generated-camera", "generated-session", Config(max_tracks=1))
        frames = (Frame(0, 0, object()), Frame(1, 31001, object()))
        result = run_frame_source(
            frames, lambda image, index, timestamp: (PerceptionObservation("old" if index == 0 else "new", "down", .9),),
            engine,
        )
        self.assertEqual(result.observations_processed, 2)
        self.assertEqual(result.events, ())
        self.assertEqual(engine.active_tracks, 1)
        self.assertEqual(set(engine._tracks), {"new"})

    def test_long_empty_stream_expires_old_tracks_without_synthetic_observations(self):
        engine = _MissingRecorder(Config(max_tracks=1, track_ttl_ms=1500))
        result = run_frame_source(
            self._frames(count=2000),
            lambda image, index, timestamp: () if index else (PerceptionObservation("person-a", "down", .9),),
            engine,
        )
        self.assertEqual(result.observations_processed, 1)
        self.assertEqual(result.events, ())
        self.assertEqual(engine.nonempty_missing, [(250, ("person-a",))])
        self.assertEqual(engine.active_tracks, 0)

    def test_missing_markers_do_not_extend_ttl_under_many_reacquisitions(self):
        engine = PersonDownEngine("generated-camera", "generated-session", Config(max_tracks=2, track_ttl_ms=1500))
        peak_tracks = 0

        def perception(image, index, timestamp):
            nonlocal peak_tracks
            peak_tracks = max(peak_tracks, engine.active_tracks)
            return () if index % 2 else (PerceptionObservation(f"person-{index}", "down", .9),)

        result = run_frame_source(self._frames(count=2000, step=500), perception, engine)
        self.assertEqual(result.observations_processed, 1000)
        self.assertEqual(result.events, ())
        self.assertLessEqual(max(peak_tracks, engine.active_tracks), 2)

    def test_engine_missing_unknown_or_expired_ids_never_creates_tracks(self):
        engine = PersonDownEngine("generated-camera", "generated-session", Config(max_tracks=1))
        engine.mark_missing(0, ("never-seen",))
        self.assertEqual(engine.active_tracks, 0)
        engine.observe(Observation(0, "old", "down", .9))
        engine.mark_missing(31001, ("old",))
        self.assertEqual(engine.active_tracks, 0)
        engine.observe(Observation(31001, "new", "down", .9))
        self.assertEqual(engine.active_tracks, 1)

    def test_engine_missing_evidence_preserves_actual_last_seen_for_ttl(self):
        engine = PersonDownEngine("generated-camera", "generated-session", Config(track_ttl_ms=1500))
        engine.observe(Observation(0, "person-a", "down", .9))
        engine.mark_missing(500, ("person-a",))
        engine.mark_missing(1400, ("person-a",))
        engine.mark_missing(1501, ())
        self.assertEqual(engine.active_tracks, 0)

    def test_engine_missing_rejects_invalid_or_nonmonotonic_inputs_without_mutation(self):
        engine = PersonDownEngine("generated-camera", "generated-session", Config(max_tracks=2))
        engine.observe(Observation(100, "person-a", "down", .9))
        before = (deepcopy(engine._tracks), engine._watermark)
        invalid = ((99, ()), (-1, ()), (True, ()), (253402300800000, ()),
                   (200, "person-a"), (200, None), (200, ("https://invalid",)),
                   (200, ("person-a", "person-a")))
        for timestamp, ids in invalid:
            with self.subTest(timestamp=timestamp, ids=ids), self.assertRaises(ValueError):
                engine.mark_missing(timestamp, ids)
            self.assertEqual((engine._tracks, engine._watermark), before)

    def test_engine_missing_input_consumption_is_bounded_before_mutation(self):
        engine = PersonDownEngine("generated-camera", "generated-session", Config(max_tracks=2))
        engine.observe(Observation(0, "person-a", "down", .9))
        before = (deepcopy(engine._tracks), engine._watermark)
        consumed = []

        def endless():
            index = 0
            while True:
                consumed.append(index)
                yield f"unknown-{index}"
                index += 1

        with self.assertRaises(RuntimeError):
            engine.mark_missing(100, endless())
        self.assertEqual(len(consumed), 3)
        self.assertEqual((engine._tracks, engine._watermark), before)

    def test_engine_missing_duplicate_evidence_timestamp_is_rejected(self):
        engine = PersonDownEngine("generated-camera", "generated-session")
        engine.observe(Observation(0, "person-a", "down", .9))
        engine.mark_missing(250, ("person-a",))
        for operation in (
            lambda: engine.mark_missing(250, ("person-a",)),
            lambda: engine.observe(Observation(250, "person-a", "down", .9)),
        ):
            with self.assertRaises(ValueError):
                operation()
        engine.observe(Observation(500, "person-a", "down", .9))


if __name__ == "__main__":
    unittest.main()
