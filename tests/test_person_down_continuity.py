"""Generated metadata regressions for the actual retained scan orchestrators."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from analytics_lab import person_down_e2e_diagnostics as e2e
from analytics_lab import person_down_orientation_diagnostics as orientation
from analytics_lab.detector_continuity import DetectionBox
from analytics_lab.evaluation import LabeledPersonDown
from analytics_lab.openpose_pose_geometry import AssociatedReferencePose
from analytics_lab.perception import Keypoint, PoseCandidate
from analytics_lab.temporal import Config, PersonDownEngine
from analytics_lab.validation import ValidationSampleSpec
from analytics_lab.video import Frame


class PersonDownContinuityTests(unittest.TestCase):
    modules = (e2e, orientation)

    def _scan(self, module, rows, *, unknown_times=(), labels=(), declared_start=None, declared_end=None):
        config = Config(min_confidence=.1, max_unknown_gap_ms=750, max_tracks=1)
        events = []
        live_engines = weakref.WeakSet()
        counts = {"engines": 0, "max_live_engines": 0, "max_active_tracks": 0}
        current_time = None

        class RecordingEngine(PersonDownEngine):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                counts["engines"] += 1
                live_engines.add(self)
                counts["max_live_engines"] = max(counts["max_live_engines"], len(live_engines))

            def observe(self, observation):
                emitted = super().observe(observation)
                events.extend(emitted)
                counts["max_active_tracks"] = max(counts["max_active_tracks"], self.active_tracks)
                return emitted

        class Frames:
            def __enter__(self):
                return iter(
                    Frame(index, timestamp, SimpleNamespace(shape=(100, 100, 3), timestamp=timestamp, x=x))
                    for index, (timestamp, x) in enumerate(rows)
                )

            def __exit__(self, *args):
                return False

        class Runtime:
            def infer(self, image):
                return (), (), 100, 100

        def detections(detector, image):
            nonlocal current_time
            current_time = image.timestamp
            if image.x is None:
                return ()
            return (DetectionBox(.9, image.x, .1, image.x + .2, .2),)

        def associated(poses, scores, **kwargs):
            if current_time in unknown_times:
                return None
            box = kwargs["selection_bbox"]
            points = (
                Keypoint("left_shoulder", box.x1 + 2, box.y1 + 4, .9),
                Keypoint("right_shoulder", box.x1 + 2, box.y1 + 6, .9),
                Keypoint("left_hip", box.x1 + 18, box.y1 + 4, .9),
                Keypoint("right_hip", box.x1 + 18, box.y1 + 6, .9),
            )
            return AssociatedReferencePose(PoseCandidate(box, points, .9), .9, 4, 4, 4, 4, 1.0)

        # Rotation metadata only: no array, decoder, inference or media dependency.
        detector = SimpleNamespace(np=SimpleNamespace(
            rot90=lambda image, **kwargs: image,
            ascontiguousarray=lambda image: image,
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated-metadata.bin"
            payload = b"generated metadata fixture; no video or pixels"
            path.write_bytes(payload)
            sample = ValidationSampleSpec(
                "generated-continuity", "generated-site", "generated-camera", "synthetic-fixture",
                path, hashlib.sha256(payload).hexdigest(),
                rows[0][0] if declared_start is None else declared_start,
                rows[-1][0] if declared_end is None else declared_end,
                tuple(labels),
            )
            with patch.object(module, "OpenCVVideoFileSource", return_value=Frames()), \
                    patch.object(module, "_person_detections", side_effect=detections), \
                    patch.object(module, "select_reference_pose", side_effect=associated), \
                    patch.object(module, "PersonDownEngine", RecordingEngine), \
                    patch.object(module, "_measured_temporal_config", return_value=config):
                args = (sample, detector, Runtime(), lambda *_: ((), ()))
                if module is e2e:
                    report, evaluated = module._scan_sample(*args, config)
                else:
                    report, evaluated = module._scan_sample(*args)
        return report, events, counts, evaluated

    def test_disconnected_short_episodes_cannot_share_duration(self):
        rows = [(t, 0.0 if t <= 1000 else .75) for t in range(0, 3001, 500)]
        for module in self.modules:
            with self.subTest(module=module.__name__):
                report, events, _, _ = self._scan(module, rows)
                self.assertEqual(events, [])
                self.assertEqual(report["evaluation"]["false_alerts"], 0)
                self.assertEqual(report["continuity"]["resets"], 1)
                self.assertEqual(report["temporal_trace"]["longest_qualified_down_run_ms"], 1500)
                self.assertEqual(report["temporal_trace"]["longest_raw_down_run_ms"], 1500)
                self.assertEqual(report["temporal_trace"]["reset_reason_counts"], {"detection_continuity_lost": 1})

    def test_empty_detection_then_reacquisition_cannot_bridge(self):
        # One missing detection between accepted samples only 500ms apart.
        rows = [(t, None if t == 1250 else (0.0 if t <= 1000 else .75)) for t in range(0, 3001, 250)]
        for module in self.modules:
            with self.subTest(module=module.__name__):
                report, events, _, _ = self._scan(module, rows)
                self.assertEqual(events, [])
                self.assertEqual(report["continuity"]["resets"], 0)  # Preserve the existing statistics.
                self.assertEqual(report["temporal_trace"]["bridged_unknown_frames"], 0)
                self.assertEqual(report["temporal_trace"]["reset_reason_counts"], {"detection_continuity_lost": 1})

    def test_continuous_detection_preserves_positive(self):
        rows = [(t, 0.0) for t in range(0, 3001, 500)]
        for module in self.modules:
            with self.subTest(module=module.__name__):
                report, events, counts, _ = self._scan(module, rows, labels=(LabeledPersonDown(0, 3000, "generated-down"),))
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["evidence"][0]["start_timestamp_ms"], 0)
                self.assertEqual(report["evaluation"]["matched_events"], 1)
                self.assertEqual(report["temporal_trace"]["reset_reason_counts"], {})
                self.assertEqual(counts["engines"], 1)

    def test_pose_unknown_with_linked_detection_preserves_configured_bridge(self):
        rows = [(t, 0.0) for t in range(0, 3001, 250)]
        for module in self.modules:
            with self.subTest(module=module.__name__):
                report, events, counts, _ = self._scan(module, rows, unknown_times=(1250, 1500))
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["evidence"][0]["start_timestamp_ms"], 0)
                self.assertEqual(report["temporal_trace"]["bridged_unknown_frames"], 2)
                self.assertEqual(report["temporal_trace"]["reset_reason_counts"], {})
                self.assertEqual(counts["engines"], 1)

    def test_reacquired_full_episode_has_fresh_start_session_and_trace(self):
        rows = [(t, 0.0 if t <= 1000 else .75) for t in range(0, 4501, 500)]
        for module in self.modules:
            with self.subTest(module=module.__name__):
                report, events, counts, _ = self._scan(module, rows, labels=(LabeledPersonDown(1500, 4500, "generated-new-down"),))
                self.assertEqual(len(events), 1)
                window = events[0]["evidence"][0]
                self.assertEqual((window["start_timestamp_ms"], window["end_timestamp_ms"]), (1500, 4500))
                self.assertTrue(window["session_id"].endswith("-2"))
                self.assertEqual(report["evaluation"]["matched_events"], 1)
                self.assertEqual(report["temporal_trace"]["longest_qualified_down_run_ms"], 3000)
                self.assertEqual(report["temporal_trace"]["longest_raw_down_run_ms"], 3000)
                self.assertEqual(report["temporal_trace"]["qualified_down_frames"], len(rows))
                self.assertEqual(counts["engines"], 2)

    def test_frequent_reacquisition_keeps_only_one_engine_track(self):
        rows = [(t, 0.0 if t % 2 else .75) for t in range(1100)]
        for module in self.modules:
            with self.subTest(module=module.__name__):
                report, events, counts, _ = self._scan(module, rows)
                self.assertEqual(events, [])
                self.assertEqual(counts["engines"], len(rows))
                self.assertLessEqual(counts["max_live_engines"], 1)
                self.assertEqual(counts["max_active_tracks"], 1)
                self.assertEqual(report["temporal_trace"]["longest_qualified_down_run_ms"], 0)

    def test_decoded_span_cannot_expand_to_declared_one_hour(self):
        rows = [(0, 0.0), (500, 0.0), (1000, 0.0)]
        for module in self.modules:
            with self.subTest(module=module.__name__):
                report, events, _, evaluated = self._scan(module, rows, declared_end=3_600_000)
                self.assertEqual(events, [])
                self.assertEqual(report["decoded_start_timestamp_ms"], 0)
                self.assertEqual(report["decoded_end_timestamp_ms"], 1000)
                self.assertEqual(report["decoded_duration_ms"], 1000)
                self.assertEqual(report["evaluation"]["duration_ms"], 1000)
                self.assertEqual((evaluated.start_timestamp_ms, evaluated.end_timestamp_ms), (0, 1000))

    def test_false_alert_rate_uses_decoded_camera_time(self):
        rows = [(t, 0.0) for t in range(0, 3001, 500)]
        for module in self.modules:
            with self.subTest(module=module.__name__):
                report, events, _, evaluated = self._scan(module, rows, declared_end=3_600_000)
                self.assertEqual(len(events), 1)
                self.assertEqual(report["evaluation"]["false_alerts"], 1)
                self.assertEqual(report["evaluation"]["false_alerts_per_camera_hour"], 1200.0)
                self.assertEqual(evaluated.result.duration_ms, 3000)

    def test_uncovered_labels_fail_closed_without_clipping(self):
        rows = [(0, 0.0), (500, 0.0), (1000, 0.0)]
        labels = (
            LabeledPersonDown(500, 2000, "partly-uncovered"),
            LabeledPersonDown(3000, 4000, "fully-uncovered"),
        )
        for module in self.modules:
            for label in labels:
                with self.subTest(module=module.__name__, label=label.label_id):
                    with self.assertRaisesRegex(ValueError, "label lies outside the evaluated video interval"):
                        self._scan(module, rows, labels=(label,), declared_end=5000)

    def test_one_frame_has_no_measurable_decoded_duration(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(RuntimeError, "must span more than one decoded timestamp"):
                    self._scan(module, [(0, 0.0)], declared_end=1000)

    def test_decoded_start_must_match_declared_start(self):
        for module in self.modules:
            for actual_start in (0, 1500):
                rows = [(actual_start + t, 0.0) for t in (0, 500, 1000)]
                with self.subTest(module=module.__name__, actual_start=actual_start):
                    with self.assertRaisesRegex(RuntimeError, "decoded video start does not match the declared sample start"):
                        self._scan(module, rows, declared_start=1000, declared_end=5000)

    def test_decoded_end_may_exceed_nominal_end_with_covered_positive(self):
        rows = [(t, 0.0) for t in range(0, 3501, 500)]
        labels = (LabeledPersonDown(0, 3000, "covered-down"),)
        for module in self.modules:
            with self.subTest(module=module.__name__):
                report, events, _, evaluated = self._scan(module, rows, labels=labels, declared_end=3000)
                self.assertEqual(len(events), 1)
                self.assertEqual(report["evaluation"]["matched_events"], 1)
                self.assertEqual(report["evaluation"]["missed_episodes"], 0)
                self.assertEqual(report["decoded_duration_ms"], 3500)
                self.assertEqual(evaluated.end_timestamp_ms, 3500)


if __name__ == "__main__":
    unittest.main()
