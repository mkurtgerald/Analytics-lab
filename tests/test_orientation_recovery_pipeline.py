"""Guarded rotation regressions using generated metadata and Python heatmaps only."""
from dataclasses import asdict, dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from analytics_lab.openvino_omz import OpenVINOOMZConfig, OpenVINOOMZPoseBackend, _OpenVINORuntime
from analytics_lab.openvino_pipeline import (
    OpenVINOOMZPerceptionAdapter, OpenVINOOMZPipelineConfig, run_frame_source_openvino_omz,
)
from analytics_lab.video import Frame


_SOURCE_BOX = (.25, .25, .75, .5)
_CCW_BOX = (.25, .25, .5, .75)
_CW_BOX = (.5, .25, .75, .75)


def _row(box=_SOURCE_BOX, score=.95):
    return (0, 0, score, *box)


@dataclass(frozen=True)
class _Image:
    index: int
    shape: tuple = (200, 400, 3)
    marker: str = "generated layout metadata; no pixels"


class _Runtime:
    runtime_version = "2026.3.1-generated"

    def __init__(self, primary=None, rotated=None):
        self.primary = {0: (_row(),)} if primary is None else primary
        self.rotated = {} if rotated is None else rotated
        self.calls = []
        self.pose_calls = []
        self.pose_failures = set()

    def detect_rows(self, image):
        self.calls.append((image.index, 0))
        return self.primary.get(image.index, ())

    def detect_rotated_rows(self, image, turn):
        self.calls.append((image.index, turn))
        value = self.rotated.get((image.index, turn), ())
        if isinstance(value, Exception):
            raise value
        return value

    def pose_heatmaps(self, image, bbox):
        self.pose_calls.append((image, bbox))
        if image.index in self.pose_failures:
            raise RuntimeError("generated pose failure")
        maps = [[[0.0] * 4 for _ in range(4)] for _ in range(19)]
        for channel, x in ((2, 0), (5, 1), (8, 2), (11, 3)):
            maps[channel][1][x] = .9
        return maps


class OrientationRecoveryPipelineTests(unittest.TestCase):
    def config(self, enabled=True):
        return OpenVINOOMZPipelineConfig(
            openvino=OpenVINOOMZConfig(detector_model="person-detection-0200"),
            orientation_recovery=enabled,
        )

    def adapter(self, runtime, *, enabled=True, backend=None):
        with patch("analytics_lab.openvino_omz.verify_artifact_set", return_value=()):
            return OpenVINOOMZPerceptionAdapter(
                None if backend is not None else "generated-artifact-metadata",
                config=self.config(enabled), backend=backend,
                runtime_factory=None if backend is not None else lambda *_: runtime,
            )

    def test_optin_is_strict_default_off_and_retail_rejects_before_construction(self):
        self.assertIs(OpenVINOOMZPipelineConfig().orientation_recovery, False)
        self.assertEqual(self.config().openvino.detection_threshold, .50)
        self.assertEqual(self.config().tracker.min_iou, .15)
        self.assertEqual(self.config().temporal.min_confidence, .7)
        self.assertEqual(self.config().temporal.down_duration_ms, 3000)
        self.assertEqual(self.config().temporal.max_unknown_gap_ms, 0)
        for value in (1, 0, None, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                OpenVINOOMZPipelineConfig(orientation_recovery=value)
        with self.assertRaises(ValueError):
            OpenVINOOMZPipelineConfig(orientation_recovery=True)
        cfg = OpenVINOOMZPipelineConfig()
        object.__setattr__(cfg, "orientation_recovery", True)
        with patch("analytics_lab.openvino_pipeline.OpenVINOOMZPoseBackend") as construct:
            with self.assertRaises(ValueError):
                OpenVINOOMZPerceptionAdapter("generated", config=cfg)
            construct.assert_not_called()

    def test_generated_positive_recovers_at_unchanged_confidence_and_persistence(self):
        images = tuple(_Image(index) for index in range(7))
        snapshots = tuple(asdict(image) for image in images)
        rotated = {(index, 1): (_row(_CCW_BOX),) for index in range(1, 7)}
        for enabled, expected_observations, expected_events in ((False, 1, 0), (True, 7, 1)):
            with self.subTest(enabled=enabled):
                runtime = _Runtime(rotated=rotated)
                with patch("analytics_lab.openvino_omz.verify_artifact_set", return_value=()):
                    result = run_frame_source_openvino_omz(
                        (Frame(index, index * 500, image) for index, image in enumerate(images)),
                        artifact_root="generated-artifact-metadata", source_id="generated-camera",
                        session_id="generated-session", config=self.config(enabled),
                        runtime_factory=lambda *_: runtime,
                    )
                self.assertEqual(result.video.frames_processed, 7)
                self.assertEqual(result.video.observations_processed, expected_observations)
                self.assertEqual(len(result.video.events), expected_events)
                self.assertEqual(len(runtime.pose_calls), expected_observations)
                for image, box in runtime.pose_calls:
                    self.assertIs(image, images[image.index])
                    self.assertEqual((box.x1, box.y1, box.x2, box.y2), (100, 50, 300, 100))
                if enabled:
                    event = result.video.events[0]
                    self.assertEqual(event["confidence"], .9)
                    self.assertEqual(event["status"], "candidate")
                    self.assertEqual(event["evidence"][0]["start_timestamp_ms"], 0)
                    self.assertEqual(event["evidence"][0]["end_timestamp_ms"], 3000)
                    self.assertEqual(runtime.calls, [(0, 0)] + [call for index in range(1, 7)
                                                              for call in ((index, 0), (index, -1), (index, 1))])
                else:
                    self.assertEqual(runtime.calls, [(index, 0) for index in range(7)])
        self.assertEqual(tuple(asdict(image) for image in images), snapshots)

    def test_both_views_map_back_and_only_one_box_reaches_pose(self):
        for turn, box in ((-1, _CW_BOX), (1, _CCW_BOX)):
            with self.subTest(turn=turn):
                runtime = _Runtime(rotated={(1, turn): (_row(box), _row(box, .8))})
                adapter = self.adapter(runtime)
                adapter(_Image(0), 0, 0)
                observed = adapter(_Image(1), 1, 500)
                self.assertEqual(len(observed), 1)
                self.assertEqual(observed[0].posture, "down")
                self.assertEqual(len(runtime.pose_calls), 2)
                box = runtime.pose_calls[-1][1]
                self.assertEqual((box.x1, box.y1, box.x2, box.y2), (100, 50, 300, 100))

    def test_cutoff_is_inclusive_and_weak_or_unlinked_proposals_abstain(self):
        for proposal, expected in ((_row(_CCW_BOX, .49), 0),
                                   (_row((0, 0, .1, .1), .99), 0),
                                   (_row(_CCW_BOX, .50), 1)):
            with self.subTest(proposal=proposal):
                runtime = _Runtime(rotated={(1, 1): (proposal,)})
                adapter = self.adapter(runtime)
                adapter(_Image(0), 0, 0)
                observed = adapter(_Image(1), 1, 500)
                self.assertEqual(len(observed), expected)
                self.assertEqual(len(runtime.pose_calls), 1 + expected)
                if expected:
                    self.assertEqual(observed[0].confidence, .50)

    def test_primary_detection_precedence_never_calls_rotations(self):
        runtime = _Runtime(primary={0: (_row(),), 1: (_row(score=.8),)},
                           rotated={(1, 1): (_row(_CCW_BOX, .99),)})
        adapter = self.adapter(runtime)
        adapter(_Image(0), 0, 0)
        self.assertEqual(adapter(_Image(1), 1, 500)[0].confidence, .8)
        self.assertEqual(runtime.calls, [(0, 0), (1, 0)])

    def test_missing_multi_person_frame_or_time_gap_cannot_seed_recovery(self):
        cases = (
            ({}, None, _Image(1), 0, 0),
            ({0: (_row(), _row((.05, .05, .15, .2)))}, _Image(0), _Image(1), 1, 500),
            ({0: (_row(),)}, _Image(0), _Image(1), 2, 500),
            ({0: (_row(),)}, _Image(0), _Image(1), 1, 751),
            ({0: (_row(),)}, _Image(0), _Image(1), 1, 0),
            ({0: (_row(),)}, _Image(0), _Image(1, (400, 200, 3)), 1, 500),
        )
        for primary, initial, image, frame_index, timestamp in cases:
            with self.subTest(primary=primary, frame_index=frame_index, timestamp=timestamp, shape=image.shape):
                runtime = _Runtime(primary=primary, rotated={(1, 1): (_row(_CCW_BOX),)})
                adapter = self.adapter(runtime)
                if initial is not None:
                    adapter(initial, 0, 0)
                self.assertEqual(adapter(image, frame_index, timestamp), ())
                self.assertTrue(all(turn == 0 for _, turn in runtime.calls))

    def test_failed_recovery_clears_prior_until_new_primary_acquisition(self):
        runtime = _Runtime(primary={0: (_row(),), 3: (_row(),)},
                           rotated={(2, 1): (_row(_CCW_BOX),), (4, 1): (_row(_CCW_BOX),)})
        adapter = self.adapter(runtime)
        sizes = [len(adapter(_Image(index), index, index * 500)) for index in range(5)]
        self.assertEqual(sizes, [1, 0, 0, 1, 1])
        self.assertNotIn((2, -1), runtime.calls)
        self.assertNotIn((2, 1), runtime.calls)

    def test_rotated_or_pose_failure_clears_history(self):
        for failure in ("rotated", "pose"):
            with self.subTest(failure=failure):
                runtime = _Runtime(
                    primary={0: (_row(),), **({1: (_row(),)} if failure == "pose" else {})},
                    rotated={(1, -1): RuntimeError("generated detector failure"), (2, 1): (_row(_CCW_BOX),)},
                )
                if failure == "pose":
                    runtime.pose_failures.add(1)
                adapter = self.adapter(runtime)
                adapter(_Image(0), 0, 0)
                with self.assertRaises(RuntimeError):
                    adapter(_Image(1), 1, 500)
                self.assertIsNone(adapter._recovery.previous)
                self.assertEqual(adapter(_Image(2), 2, 1000), ())
                self.assertNotIn((2, 1), runtime.calls)

    def test_downstream_tracker_failure_clears_newly_committed_recovery_history(self):
        adapter = self.adapter(_Runtime(primary={0: (_row(),)}))
        with patch.object(adapter._adapter.tracker, "update", side_effect=RuntimeError("generated failure")):
            with self.assertRaises(RuntimeError):
                adapter(_Image(0), 0, 0)
        self.assertIsNone(adapter._recovery.previous)

    def test_rotation_capability_is_required_only_when_recovery_is_attempted(self):
        runtime = _Runtime()
        runtime.detect_rotated_rows = None
        adapter = self.adapter(runtime)
        self.assertEqual(len(adapter(_Image(0), 0, 0)), 1)
        with self.assertRaisesRegex(RuntimeError, "does not support rotated detection"):
            adapter(_Image(1), 1, 500)
        self.assertIsNone(adapter._recovery.previous)
        self.assertEqual(adapter(_Image(2), 2, 1000), ())

    def test_clips_have_independent_history_with_one_shared_warm_backend(self):
        runtime = _Runtime(rotated={(1, 1): (_row(_CCW_BOX),)})
        factory = Mock(return_value=runtime)
        with patch("analytics_lab.openvino_omz.verify_artifact_set", return_value=()):
            backend = OpenVINOOMZPoseBackend("generated", config=self.config().openvino, runtime_factory=factory)
        first = self.adapter(runtime, backend=backend)
        second = self.adapter(runtime, backend=backend)
        first(_Image(0), 0, 0)
        self.assertEqual(len(first(_Image(1), 1, 500)), 1)
        calls_before_second = len(runtime.calls)
        self.assertEqual(second(_Image(1), 0, 0), ())
        self.assertEqual(runtime.calls[calls_before_second:], [(1, 0)])
        self.assertIs(first.backend, second.backend)
        self.assertIsNot(first._recovery, second._recovery)
        self.assertFalse(hasattr(backend, "previous"))
        factory.assert_called_once()

    def test_stateless_detection_api_rejects_unknown_rotation_and_retail_rotations(self):
        runtime = _Runtime()
        backend = self.adapter(runtime).backend
        for turn in (True, False, 2, -2, "1", None):
            with self.subTest(turn=turn), self.assertRaises(ValueError):
                backend.detect_people(_Image(0), quarter_turn=turn)
        self.assertEqual(runtime.calls, [])
        with patch("analytics_lab.openvino_omz.verify_artifact_set", return_value=()):
            retail = OpenVINOOMZPoseBackend("generated", runtime_factory=lambda *_: runtime)
        with self.assertRaises(ValueError):
            retail.detect_people(_Image(0), quarter_turn=1)
        self.assertEqual(runtime.calls, [])

    def test_native_rotation_uses_temporary_contiguous_view_with_same_detector(self):
        source = _Image(0)
        view = SimpleNamespace(shape=(400, 200, 3), marker="generated rotated view")
        contiguous = SimpleNamespace(shape=(400, 200, 3), marker="generated contiguous view")
        runtime = object.__new__(_OpenVINORuntime)
        runtime._np = SimpleNamespace(rot90=Mock(return_value=view), ascontiguousarray=Mock(return_value=contiguous))
        runtime.detect_rows = Mock(return_value=(_row(_CCW_BOX),))
        snapshot = asdict(source)
        self.assertEqual(runtime.detect_rotated_rows(source, 1), (_row(_CCW_BOX),))
        runtime._np.rot90.assert_called_once_with(source, k=1)
        runtime._np.ascontiguousarray.assert_called_once_with(view)
        runtime.detect_rows.assert_called_once_with(contiguous)
        self.assertEqual(asdict(source), snapshot)
        runtime._np.rot90.reset_mock()
        runtime.detect_rows.reset_mock()
        for turn in (0, 2, True, "1"):
            with self.subTest(turn=turn), self.assertRaises(ValueError):
                runtime.detect_rotated_rows(source, turn)
        runtime._np.rot90.assert_not_called()
        runtime.detect_rows.assert_not_called()


if __name__ == "__main__":
    unittest.main()
