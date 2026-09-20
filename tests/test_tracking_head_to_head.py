import unittest

from analytics_lab.tracking import DetectionCandidate, NormalizedBox
from analytics_lab.tracking_annotations import TrackingGroundTruthFrame
from analytics_lab.tracking_evaluation import GroundTruthObject
from analytics_lab.tracking_evidence import FrameDigest
from analytics_lab.tracking_ground_truth_package import TrackingGroundTruthPackage
from analytics_lab.tracking_head_to_head import (
    TrackingDetectionFrame,
    compare_frozen_trackers,
)


class TrackingHeadToHeadTests(unittest.TestCase):
    @staticmethod
    def _package() -> TrackingGroundTruthPackage:
        boxes = (
            NormalizedBox(0.10, 0.10, 0.30, 0.50),
            NormalizedBox(0.11, 0.10, 0.31, 0.50),
            NormalizedBox(0.12, 0.10, 0.32, 0.50),
        )
        frames = tuple(
            TrackingGroundTruthFrame(
                index,
                (GroundTruthObject("person-1", "person", box),),
            )
            for index, box in enumerate(boxes)
        )
        digests = tuple(FrameDigest(index, f"{index + 1:064x}") for index in range(3))
        return TrackingGroundTruthPackage(
            frames=frames,
            frame_digests=digests,
            annotation_sha256="a" * 64,
            frame_manifest_sha256="b" * 64,
            object_observations=3,
        )

    @staticmethod
    def _detections() -> tuple[TrackingDetectionFrame, ...]:
        boxes = (
            NormalizedBox(0.10, 0.10, 0.30, 0.50),
            NormalizedBox(0.11, 0.10, 0.31, 0.50),
            NormalizedBox(0.12, 0.10, 0.32, 0.50),
        )
        confidences = (0.90, 0.40, 0.90)
        return tuple(
            TrackingDetectionFrame(
                index,
                1000 + index * 40,
                (DetectionCandidate("person", confidence, box, 0),),
            )
            for index, (box, confidence) in enumerate(zip(boxes, confidences, strict=True))
        )

    def test_bytetrack_low_confidence_recovery_beats_frozen_iou_control(self) -> None:
        result = compare_frozen_trackers(self._package(), self._detections())

        simple = result.simple_iou.association
        byte = result.portable_bytetrack.association
        self.assertEqual(simple.matched_observations, 2)
        self.assertEqual(simple.misses, 1)
        self.assertEqual(simple.id_switches, 1)
        self.assertEqual(simple.fragmentations, 1)
        self.assertAlmostEqual(simple.continuity, 2 / 3)

        self.assertEqual(byte.matched_observations, 3)
        self.assertEqual(byte.misses, 0)
        self.assertEqual(byte.id_switches, 0)
        self.assertEqual(byte.fragmentations, 0)
        self.assertEqual(byte.continuity, 1.0)
        self.assertGreater(byte.mean_matched_iou, 0.9)

        for run in (result.simple_iou, result.portable_bytetrack):
            self.assertEqual(run.runtime.frames, 3)
            self.assertEqual(run.runtime.detection_observations, 3)
            self.assertGreater(run.runtime.throughput_fps, 0.0)
            self.assertGreaterEqual(run.runtime.elapsed_seconds, 0.0)
            self.assertGreaterEqual(run.runtime.cpu_seconds, 0.0)
            self.assertGreaterEqual(run.runtime.mean_latency_ms, 0.0)
            self.assertGreaterEqual(run.runtime.p95_latency_ms, 0.0)
            self.assertGreaterEqual(run.runtime.max_latency_ms, 0.0)
            self.assertGreaterEqual(run.runtime.peak_python_bytes, 0)
            self.assertEqual(run.runtime.max_retained_tracks, 1)

        self.assertEqual(result.simple_iou.runtime.track_observations, 2)
        self.assertEqual(result.portable_bytetrack.runtime.track_observations, 3)
        self.assertTrue(result.first_attempt)
        self.assertEqual(result.evidence_scope, "engineering_smoke_not_commercial_accuracy")

    def test_rejects_detection_frame_mismatch(self) -> None:
        detections = self._detections()
        shifted = (
            TrackingDetectionFrame(1, detections[0].timestamp_ms, detections[0].detections),
            detections[1],
            detections[2],
        )
        with self.assertRaisesRegex(ValueError, "frame order"):
            compare_frozen_trackers(self._package(), shifted)

    def test_rejects_non_increasing_timestamps(self) -> None:
        detections = self._detections()
        repeated = (
            detections[0],
            TrackingDetectionFrame(1, detections[0].timestamp_ms, detections[1].detections),
            detections[2],
        )
        with self.assertRaisesRegex(ValueError, "timestamps"):
            compare_frozen_trackers(self._package(), repeated)


if __name__ == "__main__":
    unittest.main()
