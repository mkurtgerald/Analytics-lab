"""Unit coverage for the bounded real-person face measurement evidence."""
from __future__ import annotations

import unittest

from analytics_lab.face_privacy import FaceDetection
from analytics_lab.face_real_cc0_measurement import _expected_graph_identity, _observation, _serialize_detections
from analytics_lab.tracking import NormalizedBox


class FaceRealCC0MeasurementTests(unittest.TestCase):
    def test_measurement_reuses_pinned_graph_identity_without_archive_readmission(self):
        size, digest = _expected_graph_identity()
        self.assertEqual(size, 66_606_111)
        self.assertEqual(
            digest,
            "150f3eb77b741ed24e1a19559589205ccd616b059fc976c8b8d5cefc126bf86b",
        )

    def test_serializes_raw_detections_without_identity_fields(self):
        detections = (
            FaceDetection(0.875, NormalizedBox(0.1, 0.2, 0.7, 0.8)),
            FaceDetection(0.625, NormalizedBox(0.2, 0.3, 0.4, 0.5)),
        )
        payload = _serialize_detections(detections)
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["confidence"], 0.875)
        self.assertEqual(payload[0]["box_xyxy_normalized"], [0.1, 0.2, 0.7, 0.8])
        text = repr(payload).lower()
        for forbidden in ("name", "identity", "embedding", "reid"):
            self.assertNotIn(forbidden, text)

    def test_observation_keeps_first_measurement_unscored_without_ground_truth(self):
        detections = (
            FaceDetection(0.75, NormalizedBox(0.1, 0.1, 0.9, 0.9)),
        )
        payload = _observation(
            detections=detections,
            privacy_action="blur",
            width=5184,
            height=3456,
            input_immutable=True,
            output_changed=True,
            openvino_version="2026.3.1",
            opencv_version="4.12.0",
        )
        self.assertEqual(payload["confidence_threshold"], 0.50)
        self.assertEqual(payload["inference_runs"], 1)
        self.assertEqual(payload["detection_count"], 1)
        self.assertEqual(payload["privacy_action"], "blur")
        self.assertFalse(payload["ground_truth_available"])
        self.assertEqual(
            payload["misses"], "unscored_without_independent_ground_truth"
        )
        self.assertEqual(
            payload["false_positives"], "unscored_without_independent_ground_truth"
        )
        self.assertFalse(payload["threshold_tuned_after_result"])
        self.assertIn("not face-detection accuracy", payload["claim"])

    def test_observation_preserves_zero_detection_result(self):
        payload = _observation(
            detections=(),
            privacy_action="blur",
            width=5184,
            height=3456,
            input_immutable=True,
            output_changed=False,
            openvino_version="2026.3.1",
            opencv_version="4.12.0",
        )
        self.assertEqual(payload["detection_count"], 0)
        self.assertEqual(payload["detections"], [])
        self.assertFalse(payload["output_changed"])
        self.assertEqual(payload["inference_runs"], 1)


if __name__ == "__main__":
    unittest.main()
