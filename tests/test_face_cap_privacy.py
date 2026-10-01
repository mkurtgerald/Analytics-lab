"""Synthetic cap-boundary regressions; no model, media or native runtime needed."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from analytics_lab.face_oid_ssd import (
    OID_V4_BOXES_OUTPUT, OID_V4_CLASSES_OUTPUT, OID_V4_INPUT_NAME,
    OID_V4_NUM_DETECTIONS_OUTPUT, OID_V4_SCORES_OUTPUT,
    OpenVINOOIDSSDDetector, parse_oid_v4_detections,
)
from analytics_lab.face_privacy import (
    FacePrivacyConfig, OpenCVYuNetDetector, apply_face_privacy, parse_yunet_rows,
)
from analytics_lab.tracking import NormalizedBox


def synthetic_boxes(count):
    # Distinct, non-overlapping boxes make dropping any face observable.
    return [
        NormalizedBox(x / 256, y / 256, (x + 8) / 256, (y + 8) / 256)
        for x, y in (((i % 8) * 16 + 4, (i // 8) * 16 + 4) for i in range(count))
    ]


def oid_outputs(boxes):
    return {
        OID_V4_BOXES_OUTPUT: [[[b.y_min, b.x_min, b.y_max, b.x_max] for b in boxes]],
        OID_V4_SCORES_OUTPUT: [[0.95] * len(boxes)],
        OID_V4_CLASSES_OUTPUT: [[502.0] * len(boxes)],
        OID_V4_NUM_DETECTIONS_OUTPUT: [float(len(boxes))],
    }


def yunet_rows(boxes):
    return [
        [b.x_min * 300, b.y_min * 300, (b.x_max - b.x_min) * 300,
         (b.y_max - b.y_min) * 300] + [0.0] * 10 + [0.95]
        for b in boxes
    ]


class SyntheticFrame:
    shape = (300, 300, 3)
    ndim = 3
    dtype = "uint8"

    def __init__(self):
        self.copy = Mock(return_value=object())

    def __getitem__(self, _key):
        return self


class Port:
    def __init__(self, name):
        self.name = name

    def get_names(self):
        return {self.name}


@contextmanager
def synthetic_adapter(kind, boxes, cap):
    if kind == "oid":
        outputs = oid_outputs(boxes)
        ports = [Port(name) for name in outputs]
        compiled = Mock(return_value={port: outputs[port.name] for port in ports})
        compiled.inputs = [Port(OID_V4_INPUT_NAME)]
        compiled.outputs = ports
        detector = OpenVINOOIDSSDDetector(compiled, max_faces=cap)
        # Only native preprocessing is stubbed; the real adapter/parser/privacy
        # chain executes, including port binding, output filtering and failure.
        numpy = SimpleNamespace(
            uint8="uint8", asarray=lambda frame: frame,
            ascontiguousarray=lambda frame: frame,
        )
        with patch.dict("sys.modules", {"cv2": SimpleNamespace(), "numpy": numpy}):
            yield detector, compiled
    else:
        with tempfile.TemporaryDirectory() as root:
            artifact = b"synthetic face adapter fixture, not model weights"
            path = Path(root) / "synthetic.onnx"
            path.write_bytes(artifact)
            infer = Mock(return_value=(1, yunet_rows(boxes)))
            detector = OpenCVYuNetDetector(
                path, expected_sha256=hashlib.sha256(artifact).hexdigest(),
                max_faces=cap, factory=lambda *args: SimpleNamespace(detect=infer),
            )
            yield detector, infer


class FaceCapPrivacyTests(unittest.TestCase):
    def test_at_cap_keeps_every_face_for_default_and_denied_unblur(self):
        for kind in ("oid", "yunet"):
            for cap in (1, 2, 64):
                for request_unblur in (False, True):
                    with self.subTest(kind=kind, cap=cap, request_unblur=request_unblur):
                        boxes = synthetic_boxes(cap)
                        frame = SyntheticFrame()
                        with synthetic_adapter(kind, boxes, cap) as (detector, infer):
                            with patch("analytics_lab.face_privacy._blur_regions") as blur:
                                result = apply_face_privacy(
                                    frame, detector, config=FacePrivacyConfig(max_faces=cap),
                                    request_unblur=request_unblur, authorized_unblur=False,
                                )
                        infer.assert_called_once()
                        blur.assert_called_once_with(frame, result.detections, FacePrivacyConfig(max_faces=cap))
                        self.assertIs(result.frame_bgr, blur.return_value)
                        self.assertEqual(result.audit.face_count, cap)
                        self.assertEqual(result.audit.action, "unblur_denied" if request_unblur else "blur")
                        self.assertEqual(
                            [item.box for item in result.detections],
                            sorted(boxes, key=lambda box: (box.x_min, box.y_min)),
                        )

    def test_cap_plus_one_rejects_all_privacy_paths_before_any_output(self):
        for kind in ("oid", "yunet"):
            for cap in (1, 2, 64):
                for default_blur, request_unblur, authorized_unblur in (
                    (True, False, False), (True, True, False),
                    (True, True, True), (False, False, False),
                ):
                    with self.subTest(kind=kind, cap=cap, default_blur=default_blur,
                                      request_unblur=request_unblur, authorized_unblur=authorized_unblur):
                        frame = SyntheticFrame()
                        with synthetic_adapter(kind, synthetic_boxes(cap + 1), cap) as (detector, infer):
                            with patch("analytics_lab.face_privacy._blur_regions") as blur:
                                with self.assertRaisesRegex(RuntimeError, "face count exceeds configured bound"):
                                    apply_face_privacy(
                                        frame, detector,
                                        config=FacePrivacyConfig(default_blur=default_blur),
                                        request_unblur=request_unblur,
                                        authorized_unblur=authorized_unblur,
                                    )
                        infer.assert_called_once()
                        blur.assert_not_called()
                        frame.copy.assert_not_called()

    def test_oid_cap_counts_only_admitted_faces(self):
        faces = parse_oid_v4_detections(
            boxes=[[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4],
                    [0.5, 0.5, 0.6, 0.6], [0.7, 0.7, 0.7, 0.8]]],
            scores=[[0.50, 0.99, 0.49, 0.95]], classes=[[502.0, 1.0, 502.0, 502.0]],
            num_detections=[4.0], max_faces=1,
        )
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].confidence, 0.50)

    def test_yunet_cap_counts_only_admitted_faces(self):
        faces = parse_yunet_rows(
            [[2.0, 2.0, 20.0, 20.0] + [0.0] * 10 + [0.90],
             [34.0, 2.0, 20.0, 20.0] + [0.0] * 10 + [0.89],
             [66.0, 2.0, 0.0, 20.0] + [0.0] * 10 + [0.95]],
            width=300, height=300, max_faces=1,
        )
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].confidence, 0.90)

    def test_yunet_stops_reading_on_first_overflow_face(self):
        def rows():
            yield from yunet_rows(synthetic_boxes(3))
            self.fail("parser kept consuming rows after detecting overflow")

        with self.assertRaisesRegex(RuntimeError, "face count exceeds configured bound"):
            parse_yunet_rows(rows(), width=300, height=300, max_faces=2)


if __name__ == "__main__":
    unittest.main()
