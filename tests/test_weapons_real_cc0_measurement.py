from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from analytics_lab import weapons_real_cc0_admission as admission
from analytics_lab import weapons_real_cc0_measurement as measurement
from analytics_lab.tracking import DetectionCandidate, NormalizedBox


class _Response:
    def __init__(self, payload: bytes, *, url: str, content_length: str | None = None):
        self._payload = payload
        self._offset = 0
        self._url = url
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def geturl(self) -> str:
        return self._url

    def read(self, amount: int) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        chunk = self._payload[self._offset:self._offset + amount]
        self._offset += len(chunk)
        return chunk


def _opener(payload: bytes, source: admission._Source):
    def open_request(request, timeout: int):
        del timeout
        return _Response(payload, url=request.full_url, content_length=str(len(payload)))
    return open_request


class WeaponsRealCC0MeasurementTests(unittest.TestCase):
    def test_source_download_requires_pinned_sha256(self):
        payload = b"fixture"
        source = replace(
            admission._RIFLE,
            expected_size=len(payload),
            expected_sha1=hashlib.sha1(payload).hexdigest(),
            sha256=None,
        )
        with self.assertRaisesRegex(RuntimeError, "SHA-256 is not pinned"):
            measurement._download_verified_source(source, _opener(payload, source))

    def test_source_download_reverifies_all_identities(self):
        payload = b"fixture"
        source = replace(
            admission._RIFLE,
            expected_size=len(payload),
            expected_sha1=hashlib.sha1(payload).hexdigest(),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        self.assertEqual(
            measurement._download_verified_source(source, _opener(payload, source)),
            payload,
        )

    def test_serialization_preserves_class_category_confidence_and_box(self):
        item = DetectionCandidate(
            category="rifle",
            confidence=0.75,
            box=NormalizedBox(0.1, 0.2, 0.3, 0.4),
            model_class_id=351,
        )
        self.assertEqual(
            measurement._serialize((item,)),
            [{
                "class_id": 351,
                "category": "rifle",
                "confidence": 0.75,
                "box_xyxy_normalized": [0.1, 0.2, 0.3, 0.4],
            }],
        )

    def test_measurement_contract_is_exactly_two_pinned_cc0_sources(self):
        self.assertEqual(len(admission._SOURCES), 2)
        self.assertTrue(all(source.license == "CC0-1.0" for source in admission._SOURCES))
        self.assertTrue(all(source.sha256 is not None for source in admission._SOURCES))
        self.assertEqual(measurement._FIXED_THRESHOLD, 0.50)


if __name__ == "__main__":
    unittest.main()
