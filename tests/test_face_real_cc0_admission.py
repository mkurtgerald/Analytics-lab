from __future__ import annotations

import hashlib
import unittest
from unittest import mock

from analytics_lab import face_real_cc0_admission as admission


class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, amount: int) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        chunk = self._payload[self._offset : self._offset + amount]
        self._offset += len(chunk)
        return chunk


def _opener(payload: bytes):
    def open_request(request, timeout: int):
        del request, timeout
        return _Response(payload)
    return open_request


class FaceRealCC0AdmissionTests(unittest.TestCase):
    def test_source_metadata_is_fail_closed_and_cc0(self):
        self.assertEqual(admission._EXPECTED_SIZE, 9_023_019)
        self.assertEqual(admission._EXPECTED_SHA1, "d7ac58077d135b823e71e7b38e763f082e2053bc")
        self.assertEqual(admission._SOURCE_LICENSE, "CC0-1.0")
        self.assertIsNone(admission._SOURCE_SHA256)
        self.assertLess(admission._EXPECTED_SIZE, admission._MAX_BYTES)

    def test_discovery_stream_emits_identity_without_decode_or_inference(self):
        payload = b"rights-cleared-face-source"
        with mock.patch.object(admission, "_EXPECTED_SIZE", len(payload)), \
             mock.patch.object(admission, "_EXPECTED_SHA1", hashlib.sha1(payload).hexdigest()), \
             mock.patch.object(admission, "_SOURCE_SHA256", None):
            result = admission.run(_opener(payload))
        self.assertEqual(result["source_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertFalse(result["sha256_pinned"])
        self.assertFalse(result["media_decoded"])
        self.assertFalse(result["model_used"])
        self.assertFalse(result["inference_run"])
        self.assertFalse(result["artifact_retained"])

    def test_pinned_sha256_mismatch_fails_closed(self):
        payload = b"changed"
        with mock.patch.object(admission, "_EXPECTED_SIZE", len(payload)), \
             mock.patch.object(admission, "_EXPECTED_SHA1", hashlib.sha1(payload).hexdigest()), \
             mock.patch.object(admission, "_SOURCE_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 changed"):
                admission.run(_opener(payload))

    def test_stream_bound_is_enforced(self):
        payload = b"12345"
        with mock.patch.object(admission, "_MAX_BYTES", 4):
            with self.assertRaisesRegex(RuntimeError, "bounded admission limit"):
                admission._stream_identity(_opener(payload))


if __name__ == "__main__":
    unittest.main()
