from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from analytics_lab import weapons_real_cc0_admission as admission


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
        chunk = self._payload[self._offset:self._offset + amount]
        self._offset += len(chunk)
        return chunk


def _opener(payloads: dict[str, bytes]):
    def open_request(request, timeout: int):
        del timeout
        return _Response(payloads[request.full_url])
    return open_request


class WeaponsRealCC0AdmissionTests(unittest.TestCase):
    def test_exact_two_sources_are_cc0_and_identity_bound(self):
        self.assertEqual(len(admission._SOURCES), 2)
        rifle, chair = admission._SOURCES
        self.assertEqual(rifle.license, "CC0-1.0")
        self.assertEqual(rifle.expected_size, 1_068_333)
        self.assertEqual(rifle.expected_sha1, "a4e50100507987ce58ff128d93eb2b11bccd88e2")
        self.assertEqual((rifle.width, rifle.height), (4000, 1663))
        self.assertEqual(chair.license, "CC0-1.0")
        self.assertEqual(chair.expected_size, 2_508_211)
        self.assertEqual(chair.expected_sha1, "f131416b5f2757d57b5a6fdb8b049c7bad87872b")
        self.assertEqual((chair.width, chair.height), (2848, 4272))
        self.assertTrue(all(source.url.startswith("https://upload.wikimedia.org/") for source in admission._SOURCES))

    def test_unpinned_discovery_emits_sha256_without_decode_or_inference(self):
        payloads = {
            admission._RIFLE.url: b"rifle-source",
            admission._CHAIR.url: b"chair-source",
        }
        sources = (
            replace(
                admission._RIFLE,
                expected_size=len(payloads[admission._RIFLE.url]),
                expected_sha1=hashlib.sha1(payloads[admission._RIFLE.url]).hexdigest(),
                sha256=None,
            ),
            replace(
                admission._CHAIR,
                expected_size=len(payloads[admission._CHAIR.url]),
                expected_sha1=hashlib.sha1(payloads[admission._CHAIR.url]).hexdigest(),
                sha256=None,
            ),
        )
        original = admission._SOURCES
        try:
            admission._SOURCES = sources
            result = admission.run(_opener(payloads))
        finally:
            admission._SOURCES = original
        self.assertFalse(result["all_sha256_pinned"])
        self.assertEqual(result["source_count"], 2)
        for item in result["sources"]:
            self.assertFalse(item["sha256_pinned"])
            self.assertFalse(item["media_decoded"])
            self.assertFalse(item["model_used"])
            self.assertFalse(item["inference_run"])
            self.assertFalse(item["artifact_retained"])

    def test_pinned_sha256_mismatch_fails_closed(self):
        payload = b"changed-source"
        source = replace(
            admission._RIFLE,
            expected_size=len(payload),
            expected_sha1=hashlib.sha1(payload).hexdigest(),
            sha256="0" * 64,
        )
        with self.assertRaisesRegex(RuntimeError, "SHA-256 changed"):
            admission._admit(source, _opener({source.url: payload}))

    def test_stream_bound_is_enforced(self):
        payload = b"12345"
        source = replace(admission._RIFLE, expected_size=1)
        with self.assertRaisesRegex(RuntimeError, "bounded admission limit"):
            admission._stream_identity(source, _opener({source.url: payload}))


if __name__ == "__main__":
    unittest.main()
