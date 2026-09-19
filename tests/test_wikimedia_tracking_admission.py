import hashlib
import io
import unittest
from unittest.mock import patch

from analytics_lab import wikimedia_tracking_admission as w


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, *, url: str | None = None, content_length: str | None = None):
        super().__init__(payload)
        self._url = url or w.SOURCE_URL
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class WikimediaTrackingAdmissionTests(unittest.TestCase):
    def test_hash_stream_is_bounded_and_deterministic(self):
        payload = b"abc" * 10
        digest = w._hash_stream(io.BytesIO(payload), max_bytes=len(payload))
        self.assertEqual(digest.size_bytes, len(payload))
        self.assertEqual(digest.sha1, hashlib.sha1(payload).hexdigest())
        self.assertEqual(digest.sha256, hashlib.sha256(payload).hexdigest())
        with self.assertRaises(RuntimeError):
            w._hash_stream(io.BytesIO(payload), max_bytes=len(payload) - 1)

    def test_admit_source_requires_exact_published_bytes(self):
        payload = b"reviewed-evidence-bytes"
        sha1 = hashlib.sha1(payload).hexdigest()

        def opener(request, timeout):
            self.assertEqual(request.full_url, w.SOURCE_URL)
            self.assertEqual(timeout, 30)
            return _Response(payload, content_length=str(len(payload)))

        with patch.object(w, "PUBLISHED_SIZE", len(payload)), patch.object(w, "PUBLISHED_SHA1", sha1):
            digest = w.admit_source(opener=opener)
        self.assertEqual(digest.sha256, hashlib.sha256(payload).hexdigest())

    def test_admit_source_rejects_redirect_outside_reviewed_asset(self):
        payload = b"bytes"

        def opener(request, timeout):
            return _Response(payload, url="https://example.com/not-reviewed.webm")

        with self.assertRaises(RuntimeError):
            w.admit_source(opener=opener)

    def test_admit_source_rejects_advertised_size_change_before_hashing(self):
        payload = b"bytes"

        def opener(request, timeout):
            return _Response(payload, content_length=str(w.PUBLISHED_SIZE + 1))

        with self.assertRaises(RuntimeError):
            w.admit_source(opener=opener)

    def test_admit_source_rejects_checksum_change(self):
        payload = b"changed-evidence"

        def opener(request, timeout):
            return _Response(payload, content_length=str(len(payload)))

        with patch.object(w, "PUBLISHED_SIZE", len(payload)):
            with self.assertRaises(RuntimeError):
                w.admit_source(opener=opener)


if __name__ == "__main__":
    unittest.main()
