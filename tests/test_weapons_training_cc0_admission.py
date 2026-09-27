import hashlib
import unittest

from analytics_lab import weapons_training_cc0_admission as admission


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


def _opener(
    payload: bytes,
    source: admission._Source,
    *,
    final_url: str | None = None,
    content_length: str | None = None,
):
    def open_request(request, timeout: int):
        del timeout
        return _Response(
            payload,
            url=final_url or request.full_url,
            content_length=str(len(payload)) if content_length is None else content_length,
        )
    return open_request


class WeaponsTrainingCC0AdmissionTests(unittest.TestCase):
    def test_source_set_is_exact_object_only_cc0_preselection(self):
        self.assertEqual(
            [(s.name, s.source_class, s.license) for s in admission._SOURCES],
            [
                ("walther-p38-pistol", "handgun", "CC0-1.0"),
                ("british-styled-celtic-la-tene-sword", "sword", "CC0-1.0"),
                ("cimmerian-dagger", "knife", "CC0-1.0"),
            ],
        )
        self.assertTrue(all(not source.identifiable_people for source in admission._SOURCES))
        self.assertTrue(all(source.sha256 is None for source in admission._SOURCES))

    def test_stream_identity_verifies_published_size_and_sha1(self):
        payload = b"fixture"
        source = admission._Source(
            name="fixture",
            source_class="knife",
            page="https://commons.wikimedia.org/wiki/File:Fixture.jpg",
            url="https://upload.wikimedia.org/wikipedia/commons/a/aa/Fixture.jpg",
            license="CC0-1.0",
            rights_basis="fixture rights",
            expected_size=len(payload),
            expected_sha1=hashlib.sha1(payload).hexdigest(),
            width=10,
            height=20,
            identifiable_people=False,
            third_party_caveat="none",
        )
        size, sha1, sha256 = admission._stream_identity(
            source, _opener(payload, source)
        )
        self.assertEqual(size, len(payload))
        self.assertEqual(sha1, hashlib.sha1(payload).hexdigest())
        self.assertEqual(sha256, hashlib.sha256(payload).hexdigest())

    def test_redirect_escape_fails_closed(self):
        payload = b"fixture"
        source = admission._Source(
            name="fixture",
            source_class="knife",
            page="https://commons.wikimedia.org/wiki/File:Fixture.jpg",
            url="https://upload.wikimedia.org/wikipedia/commons/a/aa/Fixture.jpg",
            license="CC0-1.0",
            rights_basis="fixture rights",
            expected_size=len(payload),
            expected_sha1=hashlib.sha1(payload).hexdigest(),
            width=10,
            height=20,
            identifiable_people=False,
            third_party_caveat="none",
        )
        with self.assertRaisesRegex(RuntimeError, "redirected outside"):
            admission._stream_identity(
                source,
                _opener(
                    payload,
                    source,
                    final_url="https://example.com/Fixture.jpg",
                ),
            )

    def test_content_length_and_sha1_drift_fail_closed(self):
        payload = b"fixture"
        source = admission._Source(
            name="fixture",
            source_class="knife",
            page="https://commons.wikimedia.org/wiki/File:Fixture.jpg",
            url="https://upload.wikimedia.org/wikipedia/commons/a/aa/Fixture.jpg",
            license="CC0-1.0",
            rights_basis="fixture rights",
            expected_size=len(payload),
            expected_sha1=hashlib.sha1(payload).hexdigest(),
            width=10,
            height=20,
            identifiable_people=False,
            third_party_caveat="none",
        )
        with self.assertRaisesRegex(RuntimeError, "Content-Length changed"):
            admission._stream_identity(
                source,
                _opener(
                    payload,
                    source,
                    content_length=str(len(payload) + 1),
                ),
            )
        changed = admission._Source(
            **{**source.__dict__, "expected_sha1": "0" * 40}
        )
        with self.assertRaisesRegex(RuntimeError, "SHA-1 changed"):
            admission._stream_identity(changed, _opener(payload, changed))

    def test_pinned_sha256_mismatch_fails_closed(self):
        payload = b"fixture"
        source = admission._Source(
            name="fixture",
            source_class="knife",
            page="https://commons.wikimedia.org/wiki/File:Fixture.jpg",
            url="https://upload.wikimedia.org/wikipedia/commons/a/aa/Fixture.jpg",
            license="CC0-1.0",
            rights_basis="fixture rights",
            expected_size=len(payload),
            expected_sha1=hashlib.sha1(payload).hexdigest(),
            width=10,
            height=20,
            identifiable_people=False,
            third_party_caveat="none",
            sha256="0" * 64,
        )
        with self.assertRaisesRegex(RuntimeError, "SHA-256 changed"):
            admission._stream_identity(source, _opener(payload, source))


if __name__ == "__main__":
    unittest.main()
