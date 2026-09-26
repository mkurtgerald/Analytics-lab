import hashlib
import unittest

from analytics_lab import weapons_rfdetr_source_admission as admission


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


def _opener(payload: bytes, *, url: str, final_url: str | None = None, content_length: str | None = None):
    def open_request(request, timeout: int):
        del timeout
        return _Response(
            payload,
            url=final_url or request.full_url,
            content_length=str(len(payload)) if content_length is None else content_length,
        )
    return open_request


class RFDETRSourceAdmissionTests(unittest.TestCase):
    def test_pinned_revision_and_source_set_are_exact(self):
        self.assertEqual(
            admission.REVISION,
            "1985300a0e8f905c70bb1772855dcf6c964e5ed5",
        )
        self.assertEqual(admission.LICENSE_ID, "Apache-2.0")
        self.assertEqual(
            [item.path for item in admission._FILES],
            [
                "LICENSE",
                "pyproject.toml",
                "src/rfdetr/config.py",
                "src/rfdetr/detr.py",
                "src/rfdetr/models/backbone/dinov2.py",
                "src/rfdetr/models/lwdetr.py",
            ],
        )

    def test_git_blob_hash_matches_git_object_rule(self):
        payload = b"abc"
        expected = hashlib.sha1(b"blob 3\0abc").hexdigest()
        self.assertEqual(admission._git_blob_sha1(payload), expected)

    def test_download_verifies_size_blob_and_redirect(self):
        payload = b"fixture"
        source = admission._PinnedSource(
            "fixture.txt",
            len(payload),
            admission._git_blob_sha1(payload),
        )
        self.assertEqual(
            admission._download(source, _opener(payload, url=source.url)),
            payload,
        )
        with self.assertRaisesRegex(RuntimeError, "redirected outside"):
            admission._download(
                source,
                _opener(
                    payload,
                    url=source.url,
                    final_url="https://example.com/fixture.txt",
                ),
            )

    def test_download_rejects_length_and_blob_drift(self):
        payload = b"fixture"
        source = admission._PinnedSource(
            "fixture.txt",
            len(payload),
            admission._git_blob_sha1(payload),
        )
        with self.assertRaisesRegex(RuntimeError, "Content-Length changed"):
            admission._download(
                source,
                _opener(payload, url=source.url, content_length=str(len(payload) + 1)),
            )
        changed = admission._PinnedSource("fixture.txt", len(payload), "0" * 40)
        with self.assertRaisesRegex(RuntimeError, "Git blob identity changed"):
            admission._download(changed, _opener(payload, url=changed.url))

    def test_required_weightless_semantics_are_fail_closed(self):
        files = {
            "LICENSE": b"Apache License",
            "pyproject.toml": (
                b'license = {text = "Apache License 2.0"}\n'
                b'version = "1.11.0"\n'
            ),
            "src/rfdetr/config.py": (
                b"class RFDETRNanoConfig(RFDETRBaseConfig):\n"
                b"    patch_size: int = 16\n"
                b"    resolution: int = 384\n"
                b"    positional_encoding_size: int = 24\n"
                b'    pretrain_weights: PathLikeStr | None = "rf-detr-nano.pth"\n'
            ),
            "src/rfdetr/detr.py": (
                b"if self.model_config.pretrain_weights is None:\n"
                b"            return\n"
            ),
            "src/rfdetr/models/backbone/dinov2.py": (
                b"if patch_size != 14:\n"
                b"    load_dinov2_weights = False\n"
                b"else WindowedDinov2WithRegistersBackbone(windowed_dino_config)\n"
            ),
            "src/rfdetr/models/lwdetr.py": (
                b"load_dinov2_weights=args.pretrain_weights is None,\n"
            ),
        }
        admission._require_static_semantics(files)
        files["src/rfdetr/detr.py"] = b"download everything"
        with self.assertRaisesRegex(RuntimeError, "weightless checkpoint early return"):
            admission._require_static_semantics(files)


if __name__ == "__main__":
    unittest.main()
