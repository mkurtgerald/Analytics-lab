import hashlib
import unittest
from unittest import mock

from analytics_lab import face_donor_facessd_rewrite_probe as probe


class FaceSSDRewriteProbeTests(unittest.TestCase):
    def test_git_blob_identity_matches_git_object_format(self):
        payload = b"abc"
        expected = hashlib.sha1(b"blob 3\0abc").hexdigest()
        self.assertEqual(probe._git_blob_sha1(payload), expected)

    def test_helper_download_rejects_wrong_size_before_blob_identity(self):
        with mock.patch("urllib.request.urlopen") as urlopen:
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.geturl.return_value = (
                "https://raw.githubusercontent.com/opencv/opencv/"
                + probe._OPENCV_COMMIT
                + "/samples/dnn/tf_text_graph_ssd.py"
            )
            response.read.return_value = b"x"
            urlopen.return_value = response
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                probe._download_helper(
                    "tf_text_graph_ssd.py",
                    2,
                    "0" * 40,
                )

    def test_helper_specs_are_exactly_pinned(self):
        self.assertEqual(probe._OPENCV_COMMIT, "49486f61fb25722cbcf586b7f4320921d46fb38e")
        self.assertEqual(
            probe._HELPERS,
            (
                ("tf_text_graph_ssd.py", 18314, "d27fd0d384f509789bf64f069203d3f9964db576"),
                ("tf_text_graph_common.py", 10055, "c82053b4fb05aa2fd40c45b1cefa550431608847"),
            ),
        )

    def test_network_result_preserves_handled_import_failure(self):
        class Handled(Exception):
            pass
        def loader(*_args):
            raise Handled()
        result = probe._network_result(loader, None, None, Handled)
        self.assertFalse(result["loaded"])
        self.assertEqual(result["result"], "opencv_error")


if __name__ == "__main__":
    unittest.main()
