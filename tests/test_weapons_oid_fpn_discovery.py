import gzip
import hashlib
import http.client
import io
import tarfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from analytics_lab import weapons_oid_fpn_discovery as discovery


CONFIG = b'''model { ssd {
 num_classes: 601
 image_resizer { keep_aspect_ratio_resizer {
  min_dimension: 512 max_dimension: 512 pad_to_max_dimension: true
 } }
} }
train_input_reader { input_path: "num_classes: 999 image_resizer { spoof }" }
# num_classes: 123
'''
OBSERVED_PAX = b"14 uid=345018\n27 mtime=1548178193.679631\n"


def pax_record(key, value):
    body = (key + "=" + value + "\n").encode("utf-8")
    length = len(body) + 2
    while len(str(length)) + 1 + len(body) != length:
        length = len(str(length)) + 1 + len(body)
    return str(length).encode() + b" " + body


def pax_control(payload):
    info = tarfile.TarInfo("PaxHeader/" + discovery._ROOT)
    info.type, info.size = tarfile.XHDTYPE, len(payload)
    return info.tobuf(format=tarfile.USTAR_FORMAT) + payload + bytes((-len(payload)) % 512)


def archive(entries=None):
    entries = entries if entries is not None else (
        (discovery._ROOT, None),
        (discovery._ROOT + "/frozen_inference_graph.pb", b"generated-graph-bytes"*50),
        (discovery._ROOT + "/pipeline.config", CONFIG),
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as result:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE if content is None else tarfile.REGTYPE
            info.size = len(content) if content is not None else 0
            result.addfile(info, io.BytesIO(content) if content is not None else None)
    return buffer.getvalue()


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Response:
    def __init__(self, payload, *, clock=None, step=0, declared=None):
        self.payload = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload)) if declared is None else declared}
        self.status = 200
        self.url = discovery._ARCHIVE_URL
        self.timeouts, self.read_sizes = [], []
        self.closed = False
        self.clock, self.step = clock, step

    def geturl(self):
        return self.url

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def read(self, size):
        self.read_sizes.append(size)
        if size < 1 or size > discovery._CHUNK:
            raise AssertionError("HTTP reads must stay bounded")
        if self.clock is not None:
            self.clock.now += self.step
        return self.payload.read(size)

    def isclosed(self):
        return self.closed

    def read1(self, size):
        return self.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class DiscoveryTests(unittest.TestCase):
    def run_fixture(self, raw=None, *, compressed=None, changes=None, response=None, clock=None, opener_hook=None):
        raw = archive() if raw is None else raw
        compressed = gzip.compress(raw, mtime=0) if compressed is None else compressed
        response = response or Response(compressed, clock=clock)
        changes = {"_EXPECTED_SIZE": len(compressed), **(changes or {})}
        calls = []
        def opener(request, *, timeout):
            calls.append((request, timeout))
            if opener_hook:
                opener_hook()
            return response
        with ExitStack() as stack:
            for name, value in changes.items():
                stack.enter_context(patch.object(discovery, name, value))
            result = discovery.run(opener=opener, clock=clock or Clock())
        return result, response, calls

    def test_complete_archive_and_regular_hashes_without_execution(self):
        raw = archive()
        compressed = gzip.compress(raw, mtime=0)
        result, response, calls = self.run_fixture(raw, compressed=compressed)
        self.assertEqual(result["archive_size"], len(compressed))
        self.assertEqual(result["archive_sha256"], hashlib.sha256(compressed).hexdigest())
        self.assertEqual(result["expanded_size"], len(raw))
        self.assertEqual(result["member_count"], 3)
        self.assertEqual(result["archive_root"], discovery._ROOT)
        self.assertEqual(result["graph"]["sha256"], hashlib.sha256(b"generated-graph-bytes"*50).hexdigest())
        config = next(item for item in result["members"] if item["name"].endswith("pipeline.config"))
        self.assertEqual(config["sha256"], hashlib.sha256(CONFIG).hexdigest())
        self.assertEqual(result["pipeline"], {"num_classes": 601, "image_resizer": {
            "type": "keep_aspect_ratio_resizer", "min_dimension": 512, "max_dimension": 512,
            "pad_to_max_dimension": True}})
        self.assertEqual(result["oid_v4_label_map_blob"], "643b9e8ed5d9239a3248b895fb32f3b51caa92f3")
        for field in ("archive_identity_pinned", "artifact_admitted", "native_installed", "model_constructed",
                      "inference_run", "artifact_retained", "accuracy_claim", "readiness_claim"):
            self.assertIs(result[field], False)
        self.assertTrue(response.closed)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0].full_url, discovery._ARCHIVE_URL)
        self.assertTrue(all(0 < value <= 30 for value in response.timeouts))
        self.assertLessEqual(max(response.read_sizes), discovery._CHUNK)

    def test_full_compressed_hash_includes_gzip_trailer_and_zero_padding(self):
        raw = archive()
        compressed = gzip.compress(raw, mtime=0) + bytes(19)
        result, _, _ = self.run_fixture(raw, compressed=compressed)
        self.assertEqual(result["archive_sha256"], hashlib.sha256(compressed).hexdigest())
        self.assertEqual(result["archive_size"], len(compressed))

    def test_fixed_resizer_metadata(self):
        config = b"model { ssd { num_classes: 601 image_resizer { fixed_shape_resizer { height: 512 width: 512 } } } }"
        result, _, _ = self.run_fixture(archive(((discovery._ROOT + "/frozen_inference_graph.pb", b"g"),
                                                (discovery._ROOT + "/pipeline.config", config))))
        self.assertEqual(result["pipeline"]["image_resizer"], {"type": "fixed_shape_resizer", "height": 512, "width": 512})

    def test_exact_observed_uid_mtime_control_before_root_directory(self):
        self.assertEqual(len(OBSERVED_PAX), 41)
        raw = pax_control(OBSERVED_PAX) + archive()
        result, _, _ = self.run_fixture(raw)
        self.assertEqual(result["member_count"], 4)
        self.assertEqual(result["archive_root"], discovery._ROOT)
        control = result["members"][0]
        self.assertEqual(control["type"], "pax_metadata")
        self.assertEqual(control["name"], "PaxHeader/" + discovery._ROOT)
        self.assertEqual(control["size"], 41)
        self.assertEqual(control["sha256"], hashlib.sha256(OBSERVED_PAX).hexdigest())
        self.assertEqual(control["payload_utf8"], OBSERVED_PAX.decode())
        self.assertEqual(control["fields"], {"uid": "345018", "mtime": "1548178193.679631"})
        self.assertFalse(control["metadata_applied"])
        self.assertEqual(control["following_member"], discovery._ROOT)
        self.assertEqual(result["archive_sha256"], hashlib.sha256(gzip.compress(raw, mtime=0)).hexdigest())

    def test_pax_whitelist_and_utf8_byte_length(self):
        payload = b"".join(pax_record(key, value) for key, value in
                           (("uid", "1"), ("gid", "2"), ("uname", "Müller"), ("gname", "group"),
                            ("mtime", "1.1"), ("atime", "2.2"), ("ctime", "3.3")))
        result, _, _ = self.run_fixture(pax_control(payload) + archive())
        self.assertEqual(result["members"][0]["fields"]["uname"], "Müller")
        self.assertEqual(set(result["members"][0]["fields"]), discovery._PAX_KEYS)

    def test_pax_override_sparse_and_unknown_keys_rejected(self):
        for key in ("path", "size", "linkpath", "GNU.sparse.map", "GNU.sparse.size", "SCHILY.filetype", "comment"):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.run_fixture(pax_control(pax_record(key, "1")) + archive())

    def test_pax_malformed_framing_duplicates_and_control_values_rejected(self):
        malformed = (b"", b"0 uid=x\n", b"01 uid=x\n", b"10000 uid=x\n", b"8 uid x\n", b"7 uid=\n",
                     OBSERVED_PAX[:-1], OBSERVED_PAX.replace(b"14 uid", b"15 uid", 1),
                     pax_record("uid", "1") + pax_record("uid", "2"),
                     pax_record("uname", "x\ny"), pax_record("uname", "x\ty"), pax_record("uname", "x\x7fy"),
                     pax_record("uname", "x\x85y"), b"9 uname=\xff\n")
        for payload in malformed:
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                self.run_fixture(pax_control(payload) + archive())

    def test_pax_consecutive_orphan_and_nonzero_padding_rejected(self):
        control = pax_control(OBSERVED_PAX)
        for raw in (control + control + archive(), control + bytes(1024)):
            with self.subTest(length=len(raw)), self.assertRaises(RuntimeError):
                self.run_fixture(raw)
        bad_padding = bytearray(control)
        bad_padding[512 + len(OBSERVED_PAX)] = 1
        with self.assertRaises(RuntimeError):
            self.run_fixture(bytes(bad_padding) + archive())

    def test_pax_header_total_and_physical_member_caps(self):
        raw_archive = archive()
        first, second = pax_control(OBSERVED_PAX), pax_control(pax_record("gid", "123"))
        raw = first + raw_archive[:512] + second + raw_archive[512:]
        total = len(OBSERVED_PAX) + len(pax_record("gid", "123"))
        result, _, _ = self.run_fixture(raw, changes={"_MAX_PAX": len(OBSERVED_PAX), "_MAX_PAX_TOTAL": total, "_MAX_MEMBERS": 5})
        self.assertEqual(result["member_count"], 5)
        self.assertEqual(result["members"][2]["following_member"], discovery._ROOT + "/frozen_inference_graph.pb")
        for changes in ({"_MAX_PAX": len(OBSERVED_PAX)-1}, {"_MAX_PAX_TOTAL": total-1}, {"_MAX_MEMBERS": 4}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                self.run_fixture(raw, changes=changes)

    def test_observed_wrapper_can_differ_from_archive_filename(self):
        root = discovery._WRAPPER_PREFIX + "_oid_v4"
        raw = archive(((root, None), (root + "/frozen_inference_graph.pb", b"generated-g"),
                       (root + "/pipeline.config", CONFIG), (root + "/saved_model", None),
                       (root + "/saved_model/saved_model.pb", b"generated-s")))
        result, _, calls = self.run_fixture(raw)
        self.assertEqual(result["archive_root"], root)
        self.assertEqual(result["graph"]["name"], root + "/frozen_inference_graph.pb")
        self.assertEqual(calls[0][0].full_url, discovery._ARCHIVE_URL)
        self.assertEqual(result["archive_sha256"], hashlib.sha256(gzip.compress(raw, mtime=0)).hexdigest())

    def test_one_leading_dot_slash_and_container_preserve_original_names(self):
        root = discovery._WRAPPER_PREFIX + "_oid_v4"
        result, _, _ = self.run_fixture(archive(((".", None), ("./" + root, None),
                                                ("./" + root + "/frozen_inference_graph.pb", b"g"),
                                                ("./" + root + "/pipeline.config", CONFIG))))
        self.assertEqual(result["archive_root"], root)
        self.assertEqual(result["members"][0]["name"], ".")
        self.assertEqual(result["members"][0]["original_name"], "./")
        self.assertEqual(result["graph"]["original_name"], "./" + root + "/frozen_inference_graph.pb")
        self.assertEqual(result["graph"]["name"], root + "/frozen_inference_graph.pb")
        self.assertEqual(result["members"][1]["original_name"], "./" + root + "/")

    def test_flat_root_and_dot_container(self):
        result, _, _ = self.run_fixture(archive(((".", None), ("./frozen_inference_graph.pb", b"g"),
                                                ("pipeline.config", CONFIG), ("saved_model", None),
                                                ("saved_model/saved_model.pb", b"s"))))
        self.assertEqual(result["archive_root"], ".")
        self.assertEqual(result["graph"]["name"], "frozen_inference_graph.pb")
        self.assertEqual(result["graph"]["original_name"], "./frozen_inference_graph.pb")

    def test_normalized_duplicates_and_double_dot_prefix_rejected(self):
        root = discovery._WRAPPER_PREFIX + "_oid_v4"
        cases = (((root + "/weights", b"a"), ("./" + root + "/weights", b"b")),
                 ((root, None), ("./" + root, None)),
                 (("././" + root + "/weights", b"a"),), ((root + "//", None),))
        for entries in cases:
            with self.subTest(entries=entries), self.assertRaises(RuntimeError):
                self.run_fixture(archive(entries))

    def test_mixed_outside_unapproved_or_nested_roots_and_ambiguous_pairs_rejected(self):
        root = discovery._WRAPPER_PREFIX + "_oid_v4"
        other = discovery._WRAPPER_PREFIX + "_other"
        graph = (root + "/frozen_inference_graph.pb", b"g")
        config = (root + "/pipeline.config", CONFIG)
        cases = ((graph, (other + "/pipeline.config", CONFIG)),
                 (graph, config, ("outside/weights", b"w")),
                 ((root + "/nested/frozen_inference_graph.pb", b"g"), (root + "/nested/pipeline.config", CONFIG)),
                 (("unapproved/frozen_inference_graph.pb", b"g"), ("unapproved/pipeline.config", CONFIG)),
                 (graph, config, (root + "/nested/frozen_inference_graph.pb", b"second-g")),
                 (graph, config, (root + "/nested/pipeline.config", CONFIG)))
        for entries in cases:
            with self.subTest(entries=entries), self.assertRaises(RuntimeError):
                self.run_fixture(archive(entries))

    def test_each_byte_and_member_ceiling_fails_closed(self):
        raw = archive()
        compressed = gzip.compress(raw, mtime=0)
        for changes in ({"_MAX_COMPRESSED": len(compressed)-1}, {"_MAX_EXPANDED": 1024},
                        {"_MAX_REGULAR": 999}, {"_MAX_CONFIG": len(CONFIG)-1}, {"_MAX_MEMBERS": 2}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                self.run_fixture(raw, compressed=compressed, changes=changes)
        result, _, _ = self.run_fixture(raw, compressed=compressed, changes={
            "_MAX_COMPRESSED": len(compressed), "_MAX_EXPANDED": len(raw), "_MAX_MEMBERS": 3,
            "_MAX_REGULAR": 1050, "_MAX_CONFIG": len(CONFIG)})
        self.assertEqual(result["member_count"], 3)
        response = Response(compressed)
        response.headers.clear()
        with self.assertRaises(RuntimeError):
            self.run_fixture(raw, compressed=compressed, response=response,
                             changes={"_MAX_COMPRESSED": len(compressed)-1})
        self.assertTrue(response.read_sizes)

    def test_unknown_control_headers_rejected_before_payload_allocation(self):
        for member_type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME,
                            tarfile.GNUTYPE_LONGLINK, tarfile.GNUTYPE_SPARSE, tarfile.SYMTYPE,
                            tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, b"9"):
            info = tarfile.TarInfo(discovery._ROOT + "/control")
            info.type, info.size = member_type, 699_999_999
            class HeaderOnly:
                def exact(self, size):
                    if size != 512:
                        raise AssertionError("control payload was accessed")
                    return info.tobuf(format=tarfile.GNU_FORMAT)
                def read(self, size):
                    raise AssertionError("control payload was read")
            with self.subTest(member_type=member_type), self.assertRaises(RuntimeError):
                discovery._members(HeaderOnly())

    def test_paths_duplicates_and_regular_link_metadata_rejected(self):
        names = ("/absolute", discovery._ROOT + "/../escape", discovery._ROOT + "/./dot",
                 discovery._ROOT + "//double", discovery._ROOT + "/back\\slash",
                 discovery._ROOT + "/newline\nfile", discovery._ROOT + "x/file")
        for name in names:
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.run_fixture(archive(((name, b"x"),)))
        with self.assertRaises(RuntimeError):
            self.run_fixture(archive(((discovery._ROOT + "/same", b"a"), (discovery._ROOT + "/same", b"b"))))
        info = tarfile.TarInfo(discovery._ROOT + "/linked")
        info.linkname = discovery._ROOT + "/target"
        with self.assertRaises(RuntimeError):
            discovery._safe_header(info.tobuf(format=tarfile.USTAR_FORMAT))

    def test_bad_header_checksum_and_utf8_rejected(self):
        raw = bytearray(archive())
        raw[0] ^= 1
        with self.assertRaises(RuntimeError):
            self.run_fixture(bytes(raw))
        raw = bytearray(archive())
        raw[0] = 255
        raw[148:156] = b" "*8
        raw[148:156] = (f"{sum(raw[:512]):06o}\0 ").encode()
        with self.assertRaises(RuntimeError):
            self.run_fixture(bytes(raw))

    def test_appended_tar_nonzero_padding_and_truncation_rejected(self):
        raw = archive()
        for malformed in (raw + archive(), raw + b"hidden", raw[:400], raw[:512+512+20]):
            with self.subTest(length=len(malformed)), self.assertRaises(RuntimeError):
                self.run_fixture(malformed)
        # Replace the second zero terminator with a nonzero block.
        content_end = 512 + 512 + 1536 + 512 + 512
        self.assertEqual(raw[content_end:content_end+1024], bytes(1024))
        malformed = raw[:content_end+512] + b"x"*512 + raw[content_end+1024:]
        with self.assertRaises(RuntimeError):
            self.run_fixture(malformed)
        # Regular-member padding, separately from end-of-archive padding.
        malformed = bytearray(raw)
        malformed[512 + 512 + 1050] = 1
        with self.assertRaises(RuntimeError):
            self.run_fixture(bytes(malformed))

    def test_gzip_crc_and_truncated_stream_rejected(self):
        compressed = bytearray(gzip.compress(archive(), mtime=0))
        compressed[-8] ^= 1
        with self.assertRaises(RuntimeError):
            self.run_fixture(compressed=bytes(compressed))
        with self.assertRaises(RuntimeError):
            self.run_fixture(compressed=bytes(compressed[:-7]))

    def test_required_files_and_pipeline_fields_fail_closed(self):
        for entries in ((), ((discovery._ROOT + "/pipeline.config", CONFIG),),
                        ((discovery._ROOT + "/frozen_inference_graph.pb", b""),
                         (discovery._ROOT + "/pipeline.config", CONFIG))):
            with self.subTest(entries=entries), self.assertRaises(RuntimeError):
                self.run_fixture(archive(entries))
        for config in (CONFIG.replace(b"num_classes: 601", b"num_classes: 0"),
                       CONFIG.replace(b"num_classes: 601", b"num_classes: 601.0"),
                       b"\xff",
                       CONFIG.replace(b"num_classes: 601", b"num_classes: 601 num_classes: 602"),
                       CONFIG.replace(b"min_dimension: 512", b"min_dimension: -1"),
                       CONFIG.replace(b"keep_aspect_ratio_resizer", b"unknown_resizer"),
                       CONFIG.split(b"train_input_reader")[0].rstrip()[:-1]):
            with self.subTest(config=config), self.assertRaises(RuntimeError):
                self.run_fixture(archive(((discovery._ROOT + "/frozen_inference_graph.pb", b"g"),
                                          (discovery._ROOT + "/pipeline.config", config))))

    def test_exact_response_url_status_encoding_and_declared_size(self):
        compressed = gzip.compress(archive(), mtime=0)
        for attr, value in (("url", discovery._ARCHIVE_URL + "?redirected"), ("status", 206)):
            response = Response(compressed)
            setattr(response, attr, value)
            with self.subTest(attr=attr), self.assertRaises(RuntimeError):
                self.run_fixture(compressed=compressed, response=response)
            self.assertEqual(response.read_sizes, [])
        for declared in ("-1", "NaN", str(len(compressed)+1)):
            with self.subTest(declared=declared), self.assertRaises(RuntimeError):
                self.run_fixture(compressed=compressed, response=Response(compressed, declared=declared))
        response = Response(compressed)
        response.headers["Content-Encoding"] = "gzip"
        with self.assertRaises(RuntimeError):
            self.run_fixture(compressed=compressed, response=response)

    def test_missing_content_length_and_actual_size_mismatch(self):
        compressed = gzip.compress(archive(), mtime=0)
        response = Response(compressed)
        response.headers.clear()
        result, _, _ = self.run_fixture(compressed=compressed, response=response)
        self.assertEqual(result["archive_size"], len(compressed))
        response = Response(compressed)
        response.headers.clear()
        with self.assertRaises(RuntimeError):
            self.run_fixture(compressed=compressed, response=response, changes={"_EXPECTED_SIZE": len(compressed)+1})

    def test_redirect_rejected_before_forwarding(self):
        with self.assertRaises(RuntimeError):
            discovery._NoRedirect().redirect_request(None, None, 302, "redirect", {},
                                                     "https://example.invalid/escaped")

    def test_deadline_and_remaining_read_timeout(self):
        compressed = gzip.compress(archive(), mtime=0)
        clock = Clock()
        response = Response(compressed, clock=clock, step=210)
        with self.assertRaises(TimeoutError):
            self.run_fixture(compressed=compressed, response=response, clock=clock)
        self.assertTrue(response.closed)
        clock = Clock()
        response = Response(compressed, clock=clock)
        _, response, calls = self.run_fixture(compressed=compressed, response=response, clock=clock,
                                             opener_hook=lambda: setattr(clock, "now", 200))
        self.assertEqual(calls[0][1], 30)
        self.assertTrue(all(0 < value <= 10 for value in response.timeouts))
        # One byte arrives per receive: read1 keeps the overall deadline active
        # while preserving the exact two-byte gzip header contract.
        response = Response(compressed)
        original = response.read
        response.read1 = lambda size: original(min(size, 1))
        result, _, _ = self.run_fixture(compressed=compressed, response=response)
        self.assertEqual(result["archive_size"], len(compressed))
        clock = Clock()
        response = Response(compressed, clock=clock, step=29)
        original = response.read
        response.read1 = lambda size: original(min(size, 1))
        with self.assertRaises(TimeoutError):
            self.run_fixture(compressed=compressed, response=response, clock=clock)
        self.assertTrue(response.closed)

    def test_malformed_reads_and_timeout_control_fail_closed(self):
        compressed = gzip.compress(archive(), mtime=0)
        for result in (bytearray(b"xx"), b"x"*100_000):
            response = Response(compressed)
            response.read = lambda size, value=result: value
            with self.subTest(type=type(result)), self.assertRaises(RuntimeError):
                self.run_fixture(compressed=compressed, response=response)
        response = Response(compressed)
        response.settimeout = None
        with self.assertRaises(RuntimeError):
            self.run_fixture(compressed=compressed, response=response)

    def test_stdlib_http_response_socket_timeout_and_read1_lifecycle(self):
        compressed = gzip.compress(archive(), mtime=0)
        wire = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(compressed)).encode() + b"\r\n\r\n" + compressed
        class MemorySocket:
            def __init__(self):
                self.timeouts = []
            def settimeout(self, timeout):
                self.timeouts.append(timeout)
            def makefile(self, mode):
                return io.BufferedReader(SocketReader(self))
        class SocketReader(io.RawIOBase):
            def __init__(self, socket):
                self._sock, self.body = socket, io.BytesIO(wire)
            def readable(self):
                return True
            def readinto(self, view):
                data = self.body.read(len(view))
                view[:len(data)] = data
                return len(data)
        socket = MemorySocket()
        response = http.client.HTTPResponse(socket, method="GET")
        response.begin()
        response.url = discovery._ARCHIVE_URL
        result, _, _ = self.run_fixture(compressed=compressed, response=response)
        self.assertEqual(result["archive_sha256"], hashlib.sha256(compressed).hexdigest())
        self.assertTrue(socket.timeouts)
        self.assertTrue(all(0 < value <= 30 for value in socket.timeouts))


if __name__ == "__main__":
    unittest.main()
