"""Generated JSON/HLS/TS and fake transport only; no network or native code."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
import io
from itertools import permutations
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from analytics_lab import public_live_hls as live


SESSION = "489801022"
CHILD_URL = live.MASTER_URL.rsplit("/", 1)[0] + f"/chunklist_w{SESSION}.m3u8"
MASTER = ('#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-STREAM-INF:BANDWIDTH=202675,'
          f'CODECS="avc1.100.41",RESOLUTION=320x240\nchunklist_w{SESSION}.m3u8\n').encode()


def media(sequence=18233, *, session=SESSION, durations=(9.595, 10.662, 9.595), target=11):
    lines = ["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{target}",
             f"#EXT-X-MEDIA-SEQUENCE:{sequence}", "#EXT-X-DISCONTINUITY-SEQUENCE:0"]
    for index, duration in enumerate(durations):
        lines += [f"#EXTINF:{duration},", f"media_w{session}_{sequence + index}.ts"]
    return ("\n".join(lines) + "\n").encode()


def ts(packets=2):
    return (b"\x47" + bytes(187)) * packets


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class Response:
    def __init__(self, url, payload, clock):
        self.url, self.payload, self.clock = url, payload, clock
        self.headers = {"Content-Length": str(len(payload)), "Date": "Sun, 18 Oct 2026 12:00:00 GMT"}
        self.status, self.position, self.closed = 200, 0, False
        self.timeouts, self.read_tick, self.chunk_size = [], 0.0, 65536
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self))

    def settimeout(self, timeout):
        if self.closed:
            raise OSError("generated closed socket")
        self.timeouts.append(timeout)

    def read1(self, count):
        self.clock.value += self.read_tick
        block = self.payload[self.position:self.position + min(count, self.chunk_size)]
        self.position += len(block)
        if self.position == int(self.headers["Content-Length"]):
            self.closed = True
        return block

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class Fixture:
    def __init__(self):
        self.clock = Clock()
        self.requests, self.responses, self.sleeps = [], [], []
        camera = {"index": "88", "inService": "true",
                  "location": {"district": "1", "locationName": live.CAMERA_NAME},
                  "imageData": {"streamingVideoURL": live.MASTER_URL}}
        self.catalog = {"data": [{"cctv": camera}]}
        self.payloads = {live.CATALOG_URL: None, live.MASTER_URL: MASTER,
                         CHILD_URL: [media(), media(18234)]}
        for number in range(18233, 18236):
            self.payloads[live.MASTER_URL.rsplit("/", 1)[0] + f"/media_w{SESSION}_{number}.ts"] = ts()

    def opener(self, request, *, timeout):
        assert request.get_header("Cache-control") == "no-cache"
        url = request.full_url
        self.requests.append((url, timeout))
        payload = self.payloads[url]
        if url == live.CATALOG_URL and payload is None:
            payload = json.dumps(self.catalog).encode()
        if isinstance(payload, list):
            payload = payload.pop(0)
        response = Response(url, payload, self.clock)
        self.responses.append(response)
        return response

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock.value += seconds

    def utc_now(self):
        return datetime(2026, 10, 18, 12, tzinfo=timezone.utc) + timedelta(seconds=self.clock.value)

    def capture(self, directory, **kwargs):
        return live.capture_live_window(directory, opener=self.opener, clock=self.clock,
                                        sleep=self.sleep, utc_now=self.utc_now, **kwargs)


@contextmanager
def destination():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        directory = root / "run" / "capture"
        directory.mkdir(parents=True)
        with patch.dict(os.environ, {"RUNNER_TEMP": str(root)}):
            yield directory


class PublicLiveHLSTests(unittest.TestCase):
    def test_actual_publisher_master_and_semantic_attribute_order(self):
        self.assertEqual(len(MASTER), 126)
        self.assertEqual(live._master(MASTER), (CHILD_URL, SESSION))
        for attributes in permutations(("BANDWIDTH=202675", 'CODECS="avc1.100.41"', "RESOLUTION=320x240")):
            payload = ("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-STREAM-INF:" + ",".join(attributes)
                       + f"\nchunklist_w{SESSION}.m3u8\n").encode()
            self.assertEqual(live._master(payload), (CHILD_URL, SESSION))
        for suffix in (b",BANDWIDTH=202675", b",AUDIO=other"):
            payload = MASTER.replace(b"RESOLUTION=320x240\n", b"RESOLUTION=320x240" + suffix + b"\n")
            with self.assertRaises(RuntimeError): live._master(payload)
        with self.assertRaises(RuntimeError): live._master(MASTER.replace(b"202675", b"500001"))

    def test_complete_fixed_camera_scope_six_gets_and_two_verified_files(self):
        fixture = Fixture()
        with destination() as directory:
            window = fixture.capture(directory)
            segment_urls = list(fixture.payloads)[4:]
            self.assertEqual([url for url, _ in fixture.requests],
                             [live.CATALOG_URL, live.MASTER_URL, CHILD_URL, *segment_urls, CHILD_URL])
            self.assertEqual(fixture.sleeps, [15.0])
            self.assertEqual(len(window.segments), 2)
            for segment in window.segments:
                self.assertEqual(segment.path.parent, directory)
                self.assertEqual(segment.path.read_bytes(), ts())
                self.assertEqual(segment.sha256, hashlib.sha256(ts()).hexdigest())
                self.assertEqual(segment.size_bytes, len(ts()))
            self.assertEqual(sorted(directory.iterdir()), sorted(item.path for item in window.segments))
            metadata = window.metadata
            self.assertTrue(metadata["sequence_advanced"])
            self.assertEqual((metadata["initial_media_sequence"], metadata["advanced_media_sequence"]), (18233, 18234))
            self.assertEqual(metadata["source_id"], "caltrans-d1-cctv-88")
            self.assertIsNone(metadata["camera_capture_utc"])
            self.assertAlmostEqual(metadata["declared_duration_seconds"], 20.257)
            self.assertEqual(metadata["source_get_count"], 6)
            self.assertEqual(metadata["selection"], "latest_two_of_three_advertised_segments")
            self.assertEqual((metadata["advertised_segment_count"], metadata["selected_segment_count"]), (3, 2))
            self.assertEqual(metadata["selected_start_sequence"], 18234)
            for key in ("catalog", "master", "initial_child", "advanced_child"):
                self.assertEqual(metadata[key]["http_date"], "Sun, 18 Oct 2026 12:00:00 GMT")
                self.assertTrue(metadata[key]["received_utc"].endswith("Z"))
            self.assertNotIn(str(directory), json.dumps(metadata))
            self.assertTrue(all(0 < timeout <= 10 for _, timeout in fixture.requests))

    def test_expired_oldest_is_never_requested_and_selected_failure_has_no_fallback(self):
        with destination() as directory:
            fixture = Fixture()
            oldest = list(fixture.payloads)[3]
            del fixture.payloads[oldest]
            window = fixture.capture(directory)
            self.assertEqual(tuple(item.sequence for item in window.segments), (18234, 18235))
            self.assertNotIn(oldest, [url for url, _ in fixture.requests])
        with destination() as directory:
            fixture = Fixture()
            selected = list(fixture.payloads)[4]
            def opener(request, *, timeout):
                if request.full_url == selected:
                    fixture.requests.append((request.full_url, timeout))
                    raise HTTPError(selected, 404, "generated unavailable selected segment", {}, None)
                return fixture.opener(request, timeout=timeout)
            with self.assertRaises(HTTPError):
                live.capture_live_window(directory, opener=opener, clock=fixture.clock, sleep=fixture.sleep)
            self.assertEqual([url for url, _ in fixture.requests],
                             [live.CATALOG_URL, live.MASTER_URL, CHILD_URL, selected])
            self.assertFalse(list(directory.iterdir()))

    def test_catalog_binding_duplicate_and_out_of_service_fail_before_playlist(self):
        for problem in ("index", "service", "district", "name", "url", "duplicate", "duplicate_key"):
            with self.subTest(problem=problem), destination() as directory:
                fixture = Fixture()
                row = fixture.catalog["data"][0]["cctv"]
                if problem == "index": row["index"] = "89"
                elif problem == "service": row["inService"] = "false"
                elif problem == "district": row["location"]["district"] = "2"
                elif problem == "name": row["location"]["locationName"] = "different camera"
                elif problem == "url": row["imageData"]["streamingVideoURL"] += "?token=x"
                elif problem == "duplicate": fixture.catalog["data"].append({"cctv": dict(row)})
                else: fixture.payloads[live.CATALOG_URL] = b'{"data":[],"data":[]}'
                with self.assertRaises(RuntimeError): fixture.capture(directory)
                self.assertEqual(len(fixture.requests), 1)
                self.assertEqual(list(directory.iterdir()), [])

    def test_master_rejects_multiple_variants_changed_profile_and_unsafe_child(self):
        for payload in (MASTER + MASTER.splitlines()[-1] + b"\n", MASTER.replace(b"320x240", b"640x480"),
                        MASTER.replace(b"chunklist_w", b"../chunklist_w"),
                        MASTER.replace(b".m3u8", b".m3u8?x=1"),
                        MASTER.replace(b"chunklist_w", b"https://other.example/chunklist_w")):
            with self.subTest(payload=payload), destination() as directory:
                fixture = Fixture(); fixture.payloads[live.MASTER_URL] = payload
                with self.assertRaises(RuntimeError): fixture.capture(directory)
                self.assertEqual(len(fixture.requests), 2)
                self.assertEqual(list(directory.iterdir()), [])

    def test_media_rejects_keys_map_ranges_endlist_and_discontinuities(self):
        tags = ('#EXT-X-KEY:METHOD=AES-128,URI="key"', '#EXT-X-MAP:URI="init"',
                "#EXT-X-BYTERANGE:188@0", "#EXT-X-ENDLIST", "#EXT-X-DISCONTINUITY",
                "#EXT-X-DISCONTINUITY-SEQUENCE:1", "#EXT-X-MEDIA-SEQUENCE:18233")
        for tag in tags:
            with self.subTest(tag=tag), destination() as directory:
                fixture = Fixture()
                fixture.payloads[CHILD_URL][0] = media().replace(b"#EXTINF:", (tag + "\n#EXTINF:").encode(), 1)
                with self.assertRaises(RuntimeError): fixture.capture(directory)
                self.assertEqual(len(fixture.requests), 3)
                self.assertFalse(list(directory.iterdir()))

    def test_media_sequence_session_count_and_duration_bounds(self):
        invalid = (media(durations=(9, 9)), media(durations=(9, 9, 9, 9)),
                   media().replace(b"_18234.ts", b"_18237.ts"),
                   media().replace(b"media_w489801022", b"media_w123"),
                   media().replace(b"media_w", b"../media_w"),
                   media().replace(b"media_w", b"%2fmedia_w"),
                   media(durations=(12, 12, 12), target=12), media(durations=(0, 9, 9)),
                   media().replace(b"9.595,", b"NaN,"),
                   media().replace(b"#EXT-X-MEDIA-SEQUENCE:18233", b"#EXT-X-MEDIA-SEQUENCE:9999999999999999999"))
        for payload in invalid:
            with self.subTest(payload=payload), destination() as directory:
                fixture = Fixture(); fixture.payloads[CHILD_URL][0] = payload
                with self.assertRaises(RuntimeError): fixture.capture(directory)
                self.assertEqual(len(fixture.requests), 3)
                self.assertFalse(list(directory.iterdir()))

    def test_observed_version_four_media_preserves_all_existing_scope_checks(self):
        fixture = media().replace(b"#EXT-X-VERSION:3", b"#EXT-X-VERSION:4")
        self.assertEqual(live._media(fixture, SESSION), live._media(media(), SESSION))
        with destination() as directory:
            source = Fixture()
            source.payloads[CHILD_URL] = [fixture, media(18234).replace(b"#EXT-X-VERSION:3", b"#EXT-X-VERSION:4")]
            self.assertEqual(len(source.capture(directory).segments), 2)
        for version in (0, 1, 2, 5, 6, 99):
            with self.subTest(version=version), self.assertRaises(RuntimeError):
                live._media(media().replace(b"#EXT-X-VERSION:3", f"#EXT-X-VERSION:{version}".encode()), SESSION)
        for tag in (b"#EXT-X-KEY:METHOD=NONE", b"#EXT-X-BYTERANGE:188@0", b"#EXT-X-DISCONTINUITY"):
            with self.subTest(tag=tag), self.assertRaises(RuntimeError):
                live._media(fixture.replace(b"#EXTINF:", tag + b"\n#EXTINF:", 1), SESSION)

    def test_transport_sync_packet_alignment_and_both_byte_caps(self):
        for problem in ("alignment", "second_sync", "individual_cap", "total_cap"):
            with self.subTest(problem=problem), destination() as directory:
                fixture = Fixture()
                urls = list(fixture.payloads)[4:]
                if problem == "alignment": fixture.payloads[urls[0]] = ts() + b"x"
                elif problem == "second_sync": fixture.payloads[urls[0]] = ts()[:188] + b"x" + bytes(187)
                elif problem == "individual_cap": fixture.payloads[urls[0]] = ts(live._SEGMENT_CAP // 188 + 1)
                else:
                    for url in urls: fixture.payloads[url] = ts(2)
                cap = 3 * 188 if problem == "total_cap" else live._TOTAL_SEGMENT_CAP
                with patch.object(live, "_TOTAL_SEGMENT_CAP", cap), self.assertRaises(RuntimeError):
                    fixture.capture(directory)
                self.assertFalse(list(directory.iterdir()))
                self.assertLessEqual(len(fixture.requests), 5)

    def test_stale_or_backwards_refresh_never_writes_a_segment(self):
        for sequence in (18233, 18232):
            with self.subTest(sequence=sequence), destination() as directory:
                fixture = Fixture(); fixture.payloads[CHILD_URL][1] = media(sequence)
                with self.assertRaisesRegex(RuntimeError, "did not advance"):
                    fixture.capture(directory)
                self.assertEqual(len(fixture.requests), 6)
                self.assertFalse(list(directory.iterdir()))

    def test_insufficient_deadline_prevents_refresh_and_files(self):
        with destination() as directory:
            fixture = Fixture()
            with self.assertRaises(TimeoutError):
                fixture.capture(directory, deadline=live.LiveDeadline(fixture.clock, 10))
            self.assertEqual(len(fixture.requests), 5)
            self.assertFalse(fixture.sleeps)
            self.assertFalse(list(directory.iterdir()))

    def test_fixed_length_stream_and_redirect_guards(self):
        url = live.MASTER_URL
        for problem in ("status", "redirect", "length", "missing", "chunked", "compressed", "short", "read1", "socket"):
            with self.subTest(problem=problem):
                clock = Clock(); response = Response(url, b"abc", clock)
                if problem == "status": response.status = 302
                elif problem == "redirect": response.url += "?redirect=1"
                elif problem == "length": response.headers["Content-Length"] = "4"
                elif problem == "missing": del response.headers["Content-Length"]
                elif problem == "chunked": response.headers["Transfer-Encoding"] = "chunked"
                elif problem == "compressed": response.headers["Content-Encoding"] = "gzip"
                elif problem == "short": response.payload = b"ab"
                elif problem == "read1": response.read1 = None
                else: response.fp = None
                with self.assertRaises(RuntimeError):
                    live.bounded_fetch(url, max_bytes=3, deadline=live.LiveDeadline(clock),
                                       allowed_urls=frozenset({url}), opener=lambda *args, **kwargs: response)
        for code in (301, 302, 303, 307, 308):
            with self.assertRaisesRegex(RuntimeError, "redirects"):
                live._NoRedirect().redirect_request(None, None, code, "redirect", {}, "https://other.example/")

    def test_exact_url_allowlist_rejects_unsafe_urls_before_open(self):
        for url in (live.MASTER_URL + "?x=1", live.MASTER_URL + "#fragment",
                    live.MASTER_URL.replace("https://", "https://user@"),
                    live.MASTER_URL.replace(".gov/", ".gov:443/"),
                    live.MASTER_URL.replace("playlist", "%70laylist"),
                    live.MASTER_URL.replace("playlist", "../playlist")):
            with self.subTest(url=url):
                with patch.object(live, "_open") as opener, self.assertRaises(ValueError):
                    live.bounded_fetch(url, max_bytes=3, deadline=live.LiveDeadline(), allowed_urls=frozenset({url}))
                opener.assert_not_called()

    def test_partial_body_progress_shrinks_timeout_and_exhausts_shared_budget(self):
        clock = Clock(); deadline = live.LiveDeadline(clock); clock.value = 170
        response = Response(live.MASTER_URL, bytes(20), clock)
        response.chunk_size, response.read_tick = 7, 4
        with self.assertRaises(TimeoutError):
            live.bounded_fetch(live.MASTER_URL, max_bytes=20, deadline=deadline,
                               allowed_urls=frozenset({live.MASTER_URL}), opener=lambda *args, **kwargs: response)
        self.assertEqual(response.timeouts, [10, 6, 2])

    def test_existing_or_noncanonical_directory_is_rejected_before_fetch(self):
        with destination() as directory:
            sentinel = directory / "keep"; sentinel.write_bytes(b"unrelated")
            fixture = Fixture()
            with self.assertRaises(ValueError): fixture.capture(directory)
            self.assertEqual(sentinel.read_bytes(), b"unrelated")
            self.assertFalse(fixture.requests)

    def test_late_file_failure_removes_only_created_segments(self):
        with destination() as directory:
            original = Path.open
            def failing_open(path, *args, **kwargs):
                if path.name == "segment-18235.ts": raise OSError("generated write failure")
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", failing_open), self.assertRaises(OSError):
                Fixture().capture(directory)
            self.assertFalse(list(directory.iterdir()))

    def test_real_http_response_closes_socket_at_exact_body_end(self):
        class MemorySocket:
            closed = False
            def makefile(self, mode): return io.BufferedReader(MemoryRaw(self))
            def settimeout(self, value):
                if self.closed: raise OSError("closed socket")
        class MemoryRaw(io.RawIOBase):
            def __init__(self, sock):
                self._sock = sock
                self.stream = io.BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc")
            def readable(self): return True
            def readinto(self, buffer):
                value = self.stream.read(len(buffer)); buffer[:len(value)] = value; return len(value)
            def close(self): self._sock.closed = True; super().close()
        sock = MemorySocket()
        response = http.client.HTTPResponse(sock, method="GET", url=live.MASTER_URL)
        response.begin(); response.url = live.MASTER_URL
        fetched = live.bounded_fetch(live.MASTER_URL, max_bytes=3, deadline=live.LiveDeadline(),
                                     allowed_urls=frozenset({live.MASTER_URL}), opener=lambda *args, **kwargs: response)
        self.assertEqual(fetched.payload, b"abc")
        self.assertTrue(sock.closed)


if __name__ == "__main__":
    unittest.main()
