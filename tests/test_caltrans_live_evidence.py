"""Generated, offline live-evidence regressions; no media or native runtime."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from analytics_lab import caltrans_live_evidence as evidence
from analytics_lab.artifacts import ArtifactSpec
from analytics_lab.public_live_hls import CAMERA_NAME
from analytics_lab.tracking import DetectionCandidate, NormalizedBox
from analytics_lab.video import Frame


class CaltransLiveEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="caltrans-generated-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.work = self.root / "evidence"
        self.steps = []
        self.segments = []
        self.detector = SimpleNamespace(runtime_version="2026.3.1")

    def _capture(self, directory, *args, **kwargs):
        self.steps.append("capture")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.segments = []
        for sequence in (100, 101, 102):
            payload = f"generated-segment-metadata-only:{sequence}".encode("ascii")
            path = directory / f"generated-{sequence}.ts"
            path.write_bytes(payload)
            self.segments.append(SimpleNamespace(
                sequence=sequence, duration_seconds=10.0, path=path,
                sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload),
            ))
        return SimpleNamespace(segments=tuple(self.segments), metadata={
            "sequence_advanced": True, "camera_capture_utc": None,
            "source_id": "caltrans-d1-cctv-88",
            "camera_name": CAMERA_NAME,
        })

    def _acquire_models(self, directory, *args, **kwargs):
        self.steps.append("models")
        root = Path(directory) / "models"
        root.mkdir()
        identities = []
        for suffix in ("xml", "bin"):
            payload = f"generated-model-identity-only:{suffix}".encode("ascii")
            (root / ("generated." + suffix)).write_bytes(payload)
            identities.append({"relative_path": "generated." + suffix,
                               "size_bytes": len(payload),
                               "sha256": hashlib.sha256(payload).hexdigest()})
        return identities

    def _detector_factory(self, directory, *args, **kwargs):
        self.steps.append("detector")
        return self.detector

    def _process(self, segment, detector, *, deadline, decoded_budget, sample_budget):
        self.assertIs(detector, self.detector)
        self.steps.append(segment.sequence)
        self.assertGreater(decoded_budget, 0)
        self.assertGreater(sample_budget, 0)
        return {
            "sequence": segment.sequence, "source_sha256": segment.sha256,
            "source_bytes": segment.size_bytes,
            "declared_duration_seconds": segment.duration_seconds,
            "decoded_frames": 800, "sampled_frames": 30,
            "detections": 1, "track_observations": 1,
            "within_segment_unique_ids": ["segment-local-1"],
            "decoded_span_ms": [0, 10000], "samples": [],
            "decoded_bgr24_chain_sha256": "a" * 64,
        }

    def _run(self, *, capture=None, models=None, process=None, phase=None,
             factory=None, clock=None):
        def forbidden_network(*args, **kwargs):
            self.fail("Offline fixture attempted network acquisition")

        with ExitStack() as stack:
            stack.enter_context(patch.dict(evidence.os.environ, {"RUNNER_TEMP": str(self.root)}))
            self.hosted = stack.enter_context(patch.object(evidence, "_require_hosted_linux"))
            stack.enter_context(patch.object(evidence, "_runtime_versions", return_value={
                "opencv": "4.12.0", "numpy": "2.2.6", "openvino": "2026.3.1",
            }))
            self.capture_call = stack.enter_context(patch.object(
                evidence, "_capture", side_effect=capture or self._capture))
            self.models_call = stack.enter_context(patch.object(
                evidence, "_acquire_models", side_effect=models or self._acquire_models))
            self.factory_call = stack.enter_context(patch.object(
                evidence, "_detector_factory", side_effect=factory or self._detector_factory))
            self.process_call = stack.enter_context(patch.object(
                evidence, "_process_segment", side_effect=process or self._process))
            if phase is not None:
                stack.enter_context(patch.object(evidence, "_phase", side_effect=phase))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return evidence.run(self.work, opener=forbidden_network, clock=clock or (lambda: 0.0))

    def test_capture_precedes_single_model_preparation_and_one_detector_for_three_segments(self):
        report = self._run()
        self.assertEqual(self.steps, ["capture", "models", "detector", 100, 101, 102])
        self.capture_call.assert_called_once()
        self.models_call.assert_called_once()
        self.factory_call.assert_called_once()
        self.assertEqual(self.process_call.call_count, 3)
        self.assertEqual(
            [(call.kwargs["decoded_budget"], call.kwargs["sample_budget"])
             for call in self.process_call.call_args_list],
            [(2400, 100), (1600, 70), (800, 40)],
        )
        self.assertFalse(self.work.exists())
        serialized = json.dumps(report, allow_nan=False, sort_keys=True)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("generated-segment-metadata-only:", serialized)
        self.assertNotIn("generated-model-identity-only:", serialized)
        self.assertEqual((report["decoded_frames"], report["sampled_frames"]), (2400, 90))
        self.assertEqual((report["detections"], report["track_observations"]), (3, 3))
        self.assertEqual(report["detector"]["instances"], 1)
        self.assertEqual(report["tracking"]["sessions"], 3)
        self.assertFalse(report["tracking"]["cross_segment_association"])
        self.assertFalse(report["tracking"]["identity_claim"])
        for key in ("precision", "recall", "false_positive_rate", "misses", "camera_capture_utc"):
            self.assertIsNone(report[key], key)
        for key in ("independent_labels", "positive_qualified", "negative_qualified", "accuracy_gate_passed",
                    "accuracy_claim", "performance_claim", "training", "recognition", "deployment",
                    "retained_media", "exported_media"):
            self.assertIs(report[key], False, key)
        self.assertTrue(report["execution_complete"])
        self.assertTrue(report["temporary_assets_removed"])
        deadline = self.capture_call.call_args.kwargs["deadline"]
        self.assertIs(self.models_call.call_args.args[1], deadline)
        for call in self.process_call.call_args_list:
            self.assertIs(call.kwargs["deadline"], deadline)

    def test_capture_freshness_and_unknown_camera_timestamp_are_required_before_models(self):
        for key, value in (("sequence_advanced", False), ("camera_capture_utc", "2026-10-18T00:00:00Z")):
            with self.subTest(key=key):
                def capture(*args, **kwargs):
                    window = self._capture(*args, **kwargs)
                    window.metadata[key] = value
                    return window
                with self.assertRaises((RuntimeError, ValueError)):
                    self._run(capture=capture)
                self.models_call.assert_not_called()
                self.factory_call.assert_not_called()
                self.process_call.assert_not_called()
                self.assertFalse(self.work.exists())

    def test_capture_must_return_exactly_three_segments(self):
        for count in (0, 2, 4):
            with self.subTest(count=count):
                def capture(*args, **kwargs):
                    window = self._capture(*args, **kwargs)
                    segments = window.segments[:count]
                    if count == 4:
                        segments += (window.segments[-1],)
                    window.segments = segments
                    return window
                with self.assertRaises((RuntimeError, ValueError)):
                    self._run(capture=capture)
                self.process_call.assert_not_called()
                self.assertFalse(self.work.exists())

    def test_camera_identity_sequence_and_duration_scope_are_required(self):
        for invalid in ("source", "sequence", "duration"):
            with self.subTest(invalid=invalid):
                def capture(*args, **kwargs):
                    window = self._capture(*args, **kwargs)
                    if invalid == "source":
                        window.metadata["source_id"] = "some-other-camera"
                    elif invalid == "sequence":
                        window.segments[1].sequence += 1
                    else:
                        for segment in window.segments:
                            segment.duration_seconds = 12.0
                    return window
                with self.assertRaises((RuntimeError, ValueError)):
                    self._run(capture=capture)
                self.models_call.assert_not_called()
                self.assertFalse(self.work.exists())

    def test_captured_segment_hash_is_reverified_before_processing(self):
        def models(*args, **kwargs):
            result = self._acquire_models(*args, **kwargs)
            self.segments[0].path.write_bytes(b"changed-generated-bytes")
            return result
        with self.assertRaises((RuntimeError, ValueError)):
            self._run(models=models)
        self.process_call.assert_not_called()
        self.assertFalse(self.work.exists())

    def test_segment_path_outside_owned_directory_is_rejected_without_removing_outside_file(self):
        outside = self.root / "outside-generated-segment.ts"
        outside.write_bytes(b"outside sentinel")
        def capture(*args, **kwargs):
            window = self._capture(*args, **kwargs)
            window.segments[0].path = outside
            window.segments[0].size_bytes = outside.stat().st_size
            window.segments[0].sha256 = hashlib.sha256(outside.read_bytes()).hexdigest()
            return window
        with self.assertRaises((RuntimeError, ValueError)):
            self._run(capture=capture)
        self.process_call.assert_not_called()
        self.assertEqual(outside.read_bytes(), b"outside sentinel")
        self.assertFalse(self.work.exists())

    def test_aggregate_decoded_and_sampled_budgets_cannot_be_exceeded(self):
        for key, value in (("decoded_frames", 801), ("sampled_frames", 34)):
            with self.subTest(key=key):
                def process(*args, **kwargs):
                    result = self._process(*args, **kwargs)
                    result[key] = value
                    return result
                with self.assertRaises((RuntimeError, ValueError)):
                    self._run(process=process)
                self.assertFalse(self.work.exists())

    def test_every_preparation_and_processing_failure_cleans_owned_assets(self):
        sentinel = self.root / "unowned.txt"
        sentinel.write_text("keep", encoding="utf-8")
        for phase in ("capture", "models", "process"):
            with self.subTest(phase=phase):
                def capture(*args, **kwargs):
                    result = self._capture(*args, **kwargs)
                    if phase == "capture":
                        raise RuntimeError("generated capture failure")
                    return result
                def models(*args, **kwargs):
                    result = self._acquire_models(*args, **kwargs)
                    if phase == "models":
                        raise RuntimeError("generated model preparation failure")
                    return result
                def process(*args, **kwargs):
                    if phase == "process":
                        raise RuntimeError("generated processing failure")
                    return self._process(*args, **kwargs)
                with self.assertRaises(RuntimeError):
                    self._run(capture=capture, models=models, process=process)
                self.assertFalse(self.work.exists())
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_compile_and_cleanup_output_failures_still_remove_assets(self):
        def factory(*args, **kwargs):
            raise RuntimeError("generated compile failure")
        with self.assertRaisesRegex(RuntimeError, "compile failure"):
            self._run(factory=factory)
        self.assertFalse(self.work.exists())
        self.process_call.assert_not_called()

        def phase(name, **kwargs):
            if name == "cleanup":
                raise BrokenPipeError("generated output failure")
        with self.assertRaises(BrokenPipeError):
            self._run(phase=phase)
        self.assertEqual(self.process_call.call_count, 3)
        self.assertFalse(self.work.exists())

    def test_capture_phase_timeout_stops_before_model_acquisition_and_cleans(self):
        now = [0.0]
        def capture(*args, **kwargs):
            window = self._capture(*args, **kwargs)
            now[0] = 76.0
            return window
        with self.assertRaises(TimeoutError):
            self._run(capture=capture, clock=lambda: now[0])
        self.models_call.assert_not_called()
        self.assertFalse(self.work.exists())

    def test_hosted_guard_requires_the_exact_linux_repository_pr_lane(self):
        valid = {"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "github-hosted",
                 "RUNNER_OS": "Linux", "GITHUB_REPOSITORY": "mkurtgerald/Analytics-lab",
                 "GITHUB_EVENT_NAME": "pull_request",
                 "GITHUB_HEAD_REF": "evidence/public-live-caltrans-generated"}
        with patch.dict(evidence.os.environ, valid, clear=True), \
                patch.object(evidence.platform, "system", return_value="Linux"):
            evidence._require_hosted_linux()
            for key in valid:
                with self.subTest(key=key), patch.dict(evidence.os.environ, {key: "wrong"}):
                    with self.assertRaises(RuntimeError):
                        evidence._require_hosted_linux()
        with patch.dict(evidence.os.environ, valid, clear=True), \
                patch.object(evidence.platform, "system", return_value="Windows"):
            with self.assertRaises(RuntimeError):
                evidence._require_hosted_linux()

    def test_work_directory_must_be_a_new_direct_child_and_preserves_existing_files(self):
        existing = self.root / "existing"
        existing.mkdir()
        sentinel = existing / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        nested = self.root / "nested"
        nested.mkdir()
        with patch.dict(evidence.os.environ, {"RUNNER_TEMP": str(self.root)}):
            self.assertEqual(evidence._work_directory(self.work), self.work)
            for invalid in (self.root, existing, nested / "child", self.root.parent / "outside"):
                with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                    evidence._work_directory(invalid)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


class _GeneratedImage:
    def __init__(self, *, shape=(240, 320, 3), dtype="uint8", payload=None):
        self.shape, self.dtype = shape, dtype
        self.payload = payload if payload is not None else bytes(320 * 240 * 3)

    def tobytes(self, order="C"):
        if order != "C":
            raise AssertionError("generated image expects C-order BGR bytes")
        return self.payload


class CaltransSegmentProcessingTests(unittest.TestCase):
    def setUp(self):
        self.segment = SimpleNamespace(sequence=100, duration_seconds=1.0,
                                       path=Path("generated.ts"), sha256="b" * 64, size_bytes=188)
        self.deadline = SimpleNamespace(check=lambda: None)
        self.candidate = DetectionCandidate("vehicle", .9, NormalizedBox(.1, .1, .6, .6), 1)

    def _process(self, frames, *, detect=None, decoded_budget=2400, sample_budget=100, fps=25.0):
        frames = tuple(frames)
        class Source:
            def __enter__(inner):
                return source
            def __exit__(inner, *args):
                return False
        class IterableSource:
            _fps = fps
            def __iter__(inner):
                return iter(frames)
        source = IterableSource()
        detector = SimpleNamespace(detect=detect or (lambda image: (self.candidate,)))
        with patch.object(evidence, "OpenCVVideoFileSource", return_value=Source()) as factory:
            result = evidence._process_segment(self.segment, detector, deadline=self.deadline,
                                              decoded_budget=decoded_budget, sample_budget=sample_budget)
        factory.assert_called_once_with(self.segment.path, start_timestamp_ms=0,
                                        max_file_bytes=4 * 1024 * 1024, max_fps=60.0)
        return result

    def test_real_decoder_loop_samples_and_hashes_bytes_with_fresh_tracking_per_segment(self):
        frames = [Frame(index, stamp, _GeneratedImage(payload=bytes([index]) * (320 * 240 * 3)))
                  for index, stamp in enumerate((0, 400, 800))]
        chain = hashlib.sha256()
        for frame in frames:
            digest = hashlib.sha256(frame.image.payload).digest()
            chain.update(frame.index.to_bytes(8, "big") + frame.timestamp_ms.to_bytes(8, "big") + digest)
        rows = [self._process(frames) for _ in range(3)]
        for row in rows:
            self.assertEqual((row["decoded_frames"], row["sampled_frames"]), (3, 3))
            self.assertEqual(row["within_segment_unique_ids"], ["bt-000001"])
            self.assertEqual(row["track_observations"], 3)
            self.assertEqual(row["decoded_span_ms"], [0, 800])
            self.assertEqual([sample["segment_relative_ms"] for sample in row["samples"]], [0, 400, 800])
            self.assertEqual(row["decoded_bgr24_chain_sha256"], chain.hexdigest())
            self.assertIs(row["decode_eof_is_complete_media_proof"], False)

    def test_sampling_period_skips_inference_but_hashes_every_decoded_frame(self):
        seen = []
        def detect(image):
            seen.append(image)
            return ()
        frames = [Frame(index, stamp, _GeneratedImage())
                  for index, stamp in enumerate((0, 40, 359, 360, 400, 720))]
        result = self._process(frames, detect=detect)
        self.assertEqual((result["decoded_frames"], result["sampled_frames"]), (6, 3))
        self.assertEqual(len(seen), 3)
        self.assertEqual([sample["frame_index"] for sample in result["samples"]], [0, 3, 5])

    def test_sparse_native_indices_use_dense_tracker_updates_for_unconfirmed_activation(self):
        frames = [Frame(index, stamp, _GeneratedImage())
                  for index, stamp in ((0, 0), (10, 400), (20, 800))]
        outputs = iter(((), (self.candidate,), (self.candidate,)))
        result = self._process(frames, detect=lambda image: next(outputs))
        self.assertEqual([sample["frame_index"] for sample in result["samples"]], [0, 10, 20])
        self.assertEqual([sample["track_observations"] for sample in result["samples"]], [0, 0, 1])
        self.assertEqual([sample["tracking_update_index"] for sample in result["samples"]], [0, 1, 2])
        self.assertEqual(result["within_segment_unique_ids"], ["bt-000001"])
        self.assertEqual(result["tracking_frame_basis"], "sample ordinal")

    def test_invalid_image_shape_dtype_or_byte_count_is_rejected_before_inference(self):
        for image in (_GeneratedImage(shape=(480, 640, 3)), _GeneratedImage(dtype="float32"),
                      _GeneratedImage(payload=b"short")):
            with self.subTest(shape=image.shape, dtype=image.dtype, size=len(image.payload)):
                def forbidden(image):
                    self.fail("Invalid generated frame reached inference")
                with self.assertRaises(RuntimeError):
                    self._process([Frame(0, 0, image)], detect=forbidden)

    def test_empty_decoder_and_local_decoded_or_sampled_overflow_reject(self):
        with self.assertRaises(RuntimeError):
            self._process([])
        frames = [Frame(index, stamp, _GeneratedImage()) for index, stamp in enumerate((0, 400))]
        for limits in ({"decoded_budget": 1}, {"sample_budget": 1}):
            with self.subTest(limits=limits), self.assertRaises(RuntimeError):
                self._process(frames, **limits)

    def test_decoder_reported_fps_must_be_finite_positive_and_at_most_sixty(self):
        for fps in (None, True, 0, -1, 60.01, float("nan"), float("inf")):
            with self.subTest(fps=fps), self.assertRaises(RuntimeError):
                self._process([Frame(0, 0, _GeneratedImage())], fps=fps)

    def test_detector_outputs_are_vehicle_only_bounded_tuples_and_cannot_mutate_input(self):
        wrong = DetectionCandidate("plate", .9, self.candidate.box, 2)
        low = DetectionCandidate("vehicle", .49, self.candidate.box, 1)
        for output in ([self.candidate], (wrong,), (low,), (self.candidate,) * 65):
            with self.subTest(output_type=type(output), count=len(output)), self.assertRaises(RuntimeError):
                self._process([Frame(0, 0, _GeneratedImage())], detect=lambda image: output)
        def mutate(image):
            image.payload = b"x" * len(image.payload)
            return (self.candidate,)
        with self.assertRaises(RuntimeError):
            self._process([Frame(0, 0, _GeneratedImage())], detect=mutate)

    def test_real_video_boundary_rejects_excessive_fps_and_releases_capture_without_native_runtime(self):
        temporary = tempfile.TemporaryDirectory(prefix="caltrans-generated-fps-")
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name).resolve() / "generated.ts"
        path.write_bytes(b"generated-not-media")
        released = []
        capture = SimpleNamespace(isOpened=lambda: True, get=lambda key: 61.0,
                                  release=lambda: released.append(True))
        fake_cv2 = SimpleNamespace(VideoCapture=lambda value: capture, CAP_PROP_FPS=1)
        self.segment.path = path
        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            with self.assertRaises(ValueError):
                evidence._process_segment(self.segment, SimpleNamespace(detect=lambda image: ()),
                                          deadline=self.deadline, decoded_budget=2400, sample_budget=100)
        self.assertEqual(released, [True])


class CaltransModelAcquisitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="caltrans-generated-model-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.deadline = SimpleNamespace(check=lambda: None)
        self.payloads = {"https://generated.invalid/generated.xml": b"x" * 806112,
                         "https://generated.invalid/generated.bin": b"b" * 806112}
        self.specs = tuple(ArtifactSpec(
            component="generated-offline-fixture", relative_path=url.rsplit("/", 1)[1],
            size_bytes=len(payload), sha384=hashlib.sha384(payload).hexdigest(),
            source_url=url, license_id="generated-test-only",
            license_url="https://generated.invalid/fixture-license",
        ) for url, payload in self.payloads.items())

    def test_exact_two_model_fetches_use_shared_deadline_and_allowlist_then_verify_local_bytes(self):
        from analytics_lab import lpr_ocr
        calls = []
        opener = object()
        def fetch(url, **kwargs):
            calls.append((url, kwargs))
            return SimpleNamespace(payload=self.payloads[url])
        with patch.object(lpr_ocr, "omz_vehicle_artifacts", return_value=self.specs), \
                patch.object(evidence, "bounded_fetch", side_effect=fetch):
            identities = evidence._acquire_models(self.root, self.deadline, opener)
        self.assertEqual([url for url, kwargs in calls], list(self.payloads))
        self.assertEqual(len(identities), 2)
        for spec, (url, kwargs), identity in zip(self.specs, calls, identities):
            self.assertIs(kwargs["deadline"], self.deadline)
            self.assertIs(kwargs["opener"], opener)
            self.assertEqual(kwargs["allowed_urls"], frozenset(self.payloads))
            self.assertEqual(kwargs["max_bytes"], spec.size_bytes)
            self.assertEqual(identity["sha384"], spec.sha384)
            self.assertEqual((self.root / "models" / spec.relative_path).read_bytes(), self.payloads[url])

    def test_same_size_changed_model_bytes_fail_before_any_artifact_is_written(self):
        from analytics_lab import lpr_ocr
        with patch.object(lpr_ocr, "omz_vehicle_artifacts", return_value=self.specs), \
                patch.object(evidence, "bounded_fetch", return_value=SimpleNamespace(payload=b"z" * 806112)) as fetch:
            with self.assertRaises(RuntimeError):
                evidence._acquire_models(self.root, self.deadline, object())
        fetch.assert_called_once()
        self.assertEqual(list((self.root / "models").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
