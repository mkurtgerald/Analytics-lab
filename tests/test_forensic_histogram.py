"""Actual generated pixels, not synthetic vectors or detector-accuracy evidence."""
from dataclasses import asdict, replace
import hashlib
import json
import math
import unittest
from unittest.mock import patch

from analytics_lab import forensic_histogram
from analytics_lab.forensic_appearance import rank_appearance_similarity
from analytics_lab.forensic_handoff import envelope_for_appearance_matches, parse_envelope_json
from analytics_lab.forensic_histogram import (
    EXTRACTION_CONFIG_SHA256, HistogramFrameBinding, extract_bgr24_histogram, extraction_config_json,
)
from analytics_lab.forensic_search import ForensicAttribute, ForensicEvidenceLink, ForensicRecord
from analytics_lab.tracking import NormalizedBox

DETECTOR_CONFIG = json.dumps({"schema": "generated-box-fixture.v1", "detector": "none"},
                             sort_keys=True, separators=(",", ":")).encode()
DETECTOR_CONFIG_SHA = hashlib.sha256(DETECTOR_CONFIG).hexdigest()
RED, GREEN, BLUE, BLACK = bytes((0, 0, 255)), bytes((0, 255, 0)), bytes((255, 0, 0)), bytes((0, 0, 0))


def fixture(name="candidate", *, source="generated_camera", width=8, height=8,
            color=RED, padding=b"", bbox=(0.0, 0.0, 1.0, 1.0), pixels=None):
    stride = width * 3 + len(padding)
    pixels = pixels if pixels is not None else (color * width + padding) * height
    # These are explicit fixture facts for this exact buffer and ROI. No other
    # event is relabeled, no detector is run, and no production authority exists.
    evidence = ForensicEvidenceLink(name + "_event", "generated_box_fixture", "1", DETECTOR_CONFIG_SHA,
                                   source_revision="project-authored-generated-fixture-v1")
    record = ForensicRecord(name, source, 1791201600000, "person", 0.9,
                           box=NormalizedBox(*bbox), track_id="generated_session:1",
                           attributes=(ForensicAttribute("fixture", "generated", 1.0, "fixture_v1"),),
                           evidence=evidence)
    frame = HistogramFrameBinding(source, evidence.event_id, record.timestamp_ms, 7, width, height,
                                  stride, hashlib.sha256(pixels).hexdigest(), bbox)
    return pixels, record, frame


def extract(values):
    pixels, record, frame = values
    return extract_bgr24_histogram(pixels, record=record, frame=frame)


class ForensicHistogramTests(unittest.TestCase):
    def test_fixed_bgr_rgb_bin_semantics_use_real_pixels(self):
        for color, expected in ((RED, 48), (GREEN, 12), (BLUE, 3), (BLACK, 0), (b"\xff\xff\xff", 63)):
            with self.subTest(color=color):
                result = extract(fixture(color=color))
                self.assertEqual(result.descriptor.values, tuple(float(i == expected) for i in range(64)))
                self.assertEqual(result.sample_count, 64)
                self.assertEqual(result.crop_bounds, (0, 0, 8, 8))

    def test_quantization_boundaries_and_channels_are_fixed(self):
        pixels = b"".join(bytes((0, 0, value)) for value in (0, 63, 64, 127, 128, 191, 192, 255))
        result = extract(fixture(width=8, height=1, pixels=pixels))
        self.assertEqual([i for i, value in enumerate(result.descriptor.values) if value], [0, 16, 32, 48])
        self.assertEqual([value for value in result.descriptor.values if value], [0.25] * 4)

    def test_floor_ceil_crop_is_exact_and_preserves_detector_bbox(self):
        row = RED * 2 + BLUE * 2
        values = fixture(width=4, height=2, pixels=row * 2, bbox=(0.26, 0.1, 0.51, 0.9))
        result = extract(values)
        self.assertEqual(result.crop_bounds, (1, 0, 3, 2))
        self.assertEqual(result.sample_count, 4)
        self.assertEqual(result.descriptor.values[48], 0.5)
        self.assertEqual(result.descriptor.values[3], 0.5)
        self.assertEqual(result.descriptor.record.box, values[1].box)

    def test_padding_is_not_sampled_but_is_part_of_buffer_identity(self):
        values = fixture(width=3, height=2, padding=BLUE * 2)
        result = extract(values)
        self.assertEqual(result.descriptor.values[48], 1.0)
        self.assertEqual(result.descriptor.values[3], 0.0)
        altered = values[0][:-1] + b"\x01"
        with self.assertRaisesRegex(ValueError, "frame identity"):
            extract_bgr24_histogram(altered, record=values[1], frame=values[2])

    def test_adjacent_float_roi_rounding_empty_on_either_axis_fails_closed(self):
        left = 1 / 3
        right = math.nextafter(left, 1.0)
        self.assertLess(left, right)
        self.assertEqual(left * 3, right * 3)
        for bbox in ((left, 0, right, 1), (0, left, 1, right), (left, left, right, right)):
            with self.subTest(bbox=bbox), self.assertRaisesRegex(ValueError, "empty pixel crop"):
                extract(fixture(width=3, height=3, bbox=bbox))

    def test_large_roi_is_deterministically_capped_at_4096_samples(self):
        # In a 128x128 crop, cell-centred sampling selects odd x/y coordinates.
        row = b"".join(BLUE if x % 2 else RED for x in range(128))
        values = fixture(width=128, height=128, pixels=row * 128)
        result = extract(values)
        self.assertEqual(result.sample_count, 4096)
        self.assertEqual(result.descriptor.values[3], 1.0)
        self.assertEqual(result, extract(values))

    def test_config_and_original_provenance_are_separate_and_not_overwritten(self):
        values = fixture()
        record = replace(values[1], evidence=replace(values[1].evidence,
            model_sha256=hashlib.sha256(b"inert generated model-identity test bytes; never executed").hexdigest()))
        result = extract((values[0], record, values[2]))
        self.assertEqual(asdict(result.descriptor.record), asdict(record))
        self.assertEqual(result.descriptor.record.evidence.config_sha256, DETECTOR_CONFIG_SHA)
        self.assertNotEqual(DETECTOR_CONFIG_SHA, result.extraction_config_sha256)
        self.assertEqual(hashlib.sha256(extraction_config_json().encode("ascii")).hexdigest(),
                         EXTRACTION_CONFIG_SHA256)
        self.assertEqual(result.extraction_config_sha256, EXTRACTION_CONFIG_SHA256)

    def test_real_pixels_rank_and_roundtrip_without_losing_source_evidence(self):
        probe = extract(fixture("probe", source="generated_probe_camera"))
        same = extract(fixture("same", source="generated_candidate_camera"))
        different = extract(fixture("different", color=BLUE))
        matches = rank_appearance_similarity(probe.descriptor_for_ranking(),
            (different.descriptor_for_ranking(), same.descriptor_for_ranking()), min_similarity=0.5)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].similarity, 1.0)
        self.assertEqual(matches[0].candidate, same.descriptor.record)
        raw = envelope_for_appearance_matches(matches).to_json()
        envelope = parse_envelope_json(raw)
        self.assertEqual(envelope.to_json(), raw)
        self.assertEqual(envelope.payload["matches"][0]["hit"]["evidence"], asdict(same.descriptor.record.evidence))
        self.assertFalse(envelope.payload["identity_claim"] or envelope.payload["authorizes_action"])

    def test_layout_dimension_length_and_mutable_buffer_rejections(self):
        pixels, record, frame = fixture()
        for changes in ({"width": 0}, {"width": True}, {"width": 8193}, {"height": -1},
                        {"height": 1.0}, {"stride_bytes": 23}, {"stride_bytes": False},
                        {"width": 8192, "height": 8192, "stride_bytes": 24576},
                        {"frame_index": -1}, {"frame_index": 2**63}, {"timestamp_ms": True},
                        {"frame_sha256": "placeholder"}, {"frame_sha256": "A" * 64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(frame, **changes)
        for invalid in (pixels[:-1], pixels + b"\0", bytearray(pixels), memoryview(pixels), None):
            with self.subTest(kind=type(invalid).__name__), self.assertRaises(ValueError):
                extract_bgr24_histogram(invalid, record=record, frame=frame)

    def test_bbox_requires_finite_exact_normalized_positive_area(self):
        _, _, frame = fixture()
        for box in ((0, 0, float("nan"), 1), (0, 0, float("inf"), 1), (False, 0, 1, 1),
                    (-0.1, 0, 1, 1), (0, 0, 1.1, 1), (0.5, 0, 0.5, 1),
                    (0.7, 0, 0.5, 1), (0, 0, 1, 10**400), [0, 0, 1, 1], (0, 0, 1)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                replace(frame, bbox=box)

    def test_altered_color_source_event_time_bbox_or_stride_cannot_use_old_binding(self):
        pixels, record, frame = fixture()
        with self.assertRaisesRegex(ValueError, "frame identity"):
            extract_bgr24_histogram(BLUE * 64, record=record, frame=frame)
        for changed in (replace(frame, source_id="other_camera"), replace(frame, event_id="other_event"),
                        replace(frame, timestamp_ms=frame.timestamp_ms + 1),
                        replace(frame, bbox=(0, 0, 0.5, 1)), replace(frame, stride_bytes=27)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                extract_bgr24_histogram(pixels, record=record, frame=changed)

    def test_forged_result_fields_cannot_reuse_the_original_metadata_digest(self):
        result = extract(fixture())
        for fields in ({"extraction_config_sha256": "0" * 64}, {"crop_bounds": (0, 0, 4, 8)},
                       {"crop_bounds": (False, 0, 8, 8)}, {"sample_count": 1}, {"sample_count": True},
                       {"binding_sha256": "0" * 64}, {"frame": replace(result.frame, frame_index=8)},
                       {"descriptor": replace(result.descriptor, record=replace(result.descriptor.record,
                                              observation_id="forged_observation"))}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(result, **fields)

    def test_original_objects_and_returned_descriptor_cannot_mutate_the_retained_result(self):
        pixels, record, frame = fixture()
        result = extract_bgr24_histogram(pixels, record=record, frame=frame)
        original = result.binding_sha256
        object.__setattr__(frame, "source_id", "changed")
        object.__setattr__(record.evidence, "config_sha256", "0" * 64)
        copy = result.descriptor_for_ranking()
        object.__setattr__(copy.record.box, "x_max", 0.5)
        self.assertEqual(result.binding_sha256, original)
        self.assertEqual(result.descriptor_for_ranking().record.evidence.config_sha256, DETECTOR_CONFIG_SHA)
        object.__setattr__(result.descriptor.record.evidence, "config_sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "metadata binding"):
            result.descriptor_for_ranking()

    def test_missing_evidence_bbox_or_forbidden_category_is_not_invented(self):
        pixels, record, frame = fixture()
        for changed in (replace(record, evidence=None), replace(record, box=None),
                        replace(record, category="face"), replace(record, category="license_plate")):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                extract_bgr24_histogram(pixels, record=changed, frame=frame)

    def test_overbound_mutated_attributes_are_rejected_before_cloning(self):
        _, record, _ = fixture()
        object.__setattr__(record, "attributes", record.attributes * 10000)
        with patch.object(forensic_histogram, "replace", wraps=replace) as clone:
            with self.assertRaises(ValueError):
                forensic_histogram._copy_record(record)
        self.assertEqual(clone.call_count, 0)

    def test_oversized_mutated_text_is_rejected_before_normalization(self):
        for field in ("category", "name", "value"):
            _, record, _ = fixture()
            target = record if field == "category" else record.attributes[0]
            object.__setattr__(target, field, " " * 1000000 + "generated")
            with self.subTest(field=field), patch.object(forensic_histogram, "replace", wraps=replace) as clone:
                with self.assertRaisesRegex(ValueError, "text bound"):
                    forensic_histogram._copy_record(record)
            self.assertEqual(clone.call_count, 0)

    def test_result_contains_metadata_only(self):
        result = extract(fixture())
        def inspect(value):
            self.assertNotIsInstance(value, (bytes, bytearray, memoryview))
            if isinstance(value, dict):
                for item in value.values(): inspect(item)
            elif isinstance(value, (tuple, list)):
                for item in value: inspect(item)
        inspect(asdict(result))


if __name__ == "__main__":
    unittest.main()
