"""Bounded model-free BGR24 crop histograms with separate extraction provenance.

Only caller-supplied immutable pixels are read. No media acquisition, detector,
storage or authorization is performed. Frame/event association is a trusted
caller fact, not something a histogram can authenticate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import re

from .forensic_appearance import AppearanceDescriptor
from .forensic_search import ForensicAttribute, ForensicEvidenceLink, ForensicRecord, _timestamp, _token
from .tracking import NormalizedBox

_MAX_FRAME_BYTES = 16 * 1024 * 1024
_MAX_DIMENSION = 8192
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG_JSON = json.dumps({
    "schema": "analytics.bgr24-histogram-extraction.v1",
    "input": "BGR24-uint8-top-down", "bins_per_rgb_channel": 4,
    "bin_index": "16*(R>>6)+4*(G>>6)+(B>>6)",
    "crop": "normalized-xyxy-floor-left-top-ceil-right-bottom-exclusive",
    "sampling": "cell-center:origin+((2*i+1)*extent)//(2*min(extent,64))",
    "max_samples_per_axis": 64, "normalization": "count/sample-count",
}, sort_keys=True, separators=(",", ":"))
EXTRACTION_CONFIG_SHA256 = hashlib.sha256(_CONFIG_JSON.encode("ascii")).hexdigest()


def extraction_config_json() -> str:
    """Return the exact non-secret configuration bytes hashed by this producer."""
    return _CONFIG_JSON


def _integer(value: object, minimum: int, maximum: int, name: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside the supported integer bound")
    return value


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} requires lowercase SHA-256")
    return value


@dataclass(frozen=True)
class HistogramFrameBinding:
    """Independently supplied source-event, frame, layout and selected ROI facts.

    The digest covers the entire declared buffer including row padding. A caller
    must obtain this binding from the same authorized frame/event operation,
    never from a result payload or a different event with a similar identifier.
    """

    source_id: str
    event_id: str
    timestamp_ms: int
    frame_index: int
    width: int
    height: int
    stride_bytes: int
    frame_sha256: str
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _token(self.source_id, "frame source_id")
        _token(self.event_id, "frame event_id")
        _timestamp(self.timestamp_ms, "frame timestamp_ms")
        _integer(self.frame_index, 0, 2**63 - 1, "frame_index")
        _integer(self.width, 1, _MAX_DIMENSION, "width")
        _integer(self.height, 1, _MAX_DIMENSION, "height")
        _integer(self.stride_bytes, self.width * 3, _MAX_FRAME_BYTES, "stride_bytes")
        if self.height * self.stride_bytes > _MAX_FRAME_BYTES:
            raise ValueError("frame buffer exceeds the supported byte bound")
        _digest(self.frame_sha256, "frame identity")
        if type(self.bbox) is not tuple or len(self.bbox) != 4:
            raise ValueError("bbox requires an immutable normalized xyxy tuple")
        for value in self.bbox:
            if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError("bbox coordinates must be finite normalized numbers")
        if not (self.bbox[0] < self.bbox[2] and self.bbox[1] < self.bbox[3]):
            raise ValueError("bbox must have positive area")
        object.__setattr__(self, "bbox", tuple(float(value) for value in self.bbox))


def _crop(frame: HistogramFrameBinding) -> tuple[int, int, int, int]:
    left, top, right, bottom = frame.bbox
    crop = (math.floor(left * frame.width), math.floor(top * frame.height),
            math.ceil(right * frame.width), math.ceil(bottom * frame.height))
    # Distinct normalized floats can multiply to the same integer pixel edge.
    # Preserve the declared rounding rule; never widen an empty crop silently.
    if crop[0] >= crop[2] or crop[1] >= crop[3]:
        raise ValueError("normalized bbox rounds to an empty pixel crop")
    return crop


def _copy_record(record: ForensicRecord) -> ForensicRecord:
    if (type(record) is not ForensicRecord or type(record.evidence) is not ForensicEvidenceLink
            or type(record.box) is not NormalizedBox or type(record.attributes) is not tuple
            or len(record.attributes) > 64
            or any(type(item) is not ForensicAttribute for item in record.attributes)):
        raise ValueError("histogram extraction requires an evidence-linked ForensicRecord")
    # Already-admitted records are normalized. Bound mutated text before the
    # constructors' whitespace normalization can scan or copy an oversized value.
    texts = (record.category, *(value for item in record.attributes for value in (item.name, item.value)))
    if any(type(value) is not str or len(value) > 128 for value in texts):
        raise ValueError("histogram record metadata exceeds the supported text bound")
    try:
        copied = replace(record, evidence=replace(record.evidence), box=replace(record.box),
                         attributes=tuple(replace(item) for item in record.attributes))
    except (TypeError, OverflowError, AttributeError) as exc:
        raise ValueError("invalid histogram record metadata") from exc
    if copied.category in {"face", "license_plate"}:
        raise ValueError("histogram extraction category is not permitted")
    return copied


def _bind_record(record: ForensicRecord, frame: HistogramFrameBinding) -> None:
    if (record.source_id != frame.source_id or record.timestamp_ms != frame.timestamp_ms
            or record.evidence.event_id != frame.event_id
            or (record.box.x_min, record.box.y_min, record.box.x_max, record.box.y_max) != frame.bbox):
        raise ValueError("frame must bind the exact record source, time, evidence event and bbox")


def _binding_digest(descriptor: AppearanceDescriptor, frame: HistogramFrameBinding,
                    config_sha256: str, crop: tuple[int, int, int, int], count: int) -> str:
    value = {"schema": "analytics.histogram-extraction-binding.v1",
             "descriptor": asdict(descriptor), "frame": asdict(frame),
             "extraction_config_sha256": config_sha256, "crop_bounds": crop, "sample_count": count}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HistogramExtraction:
    """Metadata-only extraction result; its digest detects inconsistent claims.

    It is not an authenticity signature. Original detector evidence is retained
    unchanged and its config/model hashes are never replaced by extractor hashes.
    """

    descriptor: AppearanceDescriptor
    frame: HistogramFrameBinding
    extraction_config_sha256: str
    crop_bounds: tuple[int, int, int, int]
    sample_count: int
    binding_sha256: str

    def __post_init__(self) -> None:
        if type(self.frame) is not HistogramFrameBinding or type(self.descriptor) is not AppearanceDescriptor:
            raise ValueError("typed histogram frame and descriptor required")
        object.__setattr__(self, "frame", replace(self.frame))
        object.__setattr__(self, "descriptor", replace(self.descriptor, record=_copy_record(self.descriptor.record)))
        self._validate()

    def _validate(self) -> None:
        if type(self.frame) is not HistogramFrameBinding or type(self.descriptor) is not AppearanceDescriptor:
            raise ValueError("typed histogram frame and descriptor required")
        replace(self.frame)
        descriptor = replace(self.descriptor, record=_copy_record(self.descriptor.record))
        _bind_record(descriptor.record, self.frame)
        if self.extraction_config_sha256 != EXTRACTION_CONFIG_SHA256:
            raise ValueError("unsupported histogram extraction configuration")
        crop = _crop(self.frame)
        if (type(self.crop_bounds) is not tuple or self.crop_bounds != crop
                or any(type(value) is not int for value in self.crop_bounds)):
            raise ValueError("crop bounds do not match the exact frame/ROI binding")
        expected_count = min(crop[2] - crop[0], 64) * min(crop[3] - crop[1], 64)
        _integer(self.sample_count, 1, 4096, "sample_count")
        if self.sample_count != expected_count or len(descriptor.values) != 64:
            raise ValueError("histogram sample count or dimension is inconsistent")
        _digest(self.binding_sha256, "extraction binding")
        if self.binding_sha256 != _binding_digest(descriptor, self.frame, self.extraction_config_sha256,
                                                  self.crop_bounds, self.sample_count):
            raise ValueError("histogram extraction metadata binding changed")

    def descriptor_for_ranking(self) -> AppearanceDescriptor:
        """Revalidate metadata integrity and return an independent descriptor."""
        self._validate()
        return replace(self.descriptor, record=_copy_record(self.descriptor.record))


def extract_bgr24_histogram(pixels: bytes, *, record: ForensicRecord,
                            frame: HistogramFrameBinding) -> HistogramExtraction:
    """Extract up to 4096 deterministic pixel samples from one immutable ROI.

    Reads at most 16 MiB once for full-buffer identity, then at most 4096 BGR
    triplets for the histogram. Mutable buffers are rejected, avoiding changes
    between identity verification and sampling. No input pixels are retained.
    """
    if type(frame) is not HistogramFrameBinding:
        raise ValueError("typed histogram frame binding required")
    frame = replace(frame)
    record = _copy_record(record)
    _bind_record(record, frame)
    if type(pixels) is not bytes or len(pixels) != frame.height * frame.stride_bytes:
        raise ValueError("immutable pixel bytes must match the exact frame layout")
    if hashlib.sha256(pixels).hexdigest() != frame.frame_sha256:
        raise ValueError("pixel buffer does not match the supplied frame identity")
    crop = _crop(frame)
    left, top, right, bottom = crop
    width, height = right - left, bottom - top
    columns, rows = min(width, 64), min(height, 64)
    bins = [0] * 64
    for row in range(rows):
        y = top + ((2 * row + 1) * height) // (2 * rows)
        for column in range(columns):
            x = left + ((2 * column + 1) * width) // (2 * columns)
            offset = y * frame.stride_bytes + x * 3
            b, g, r = pixels[offset:offset + 3]
            bins[16 * (r >> 6) + 4 * (g >> 6) + (b >> 6)] += 1
    count = columns * rows
    descriptor = AppearanceDescriptor("analytics.appearance-descriptor.v1", "model_free.color_histogram.v1",
                                      record, tuple(value / count for value in bins))
    binding = _binding_digest(descriptor, frame, EXTRACTION_CONFIG_SHA256, crop, count)
    return HistogramExtraction(descriptor, frame, EXTRACTION_CONFIG_SHA256, crop, count, binding)
