"""Select the next bounded Figshare generalization pair from archive metadata only.

This selector never downloads member payloads.  It operates only on the reviewed
central-directory map and chooses one difficult ADL negative plus one fall-class
positive that are disjoint from the first Figshare pair by subject and location.
The result is candidate metadata for a later exact admission step, not accuracy
evidence and not authorization to train on the clips.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .figshare_acquisition import ZipMember

_MEMBER_RE = re.compile(
    r"^VideoDataset/(?P<source_class>ADL|Fall)/"
    r"SBJ_(?P<subject>\d{2})_LOC(?P<location>\d+)/"
    r"ACT(?P<activity>\d+)_R_(?P<repetition>\d+)/[^/]+\.mp4$"
)

# The first admitted Figshare pair used Subject 01 / Location 3 for ACT25 and
# Subject 10 / Location 3 for ACT10.  The next evidence pair intentionally moves
# off both subjects and off Location 3 before any payload is admitted.
_EXCLUDED_SUBJECTS = frozenset({"01", "10"})
_EXCLUDED_LOCATIONS = frozenset({"3"})

# ACT19 (Laying) is a deliberately difficult ADL negative.  ACT4 (Fall on the
# back) is a distinct fall morphology from the first ACT10 sit-on-chair fall.
_NEGATIVE_ACTIVITY = 19
_POSITIVE_ACTIVITY = 4
_ACTIVITY_NAMES = {
    19: "Laying",
    4: "Fall on the back",
}


@dataclass(frozen=True)
class GeneralizationMember:
    """Parsed public archive metadata for one candidate video member."""

    role: str
    source_class: str
    subject_id: str
    location_id: str
    activity_code: int
    repetition: int
    member: ZipMember

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source_class": self.source_class,
            "subject_id": self.subject_id,
            "location_id": self.location_id,
            "activity_code": f"ACT{self.activity_code}",
            "activity_name": _ACTIVITY_NAMES[self.activity_code],
            "repetition": self.repetition,
            "name": self.member.name,
            "compressed_size": self.member.compressed_size,
            "uncompressed_size": self.member.uncompressed_size,
            "compression_method": self.member.compression_method,
            "crc32": f"{self.member.crc32:08x}",
            "local_header_offset": self.member.local_header_offset,
        }


def _candidate(member: ZipMember) -> GeneralizationMember | None:
    if not isinstance(member, ZipMember) or not member.is_bounded_video:
        return None
    match = _MEMBER_RE.fullmatch(member.name)
    if match is None:
        return None
    subject = match.group("subject")
    location = match.group("location")
    if subject in _EXCLUDED_SUBJECTS or location in _EXCLUDED_LOCATIONS:
        return None
    source_class = match.group("source_class")
    activity = int(match.group("activity"))
    if source_class == "ADL" and activity == _NEGATIVE_ACTIVITY:
        role = "negative"
    elif source_class == "Fall" and activity == _POSITIVE_ACTIVITY:
        role = "positive"
    else:
        return None
    return GeneralizationMember(
        role=role,
        source_class=source_class,
        subject_id=subject,
        location_id=location,
        activity_code=activity,
        repetition=int(match.group("repetition")),
        member=member,
    )


def select_next_generalization_pair(members: tuple[ZipMember, ...]) -> dict[str, Any]:
    """Return the smallest subject/location-disjoint ACT19/ACT4 metadata pair.

    Selection minimizes total uncompressed bytes, then maximum member size, then
    stable path order.  This keeps the later admission/evaluation cost bounded
    without tuning the analytics around any observed model result.
    """
    if not isinstance(members, tuple) or not all(isinstance(item, ZipMember) for item in members):
        raise TypeError("members must be a tuple of ZipMember")

    parsed = tuple(item for member in members if (item := _candidate(member)) is not None)
    negatives = tuple(item for item in parsed if item.role == "negative")
    positives = tuple(item for item in parsed if item.role == "positive")
    pairs = [
        (negative, positive)
        for negative in negatives
        for positive in positives
        if negative.subject_id != positive.subject_id
        and negative.location_id != positive.location_id
    ]
    if not pairs:
        raise RuntimeError("no bounded subject/location-disjoint ACT19/ACT4 Figshare pair")

    negative, positive = min(
        pairs,
        key=lambda pair: (
            pair[0].member.uncompressed_size + pair[1].member.uncompressed_size,
            max(pair[0].member.uncompressed_size, pair[1].member.uncompressed_size),
            pair[0].member.name,
            pair[1].member.name,
        ),
    )
    return {
        "selection_scope": "central-directory metadata only; no member payload fetched",
        "commercial_accuracy_claim": False,
        "selection_policy": {
            "negative_activity_code": "ACT19",
            "negative_activity_name": _ACTIVITY_NAMES[_NEGATIVE_ACTIVITY],
            "positive_activity_code": "ACT4",
            "positive_activity_name": _ACTIVITY_NAMES[_POSITIVE_ACTIVITY],
            "excluded_subject_ids": sorted(_EXCLUDED_SUBJECTS),
            "excluded_location_ids": sorted(_EXCLUDED_LOCATIONS),
            "require_subject_disjoint_pair": True,
            "require_location_disjoint_pair": True,
            "optimization": "minimum total uncompressed bytes; deterministic metadata tie-breaks",
        },
        "members": [negative.as_dict(), positive.as_dict()],
    }
