"""Select the next bounded Figshare generalization pair from archive metadata only.

This selector never downloads member payloads. It operates only on the reviewed
central-directory map and chooses a difficult floor-transition ADL negative plus
a kneeling-fall positive. Subjects already used by the first three Figshare
pairs are excluded, the two candidates must be subject/location-disjoint, and at
least one candidate must come from a location not exercised by those pairs.
The result is candidate metadata for later exact admission, not accuracy evidence
and not authorization to train on the clips.
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

# Measured pairs used Subjects 01/10, 06/03 and 02/09 across Locations 1/2/3/5.
# The next pair must move to untouched subjects and include a location outside
# that exercised set before any member payload or model output is inspected.
_EXCLUDED_SUBJECTS = frozenset({"01", "02", "03", "06", "09", "10"})
_USED_LOCATIONS = frozenset({"1", "2", "3", "5"})

# The published activity map identifies ACT20 as Standing up from laying and
# ACT6 as Fall on knees. This is a deliberately difficult floor-transition
# boundary chosen before inspecting any model output.
_NEGATIVE_ACTIVITY = 20
_POSITIVE_ACTIVITY = 6
_ACTIVITY_NAMES = {
    20: "Standing up from laying",
    6: "Fall on knees",
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
    if subject in _EXCLUDED_SUBJECTS:
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
        location_id=match.group("location"),
        activity_code=activity,
        repetition=int(match.group("repetition")),
        member=member,
    )


def select_next_generalization_pair(members: tuple[ZipMember, ...]) -> dict[str, Any]:
    """Return a small untouched ACT20/ACT6 pair with new-location coverage.

    Selection is metadata-only. It first maximizes the count of members from
    previously unused locations, then minimizes total uncompressed bytes,
    maximum member size, and stable path order. No model output can influence
    the choice.
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
        and (
            negative.location_id not in _USED_LOCATIONS
            or positive.location_id not in _USED_LOCATIONS
        )
    ]
    if not pairs:
        raise RuntimeError(
            "no bounded subject/location-disjoint ACT20/ACT6 Figshare pair with a novel location"
        )

    def sort_key(pair: tuple[GeneralizationMember, GeneralizationMember]) -> tuple[Any, ...]:
        novel_locations = sum(item.location_id not in _USED_LOCATIONS for item in pair)
        return (
            -novel_locations,
            pair[0].member.uncompressed_size + pair[1].member.uncompressed_size,
            max(pair[0].member.uncompressed_size, pair[1].member.uncompressed_size),
            pair[0].member.name,
            pair[1].member.name,
        )

    negative, positive = min(pairs, key=sort_key)
    return {
        "selection_scope": "central-directory metadata only; no member payload fetched",
        "commercial_accuracy_claim": False,
        "selection_policy": {
            "negative_activity_code": "ACT20",
            "negative_activity_name": _ACTIVITY_NAMES[_NEGATIVE_ACTIVITY],
            "positive_activity_code": "ACT6",
            "positive_activity_name": _ACTIVITY_NAMES[_POSITIVE_ACTIVITY],
            "excluded_subject_ids": sorted(_EXCLUDED_SUBJECTS),
            "previously_used_location_ids": sorted(_USED_LOCATIONS),
            "require_subject_disjoint_pair": True,
            "require_location_disjoint_pair": True,
            "require_at_least_one_novel_location": True,
            "optimization": (
                "maximize novel-location members, then minimum total uncompressed bytes; "
                "deterministic metadata tie-breaks"
            ),
        },
        "members": [negative.as_dict(), positive.as_dict()],
    }
