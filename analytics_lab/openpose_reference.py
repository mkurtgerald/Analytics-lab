"""Evidence-only Open Model Zoo OpenPose decoder semantics.

Portions of :class:`ReferenceOpenPoseDecoder` are adapted from Intel Open
Model Zoo ``demos/common/python/model_zoo/model_api/models/open_pose.py`` at
commit ``6697dead54ed1cdd664b0313189c2cb52ee6335e``.

Copyright (C) 2020-2024 Intel Corporation
SPDX-License-Identifier: Apache-2.0

This module is not the production pose backend. It exists to compare the
repository's simplified detector-crop/heatmap-peak baseline against the pinned
upstream NMS + part-affinity-field grouping semantics on the same reviewed
model outputs and rights-bound evidence clips.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Sequence

from .perception import BBox, Keypoint, PoseCandidate

_REQUIRED_COCO = (
    ("left_shoulder", 5),
    ("right_shoulder", 6),
    ("left_hip", 11),
    ("right_hip", 12),
)
_OUTPUT_SCALE = 8.0


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class DecodedReferencePose:
    candidate: PoseCandidate
    decoder_score: float
    required_points: int
    required_points_inside_selection: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, PoseCandidate):
            raise ValueError("candidate must be a PoseCandidate")
        object.__setattr__(self, "decoder_score", _finite(self.decoder_score, "decoder score"))
        if self.decoder_score < 0:
            raise ValueError("decoder score must be nonnegative")
        for name in ("required_points", "required_points_inside_selection"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= len(_REQUIRED_COCO):
                raise ValueError(f"{name} must be a bounded integer")
        if self.required_points_inside_selection > self.required_points:
            raise ValueError("inside required-point count cannot exceed required-point count")


def candidate_from_coco_pose(
    pose: Sequence[Sequence[float]],
    *,
    decoder_score: float,
    selection_bbox: BBox,
    frame_width: int,
    frame_height: int,
    resized_width: int,
    resized_height: int,
) -> DecodedReferencePose:
    """Map one upstream COCO-order pose from heatmap space to the source frame."""
    if not isinstance(selection_bbox, BBox):
        raise ValueError("selection_bbox must be a BBox")
    frame_width = _positive_int(frame_width, "frame_width")
    frame_height = _positive_int(frame_height, "frame_height")
    resized_width = _positive_int(resized_width, "resized_width")
    resized_height = _positive_int(resized_height, "resized_height")
    if len(pose) < 17:
        raise ValueError("decoded pose must contain 17 COCO keypoints")
    scale_x = (frame_width / resized_width) * _OUTPUT_SCALE
    scale_y = (frame_height / resized_height) * _OUTPUT_SCALE
    keypoints: list[Keypoint] = []
    inside = 0
    for name, index in _REQUIRED_COCO:
        raw = pose[index]
        if len(raw) < 3:
            raise ValueError("decoded keypoint must contain x, y and confidence")
        x = _finite(raw[0], "decoded keypoint x")
        y = _finite(raw[1], "decoded keypoint y")
        confidence = _finite(raw[2], "decoded keypoint confidence")
        if confidence <= 0:
            continue
        confidence = min(1.0, confidence)
        px = x * scale_x
        py = y * scale_y
        point = Keypoint(name, px, py, confidence)
        keypoints.append(point)
        if (selection_bbox.x1 <= px <= selection_bbox.x2
                and selection_bbox.y1 <= py <= selection_bbox.y2):
            inside += 1
    candidate = PoseCandidate(selection_bbox, tuple(keypoints), 1.0)
    return DecodedReferencePose(
        candidate=candidate,
        decoder_score=decoder_score,
        required_points=len(keypoints),
        required_points_inside_selection=inside,
    )


def select_reference_pose(
    poses: Sequence[Sequence[Sequence[float]]],
    scores: Sequence[float],
    *,
    selection_bbox: BBox,
    frame_width: int,
    frame_height: int,
    resized_width: int,
    resized_height: int,
) -> DecodedReferencePose | None:
    if len(poses) != len(scores):
        raise ValueError("pose and score counts must match")
    candidates = [
        candidate_from_coco_pose(
            pose,
            decoder_score=float(score),
            selection_bbox=selection_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            resized_width=resized_width,
            resized_height=resized_height,
        )
        for pose, score in zip(poses, scores)
    ]
    candidates = [item for item in candidates if item.required_points > 0]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -item.required_points_inside_selection,
            -item.required_points,
            -item.decoder_score,
        )
    )
    return candidates[0] if candidates[0].required_points_inside_selection > 0 else None


class ReferenceOpenPoseDecoder:
    """Pinned OMZ OpenPose NMS + PAF grouping, adapted for evidence diagnostics."""

    BODY_PARTS_KPT_IDS = (
        (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
        (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
        (0, 15), (15, 17), (2, 16), (5, 17),
    )
    BODY_PARTS_PAF_IDS = (12, 20, 14, 16, 22, 24, 0, 2, 4, 6, 8, 10, 28, 30, 34, 32, 36, 18, 26)

    def __init__(self, num_joints: int = 18, *, max_points: int = 100,
                 score_threshold: float = 0.1, min_paf_alignment_score: float = 0.05,
                 delta: float = 0.5) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is required for the reference OpenPose decoder") from exc
        self.np = np
        self.num_joints = num_joints
        self.skeleton = self.BODY_PARTS_KPT_IDS
        self.paf_indices = self.BODY_PARTS_PAF_IDS
        self.max_points = max_points
        self.score_threshold = score_threshold
        self.min_paf_alignment_score = min_paf_alignment_score
        self.delta = delta
        self.points_per_limb = 10
        self.grid = np.arange(self.points_per_limb, dtype=np.float32).reshape(1, -1, 1)

    def heatmap_nms(self, heatmaps: Any) -> Any:
        np = self.np
        heatmaps = np.asarray(heatmaps)
        if heatmaps.shape != (1, 19, 32, 57):
            raise RuntimeError("reference decoder requires reviewed 1x19x32x57 heatmaps")
        padded = np.pad(heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
                        mode="constant", constant_values=-np.inf)
        pooled = np.full_like(heatmaps, -np.inf)
        height, width = heatmaps.shape[-2:]
        for dy in range(3):
            for dx in range(3):
                pooled = np.maximum(pooled, padded[:, :, dy:dy + height, dx:dx + width])
        return heatmaps * (heatmaps == pooled)

    def __call__(self, heatmaps: Any, pafs: Any) -> tuple[Any, Any]:
        np = self.np
        heatmaps = np.asarray(heatmaps)
        pafs = np.asarray(pafs)
        if heatmaps.shape != (1, 19, 32, 57) or pafs.shape != (1, 38, 32, 57):
            raise RuntimeError("reference decoder output shapes changed")
        keypoints = self.extract_points(heatmaps, self.heatmap_nms(heatmaps))
        pafs = np.transpose(pafs, (0, 2, 3, 1))
        _, _, height, width = heatmaps.shape
        if self.delta > 0:
            for points in keypoints:
                points[:, :2] += self.delta
                np.clip(points[:, 0], 0, width - 1, out=points[:, 0])
                np.clip(points[:, 1], 0, height - 1, out=points[:, 1])
        pose_entries, all_keypoints = self.group_keypoints(keypoints, pafs, pose_entry_size=self.num_joints + 2)
        poses, scores = self.convert_to_coco_format(pose_entries, all_keypoints)
        if len(poses) > 0:
            poses = np.asarray(poses, dtype=np.float32).reshape((-1, 17, 3))
            scores = np.asarray(scores, dtype=np.float32)
        else:
            poses = np.empty((0, 17, 3), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)
        return poses, scores

    def extract_points(self, heatmaps: Any, nms_heatmaps: Any) -> list[Any]:
        np = self.np
        batch_size, channels_num, height, width = heatmaps.shape
        if batch_size != 1 or channels_num < self.num_joints:
            raise RuntimeError("unsupported reference decoder heatmap shape")
        xs, ys, scores = self.top_k(nms_heatmaps)
        masks = scores > self.score_threshold
        all_keypoints = []
        keypoint_id = 0
        for index in range(self.num_joints):
            mask = masks[0, index]
            x = xs[0, index][mask].ravel()
            y = ys[0, index][mask].ravel()
            score = scores[0, index][mask].ravel()
            count = len(x)
            if count == 0:
                all_keypoints.append(np.empty((0, 4), dtype=np.float32))
                continue
            x, y = self.refine(heatmaps[0, index], x, y)
            np.clip(x, 0, width - 1, out=x)
            np.clip(y, 0, height - 1, out=y)
            points = np.empty((count, 4), dtype=np.float32)
            points[:, 0] = x
            points[:, 1] = y
            points[:, 2] = score
            points[:, 3] = np.arange(keypoint_id, keypoint_id + count)
            keypoint_id += count
            all_keypoints.append(points)
        return all_keypoints

    def top_k(self, heatmaps: Any) -> tuple[Any, Any, Any]:
        np = self.np
        n, k, _, width = heatmaps.shape
        flat = heatmaps.reshape(n, k, -1)
        count = min(self.max_points, flat.shape[2])
        ind = flat.argpartition(-count, axis=2)[:, :, -count:]
        scores = np.take_along_axis(flat, ind, axis=2)
        subind = np.argsort(-scores, axis=2)
        ind = np.take_along_axis(ind, subind, axis=2)
        scores = np.take_along_axis(scores, subind, axis=2)
        y, x = np.divmod(ind, width)
        return x, y, scores

    def refine(self, heatmap: Any, x: Any, y: Any) -> tuple[Any, Any]:
        np = self.np
        height, width = heatmap.shape[-2:]
        valid = np.logical_and(np.logical_and(x > 0, x < width - 1),
                               np.logical_and(y > 0, y < height - 1))
        xx = x[valid]
        yy = y[valid]
        dx = np.sign(heatmap[yy, xx + 1] - heatmap[yy, xx - 1]).astype(np.float32) * 0.25
        dy = np.sign(heatmap[yy + 1, xx] - heatmap[yy - 1, xx]).astype(np.float32) * 0.25
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x[valid] += dx
        y[valid] += dy
        return x, y

    def is_disjoint(self, pose_a: Any, pose_b: Any) -> bool:
        np = self.np
        pose_a = pose_a[:-2]
        pose_b = pose_b[:-2]
        return bool(np.all(np.logical_or.reduce((pose_a == pose_b, pose_a < 0, pose_b < 0))))

    def update_poses(self, kpt_a_id: int, kpt_b_id: int, all_keypoints: Any,
                     connections: list[tuple[int, int, float]], pose_entries: list[Any],
                     pose_entry_size: int) -> list[Any]:
        np = self.np
        for connection in connections:
            pose_a_idx = -1
            pose_b_idx = -1
            for index, pose in enumerate(pose_entries):
                if pose[kpt_a_id] == connection[0]:
                    pose_a_idx = index
                if pose[kpt_b_id] == connection[1]:
                    pose_b_idx = index
            if pose_a_idx < 0 and pose_b_idx < 0:
                pose_entry = np.full(pose_entry_size, -1, dtype=np.float32)
                pose_entry[kpt_a_id] = connection[0]
                pose_entry[kpt_b_id] = connection[1]
                pose_entry[-1] = 2
                pose_entry[-2] = np.sum(all_keypoints[list(connection[0:2]), 2]) + connection[2]
                pose_entries.append(pose_entry)
            elif pose_a_idx >= 0 and pose_b_idx >= 0 and pose_a_idx != pose_b_idx:
                pose_a = pose_entries[pose_a_idx]
                pose_b = pose_entries[pose_b_idx]
                if self.is_disjoint(pose_a, pose_b):
                    pose_a += pose_b
                    pose_a[:-2] += 1
                    pose_a[-2] += connection[2]
                    del pose_entries[pose_b_idx]
            elif pose_a_idx >= 0 and pose_b_idx >= 0:
                pose_entries[pose_a_idx][-2] += connection[2]
            elif pose_a_idx >= 0:
                pose = pose_entries[pose_a_idx]
                if pose[kpt_b_id] < 0:
                    pose[-2] += all_keypoints[connection[1], 2]
                pose[kpt_b_id] = connection[1]
                pose[-2] += connection[2]
                pose[-1] += 1
            elif pose_b_idx >= 0:
                pose = pose_entries[pose_b_idx]
                if pose[kpt_a_id] < 0:
                    pose[-2] += all_keypoints[connection[0], 2]
                pose[kpt_a_id] = connection[0]
                pose[-2] += connection[2]
                pose[-1] += 1
        return pose_entries

    def connections_nms(self, a_idx: Any, b_idx: Any, affinity_scores: Any) -> tuple[Any, Any, Any]:
        np = self.np
        order = affinity_scores.argsort()[::-1]
        affinity_scores = affinity_scores[order]
        a_idx = a_idx[order]
        b_idx = b_idx[order]
        accepted = []
        has_a: set[int] = set()
        has_b: set[int] = set()
        for position, (a_value, b_value) in enumerate(zip(a_idx, b_idx)):
            a_int = int(a_value)
            b_int = int(b_value)
            if a_int not in has_a and b_int not in has_b:
                accepted.append(position)
                has_a.add(a_int)
                has_b.add(b_int)
        idx = np.asarray(accepted, dtype=np.int32)
        return a_idx[idx], b_idx[idx], affinity_scores[idx]

    def group_keypoints(self, all_keypoints_by_type: list[Any], pafs: Any,
                        pose_entry_size: int = 20) -> tuple[Any, Any]:
        np = self.np
        all_keypoints = np.concatenate(all_keypoints_by_type, axis=0)
        pose_entries: list[Any] = []
        for part_id, paf_channel in enumerate(self.paf_indices):
            kpt_a_id, kpt_b_id = self.skeleton[part_id]
            kpts_a = all_keypoints_by_type[kpt_a_id]
            kpts_b = all_keypoints_by_type[kpt_b_id]
            n = len(kpts_a)
            m = len(kpts_b)
            if n == 0 or m == 0:
                continue
            a = np.broadcast_to(kpts_a[:, :2][None], (m, n, 2))
            b = kpts_b[:, :2]
            vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)
            steps = (1 / (self.points_per_limb - 1) * vec_raw)
            points = steps * self.grid + a.reshape(-1, 1, 2)
            points = points.round().astype(dtype=np.int32)
            x = points[..., 0].ravel()
            y = points[..., 1].ravel()
            part_pafs = pafs[0, :, :, paf_channel:paf_channel + 2]
            field = part_pafs[y, x].reshape(-1, self.points_per_limb, 2)
            vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
            vec = vec_raw / (vec_norm + 1e-6)
            affinity_scores = (field * vec).sum(-1).reshape(-1, self.points_per_limb)
            valid_affinity_scores = affinity_scores > self.min_paf_alignment_score
            valid_num = valid_affinity_scores.sum(1)
            affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6)
            success_ratio = valid_num / self.points_per_limb
            valid_limbs = np.where(np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
            if len(valid_limbs) == 0:
                continue
            b_idx, a_idx = np.divmod(valid_limbs, n)
            affinity_scores = affinity_scores[valid_limbs]
            a_idx, b_idx, affinity_scores = self.connections_nms(a_idx, b_idx, affinity_scores)
            connections = list(zip(kpts_a[a_idx, 3].astype(np.int32),
                                   kpts_b[b_idx, 3].astype(np.int32), affinity_scores))
            if connections:
                pose_entries = self.update_poses(kpt_a_id, kpt_b_id, all_keypoints,
                                                 connections, pose_entries, pose_entry_size)
        pose_entries_array = np.asarray(pose_entries, dtype=np.float32).reshape(-1, pose_entry_size)
        pose_entries_array = pose_entries_array[pose_entries_array[:, -1] >= 3]
        return pose_entries_array, all_keypoints

    def convert_to_coco_format(self, pose_entries: Any, all_keypoints: Any) -> tuple[Any, Any]:
        np = self.np
        coco_keypoints = []
        scores = []
        reorder_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        for pose in pose_entries:
            if len(pose) == 0:
                continue
            keypoints = np.zeros(17 * 3)
            person_score = pose[-2]
            for keypoint_id, target_id in zip(pose[:-2], reorder_map):
                if target_id < 0:
                    continue
                cx = cy = score = 0.0
                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                keypoints[target_id * 3 + 0] = cx
                keypoints[target_id * 3 + 1] = cy
                keypoints[target_id * 3 + 2] = score
            coco_keypoints.append(keypoints)
            scores.append(person_score * max(0, (pose[-1] - 1)))
        return np.asarray(coco_keypoints), np.asarray(scores)
