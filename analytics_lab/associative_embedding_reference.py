"""Evidence-only Associative Embedding decoder for the pinned OMZ pose candidate.

Adapted from Open Model Zoo's Apache-2.0
``demos/common/python/model_zoo/model_api/models/hpe_associative_embedding.py``
at commit ``6697dead54ed1cdd664b0313189c2cb52ee6335e``.

Copyright (C) 2020-2024 Intel Corporation.
Licensed under the Apache License, Version 2.0.

This module is intentionally evidence-only. It does not download models, select
thresholds from validation outcomes, or promote the candidate into a shipping
runtime. SciPy is imported lazily so ordinary repository tests do not acquire a
new runtime dependency merely by importing Analytics Lab.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class _Pose:
    def __init__(self, num_joints: int, tag_size: int = 1) -> None:
        self.num_joints = num_joints
        self.tag_size = tag_size
        self.pose = np.zeros((num_joints, 3 + tag_size), dtype=np.float32)
        self.pose_tag = np.zeros(tag_size, dtype=np.float32)
        self.valid_points_num = 0
        self.c = np.zeros(2, dtype=np.float32)

    def add(self, idx: int, joint: Any, tag: Any) -> None:
        self.pose[idx] = joint
        self.c = self.c * self.valid_points_num + joint[:2]
        self.pose_tag = self.pose_tag * self.valid_points_num + tag
        self.valid_points_num += 1
        self.c /= self.valid_points_num
        self.pose_tag /= self.valid_points_num

    @property
    def tag(self):
        return self.pose_tag if self.valid_points_num > 0 else None

    @property
    def center(self):
        return self.c if self.valid_points_num > 0 else None


class AssociativeEmbeddingDecoder:
    """Pinned OMZ AE grouping decoder with fixed reviewed evidence settings."""

    def __init__(self) -> None:
        self.num_joints = 17
        self.max_num_people = 30
        self.detection_threshold = 0.1
        self.tag_threshold = 1.0
        self.pose_threshold = 0.0
        self.use_detection_val = True
        self.ignore_too_much = False
        self.joint_order = (0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16)
        self.do_adjust = True
        self.do_refine = True
        self.dist_reweight = True
        self.delta = 0.0

    @staticmethod
    def _max_match(scores: Any) -> Any:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise RuntimeError("scipy==1.17.1 is required for the reviewed AE evidence decoder") from exc
        rows, cols = linear_sum_assignment(scores)
        return np.stack((rows, cols), axis=1)

    def _match_by_tag(self, inp: tuple[Any, Any, Any]) -> tuple[Any, Any]:
        tag_k, loc_k, val_k = inp
        embd_size = tag_k.shape[2]
        all_joints = np.concatenate((loc_k, val_k[..., None], tag_k), -1)
        poses: list[_Pose] = []
        for idx in self.joint_order:
            tags = tag_k[idx]
            joints = all_joints[idx]
            mask = joints[:, 2] > self.detection_threshold
            tags = tags[mask]
            joints = joints[mask]
            if not poses:
                for tag, joint in zip(tags, joints):
                    pose = _Pose(self.num_joints, embd_size)
                    pose.add(idx, joint, tag)
                    poses.append(pose)
                continue
            if joints.shape[0] == 0 or (self.ignore_too_much and len(poses) == self.max_num_people):
                continue
            pose_tags = np.stack([pose.tag for pose in poses], axis=0)
            diff = tags[:, None] - pose_tags[None, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)
            if self.dist_reweight:
                centers = np.stack([pose.center for pose in poses], axis=0)[None]
                dists = np.linalg.norm(joints[:, :2][:, None, :] - centers, ord=2, axis=2)
                close = diff_normed < self.tag_threshold
                min_dists = np.min(dists, axis=0, keepdims=True)
                dists /= min_dists + 1e-10
                diff_normed[close] *= dists[close]
            if self.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]
            num_added, num_grouped = diff.shape
            if num_added > num_grouped:
                diff_normed = np.pad(
                    diff_normed,
                    ((0, 0), (0, num_added - num_grouped)),
                    mode="constant",
                    constant_values=1e10,
                )
            for row, col in self._max_match(diff_normed):
                if row < num_added and col < num_grouped and diff_saved[row][col] < self.tag_threshold:
                    poses[col].add(idx, joints[row], tags[row])
                else:
                    pose = _Pose(self.num_joints, embd_size)
                    pose.add(idx, joints[row], tags[row])
                    poses.append(pose)
        answer = np.asarray([pose.pose for pose in poses], dtype=np.float32).reshape(
            -1, self.num_joints, 3 + embd_size
        )
        pose_tags = np.asarray([pose.tag for pose in poses], dtype=np.float32).reshape(-1, embd_size)
        return answer, pose_tags

    def top_k(self, heatmaps: Any, tags: Any) -> tuple[Any, Any, Any]:
        n, keypoints, height, width = heatmaps.shape
        flattened = heatmaps.reshape(n, keypoints, -1)
        indices = flattened.argpartition(-self.max_num_people, axis=2)[:, :, -self.max_num_people:]
        values = np.take_along_axis(flattened, indices, axis=2)
        order = np.argsort(-values, axis=2)
        indices = np.take_along_axis(indices, order, axis=2)
        values = np.take_along_axis(values, order, axis=2)
        tags = tags.reshape(n, keypoints, width * height, -1)
        selected_tags = [np.take_along_axis(tags[..., i], indices, axis=2) for i in range(tags.shape[3])]
        tag_k = np.stack(selected_tags, axis=3)
        x = indices % width
        y = indices // width
        loc_k = np.stack((x, y), axis=3)
        return tag_k, loc_k, values

    @staticmethod
    def adjust(answer: Any, heatmaps: Any) -> Any:
        height, width = heatmaps.shape[-2:]
        for batch_idx, people in enumerate(answer):
            for person in people:
                for keypoint, joint in enumerate(person):
                    heatmap = heatmaps[batch_idx, keypoint]
                    px, py = int(joint[0]), int(joint[1])
                    if 1 < px < width - 1 and 1 < py < height - 1:
                        diff = np.array([
                            heatmap[py, px + 1] - heatmap[py, px - 1],
                            heatmap[py + 1, px] - heatmap[py - 1, px],
                        ])
                        joint[:2] += np.sign(diff) * 0.25
        return answer

    @staticmethod
    def refine(heatmap: Any, tag: Any, keypoints: Any, pose_tag: Any = None) -> Any:
        count, height, width = heatmap.shape
        if len(tag.shape) == 3:
            tag = tag[..., None]
        if pose_tag is None:
            tags = []
            for idx in range(count):
                if keypoints[idx, 2] > 0:
                    x, y = keypoints[idx][:2].astype(int)
                    tags.append(tag[idx, y, x])
            if not tags:
                return keypoints
            previous_tag = np.mean(tags, axis=0)
        else:
            previous_tag = pose_tag
        for idx, (joint_heatmap, joint_tag) in enumerate(zip(heatmap, tag)):
            if keypoints[idx, 2] > 0:
                continue
            diff = np.abs(joint_tag[..., 0] - previous_tag) + 0.5
            diff = diff.astype(np.int32).astype(joint_heatmap.dtype)
            diff -= joint_heatmap
            flat = diff.argmin()
            y, x = np.divmod(flat, joint_heatmap.shape[-1])
            value = joint_heatmap[y, x]
            if value > 0:
                keypoints[idx, :3] = x, y, value
                if 1 < x < width - 1 and 1 < y < height - 1:
                    gradient = np.array([
                        joint_heatmap[y, x + 1] - joint_heatmap[y, x - 1],
                        joint_heatmap[y + 1, x] - joint_heatmap[y - 1, x],
                    ])
                    keypoints[idx, :2] += np.sign(gradient) * 0.25
        return keypoints

    def __call__(self, heatmaps: Any, tags: Any, *, nms_heatmaps: Any | None = None) -> tuple[Any, Any]:
        if tuple(heatmaps.shape) != (1, 17, 144, 144):
            raise RuntimeError("AE heatmap output shape changed")
        if tuple(tags.shape) != (1, 17, 144, 144, 1):
            raise RuntimeError("AE embedding output shape changed")
        nms = heatmaps if nms_heatmaps is None else nms_heatmaps
        tag_k, loc_k, val_k = self.top_k(nms, tags)
        grouped = tuple(map(self._match_by_tag, zip(tag_k, loc_k, val_k)))
        if not grouped:
            return np.empty((0, 17, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        answer, answer_tags = map(list, zip(*grouped))
        heatmaps = np.abs(heatmaps)
        if self.do_adjust:
            answer = self.adjust(answer, heatmaps)
        answer = answer[0]
        scores = np.asarray([person[:, 2].mean() for person in answer])
        mask = scores > self.pose_threshold
        answer = answer[mask]
        scores = scores[mask]
        if self.do_refine:
            heatmap_numpy = heatmaps[0]
            tag_numpy = tags[0]
            for idx, pose in enumerate(answer):
                answer[idx] = self.refine(heatmap_numpy, tag_numpy, pose, answer_tags[0][idx])
        return answer, scores
