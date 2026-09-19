# Third-Party Component Register

Every external library, repository, model, weight file, dataset, code fragment, or other donor component used by Analytics Lab must be recorded here before it is merged into a production candidate.

## Policy

Preferred licenses:
- Apache-2.0
- MIT
- BSD-2-Clause
- BSD-3-Clause

Components under copyleft, source-reciprocal, non-commercial, research-only, custom, or unclear licenses require explicit review before use. GPL, AGPL, LGPL, SSPL, BUSL, Commons Clause, CC-BY-NC, research-only model terms, and unknown licenses are not automatically approved.

## Required fields

| Component | Upstream URL | Version / Commit | License | Purpose | Modified? | Redistribution Requirements | Approval |
|---|---|---|---|---|---|---|---|
| Open Model Zoo `person-detection-retail-0013` FP16 artifact manifest | `https://github.com/openvinotoolkit/open_model_zoo` | `6697dead54ed1cdd664b0313189c2cb52ee6335e` | Apache-2.0 | Candidate local person-detector artifact identity | No donor code/model vendored | Preserve applicable Apache-2.0 license/notices; product redistribution not enabled by this admission | Engineering local-evaluation baseline; release review pending |
| Open Model Zoo `human-pose-estimation-0001` FP16 artifact manifest | `https://github.com/openvinotoolkit/open_model_zoo` | `6697dead54ed1cdd664b0313189c2cb52ee6335e` | Apache-2.0 | Candidate local multi-person pose artifact identity | No donor code/model vendored | Preserve applicable Apache-2.0 license/notices; product redistribution not enabled by this admission | Engineering local-evaluation baseline; release review pending |
| Open Model Zoo OpenPose decoder reference (`demos/common/python/model_zoo/model_api/models/open_pose.py`) | `https://github.com/openvinotoolkit/open_model_zoo` | `6697dead54ed1cdd664b0313189c2cb52ee6335e` | Apache-2.0 | Reviewed keypoint channels plus evidence-only full-frame preserve-aspect, NMS and PAF-grouping diagnostic | Adapted diagnostic code in `analytics_lab/openpose_reference.py`; production backend remains separate | Preserve Intel copyright, Apache-2.0 attribution/notice, pinned source commit and modification record | Engineering evidence diagnostic; accuracy/release review pending |
| OpenVINO Runtime Python API | `https://github.com/openvinotoolkit/openvino` | tag `2026.3.1`, commit `759c5a6ab8c066af5f4bc5ebd04643706012a37d` | Apache-2.0 | Optional local CPU inference runtime for verified OpenVINO IR artifacts | Runtime is not vendored; adapter targets its public API | Preserve applicable Apache-2.0 license/notices; platform wheel hashes/native dependency notices must be pinned before product packaging | Engineering API baseline; packaging/release review pending |
| FoundationVision/ByteTrack portable association slice | `https://github.com/FoundationVision/ByteTrack` | `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` | MIT | Detector-neutral high/low-confidence two-stage association and bounded session-local track lifecycle behind `analytics_lab.tracking` | Yes — adapted in `analytics_lab/bytetrack.py`; the upstream detector/Torch, Kalman/SciPy, LAP, `cython_bbox`, OpenCV and native-extension paths are excluded from this first slice | Preserve Yifu Zhang 2021 MIT notice; full notice retained in `third_party/licenses/ByteTrack-MIT.txt`; no donor model, weights or data admitted | Engineering integration baseline only; full upstream parity and commercial accuracy are not claimed; rights-cleared real-video tracking evaluation pending |

## Reviewed alternatives / future donor extensions — not incorporated

These are research candidates or extensions only; listing them does not admit, vendor, package, or redistribute their code, dependencies, models or weights.

| Candidate | Exact revision | Repository license | Fit | Status / blocker |
|---|---|---|---|---|
| tryolabs/norfair | `e517b4236f6b67a6ecf342f5df1fccb7788dbc54` | BSD-3-Clause | Detector-agnostic generic tracking | Fallback candidate. Dependency rights and runtime footprint are not yet pinned. |
| Open Model Zoo multi-camera multi-target tracking demo | `6697dead54ed1cdd664b0313189c2cb52ee6335e` | Apache-2.0 | Person/vehicle tracking with re-identification | Retained as a reference and possible future appearance-search donor; heavier and ReID-coupled, so not the first single-camera tracking integration. |
| FoundationVision/ByteTrack full Kalman/LAP association path | `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` | MIT donor code; transitive components require separate review | Motion prediction plus global linear assignment beyond the portable first slice | Do not admit unless measured tracking evidence shows the portable slice is limited specifically by motion prediction or assignment. Pin each dependency, license/notice, Python 3.11 Linux/Windows packaging and resource cost first. `cython-bbox` is not admitted because the reviewed package metadata/portable binary path is insufficient for this fail-closed commercial route. |

## Rules

1. Never copy donor code without recording its source and license.
2. Preserve all required copyright and attribution notices.
3. Treat model weights and datasets separately from source-code licenses.
4. Do not assume a GitHub repository is reusable merely because its source is public.
5. Do not commit third-party datasets or weights unless redistribution rights are clear.
6. Prefer replaceable adapters around donor components so the project can swap them later.
