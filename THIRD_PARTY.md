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
| Open Model Zoo OpenPose decoder reference (`demos/common/python/model_zoo/model_api/models/open_pose.py`) | `https://github.com/openvinotoolkit/open_model_zoo` | `6697dead54ed1cdd664b0313189c2cb52ee6335e` | Apache-2.0 | Reference for reviewed keypoint channel IDs and quarter-pixel peak refinement in detector-isolated crop baseline | Concepts adapted; full PAF decoder is not vendored | Preserve attribution and Apache-2.0 notice for adapted donor-derived material | Engineering integration baseline; accuracy/release review pending |
| OpenVINO Runtime Python API | `https://github.com/openvinotoolkit/openvino` | tag `2026.3.1`, commit `759c5a6ab8c066af5f4bc5ebd04643706012a37d` | Apache-2.0 | Optional local CPU inference runtime for verified OpenVINO IR artifacts | Runtime is not vendored; adapter targets its public API | Preserve applicable Apache-2.0 license/notices; platform wheel hashes/native dependency notices must be pinned before product packaging | Engineering API baseline; packaging/release review pending |

## Rules

1. Never copy donor code without recording its source and license.
2. Preserve all required copyright and attribution notices.
3. Treat model weights and datasets separately from source-code licenses.
4. Do not assume a GitHub repository is reusable merely because its source is public.
5. Do not commit third-party datasets or weights unless redistribution rights are clear.
6. Prefer replaceable adapters around donor components so the project can swap them later.
