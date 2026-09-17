# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed Open Model Zoo person detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed OMZ artifact identities, OpenVINO runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video. Media is re-hashed after sample execution so changed media cannot inherit stale evidence.

## Current acceptance-moving work
The source-rights registry now includes a second real-world source that can be acquired as individual small video files instead of requiring a multi-gigabyte archive. `analytics_lab.source_rights` records exact source/version, media origin, license, provenance reference, commercial training/evaluation eligibility, attribution requirement and the requirement for exact local asset identity. Unknown sources and unsupported purposes fail closed. Synthetic sources cannot satisfy a request that explicitly requires real-world evidence.

This registry does **not** download media, waive per-file rights review, or turn a source-page license into an accuracy claim. Every selected local clip still must be SHA-256 checksummed after acquisition and passed through the existing rights-bound validation suite.

## Reviewed data-source baseline
### UE4 Fall Detection Dataset — synthetic positive/negative development source
- Primary repository: `carolinehuang033/UE4_Fall_Detection_Dataset`.
- Pinned revision: `55041766dea68eaddc1df1c06aabd0a51931a22a`.
- Primary `LICENSE.txt` at that revision states CC BY 4.0 and expressly permits adaptation for any purpose, including commercially, with attribution.
- Media origin: **synthetic** (Unreal Engine 4 generated).
- Eligible engineering use: commercial training/evaluation subject to attribution and exact-asset identity.
- Limitation: synthetic observations and videos do not establish real-video accuracy.

### GMDCSA-24 — preferred bounded real-world validation source
- Primary data repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`.
- Repository `LICENSE` at that revision is MIT and requires preservation of its copyright/permission notice. The associated Data in Brief paper is open access under CC BY 4.0, identifies the dataset DOI `10.5281/zenodo.13354453`, states that the dataset can be used to train or test a fall-detection system, and reports 81 fall plus 79 ADL clips from four subjects in three home setups.
- Media origin: **real_world**, staged by consenting actors.
- Bounded-acquisition advantage: the GitHub repository exposes individual MP4 clips rather than requiring a full archive pull.
- Initial two-clip seed is deliberately small and adversarially useful:
  - positive: `Subject 1/Fall/05.mp4`, repository blob `4e13ed24f6c2af7062b64b1920ef706c51efe94b`, reported size 5,872,655 bytes; annotation describes walking followed by a right-side fall, with falling labeled from 1.8 s through 5 s;
  - normal hard negative: `Subject 1/ADL/15.mp4`, repository blob `903da9245132cf70c10124edd0625c958f702cb8`, reported size 7,233,079 bytes; annotation describes walking, picking an object from the ground and sitting, a useful fall-like negative.
- The Git blob identities and byte counts identify the intended upstream files but do **not** replace the required SHA-256 of the exact locally acquired bytes.
- Engineering rights review is not a legal opinion; commercial release still requires the repository's existing legal/IP review boundary.

### Figshare Video-Based Fall Detection Dataset with 2017 Activities from 29 Subjects — real-world secondary source
- Figshare article: `28596332`, version 2, posted 2025-03-14.
- Creator: Ivan Ursul.
- Source page reports 2,017 recordings from subjects, including 999 falls and 1,017 activities of daily living.
- Source page license: CC BY 4.0.
- Media origin: **real_world**.
- Eligible source-level engineering use: commercial training/evaluation under the stated CC BY 4.0 terms, with attribution.
- Acquisition boundary: the complete dataset is about 2.36 GB and must **not** be pulled wholesale under the current bounded-resource policy. Keep it as a broader secondary source after the small GMDCSA-24 seed proves the runtime path.

Previously reviewed UR Fall remains excluded from the commercial path because its official source states non-commercial terms. Earlier Roboflow mirrors with unclear upstream image provenance remain hold-only rather than silently promoted.

## Reviewed perception/runtime baseline
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`.
- Pinned FP16 artifacts: `person-detection-retail-0013` and `human-pose-estimation-0001`, admitted only by exact size/SHA-384 through the local artifact manifest.
- OpenVINO Runtime baseline: `2026.3.1`, Apache-2.0 source lineage. The code intentionally requires the `2026.3.1` runtime prefix.
- RTMLib remains code-only; its default HumanArt-trained weights remain hold/do-not-ship.

## Efficiency / execution ledger
One worker, one acceptance-moving work item, at most one implementation PR. Current work item: make bounded real-world acquisition immediately executable without pulling a multi-gigabyte corpus. Live base at intake: `b37fe4250b49fcbb202ca6b29f3341674334107f`; its main Analytics quality run completed successfully on the first attempt. Open implementation PRs at intake: 0. Active runs for the base/head at mutation time: 0. Unchanged retries: 0. CI-triggering requests before the implementation PR: 0. Consecutive sessions without tested acceptance improvement: 0. Fresh pre-mutation guardrail preflight allowed implementation.

The proposed source record remains fail-closed through the existing registry tests: it must be a pinned real-world source and still requires exact acquired-media identity before inference. Exact-head Linux, Windows and Analytics quality-gate checks remain authoritative before merge.

## Reproduce
Dependency-free repository checks:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

With externally provisioned reviewed artifacts/runtime and an authorized local manifest:

```sh
python -m analytics_lab.validation_cli --manifest /path/to/local-validation.json
```

## Next executable step
Acquire only the two pinned GMDCSA-24 seed clips above, compute SHA-256 for the exact acquired bytes, record attribution, and keep them outside public GitHub. Provision OpenVINO `2026.3.1` plus the exact pinned OMZ FP16 artifacts in the same no-spend environment. Run actual CPU inference through the existing validation CLI and record decoded duration, positives, misses, false alerts/camera-hour, alert delay, throughput, tracking continuity and concrete failure cases.

If the initial positive is missed, identify whether failure occurs at person detection, pose, track continuity, posture classification or temporal persistence before changing the stack. If prone-person detector recall is inadequate, quantify that detector-stage failure before comparing at most three rights-cleared alternatives. Do not retrain a commodity detector from scratch without measured need.

## Outstanding commercial-release gates
Actual reviewed runtime execution; exact file-level positive/negative real-video evidence; false-alert and missed-event measurements; documented failure cases; comparable latency/resource evidence; platform wheel/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and owner release approval. Synthetic/stub tests do not establish video accuracy or commercial readiness.
