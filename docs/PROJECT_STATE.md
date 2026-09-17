# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed Open Model Zoo person detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed OMZ artifact identities, OpenVINO runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video. Media is re-hashed after sample execution so changed media cannot inherit stale evidence.

## Current acceptance-moving work
A pinned source-rights registry now makes the data-selection boundary executable instead of relying on an informal dataset list. `analytics_lab.source_rights` records exact source/version, media origin, license, provenance reference, commercial training/evaluation eligibility, attribution requirement and the requirement for exact local asset identity. Unknown sources and unsupported purposes fail closed. Synthetic sources cannot satisfy a request that explicitly requires real-world evidence.

This registry does **not** download media, waive per-file rights review, or turn a source-page license into an accuracy claim. Every selected local clip still must be checksummed and passed through the existing rights-bound validation suite.

## Reviewed data-source baseline
### UE4 Fall Detection Dataset — synthetic positive/negative development source
- Primary repository: `carolinehuang033/UE4_Fall_Detection_Dataset`.
- Pinned revision: `55041766dea68eaddc1df1c06aabd0a51931a22a`.
- Primary `LICENSE.txt` at that revision states CC BY 4.0 and expressly permits adaptation for any purpose, including commercially, with attribution.
- Media origin: **synthetic** (Unreal Engine 4 generated).
- Eligible engineering use: commercial training/evaluation subject to attribution and exact-asset identity.
- Limitation: synthetic observations and videos do not establish real-video accuracy.

### Figshare Video-Based Fall Detection Dataset with 2017 Activities from 29 Subjects — real-world candidate source
- Figshare article: `28596332`, version 2, posted 2025-03-14.
- Creator: Ivan Ursul.
- Source page reports 2,017 recordings from subjects, including 999 falls and 1,017 activities of daily living.
- Source page license: CC BY 4.0.
- Media origin: **real_world**.
- Eligible source-level engineering use: commercial training/evaluation under the stated CC BY 4.0 terms, with attribution.
- Acquisition boundary: the complete dataset is about 2.36 GB and must **not** be pulled wholesale under the current bounded-resource policy. The next execution must select a small file-level subset, record exact file identity/hash, and retain only the bounded authorized clips needed for the run.

Previously reviewed UR Fall remains excluded from the commercial path because its official source states non-commercial terms. Earlier Roboflow mirrors with unclear upstream image provenance remain hold-only rather than silently promoted.

## Reviewed perception/runtime baseline
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`.
- Pinned FP16 artifacts: `person-detection-retail-0013` and `human-pose-estimation-0001`, admitted only by exact size/SHA-384 through the local artifact manifest.
- OpenVINO Runtime baseline: `2026.3.1`, Apache-2.0 source lineage. The code intentionally requires the `2026.3.1` runtime prefix.
- RTMLib remains code-only; its default HumanArt-trained weights remain hold/do-not-ship.

## Efficiency / execution ledger
One worker, one acceptance-moving work item, at most one implementation PR. Current work item: executable source-rights selection for bounded commercial validation/training inputs. Tested base at start: `9b81168bb43ee4e1614804be1143af25f1efda0c`; its main Analytics quality run completed successfully. Open implementation PRs at intake: 0. Queued/running repository runs at mutation time: 0. Unchanged retries: 0. CI-triggering requests before the implementation PR: 0. Consecutive sessions without tested acceptance improvement: 0.

Focused local regression before repository mutation passed 4/4 checks: reviewed synthetic source accepted for commercial training; synthetic source rejected when real-world evidence is required; reviewed real-world source accepted for evaluation; unknown source/purpose and invalid origin fail closed. Exact-head Linux, Windows and Analytics quality-gate checks remain authoritative before merge.

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
Acquire a **bounded file-level subset** of the reviewed Figshare real-world source (at least one positive fall clip and normal-negative clips) without downloading the 2.36 GB corpus. Record exact file identities, hashes and attribution. In the same no-spend environment, provision OpenVINO `2026.3.1` plus the exact pinned OMZ FP16 artifacts and run actual CPU inference through the existing validation CLI. Record decoded duration, positives, misses, false alerts/camera-hour, alert delay, throughput, tracking continuity and failure cases. Keep synthetic UE4 material in a separate evidence class for training/stress testing only.

If prone-person recall is inadequate, quantify the detector-stage failure before comparing at most three rights-cleared alternatives. Do not retrain a commodity detector from scratch without that measured need.

## Outstanding commercial-release gates
Actual reviewed runtime execution; exact file-level positive/negative real-video evidence; false-alert and missed-event measurements; documented failure cases; comparable latency/resource evidence; platform wheel/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and owner release approval. Synthetic/stub tests do not establish video accuracy or commercial readiness.
