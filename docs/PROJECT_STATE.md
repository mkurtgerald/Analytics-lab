# Project state — 2026-09-18

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current person-down path
The retained person-down **candidate** path under measured validation is:

`authorized local video -> reviewed detector -> spatial continuity/orientation recovery -> reviewed OpenPose model -> bounded pose association -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> labeled evaluation/aggregation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media hashes, reviewed model/runtime identities, decoded coverage, continuity, timing and evaluation metrics without retaining video in public GitHub.

## Real-video evidence to date
### Subject 1 — first matched engineering seed
PR #33 produced the first matched staged-real person-down candidate on the pinned Subject-1 pair: **1 positive episode, 1 candidate, 1 match, 0 misses, 0 hard-negative false alerts**, about **0.0036 decoded camera-hours**, and **4024 ms** alert delay. The retained 3000 ms persistence requirement stayed intact. This remains a tiny engineering seed, not a commercial accuracy claim.

### Subject 2 — detector recovery, then posture fragmentation
PR #34 exposed a disjoint positive miss and only **94/138 = 68.12%** labeled-fall detector selection while the prone-normal negative produced **0 candidates / 0 false alerts**. PR #35 retained bounded same-detector orientation recovery using exact pinned `person-detection-0200`, raising labeled-fall coverage to **126/138 = 91.30%**, associated poses **98 -> 135**, qualified `down` frames **31 -> 57**, and longest qualified run **768 ms -> 1472 ms** without negative alert regression.

PR #36 then converted only measured near-diagonal non-decisive geometry (`|horizontal_fraction - vertical_fraction| <= 0.02`) from `other` to `unknown`. It never promotes to `down`, never bridges `upright`, and leaves persistence/association/confidence unchanged. The longest Subject-2 positive run improved **1472 ms -> 2544 ms**; the positive still misses because the remaining interruption is genuinely upright. Subject 2 remains held out and is not tuned further merely to manufacture a match.

### Subject 3 — association failure and rejected crop recovery
PR #37 measured a third disjoint pair. Exact identities are:
- positive `Subject 3/Fall/03.mp4` SHA-256 `267abab9b0ca7f4ef8425e2bfb2272ffeae1165fc2e3523ca1a6e0a62046b912`;
- hard negative `Subject 3/ADL/08.mp4` SHA-256 `b383bb83a22b286cfc65fdd934137ee8b29e84aef7be0b540271d7c69d10fa8b`.

The positive produced **0 candidates / 0 matches / 1 miss** and the sitting-to-sleeping-on-floor negative produced **0 candidates / 0 false alerts** across **10,432 ms / 0.00289778 camera-hours**. Detector coverage reached **89/105 = 84.76%** labeled-fall frames, but safe pose overlap reached only **50/105**. Longest qualified positive run was **1184 ms**.

PR #38 tested one bounded selected-person crop using the same OpenPose model after full-frame association failure. It improved raw association to 73/105 but regressed the positive longest qualified run **1184 ms -> 512 ms**, added about 10.29 s pose inference across the tiny pair, and was closed unmerged even though CI was green. This crop path is closed.

### Subject 4 — detector/association cleared; pose/posture now dominant
PR #39 measured the unchanged retained path on a fourth disjoint pair. Its exact head passed Linux, **226 tests**, the bounded real-video CPU lane, Windows and Analytics quality on attempt 1.

Exact Subject-4 evidence:
- positive `Subject 4/Fall/03.mp4` SHA-256 `4f178dea77bf9abb9036bcd7fedfbe345358fc2d63e1704dc49b7ec1a621e2f7`;
- dynamic prone/push-up negative `Subject 4/ADL/07.mp4` SHA-256 `6e410abbc6d4ef7bf52ac835b952a74a94311d0977fdfddb21885477602b7b69`;
- decoded evidence **11,632 ms / 0.0032311111 camera-hours**.

The positive produced **0 candidates / 0 matches / 1 miss**; the push-up negative produced **0 candidates / 0 false alerts**. On the positive labeled window, detector selection and safe pose association both reached **129/129 = 100%**. The temporal input contained **55 qualified down observations**, but the longest qualified run was only **1312 ms** because six `geometry_not_decisive` observations reset as `other`. The negative produced **44 down-like observations**, longest qualified run **464 ms**, and still emitted no alert. Retained-path throughput was about **9.67 FPS** aggregate. This isolates pose/posture quality as the next measured wall and argues against weakening the unchanged 3000 ms persistence rule.

## Stalled-path stop and alternative pose candidate
After three consecutive sessions without tested acceptance improvement on the current OpenPose/posture path (#37 no-change miss, #38 rejected correction, #39 no-change miss), further holdout-specific tuning stopped.

PR #40 merged to `main` at `20f52a58c0df4134752be4b78370f4385f25fec3` and closed the next integration/provenance boundary without promoting a model. It pins Open Model Zoo `human-pose-estimation-0005` FP16 at OMZ commit `6697dead54ed1cdd664b0313189c2cb52ee6335e` under Apache-2.0, its Apache-2.0 Associative Embedding reference decoder, and bounded evidence dependency `scipy==1.17.1` (BSD).

Pinned candidate artifacts:
- XML: **1,063,570 bytes**, SHA-384 `37595cec7cb044266eb7cb934fcf596d5b0382b12a03f5462f046a52aba3f9c96097377773618ca957d7b6941a12334b`;
- BIN: **19,039,904 bytes**, SHA-384 `ef4ab20cd0695a4b86789607acc6eb636d07fbb6786f30300ea713bb94a9110ed1ecec5db5577398ddc13663cc2ce690`.

Donor metadata only: candidate `0005` is EfficientHRNet/Associative Embedding, 17 COCO keypoints, published **45.6% COCO AP / 5.9206 GFLOPs**. Retained `0001` is published **42.8% / 15.435 GFLOPs**. Those figures do not establish Analytics Lab accuracy or runtime performance.

## Current acceptance-moving work — same-video 0001 vs 0005 A/B
Live `main` at intake is `20f52a58c0df4134752be4b78370f4385f25fec3`. Its exact-head post-merge Analytics quality run completed successfully on attempt 1. There were **0 open implementation PRs** and **0 active runs** for the live main head before this branch was prepared.

The one work item is a bounded same-video A/B on the exact Subject-4 positive/push-up-negative pair. The branch changes only pose model/decoder for the candidate arm. Both arms reuse the same pinned `person-detection-0200`, orientation recovery, continuity selection, safe overlap/edge-gap association policy, near-diagonal ambiguity handling, posture semantics, **3000 ms persistence**, **750 ms unknown-gap budget**, evaluator, media hashes and CPU device.

The candidate evidence adapter:
- downloads only the two separately reviewed `0005` artifacts into the ephemeral runner temp tree;
- verifies exact size/SHA-384/license metadata before OpenVINO opens them;
- uses preserve-aspect resize plus right/bottom zero padding into reviewed 288x288 input, without geometric distortion;
- uses the pinned OMZ Associative Embedding grouping behavior with `scipy==1.17.1` only in the evidence lane;
- maps the candidate model's 2x output stride into the retained geometry adapter without changing downstream association thresholds;
- emits only aggregate A/B evidence and retains no media/model artifacts.

Retention rule: keep `0005` only if it materially improves the positive acceptance target against the same-run `0001` baseline **and** the push-up negative remains **0 candidate events / 0 false alerts**. Record positive/negative candidates, matches, misses, false alerts, detector coverage, association/posture fragmentation, longest qualified run, alert delay if any, pose inference/decode time, total elapsed time and FPS. If the positive does not improve, reject the candidate regardless of green synthetic tests or donor-paper metrics. Do not weaken temporal/posture rules to accommodate it.

## Evidence/data provenance
### GMDCSA-24
- Primary repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`, aligned with v2.1.
- Repository license at that revision: MIT.
- Versioned dataset provenance: Zenodo v2.1 DOI `10.5281/zenodo.13354453`; paper DOI `10.1016/j.dib.2024.110892`.
- Subject-1 positive SHA-256: `1aad4e2ee93b5498283fdc4a0478a5913dffc2ac8e2538574cd821ffb1b485f1`.
- Subject-1 hard negative SHA-256: `2cf0d421cfd8e34280bf02afc67a4f1d1abee3cd87a23f11e69ef393fedc6fdb`.
- Subject-2 positive SHA-256: `0448c122dbfdffc423cc6e282cb54f60d801fc1c741de9aafca3f051db7c1abb`.
- Subject-2 prone-normal SHA-256: `24559694cceadbcf1bb34f217967c233e412b97e25f9dd9be3f733f4e16067df`.
- Subject-3 hashes are recorded above.
- Subject-4 hashes are recorded above.
- Media remains ephemeral/outside public GitHub and is re-hashed before measured execution.

### Runtime/model provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Retained pose: `human-pose-estimation-0001` FP16, exact size/hash reviewed.
- Candidate pose: `human-pose-estimation-0005` FP16, exact size/hash above.
- Evidence detector: `person-detection-0200` FP16, exact size/hash reviewed.
- OpenVINO Runtime: `2026.3.1`; OpenCV: `opencv-python-headless==4.12.0.88`; candidate decoder dependency: `scipy==1.17.1`.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. No training job, paid resource, self-hosted runner, home/customer media, duplicate agent, fourth detector or second implementation lane is introduced.

At intake: open implementation PRs **0**, active runs for live main **0**, unchanged retries **0**, CI-triggering requests **0/2**. The previous OpenPose/posture tuning path hit the three-session stop threshold and was stopped; PR #40 closed the alternative-pose rights/integration boundary, so this is the first executable measurement on the new bounded path rather than a reset of the old stalled path. Live main/open-PR/run facts were connector-verified. The connector-only worker cannot execute a repository-local clone/preflight, so no local machine preflight or test pass is claimed; exact-head hosted CI/evidence is the execution authority.

## Reproduce
Repository checks:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Rights-bound evidence path in an authorized no-spend internet-connected environment:

```sh
python -m pip install --no-cache-dir 'openvino==2026.3.1' 'opencv-python-headless==4.12.0.88' 'scipy==1.17.1'
python -m analytics_lab.validation_seed --output-dir /path/to/private-validation
python -m analytics_lab.person_down_pose_ab --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
```

## Next executable decision
Open exactly one `evidence/person-down-pose-ab-0005` PR and let the exact unchanged head run Linux, Windows, Analytics quality and the bounded real-video CPU A/B once. Read the same-run baseline/candidate output before any further mutation.

If `0005` materially improves the positive while the push-up negative remains zero-candidate/zero-false-alert and CPU cost stays inside the bounded envelope, retain the evidence adapter and merge only on the unchanged green head. If it does not improve the target, close it unmerged and record the negative result. A deterministic execution failure requires one root-cause correction plus regression; an unchanged retry is allowed only for diagnosed transient infrastructure failure.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across genuinely different cameras/sites and multi-person scenes; meaningful false-alert/miss measurements; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; platform/native dependency provenance; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Small GMDCSA-24 subject rotations do not establish commercial accuracy or commercial readiness.
