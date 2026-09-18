# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current person-down path
The person-down **candidate** path under measured validation is:

`authorized local video -> reviewed detector -> spatial continuity/orientation recovery -> reviewed OpenPose model -> bounded full-frame pose association -> optional bounded selected-person crop using the same pose model after full-frame association failure -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> labeled evaluation/aggregation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media hashes, reviewed model/runtime identities, decoded coverage, continuity, timing and evaluation metrics without retaining video in public GitHub.

## Accepted staged-real milestones
### Subject 1
PR #33 produced the first matched staged-real person-down candidate on the pinned Subject-1 pair: **1 positive episode, 1 candidate, 1 match, 0 misses, 0 hard-negative false alerts** across about **0.0036 decoded camera-hours**. Alert delay was **4024 ms**. The retained path used the unchanged 3000 ms persistence requirement plus bounded unknown-gap tolerance; `upright`, `other`, low-confidence `down`, expiry and over-bound gaps remained fail-closed. This is a tiny engineering seed, not a commercial accuracy claim.

### Subject 2
PR #34 exposed a disjoint generalization miss: **0 candidates / 0 matches / 1 miss**, with **0 hard-negative false alerts** and only **94/138 = 68.12%** labeled-fall detector selection.

PR #35 retained same-detector +/-90-degree orientation recovery only after a primary miss with prior continuity. On Subject 2, labeled-fall detector coverage rose to **126/138 = 91.30%**, associated poses **98 -> 135**, qualified `down` frames **31 -> 57**, and longest qualified run **768 ms -> 1472 ms**, while the prone-normal negative stayed at **0 candidate events / 0 false alerts**. Evidence-only throughput was about **10.20 FPS**.

PR #36 then removed only one measured false reset: near-diagonal non-decisive torso geometry (`|horizontal_fraction - vertical_fraction| <= 0.02`) is treated as `unknown`, never promoted to `down`. The longest qualified Subject-2 positive run improved **1472 ms -> 2544 ms** while the negative remained **0 candidates / 0 false alerts**. The remaining decisive break is genuinely upright and stays fail-closed. Subject 2 remains held out and must not be tuned further merely to manufacture a match.

### Subject 3 — no-change generalization measurement
PR #37 merged to live `main` at `fe0a5054fd01f147631b89452300d971a73a9115`. Its exact PR head passed Linux, **226 tests**, Windows, the Analytics quality gate and the bounded rights-cleared real-video CPU lane on attempt 1; post-merge main verification also passed.

Rights-cleared Subject-3 evidence:
- positive `Subject 3/Fall/03.mp4` SHA-256 `267abab9b0ca7f4ef8425e2bfb2272ffeae1165fc2e3523ca1a6e0a62046b912`;
- hard negative `Subject 3/ADL/08.mp4` SHA-256 `b383bb83a22b286cfc65fdd934137ee8b29e84aef7be0b540271d7c69d10fa8b`;
- decoded evidence **10,432 ms / 0.00289778 camera-hours** across the two independent clip streams.

The unchanged best-known path produced **1 positive episode, 0 candidates, 0 matches, 1 miss** and **0 hard-negative candidate events / 0 false alerts**. Detector orientation recovery selected **89/105 = 84.76%** labeled-fall frames with about **94.44% linked-transition continuity**. The larger loss was pose association: only **50/105** labeled-fall frames had safe full-frame pose overlap. Another 39 selected frames had a decoded pose, but nearest pose geometry was far outside the safe association bound (edge-gap approximately **0.567 minimum, 1.206 median, 1.524 p90**), so widening that association threshold is not justified.

The positive produced **22 qualified `down` frames** and a longest qualified run of **1184 ms** versus the unchanged **3000 ms** requirement. The sleeping-on-floor negative produced **27 down-like frames** but only a **912 ms** longest run and still emitted no alert. Do not lower persistence to make the positive pass.

## Current acceptance-moving work — bounded selected-person pose crop
Live `main` at intake is `fe0a5054fd01f147631b89452300d971a73a9115`. There are **0 open implementation PRs** and no active run for that exact main head; Linux, Windows and Analytics quality are completed/success on main. The one prepared branch is `evidence/person-down-pose-crop-fallback`.

The measured Subject-3 defect supports one change of approach at the pose layer. After a continuity-selected person exists and **only after the existing full-frame OpenPose association plus its existing bounded fallback fail**, run the exact same pinned `human-pose-estimation-0001` model and decoder on a bounded crop around that already-selected detector box. The crop adds a fixed 20% context margin. If a wide crop would exceed the reviewed 456x256 OpenPose aspect ratio, consume available source-frame vertical context first, then add only the minimum black vertical padding; do not distort geometry. Apply the same safe pose association logic inside the crop. No detector, weights, detector threshold, pose weights, posture threshold, 3000 ms persistence, 750 ms unknown-gap budget or training data changes are allowed in this item.

Predeclared decision rule: retain the crop fallback only if the exact same Subject-3 positive materially improves pose association / qualified temporal continuity and the sleeping-on-floor negative remains at **0 candidate events / 0 false alerts**. A matched positive is welcome but not required to demonstrate improvement; no change is retained for mere test-count growth. Record crop attempts/associations, additional pose inference/decode cost, total FPS, full end-to-end candidate/match/miss/false-alert results and prior fragmentation metrics. If the crop increases hard-negative alerting or produces no measured acceptance movement, remove/abandon it rather than loosening temporal policy.

## Evidence/data provenance
### GMDCSA-24
- Primary repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`, authored 2024-08-21 and aligned with v2.1.
- Repository license at that revision: MIT.
- Versioned dataset provenance: Zenodo v2.1 DOI `10.5281/zenodo.13354453`; associated paper DOI `10.1016/j.dib.2024.110892`.
- Subject-1 positive SHA-256: `1aad4e2ee93b5498283fdc4a0478a5913dffc2ac8e2538574cd821ffb1b485f1`.
- Subject-1 hard negative SHA-256: `2cf0d421cfd8e34280bf02afc67a4f1d1abee3cd87a23f11e69ef393fedc6fdb`.
- Held-out Subject-2 positive SHA-256: `0448c122dbfdffc423cc6e282cb54f60d801fc1c741de9aafca3f051db7c1abb`.
- Held-out Subject-2 prone-normal SHA-256: `24559694cceadbcf1bb34f217967c233e412b97e25f9dd9be3f733f4e16067df`.
- Subject-3 hashes are recorded above.
- Media remains ephemeral/outside public GitHub and is re-hashed before and after measured execution.

### Runtime/model provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Pose artifact: `human-pose-estimation-0001` FP16, exact size/hash reviewed.
- Evidence detector: `person-detection-0200` FP16, exact size/hash reviewed.
- OpenVINO Runtime: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- Reference OpenPose decoder source is pinned to the same OMZ commit with attribution preserved.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. No additional model family, training job, paid resource, self-hosted runner, home/customer media or duplicate agent is introduced.

At this work-item intake: open implementation PRs **0**, active runs for live main **0**, unchanged retries **0**, CI-triggering requests **0/2**, consecutive sessions without tested acceptance improvement **1** (PR #37 was a valid no-change generalization measurement). This item therefore changes approach at the measured pose layer instead of repeating the same path. Live main/open-PR/check facts were tool-verified. The connector-only environment cannot execute a local repository clone/runtime, so no local `guardrails.py preflight` or local machine test pass is claimed; exact-head hosted CI/evidence remains the execution authority.

## Reproduce
Repository checks:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Rights-bound evidence path in an authorized internet-connected no-spend environment:

```sh
python -m analytics_lab.validation_seed --output-dir /path/to/private-validation
python -m analytics_lab.validation_cli --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.detector_diagnostics --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.detector_thresholds --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.pose_diagnostics --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.openpose_diagnostics_v2 --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.person_down_e2e_diagnostics --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.person_down_orientation_diagnostics --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
```

## Next executable decision
Open one evidence PR from `evidence/person-down-pose-crop-fallback` and run exact-head Linux/Windows/quality plus the bounded Subject-3 real-video lane once. Compare the crop-enabled final diagnostic to the already-recorded Subject-3 baseline on the exact same hashes. Retain only measured acceptance movement without hard-negative candidate/false-alert regression. One unchanged retry is permitted only for a diagnosed transient infrastructure fault; deterministic failures require correction before rerun. Merge only when the final unchanged head is green on Linux, Windows and Analytics quality and the real-video result satisfies the predeclared decision rule.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across genuinely different cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; false-alert and missed-event measurements at meaningful scale; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; platform/native dependency provenance; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Small GMDCSA-24 subject rotations do not establish commercial accuracy or commercial readiness.
