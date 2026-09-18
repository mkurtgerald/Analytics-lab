# Project state — 2026-09-18

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current person-down path
The person-down **candidate** path under measured validation is:

`authorized local video -> reviewed detector -> spatial continuity/orientation recovery -> reviewed OpenPose model -> bounded pose association -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> labeled evaluation/aggregation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media hashes, reviewed model/runtime identities, decoded coverage, continuity, timing and evaluation metrics without retaining video in public GitHub.

## Accepted Subject-1 milestone
PR #33 was accepted and merged at `5a63abd6201a7739d7c4cd8eb95a46fdbc77e67d`. Its exact head passed Linux, Windows, the Analytics quality gate and the bounded rights-cleared real-video lane on attempt 1.

On the pinned Subject-1 pair, the retained evidence-only configuration produced the first matched staged-real person-down candidate: **1 positive episode, 1 candidate, 1 match, 0 misses, 0 hard-negative false alerts**, across about **0.0036 decoded camera-hours**. Alert delay was **4024 ms** from one matched event. The positive had 85 qualifying `down` frames. Bounded unknown-gap tolerance recovered a 3008 ms / 70-sample qualifying run while `upright`, `other`, low-confidence `down`, expiry and over-bound gaps stayed fail-closed.

This remains a tiny engineering seed, not a commercial accuracy claim.

## Held-out Subject-2 evidence
PR #34 first measured the disjoint Subject-2 pair and exposed a legitimate generalization failure: the positive emitted **0 candidates / 0 matches / 1 miss**, the prone sleeping normal emitted **0 candidates / 0 false alerts**, and labeled-fall detector selection was only **94/138 = 68.12%**.

PR #35 retained bounded same-detector orientation recovery using the exact pinned `person-detection-0200`. On Subject 2 this raised labeled-fall detector coverage to **126/138 = 91.30%**, associated poses **98 -> 135**, qualified `down` frames **31 -> 57**, and longest qualified run **768 ms -> 1472 ms** while the prone-normal negative stayed at **0 candidate events / 0 false alerts**. Evidence-only throughput was about **10.20 FPS**.

PR #36 then isolated one safely correctable near-diagonal `geometry_not_decisive` reset plus one genuinely upright reset. The retained correction converts only near-diagonal non-decisive geometry (`|horizontal_fraction - vertical_fraction| <= 0.02`) from `other` to `unknown`; it never promotes to `down`, never bridges `upright`, and changes no persistence, detector, model, association or confidence threshold.

On the same Subject-2 pair the correction improved the longest qualified positive run **1472 ms -> 2544 ms** while fall-window detector coverage remained **126/138 = 91.30%**, qualified `down` remained **57**, and the prone-normal negative stayed **0 candidates / 0 false alerts**. The positive remains **0 candidates / 0 matches / 1 miss** because the remaining decisive interruption is genuinely upright. Subject 2 stays held out from training/adaptation and is not tuned further merely to force a match.

## Held-out Subject-3 evidence and rejected crop path
PR #37 measured the unchanged best-known path on a third disjoint GMDCSA-24 pair and merged to live `main` at `fe0a5054fd01f147631b89452300d971a73a9115` after Linux, Windows, Analytics quality and bounded real-video CPU evidence passed on attempt 1.

Subject-3 exact evidence identities:
- positive `Subject 3/Fall/03.mp4` SHA-256 `267abab9b0ca7f4ef8425e2bfb2272ffeae1165fc2e3523ca1a6e0a62046b912`;
- hard negative `Subject 3/ADL/08.mp4` SHA-256 `b383bb83a22b286cfc65fdd934137ee8b29e84aef7be0b540271d7c69d10fa8b`.

Decoded evidence was **10,432 ms / 0.00289778 camera-hours**. The positive produced **0 candidates / 0 matches / 1 miss** and the sitting-to-sleeping-on-floor negative produced **0 candidate events / 0 false alerts**. Orientation-aware detector selection reached **89/105 = 84.76%** of labeled-fall frames, but only **50/105** had safe full-frame pose overlap. The positive produced 22 qualified `down` frames with a longest qualified run of **1184 ms**; the hard negative produced 27 down-like frames but only a **912 ms** longest run and did not alert.

PR #38 then tested one bounded selected-person OpenPose crop fallback using the same pinned pose model after safe full-frame association failed. Hosted Linux, 230 tests, Windows, Analytics quality and the real-video lane all passed on attempt 1, but the evidence failed the predeclared acceptance rule and PR #38 was closed unmerged. Association rose to 73/105 labeled frames, yet the positive longest qualified run regressed **1184 ms -> 512 ms** because three crop-associated poses were genuinely `upright`; the hard negative remained at **0 candidate events / 0 false alerts**. Evidence-only throughput fell to about **7.65 FPS**, with ~10.29 s of added pose inference across the tiny pair. Green CI did not override the failed acceptance target.

The crop path is therefore closed. Do not bridge contradictory upright evidence, weaken persistence, widen unsafe association, add a fourth detector, or train from this holdout.

## Current acceptance-moving work — disjoint Subject-4 generalization
Live `main` at intake is `fe0a5054fd01f147631b89452300d971a73a9115`; its post-merge Analytics quality run completed successfully on attempt 1. There are **0 open implementation PRs** and no active main-head run at intake. PR #38 is closed unmerged, so no implementation branch is active.

Two consecutive sessions did not produce a tested acceptance improvement (#37 was a no-change generalization measurement; #38 was a rejected correction). Per policy, the next step changes approach and shrinks to the smallest executable measurement rather than further tuning Subject 3.

The active PR-only evidence subset rotates to **Subject 4** from the already-reviewed GMDCSA-24 source while preserving the two-clip / <16 MiB media envelope and every existing runtime/model pin. No detector, pose model, tracking rule, association bound, posture rule, temporal threshold, training job or donor component changes in this measurement.

Selected Subject-4 evidence:
- positive `Subject 4/Fall/03.mp4`: **6 s**, source annotation `Walking then falling (forward) view 1`, classes `Falling (FW)[1.7 to 6]; Walking[0 to 1.7]`, exact pinned Git blob `3076cb2d10c3effc103fc501434ed97ebd27ec75`, **2,421,154 bytes**;
- hard negative `Subject 4/ADL/07.mp4`: **4 s**, source annotation `Doing exercise (push-up); view 3`, class `Exercising[0 to 4]`, exact pinned Git blob `8a6aad020ff6f70b01eb3ecd952b5ec445e0a90d`, **5,575,628 bytes**.

The pair totals **7,996,782 bytes**, well below the existing 16 MiB aggregate envelope, and is disjoint from Subjects 1–3. The push-up clip is intentionally posture-confusable and dynamic, making it a stronger normal-negative than another sleeping clip. Exact SHA-256 identities must be computed from admitted bytes before inference; media remains ephemeral/outside public GitHub.

Decision rule: run the unchanged retained path first. If it emits a matched positive with zero hard-negative candidate/false alerts, preserve the algorithm and expand later. If it misses or false-alerts, isolate the dominant measured layer before changing anything: detector recall -> pose association/quality -> posture classification -> temporal fragmentation. Do not weaken 3000 ms persistence, bridge genuine `upright`/`other` evidence, widen unsafe association, add a fourth detector, or train merely to make this pair pass.

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
- Held-out Subject-3 positive SHA-256: `267abab9b0ca7f4ef8425e2bfb2272ffeae1165fc2e3523ca1a6e0a62046b912`.
- Held-out Subject-3 hard-negative SHA-256: `b383bb83a22b286cfc65fdd934137ee8b29e84aef7be0b540271d7c69d10fa8b`.
- Subject-4 active pair: exact pinned Git blobs/byte counts above; SHA-256 is populated only from acquired bytes during the evidence run.
- Media remains ephemeral/outside public GitHub and is re-hashed before measured execution.

### Runtime/model provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Pose artifact: `human-pose-estimation-0001` FP16, exact size/hash reviewed.
- Evidence detector: `person-detection-0200` FP16, exact size/hash reviewed.
- OpenVINO Runtime: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- Reference OpenPose decoder source is pinned to the same OMZ commit with attribution preserved.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. No additional model family, training job, paid resource, self-hosted runner, home/customer media or duplicate agent is introduced.

At this work-item intake: open implementation PRs **0**, active runs for live main **0**, unchanged retries **0**, CI-triggering requests **0/2**, consecutive sessions without tested acceptance improvement **2**. This item explicitly changes approach by moving off Subject 3 and shrinking to a measurement-only Subject-4 pair. Live main/open-PR/run counts and source paths/labels/blob sizes were tool-verified. The connector-only worker cannot execute a local repository clone/runtime because outbound Git resolution is unavailable, so no local machine preflight/test pass is claimed.

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
Open one evidence PR from `evidence/person-down-heldout-s4` and run exact-head Linux/Windows/quality plus the bounded Subject-4 real-video lane once. Record exact SHA-256 identities, positives/candidates/matches/misses/false alerts, decoded camera-hours, alert delay when present, detector coverage, association/posture counts, longest qualified run, reset causes and CPU throughput.

Do not tune before reading this disjoint result. If the unchanged path matches the positive while the push-up negative remains at zero candidate/false alerts, this is a useful generalization gain and the next step is another small disjoint pair rather than immediate tuning. If it fails, change only the largest measured error source and rerun the same pair plus retained negative expectations. Merge only when the final unchanged head is green on Linux, Windows and Analytics quality and the real-video evidence is accepted under the predeclared decision rule. Main post-merge verification intentionally does not reacquire real-video/model evidence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across genuinely different cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; false-alert and missed-event measurements at meaningful scale; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; platform/native dependency provenance; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Small GMDCSA-24 subject rotations do not establish commercial accuracy or commercial readiness.
