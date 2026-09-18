# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current person-down path
The person-down **candidate** path under measured validation is:

`authorized local video -> reviewed detector -> spatial continuity/orientation recovery -> reviewed OpenPose model -> bounded pose association -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> labeled evaluation/aggregation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media hashes, reviewed model/runtime identities, decoded coverage, continuity, timing and evaluation metrics without retaining video in public GitHub.

## Accepted Subject-1 milestone
PR #33 was accepted and merged at `5a63abd6201a7739d7c4cd8eb95a46fdbc77e67d`. Its exact head passed Linux, Windows, the Analytics quality gate and the bounded rights-cleared real-video lane on attempt 1.

On the pinned Subject-1 pair, the retained evidence-only configuration produced the first matched staged-real person-down candidate: **1 positive episode, 1 candidate, 1 match, 0 misses, 0 hard-negative false alerts**, across about **0.0036 decoded camera-hours**. Alert delay was **4024 ms** from one matched event. The positive had 85 qualifying `down` frames. Bounded unknown-gap tolerance bridged 21 `unknown` frames, longest bridged gap 96 ms, and recovered a 3008 ms / 70-sample qualifying run while `upright`, `other`, low-confidence `down`, expiry and over-bound gaps stayed fail-closed.

This remains a tiny engineering seed, not a commercial accuracy claim.

## Held-out Subject-2 evidence
PR #34 first measured the disjoint Subject-2 pair and exposed a legitimate generalization failure: the positive emitted **0 candidates / 0 matches / 1 miss**, the prone sleeping normal emitted **0 candidates / 0 false alerts**, and labeled-fall detector selection was only **94/138 = 68.12%**.

PR #35 retained a bounded same-detector orientation recovery using the exact pinned `person-detection-0200`: only after a primary-orientation miss and only with a prior continuity box, the same detector may run on +/-90-degree views. A mapped candidate must still clear the existing 0.10 confidence and 0.05 IoU continuity floors. On Subject 2 this raised labeled-fall detector coverage to **126/138 = 91.30%**, associated poses **98 -> 135**, qualified `down` frames **31 -> 57**, and longest qualified run **768 ms -> 1472 ms** while the prone-normal negative stayed at **0 candidate events / 0 false alerts**. Evidence-only throughput was about **10.20 FPS**.

PR #36 then measured the remaining fragmentation. The exact tested head passed Linux, 226 tests, Windows, the Analytics quality gate and the bounded real-video lane on attempt 1, and merged to live `main` at `5201af0b28af5440636c22bd79fbd0a17a5160c0`; post-merge main verification also passed. Measurement isolated one safely correctable near-diagonal `geometry_not_decisive` reset plus one genuinely upright reset. The retained correction converts only near-diagonal non-decisive geometry (`|horizontal_fraction - vertical_fraction| <= 0.02`) from `other` to `unknown`; it never promotes to `down`, never bridges `upright`, and changes no persistence, detector, model, association or confidence threshold.

On the same Subject-2 pair the correction improved the longest qualified positive run **1472 ms -> 2544 ms** while fall-window detector coverage remained **126/138 = 91.30%**, qualified `down` remained **57**, and the prone-normal negative stayed **0 candidates / 0 false alerts** with a 960 ms longest qualified run. The positive remains **0 candidates / 0 matches / 1 miss** because the remaining decisive interruption is genuinely upright. Broad association relaxation is not supported by the measured unmatched-pose geometry, and the 3000 ms persistence requirement remains unchanged.

Subject 2 stays held out from future training/adaptation. Do not tune further around that clip merely to force a match.

## Current acceptance-moving work — disjoint Subject-3 generalization
Live `main` at intake is `5201af0b28af5440636c22bd79fbd0a17a5160c0`; its post-merge Analytics quality run completed successfully on attempt 1. There are **0 open implementation PRs** and no active run for the main head at intake. The one prepared branch is `evidence/person-down-heldout-s3`.

The active PR-only evidence subset rotates to another subject from the already-reviewed GMDCSA-24 source while preserving the two-clip / <16 MiB media envelope and all existing runtime/model pins. No detector, pose model, tracking rule, association bound, posture rule, temporal threshold, training job or donor component changes in this first measurement.

Selected Subject-3 evidence:

- positive `Subject 3/Fall/03.mp4`: backward floor fall after standing; source annotation `Falling (BW)[1.5 to 5]`; exact pinned Git blob `41daaee74ceb00e8db32f34525a57c704994f0b5`, **5,753,609 bytes**, clip end 5000 ms;
- hard negative `Subject 3/ADL/08.mp4`: sitting to sleeping on the floor; source annotation `Sitting[0 to 0.8]; Sleeping[0.8 to 4]`; exact pinned Git blob `a38dca4aed2a27d930354beeb7d36911a7b8dd55`, **5,172,948 bytes**, clip end 4000 ms.

The pair totals **10,926,557 bytes**, remains below the existing 16 MiB aggregate envelope, and is disjoint from the retained Subject-1 and Subject-2 clips. Exact SHA-256 identities must be computed after acquisition and before inference; media remains ephemeral/outside public GitHub.

First decision rule: run the unchanged best-known path and record the real result before tuning. If it emits a matched positive with zero hard-negative candidate/false alert, preserve the algorithm and expand to another bounded disjoint pair later. If it misses or false-alerts, the next work item attacks only the measured dominant layer: detector recall -> pose association/quality -> posture classification -> temporal fragmentation. Do not weaken the 3000 ms persistence requirement, bridge upright/other evidence, broaden unsafe association, add a fourth detector, or train without multiple disjoint measurements justifying it.

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
- Subject-3 active pair: exact pinned Git blobs/byte counts above; SHA-256 is intentionally populated only from the acquired bytes during the evidence run.
- Media remains ephemeral/outside public GitHub and is re-hashed before and after measured execution.

### Runtime/model provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Pose artifact: `human-pose-estimation-0001` FP16, exact size/hash reviewed.
- Evidence detector: `person-detection-0200` FP16, exact size/hash reviewed.
- OpenVINO Runtime: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- Reference OpenPose decoder source is pinned to the same OMZ commit with attribution preserved.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. No additional model family, training job, paid resource, self-hosted runner, home/customer media or duplicate agent is introduced.

At this work-item intake: open implementation PRs **0**, active runs for live main **0**, unchanged retries **0**, CI-triggering requests **0/2**, consecutive sessions without tested acceptance progress **0**. Live main/open-PR/run counts were tool-verified. Branch preparation is batched before opening the one PR so exact-head CI/evidence runs once.

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
Open one evidence PR from `evidence/person-down-heldout-s3` and run exact-head Linux/Windows/quality plus the bounded Subject-3 real-video lane once. Do not tune before reading the first disjoint result. Record exact SHA-256 identities, positives/candidates/matches/misses/false alerts, decoded camera-hours, alert delay when present, detector coverage, association/posture counts, longest qualified run, reset causes and CPU throughput.

Only if that measurement isolates a clear dominant defect may one coherent bounded correction be made on the same PR and compared against the same Subject-3 pair plus retained hard-negative expectations. Merge only when the final unchanged head is green on Linux, Windows and Analytics quality and the real-video evidence shows a tested acceptance improvement or a valid no-change generalization result without hard-negative alert regression. Main post-merge verification intentionally does not reacquire real-video/model evidence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across genuinely different cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; false-alert and missed-event measurements at meaningful scale; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; platform/native dependency provenance; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Small GMDCSA-24 subject rotations do not establish commercial accuracy or commercial readiness.
