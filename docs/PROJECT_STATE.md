# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current person-down path
The person-down **candidate** path under measured validation is:

`authorized local video -> reviewed detector -> spatial continuity -> reviewed OpenPose model -> bounded pose association -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> labeled evaluation/aggregation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media hashes, reviewed model/runtime identities, decoded coverage, continuity, timing and evaluation metrics without retaining video in public GitHub.

## Accepted Subject-1 milestone
PR #33 was accepted and merged at `5a63abd6201a7739d7c4cd8eb95a46fdbc77e67d`. Its exact head passed Linux, Windows, the Analytics quality gate and the bounded rights-cleared real-video lane on attempt 1.

On the pinned Subject-1 pair, the retained evidence-only configuration produced the first matched staged-real person-down candidate: **1 positive episode, 1 candidate, 1 match, 0 misses, 0 hard-negative false alerts**, across about **0.0036 decoded camera-hours**. Alert delay was **4024 ms** from one matched event. The positive had 85 qualifying `down` frames. Bounded unknown-gap tolerance bridged 21 `unknown` frames, longest bridged gap 96 ms, and recovered a 3008 ms / 70-sample qualifying run while `upright`, `other`, low-confidence `down`, expiry and over-bound gaps stayed fail-closed.

This remains a tiny engineering seed, not a commercial accuracy claim.

## Held-out Subject-2 baseline — accepted evidence, algorithm not promoted
PR #34 was merged to `main` at `49a558ee2df95afb990de377c1484c40892b7afc`. The unchanged exact PR head passed Linux, 217 repository tests, Windows, the Analytics quality gate, and the bounded real-video lane on attempt 1.

The disjoint Subject-2 pair produced a legitimate generalization failure:

- positive `Subject 2/Fall/01.mp4`: **1 labeled positive, 0 candidates, 0 matches, 1 miss**;
- prone sleeping normal `Subject 2/ADL/12.mp4`: **0 candidates, 0 false alerts**;
- decoded camera-hours: **0.0038578** across the two clips.

At the retained 0.10 detector floor, the labeled fall interval selected only **94/138 frames = 68.12%**, below the predeclared 85% continuity coverage bar. The prone-normal negative retained high detector coverage and still emitted zero candidate/false alert. The positive longest qualified down run was only 768 ms; the negative reached 960 ms, so weakening the unchanged 3000 ms persistence requirement was rejected.

The prior bounded donor shootout already consumed the permitted three detector alternatives and retained `person-detection-0200`; no fourth detector search is authorized by this work item. Subject 2 remains held out from any future training/adaptation data.

## Accepted same-detector orientation recovery
PR #35 was merged to live `main` at `5a9cf9c27df395bf28a481adf53b8433a54bbc9a`. The exact tested head `80e790cc0eb2bc63040177f9c7d6a2772455b85e` passed Linux, Windows, the Analytics quality gate, and the bounded rights-cleared real-video lane on attempt 1. Post-merge `main` verification also passed.

The retained evidence-only correction keeps the exact pinned `person-detection-0200` weights/runtime. Only on a primary-orientation miss, and only with a prior continuity box, the same detector runs on bounded +/-90-degree views. A mapped box is admitted only if it clears the existing 0.10 confidence floor and 0.05 spatial continuity IoU floor. Primary-orientation detections always win. No new detector, weights, donor, training framework, data source, or production default was added.

On held-out Subject 2:

- labeled-fall selection coverage improved from the same-run baseline **92/138 = 66.67%** to **126/138 = 91.30%**;
- 34 fall-window frames were recovered by the bounded same-detector orientation fallback;
- associated frames improved **98 -> 135**;
- qualified `down` frames improved **31 -> 57**;
- longest qualified run improved **768 ms -> 1472 ms**;
- the positive still produced **0 candidates / 0 matches / 1 miss**;
- the prone-normal negative remained **0 candidates / 0 false alerts**;
- overall evidence-only throughput was about **10.20 FPS**.

Detector coverage is therefore no longer the dominant Subject-2 blocker. The next measured wall is the remaining pose-association/posture fragmentation that breaks persistence after the detector recovers the person.

## Current acceptance-moving work — fragmentation measurement
Live `main` at intake is `5a9cf9c27df395bf28a481adf53b8433a54bbc9a`; there are **0 open implementation PRs** and no active run for that main head. The one prepared evidence branch is `evidence/person-down-fragmentation-diagnostics`.

This work changes **measurement only**. It does not alter detector, pose decoder, association thresholds, posture thresholds, temporal persistence, model artifacts, media set, or production behavior. The existing orientation diagnostic now records only aggregate, non-frame-retaining evidence for:

- association outcomes (`baseline_overlap`, bounded edge-gap recovery, unmatched/ambiguous cases, missing selected detection);
- nearest unmatched pose geometry when a decoded pose exists;
- posture and posture-basis counts by labeled window;
- the exact temporal reset reasons that interrupt an active qualifying down run;
- association reason, required-keypoint count, and posture geometry for those decisive reset frames.

No frame timestamps, images, clips, model artifacts or identity data are emitted. The goal is to determine whether the remaining Subject-2 miss is dominated by association loss or by genuinely conflicting posture geometry. Only after that exact-head real-video measurement may one bounded correction be retained. A correction must improve the same held-out positive without creating a prone-normal candidate/false alert. Persistence relaxation, a fourth detector, and Subject-2 training admission remain out of scope.

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
- Media remains ephemeral/outside public GitHub and is re-hashed before and after measured execution.

### Runtime/model provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Pose artifact: `human-pose-estimation-0001` FP16, exact size/hash reviewed.
- Evidence detector: `person-detection-0200` FP16, exact size/hash reviewed.
- OpenVINO Runtime: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- Reference OpenPose decoder source is pinned to the same OMZ commit with attribution preserved.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. No additional model family, training job, paid resource, self-hosted runner, home/customer media or duplicate agent is introduced.

At this work-item intake: open implementation PRs **0**, active runs for live main **0**, unchanged retries **0**, CI-triggering requests **0/2**, consecutive sessions without tested acceptance progress **0**. Live main/open-PR/run counts were tool-verified. The repository-local preflight logic was executed against that verified snapshot and allowed the implementation action. Branch preparation is batched before opening the one PR so exact-head CI/evidence runs once.

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
Open one evidence PR from `evidence/person-down-fragmentation-diagnostics` and run exact-head Linux/Windows/quality plus the existing bounded Subject-2 real-video lane once. Read the new aggregate `fragmentation_windows.during` evidence. If one measured association or posture defect clearly dominates, make at most one bounded correction on this same branch, add regression coverage, and compare against the same held-out positive and prone-normal negative. If the measurement does not isolate a safe correction, do not guess or relax persistence; record the blocker and move to the next evidence-supported path.

Merge only when the final unchanged head is green on Linux, Windows and Analytics quality and the bounded real-video evidence shows a tested acceptance improvement without hard-negative alert regression. Main post-merge verification intentionally does not reacquire real-video/model evidence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across genuinely different cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; false-alert and missed-event measurements at meaningful scale; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; platform/native dependency provenance; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Small GMDCSA-24 subject rotations do not establish commercial accuracy or commercial readiness.
