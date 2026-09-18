# Project state — 2026-09-18

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Retained person-down path
The retained evidence path is:

`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media/model/runtime identities and aggregate measurements without retaining source video in public GitHub.

## Accepted evidence to date
### Subject 1 seed
PR #33 produced the first matched staged-real candidate on the pinned Subject-1 pair: **1 labeled positive, 1 candidate, 1 match, 0 misses, 0 hard-negative false alerts**, about **0.0036 decoded camera-hours**, and **4024 ms** alert delay. The positive contained 85 qualifying `down` observations and a 3008 ms qualifying run after bounded unknown-gap handling. This is a tiny engineering seed, not a commercial accuracy claim.

### Held-out Subject 2
PR #35 retained same-detector orientation recovery, raising labeled-fall detector coverage from **66.67% to 91.30%**, associated poses **98 -> 135**, qualified `down` observations **31 -> 57**, and longest qualified run **768 -> 1472 ms**, while the prone sleeping normal remained **0 candidates / 0 false alerts**.

PR #36 retained one narrow posture-ambiguity correction: only near-diagonal `geometry_not_decisive` observations (`|horizontal_fraction - vertical_fraction| <= 0.02`) become `unknown`; they are never promoted to `down`. The Subject-2 longest run improved **1472 -> 2544 ms**, but one genuinely upright observation still correctly resets the run. The positive remains **0 candidates / 1 miss** and Subject 2 stays held out from training/adaptation.

### Held-out Subject 3
PR #37 measured the unchanged retained path. The positive produced **0 candidates / 1 miss** and the sitting-to-sleeping-on-floor negative produced **0 candidates / 0 false alerts**. Fall-window detector selection was **89/105 = 84.76%**, safe full-frame pose association **50/105**, and longest qualified positive run **1184 ms**.

PR #38 tested a selected-person crop fallback with the same OpenPose model. Association improved but the actual target regressed **1184 -> 512 ms**, introduced genuine upright contradictions, and reduced throughput. It was closed unmerged.

### Held-out Subject 4
PR #39 measured the unchanged retained path on `Subject 4/Fall/03.mp4` plus the dynamic push-up negative `Subject 4/ADL/07.mp4`. Exact admitted SHA-256 identities were:
- positive `4f178dea77bf9abb9036bcd7fedfbe345358fc2d63e1704dc49b7ec1a621e2f7`;
- negative `6e410abbc6d4ef7bf52ac835b952a74a94311d0977fdfddb21885477602b7b69`.

Decoded evidence was **11,632 ms / 0.0032311111 camera-hours**. The positive produced **0 candidates / 1 miss**; the negative produced **0 candidates / 0 false alerts**. Detector selection, continuity and safe pose association were all **129/129 = 100%** in the labeled fall window, decisively localizing the remaining failure downstream. The positive had **55 qualified down observations / 1312 ms longest run**; the push-up negative had **44 down-like observations / 464 ms longest run** and did not alert. This argues against weakening the 3000 ms persistence rule.

### Untouched-file Subject 1 robustness rotation
PR #43 measured the unchanged retained path on `Subject 1/Fall/11.mp4` plus `Subject 1/ADL/16.mp4` and merged to `main` at `b8d800498086a1d929846ae429b900fb62128299`. Exact admitted SHA-256 identities were:
- positive `c25e4fc9b669cb588b6145d2cfd94f2d403083293c08bc85fa863111372bfea7`;
- negative `e4e8574c3f9da325757a17f123807ff2ffe31abca4bbbb146a67e5d5f82dad77`.

The final retained orientation/posture path produced **1 candidate / 1 match / 0 misses** on the positive with **3992 ms** alert delay, **150/150 detector coverage** through the labeled interval, **175/181** safe pose associations overall, **107 qualified down observations**, and a **4096 ms** longest qualified run. The ground-reach negative remained **0 candidates / 0 false alerts** with only a **128 ms** longest qualified run. Aggregate decoded evidence was **9552 ms / 0.0026533333 camera-hours**, and the final evidence path ran at about **10.04 FPS** on hosted CPU. This is useful robustness evidence only; it does not erase Subjects 2–4 misses or establish commercial accuracy.

## Rejected alternate pose path
PR #40 merged a rights/resource review for OMZ `human-pose-estimation-0005` only; it did not promote the model.

PR #41 ran the exact same Subject-4 A/B. The retained `0001` baseline reproduced **55 qualified down / 1312 ms** and zero negative alerts. `0005` remained **0 candidates / 1 miss**, collapsed to **1 qualified down / 0 ms**, and had materially worse aggregate CPU throughput and decode/inference cost. It was closed unmerged.

PR #42 tested `human-pose-estimation-0006`, which shares the reviewed EfficientHRNet/AE weights with `0005` at higher input resolution. It again failed the acceptance rule: **0 candidates / 1 miss**, **1 qualified down / 0 ms**, with **102 labeled-positive observations** failing the unchanged required-keypoint-confidence floor. The push-up negative remained **0 false alerts**. Although aggregate throughput improved to about **11.65 FPS** versus **9.67 FPS** baseline, usable person-down evidence was much worse. PR #42 was closed unmerged.

The EfficientHRNet/AE resolution-stepping path is closed. Do not retry `0005`/`0006`, tune them around Subject 4, lower required-keypoint confidence, bridge genuine `upright`/`other`, weaken 3000 ms persistence, extend the 750 ms unknown-gap budget, or shop another person detector.

## Live repository intake
Live `main` at this work item's start is `b8d800498086a1d929846ae429b900fb62128299`, the merge of PR #43. Its post-merge Analytics quality run #104 completed successfully on attempt 1. At intake there were **0 open implementation PRs** and **0 active runs for the live main head**.

The first untouched robustness rotation succeeded without tuning, so the next acceptance-moving action is another disjoint measurement with the algorithm still unchanged. The other registered real-world source remains eligible in the rights registry, but a reviewed bounded per-video acquisition map has not yet been established for the existing five-minute hosted evidence lane. Until that boundary is closed, the smallest no-spend measurement is another exact-file pair from pinned GMDCSA-24.

## Current acceptance-moving work — Subject 2 night/cross-view robustness pair
This is **not new-subject generalization** and it is **not Subject-2 tuning**. It deliberately revisits Subject 2 using two files never admitted before, with a different positive viewpoint/lighting and a different hard-negative transition. The previously missed Subject-2 side-fall clip remains held out and untouched.

Active pair at pinned GMDCSA-24 revision `5abac7693229900cf80f722e878fbb119211fc1c`:
- positive `Subject 2/Fall/13.mp4`: **8 s**, night, backward fall in side view, source class `Falling (BW)[2.5 to 8]; Standing[0 to 2.5]`, Git blob `dfc8b89268975d8200782a2d396309dfe76c13b0`, **1,617,503 bytes**;
- hard negative `Subject 2/ADL/10.mp4`: **7 s**, standing up from the ground and closing the curtain, then walking, source classes `Standing[0.1 to 2.1]; Walking[2.1 to 5]`, Git blob `f50ccfcbaf330a5128247baa828c7464932accab`, **7,729,870 bytes**.

The pair totals **9,347,373 bytes**, stays below the existing 16 MiB aggregate evidence envelope, and is file-disjoint from every previously admitted pair. The positive provides **5.5 seconds of labeled fall time**, so the unchanged 3000 ms persistence rule has a fair opportunity to succeed rather than being tested against a mathematically too-short label window. Exact SHA-256 identities must be computed from admitted bytes before inference; media remains ephemeral and outside public GitHub.

### Predeclared measurement rule
Run the **unchanged retained path first**. Record:
- exact admitted SHA-256 identities and decoded camera-hours;
- positive episodes, candidates, matches, misses and alert delay if matched;
- negative candidates and false alerts;
- labeled-window detector coverage and continuity;
- safe pose association coverage;
- posture/basis and required-keypoint-confidence failure counts;
- qualified `down` observations and longest qualifying run;
- exact temporal reset causes;
- pose inference/decode time, total elapsed time and aggregate FPS.

If this untouched night side-view positive emits a legitimate candidate while the floor-to-standing negative remains clean, preserve the algorithm and expand evidence rather than tune. If it misses, use the aggregate diagnostics to identify the dominant layer, but do **not** tune around Subject 2. A repeated pose-confidence/posture-fragmentation failure across this disjoint clip would strengthen the case for a separately planned, rights-pinned adaptation/training program.

## Evidence/data provenance
### GMDCSA-24
- repository `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`;
- pinned revision `5abac7693229900cf80f722e878fbb119211fc1c`;
- reviewed repository license: MIT;
- versioned provenance recorded as Zenodo DOI `10.5281/zenodo.13354453` and paper DOI `10.1016/j.dib.2024.110892`;
- commercial evaluation/training eligibility is recorded in `analytics_lab/source_rights.py` as an engineering provenance gate, not a legal opinion.

Historical exact media hashes remain pinned in repository validation code and prior accepted PR evidence. The active pair is admitted only after exact Git-blob verification and local SHA-256 hashing.

### Runtime/model provenance
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0;
- retained pose model `human-pose-estimation-0001` FP16;
- retained evidence detector `person-detection-0200` FP16;
- OpenVINO Runtime `2026.3.1`;
- evidence video decoder `opencv-python-headless==4.12.0.88`;
- corrected OpenPose decoder is pinned to the same OMZ revision with attribution preserved.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation/evidence PR. No new model family, training job, paid resource, self-hosted runner, home/customer media, second framework or duplicate agent is introduced.

For this evidence item at intake: open implementation PRs **0**, active runs for live main **0**, unchanged retries **0**, CI-triggering requests **0/2** before the PR, and sessions without tested progress **0** because PR #43 produced an accepted robustness measurement. The preflight allowed the bounded implementation action. Branch preparation does not trigger CI; one PR opening will consume the first CI-triggering request.

## Reproduce
Repository checks:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Rights-bound evidence path in the authorized hosted CPU lane:

```sh
python -m analytics_lab.validation_seed --output-dir /path/to/private-validation
python -m analytics_lab.validation_cli --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.person_down_orientation_diagnostics --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
```

## Next executable decision
Open one `evidence/*` PR containing only the bounded active-pair rotation, its regression tests and this state update. Run the exact unchanged PR head once through Linux, Windows, Analytics quality and the rights-cleared real-video CPU lane. Do not tune before reading the result.

If the positive matches and the floor-to-standing negative remains at zero candidate/false alerts, merge the reproducible measurement and select the next disjoint evidence item. If it misses, merge only if the measurement itself is reproducible and all required gates are green; record the dominant measured failure and move away from Subject-2-specific tuning. Any algorithm correction requires independent evidence that the same failure mode reproduces beyond this subject.

## Outstanding commercial-release gates
Substantially broader held-out positive/negative real-video evidence across genuinely different subjects, cameras, sites, resolutions, viewpoints, lighting and multi-person scenes; meaningful false-alert/camera-hour and miss measurements; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.