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

## Rejected alternate pose path
PR #40 merged a rights/resource review for OMZ `human-pose-estimation-0005` only; it did not promote the model.

PR #41 then ran the exact same Subject-4 A/B. The retained `0001` baseline reproduced **55 qualified down / 1312 ms** and zero negative alerts. `0005` remained **0 candidates / 1 miss**, collapsed to **1 qualified down / 0 ms**, and had materially worse aggregate CPU throughput and decode/inference cost. It was closed unmerged.

PR #42 tested `human-pose-estimation-0006`, which shares the reviewed EfficientHRNet/AE weights with `0005` at higher input resolution. The exact head passed repository tests, Linux, Windows, Analytics quality and the bounded real-video lane, but the candidate again failed the acceptance rule: **0 candidates / 1 miss**, **1 qualified down / 0 ms**, with **102 labeled-positive observations** failing the unchanged required-keypoint-confidence floor. The push-up negative remained **0 false alerts**. Although aggregate throughput improved to about **11.65 FPS** versus **9.67 FPS** baseline, usable person-down evidence was much worse. PR #42 was closed unmerged.

The EfficientHRNet/AE resolution-stepping path is closed. Do not retry `0005`/`0006`, tune them around Subject 4, lower required-keypoint confidence, bridge genuine `upright`/`other`, weaken 3000 ms persistence, extend the 750 ms unknown-gap budget, or shop another person detector.

## Live repository intake for the next work item
Live `main` is `20f52a58c0df4134752be4b78370f4385f25fec3`. Its post-merge Analytics quality run #94 completed successfully on attempt 1. At intake there are **0 open implementation PRs** and **0 active runs for the live main head**.

The previous alternate-pose work item is stopped after the rejected #41/#42 measurements. The next item therefore changes approach back to broader evidence, as required by the efficiency policy, rather than continuing model or threshold surgery.

## Current acceptance-moving work — untouched cross-view robustness pair
GMDCSA-24 contains four source subjects and Subjects 1–4 have all already been exercised. The existing bounded evidence lane does not yet have a reviewed exact-file acquisition mapping for the other registered real-world source, so the smallest executable no-spend measurement is an **untouched-file** pair from the already-approved/pinned GMDCSA-24 source. This is explicitly **within-subject cross-view/activity robustness evidence, not new-subject generalization**.

Active pair at pinned GMDCSA-24 revision `5abac7693229900cf80f722e878fbb119211fc1c`:
- positive `Subject 1/Fall/11.mp4`: **6 s**, standing to backward fall, source class `Falling (BW)[1 to 6]; Standing[0 to 1]`, Git blob `791707e0f4fd062796eb147e55f33eb80455ba38`, **5,961,144 bytes**;
- hard negative `Subject 1/ADL/16.mp4`: **3 s**, picking a mobile phone from the ground while sitting on a chair, Git blob `0991a3657ee6458f470a9bc419d901f192b43d7a`, **3,582,225 bytes**.

The pair totals **9,543,369 bytes**, remains below the existing 16 MiB aggregate evidence envelope, and is file-disjoint from every previously admitted pair. Exact SHA-256 identities must be computed from admitted bytes before inference; media remains ephemeral and outside public GitHub.

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

If this untouched backward-fall clip emits a legitimate candidate while the ground-reach negative remains clean, preserve the algorithm and expand evidence rather than tune. If it misses, isolate the largest measured failure layer before any correction. Do not alter detector identity/threshold, association bounds, posture confidence semantics, 3000 ms persistence, 750 ms unknown-gap budget, or training data to make this pair pass.

A same-subject success does not erase the misses on Subjects 2–4 and does not establish commercial readiness. Conversely, another miss that localizes to pose-confidence/posture fragmentation strengthens the case for a separately planned, rights-pinned adaptation/training program rather than more holdout-specific threshold changes.

## Evidence/data provenance
### GMDCSA-24
- repository `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`;
- pinned revision `5abac7693229900cf80f722e878fbb119211fc1c`;
- reviewed repository license: MIT;
- versioned provenance recorded as Zenodo DOI `10.5281/zenodo.13354453` and paper DOI `10.1016/j.dib.2024.110892`;
- commercial evaluation/training eligibility is recorded in `analytics_lab/source_rights.py` as an engineering provenance gate, not a legal opinion.

Historical exact media hashes remain pinned in the repository validation code and prior accepted PR evidence. The active pair is admitted only after exact Git-blob verification and local SHA-256 hashing.

### Runtime/model provenance
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0;
- retained pose model `human-pose-estimation-0001` FP16;
- retained evidence detector `person-detection-0200` FP16;
- OpenVINO Runtime `2026.3.1`;
- evidence video decoder `opencv-python-headless==4.12.0.88`;
- corrected OpenPose decoder is pinned to the same OMZ revision with attribution preserved.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation/evidence PR. No new model family, training job, paid resource, self-hosted runner, home/customer media, second framework or duplicate agent is introduced.

For this fresh evidence item: open implementation PRs **0** before branch/PR creation, active runs for live main **0**, unchanged retries **0**, CI-triggering requests **0/2** before the PR, and sessions without progress for this new evidence item **0** because the prior stalled alternate-pose path was explicitly stopped rather than renamed or continued. The smallest safe step is the two-clip measurement above.

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

If the positive matches and the ground-reach negative remains at zero candidate/false alerts, merge the reproducible measurement and select the next disjoint evidence item. If it misses or false-alerts, identify the dominant layer and make at most one coherent evidence-driven correction on the same work item, rerunning the same pair plus retained hard-negative expectations. Merge only an unchanged exact head with all required gates green and an accepted evidence result.

## Outstanding commercial-release gates
Substantially broader held-out positive/negative real-video evidence across genuinely different subjects, cameras, sites, resolutions, viewpoints, lighting and multi-person scenes; meaningful false-alert/camera-hour and miss measurements; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.
