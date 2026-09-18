# Project state — 2026-09-18

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Retained person-down path
The retained evidence path is:

`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media/model/runtime identities and aggregate measurements without retaining source video in public GitHub.

## Accepted evidence frontier
### Subject 1 seed
PR #33 produced the first matched staged-real candidate on the pinned Subject-1 pair: **1 labeled positive, 1 candidate, 1 match, 0 misses, 0 hard-negative false alerts**, about **0.0036 decoded camera-hours**, and **4024 ms** alert delay. The positive contained 85 qualifying `down` observations and a 3008 ms qualifying run after bounded unknown-gap handling. This remains a tiny engineering seed, not a commercial accuracy claim.

### Held-out Subject 2
PR #35 retained orientation recovery, raising labeled-fall detector coverage from **66.67% to 91.30%**, associated poses **98 -> 135**, qualified `down` observations **31 -> 57**, and longest qualified run **768 -> 1472 ms**, while the prone sleeping negative remained **0 candidates / 0 false alerts**.

PR #36 retained one narrow posture-ambiguity correction: only near-diagonal `geometry_not_decisive` observations (`|horizontal_fraction - vertical_fraction| <= 0.02`) become `unknown`; they are never promoted to `down`. The Subject-2 longest run improved **1472 -> 2544 ms**, but one genuinely upright observation still correctly resets the run. The positive remains **0 candidates / 1 miss** and Subject 2 stays held out from training/adaptation.

### Held-out Subject 3
PR #37 measured the unchanged retained path. The positive produced **0 candidates / 1 miss** and the sitting-to-sleeping-on-floor negative produced **0 candidates / 0 false alerts**. Fall-window detector selection was **89/105 = 84.76%**, safe full-frame pose association **50/105**, and longest qualified positive run **1184 ms**.

PR #38 tested a selected-person crop fallback with the same OpenPose model. Association improved but the actual target regressed **1184 -> 512 ms**, introduced genuine upright contradictions, and reduced throughput. It was closed unmerged.

### Held-out Subject 4
PR #39 measured the unchanged retained path on `Subject 4/Fall/03.mp4` plus the dynamic push-up negative `Subject 4/ADL/07.mp4`. The positive produced **0 candidates / 1 miss**; the negative produced **0 candidates / 0 false alerts**. Detector selection, continuity and safe pose association were all **129/129 = 100%** in the labeled fall window. The positive had **55 qualified down observations / 1312 ms longest run**; the push-up negative had **44 down-like observations / 464 ms longest run** and did not alert. This argues against weakening the 3000 ms persistence rule.

### Untouched-file Subject 1 robustness rotation
PR #43 measured `Subject 1/Fall/11.mp4` plus `Subject 1/ADL/16.mp4` with no algorithm tuning. The positive produced **1 candidate / 1 match / 0 misses**, **3992 ms** alert delay, **150/150 detector coverage**, **175/181** safe pose associations overall, **107 qualified down observations**, and a **4096 ms** longest qualified run. The ground-reach negative remained **0 candidates / 0 false alerts** with a **128 ms** longest qualified run. Aggregate evidence throughput was about **10.04 FPS** on hosted CPU.

### Untouched-file Subject 2 night/cross-view rotation
PR #44 merged to `main` at `9bb42dfc8841f55717fb445855f5b276cfc6d320`. The unchanged retained path missed the positive (**0 candidates / 1 miss**) while the floor-to-standing/walking negative remained **0 candidates / 0 false alerts**. Positive detector selection remained strong at **76/82 = 92.7%** in the labeled interval, but safe pose association fell to **32/82**, with **50/82 `no_associated_pose`** observations and a **736 ms** longest qualified down run. The negative had **226/226** safe pose associations and only a **96 ms** longest down-like run. Broadening association was rejected because the unmatched decoded poses were not borderline matches.

These measurements increasingly localize the limiting robustness problem downstream of detector continuity, primarily around pose representation/association/posture continuity under difficult view/lighting conditions. The detector path remains retained.

## Rejected alternate pose path
PR #40 merged a rights/resource review for OMZ `human-pose-estimation-0005` only; it did not promote the model.

PR #41 ran the exact same Subject-4 A/B. The retained `0001` baseline reproduced **55 qualified down / 1312 ms** and zero negative alerts. `0005` remained **0 candidates / 1 miss**, collapsed to **1 qualified down / 0 ms**, and was closed unmerged.

PR #42 tested `human-pose-estimation-0006`, which shares the reviewed EfficientHRNet/AE weights with `0005` at higher input resolution. It again failed the acceptance rule: **0 candidates / 1 miss**, **1 qualified down / 0 ms**, with **102 labeled-positive observations** failing the unchanged required-keypoint-confidence floor. The push-up negative remained **0 false alerts**. Although aggregate throughput improved to about **11.65 FPS** versus **9.67 FPS** baseline, usable person-down evidence was much worse. PR #42 was closed unmerged.

The EfficientHRNet/AE resolution-stepping path is closed. Do not retry `0005`/`0006`, tune them around Subject 4, lower required-keypoint confidence, bridge genuine `upright`/`other`, weaken 3000 ms persistence, extend the 750 ms unknown-gap budget, or shop another person detector.

## Stalled same-source rotation closed
PR #45 attempted another file-disjoint GMDCSA-24 robustness pair on Subject 3. Two exact-head runs passed repository guardrails/synthetic regressions but failed deterministically at the `validation_cli` boundary before any real-video evidence was emitted. The single evidence-bound correction did not remove the failure. PR #45 was therefore closed unmerged rather than spending a third session on the same path. No detector/pose/temporal conclusion is drawn from that failure.

This satisfies the efficiency-policy requirement to change approach after repeated sessions without tested acceptance improvement. Do not reopen PR #45, retry the same pair, or use it as justification for threshold/model changes.

## Current acceptance-moving work — bounded Figshare acquisition boundary
The next approach is broader-source evidence rather than additional tuning on the four-subject GMDCSA source.

The already-reviewed real-world source registry contains:
- source: `figshare-fall-2017-activities`;
- title: `Video-Based Fall Detection Dataset with 2017 Activities from 29 Subjects`;
- pinned version: `figshare:28596332:v2:2025-03-14`;
- reviewed license: `CC-BY-4.0`;
- provenance: `figshare:28596332:version-2`;
- commercial training/evaluation eligibility recorded as an engineering provenance gate, not a legal opinion.

Public source metadata exposes Figshare article **28596332 version 2** and reviewed file id **52990358**. The source is approximately **2.36 GB** as published, so downloading the monolithic object is outside the existing five-minute / bounded-evidence design and is not authorized merely because the repository is public.

The current implementation work therefore adds a fail-closed acquisition adapter that can:
1. request only bounded Figshare article JSON metadata;
2. bind the exact reviewed article/version/file id, title, CC BY 4.0 license metadata, file size, HTTPS download URL and provider-supplied MD5 identity;
3. require exact HTTP `206` byte-range responses and reject servers that fall back to full-body `200` responses;
4. inspect only a bounded ZIP tail and central directory (maximum 128 KiB tail and 2 MiB central index);
5. reject multi-disk, encrypted, ZIP64-sentinel or unsafe-path cases rather than broadening the parser silently; and
6. identify only archive video members already inside the existing **16 MiB per-item ceiling**, without downloading member payloads.

This change does **not** admit any Figshare video, run inference, train a model, retain media, alter the retained person-down algorithm, change CI resource ceilings, or make a commercial-accuracy claim. Offline unit tests use generated ZIP bytes only and are synthetic acquisition-contract tests, not video-accuracy evidence.

## Evidence/data provenance
### GMDCSA-24
- repository `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`;
- pinned revision `5abac7693229900cf80f722e878fbb119211fc1c`;
- reviewed repository license: MIT;
- versioned provenance: Zenodo DOI `10.5281/zenodo.13354453` and paper DOI `10.1016/j.dib.2024.110892`.

### Figshare 2017-activity video source
- Figshare article `28596332`, version 2 dated 2025-03-14;
- reviewed source id `figshare-fall-2017-activities`;
- reviewed license `CC-BY-4.0`;
- reviewed provenance `figshare:28596332:version-2`;
- exact media members are **not admitted yet**. A future member may enter evaluation only after an exact bounded archive/member map, attribution/provenance record, size and cryptographic identity are recorded.

### Runtime/model provenance
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0;
- retained pose model `human-pose-estimation-0001` FP16;
- retained evidence detector `person-detection-0200` FP16;
- OpenVINO Runtime `2026.3.1`;
- evidence video decoder `opencv-python-headless==4.12.0.88`;
- corrected OpenPose decoder pinned to the same OMZ revision with attribution preserved.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. No new model family, training job, paid resource, self-hosted runner, home/customer media, second framework or duplicate agent is introduced.

Live base for this approach is `main` at `9bb42dfc8841f55717fb445855f5b276cfc6d320`, whose main Analytics quality run #106 completed successfully on attempt 1. At intake there were **0 open implementation PRs** and **0 active runs for the live main head**. The prior stalled GMDCSA rotation is stopped; this is a changed acquisition approach, not a renamed retry.

## Reproduce
Repository checks:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Existing rights-bound GMDCSA evidence path:

```sh
python -m analytics_lab.validation_seed --output-dir /path/to/private-validation
python -m analytics_lab.validation_cli --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.person_down_orientation_diagnostics --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
```

The Figshare adapter is intentionally not wired to ordinary CI network access in this change. Its acquisition contract is validated offline first; any live range probe must remain explicitly bounded and reviewed before media admission.

## Next executable decision
Run the exact current acquisition-branch head through Linux, Windows and the Analytics quality gate once. If exact-head CI is green, merge this bounded acquisition adapter on the unchanged tested base.

After merge, the next acceptance-moving step is a small, explicitly bounded live metadata/range probe against the reviewed Figshare article/file identity. If the source supports exact ranges and the archive index can be obtained inside the hard ceilings, pin the smallest disjoint positive/negative member pair with exact provenance/attribution, member size and hashes before downloading those members. If the source ignores ranges, requires ZIP64 beyond the reviewed parser, or does not expose sufficiently small disjoint members, stop that acquisition path rather than downloading the 2.36 GB object.

Only after a member pair is admitted should the unchanged retained detector -> orientation recovery -> OpenPose decode -> bounded association -> posture -> temporal evaluator run on it.

## Outstanding commercial-release gates
Substantially broader held-out positive/negative real-video evidence across genuinely different subjects, cameras, sites, resolutions, viewpoints, lighting and multi-person scenes; meaningful false-alert/camera-hour and miss measurements; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.
