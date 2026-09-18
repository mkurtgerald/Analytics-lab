# Project state — 2026-09-18

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Retained person-down path
The retained evidence path is:

`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media/model/runtime identities and aggregate measurements without retaining source video in public GitHub.

## Accepted evidence frontier
### Subject 1 seed and robustness rotation
PR #33 produced the first matched staged-real candidate on the pinned Subject-1 pair: **1 labeled positive, 1 candidate, 1 match, 0 misses, 0 hard-negative false alerts**, about **0.0036 decoded camera-hours**, and **4024 ms** alert delay. The positive contained 85 qualifying `down` observations and a 3008 ms qualifying run. This remains a tiny engineering seed, not a commercial accuracy claim.

PR #43 measured an untouched Subject-1 pair without tuning. The positive produced **1 candidate / 1 match / 0 misses**, **3992 ms** alert delay, **150/150 detector coverage**, **175/181** safe pose associations overall, **107 qualified down observations**, and a **4096 ms** longest qualified run. The ground-reach negative remained **0 candidates / 0 false alerts** with a **128 ms** longest qualified run. Aggregate hosted-CPU evidence throughput was about **10.04 FPS**.

### Held-out Subject 2
PR #35 retained orientation recovery, raising labeled-fall detector coverage from **66.67% to 91.30%**, associated poses **98 -> 135**, qualified `down` observations **31 -> 57**, and longest qualified run **768 -> 1472 ms**, while the prone sleeping negative remained **0 candidates / 0 false alerts**.

PR #36 retained one narrow posture-ambiguity correction: only near-diagonal `geometry_not_decisive` observations (`|horizontal_fraction - vertical_fraction| <= 0.02`) become `unknown`; they are never promoted to `down`. The Subject-2 longest run improved **1472 -> 2544 ms**, but one genuinely upright observation still correctly resets the run. The positive remains **0 candidates / 1 miss** and Subject 2 stays held out from training/adaptation.

PR #44 measured an untouched Subject-2 night/cross-view pair. The positive remained **0 candidates / 1 miss** while the floor-to-standing/walking negative remained **0 candidates / 0 false alerts**. Positive detector selection was **76/82 = 92.7%**, but safe pose association fell to **32/82**, with **50/82 `no_associated_pose`** observations and a **736 ms** longest qualified down run. The negative had **226/226** safe pose associations and only a **96 ms** longest down-like run. Broadening association was rejected because the unmatched decoded poses were not borderline matches.

### Held-out Subject 3
PR #37 measured the unchanged retained path. The positive produced **0 candidates / 1 miss** and the sitting-to-sleeping-on-floor negative produced **0 candidates / 0 false alerts**. Fall-window detector selection was **89/105 = 84.76%**, safe full-frame pose association **50/105**, and longest qualified positive run **1184 ms**.

PR #38 tested a selected-person crop fallback with the same OpenPose model. Association improved but the actual target regressed **1184 -> 512 ms**, introduced genuine upright contradictions, and reduced throughput. It was closed unmerged.

### Held-out Subject 4
PR #39 measured the unchanged retained path on a fall plus dynamic push-up negative. The positive produced **0 candidates / 1 miss**; the negative produced **0 candidates / 0 false alerts**. Detector selection, continuity and safe pose association were all **129/129 = 100%** in the labeled fall window. The positive had **55 qualified down observations / 1312 ms longest run**; the push-up negative had **44 down-like observations / 464 ms longest run** and did not alert. This argues against weakening the 3000 ms persistence rule.

These measurements increasingly localize the limiting robustness problem downstream of detector continuity, primarily around pose representation/association/posture continuity under difficult view/lighting conditions. The detector path remains retained.

## Rejected alternate pose path
PR #41 measured OMZ `human-pose-estimation-0005` against the retained Subject-4 baseline and rejected it after usable person-down evidence collapsed to **1 qualified down / 0 ms** while the hard negative remained clean.

PR #42 measured `human-pose-estimation-0006`. It again produced **0 candidates / 1 miss**, **1 qualified down / 0 ms**, with **102 labeled-positive observations** failing the unchanged required-keypoint-confidence floor. The hard negative remained **0 false alerts**. Although throughput improved to about **11.65 FPS** versus **9.67 FPS** baseline, usable person-down evidence was materially worse. The EfficientHRNet/AE resolution-stepping path is closed.

Do not retry `0005`/`0006`, tune them around Subject 4, lower required-keypoint confidence, bridge genuine `upright`/`other`, weaken 3000 ms persistence, extend the 750 ms unknown-gap budget, or shop another person detector.

## Stalled same-source rotation closed
PR #45 attempted another file-disjoint GMDCSA-24 robustness pair on Subject 3. Two exact-head runs passed repository guardrails/synthetic regressions but failed deterministically at the `validation_cli` boundary before any real-video evidence was emitted. The single evidence-bound correction did not remove the failure. PR #45 was closed unmerged rather than spending a third session on the same path.

## Current acceptance-moving work — broader-source Figshare generalization
The broader-source path uses the already-reviewed real-world registry entry:
- source: `figshare-fall-2017-activities`;
- title: `Video-Based Fall Detection Dataset with 2017 Activities from 29 Subjects`;
- pinned version: `figshare:28596332:v2:2025-03-14`;
- reviewed license: `CC-BY-4.0`;
- provenance: `figshare:28596332:version-2`.

PR #46 merged the fail-closed Figshare acquisition adapter. It binds article **28596332 version 2** and file id **52990358**, requires exact HTTP `206` byte ranges, limits article metadata to 512 KiB and ZIP tail reads to 128 KiB, rejects unsupported archive structures, and never downloads member payloads during index inspection.

PR #47 merged the measured archive-map boundary into `main` at `8016f6678301542628d18b2439f2fa56411c98ab`. The reviewed `VideoDataset.zip` is **2,529,520,868 bytes**, provider MD5 `c784167d08f2fa94e3afd36cec758e1f`, with **22,397** classic-ZIP entries and a **2,941,785-byte** central directory at offset **2,526,579,061**. The directory was fetched in exactly two requests of **2,097,152 + 844,633 bytes** under the existing 2 MiB per-request ceiling. It parsed **20,324** non-directory members and **2,022** bounded MP4 entries; no member payload was fetched by the index probe.

The smallest clear file-disjoint pair from that public map was pinned before payload admission:
- negative: `VideoDataset/ADL/SBJ_01_LOC3/ACT25_R_1/20240923130459.mp4`, compressed **277,756**, uncompressed **278,497**, CRC32 `44ac1304`, local-header offset **1,084,380,237**;
- positive: `VideoDataset/Fall/SBJ_10_LOC3/ACT10_R_2/20240915184434.mp4`, compressed **359,774**, uncompressed **360,498**, CRC32 `41961303`, local-header offset **1,738,471,658**.

PR #48 merged the exact bounded member-admission boundary to `main` at `6e6c6121e2f45b3daf61f0dac849a77e7002f98f`. Post-merge Analytics quality run #118 passed on attempt 1. The live member-admission step fetched only exact local-header/name/payload ranges for the two pinned deflate members, verified central/local ZIP identity relationships, declared size, CRC32 and the existing 16 MiB item ceiling, computed SHA-256 over the uncompressed video bytes, and wrote only to ephemeral runner storage. Exact admitted identities are:
- negative SHA-256 `7e6f026e68c280234ac34764a26b7f073e6f1367a259ed756a30a663542d3c92`;
- positive SHA-256 `8c7c13e1a9a5321e25b4203d35e65e072e2e6fcfd21ed806d7fd17b58b4438fd`.

PR #49 is the first unchanged retained-pipeline measurement on that broader source. The source publication maps **ACT10** to `Sit on chair, fall` (Fall) and **ACT25** to `Descend` (ADL), but no independent frame-level positive interval has been established for these exact clips. The diagnostic therefore scores the ADL clip normally for candidate/false-alert behavior while treating the Fall clip as **diagnostic-only** for detector coverage, pose association, posture/temporal fragmentation, candidate behavior, decoded duration and CPU cost. It deliberately does **not** score positive match/miss or alert delay from a directory/activity label. Exact admitted SHA-256 identities are checked before model provisioning so changed media fails closed before consuming inference resources.

Exact-head Analytics quality run #119 at `1b4ecb02bb2701f29cf7715da656bf89462b709b` passed Linux, Windows and the aggregate quality gate on attempt 1. Linux passed all **43 guardrail regressions**, all **246 synthetic/unit tests** with one optional OpenCV skip, synthetic replay, the bounded Figshare index probe and the new hosted-CPU person-down evidence step. No unchanged retry was used.

The cross-source measurement is diagnostic but decisive about stage attribution:
- **ACT25 ADL negative:** 56/56 detector selections, 56/56 safe pose associations, 56/56 `upright` posture observations, **0 candidate events / 0 false alerts**, decoded duration **1833 ms = 0.000509 camera-hours**, and about **10.03 FPS** evidence throughput.
- **ACT10 Fall-class positive:** 57/57 detector selections and 56/56 linked detector transitions (**100% detector coverage/continuity**), but only **32/57 safe pose associations**; **25/57** frames ended `no_candidate_within_bound` / `no_associated_pose`. Posture output was **1 down, 4 other, 32 unknown, 20 upright**; only one qualified-down observation occurred, so the longest qualified run was **0 ms** and the clip produced **0 candidate events**. Because no independent frame-level positive interval exists, this is **not scored as a miss** and alert delay remains unscored.
- Across both clips the unchanged evidence path processed **113 frames at about 10.02 FPS**. The retained detector/orientation stage remained strong while pose association/posture continuity collapsed on the Fall-class clip.

This independent-source result repeats the downstream weakness already seen on difficult GMDCSA held-outs: detection/continuity can be high while OpenPose association and usable posture continuity fail. It therefore strengthens the justification for a bounded, commercially clean **pose representation/adaptation decision** rather than weakening association bounds, confidence floors, 3000 ms persistence or the 750 ms unknown-gap budget. One short pair is still not commercial accuracy evidence.

The branch `evidence/figshare-person-down-generalization` adds only this bounded evidence adapter, regression tests and a hosted-Linux evidence step using the already-pinned OpenVINO `2026.3.1`, headless OpenCV decoder, `person-detection-0200`, OpenPose `human-pose-estimation-0001`, unchanged association/posture logic, **3000 ms** persistence and **750 ms** unknown-gap budget. Media remains ephemeral and no accuracy or injury/cause/fault/intent inference is permitted.

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
- activity mapping reference: `Vision Transformer Based Fall Detection: A Spatial Temporal Attention Mechanism for Robust Video Analysis`, DOI `10.30970/eli.33.12`, CC-BY-4.0;
- exact ephemeral media identities admitted by PR #48: negative SHA-256 `7e6f026e68c280234ac34764a26b7f073e6f1367a259ed756a30a663542d3c92`, positive SHA-256 `8c7c13e1a9a5321e25b4203d35e65e072e2e6fcfd21ed806d7fd17b58b4438fd`;
- media remains outside public GitHub; the Fall clip has only a source clip/activity class for current purposes, not a frame-level positive interval.

### Runtime/model provenance
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0;
- retained pose model `human-pose-estimation-0001` FP16;
- retained evidence detector `person-detection-0200` FP16;
- OpenVINO Runtime `2026.3.1`;
- evidence video decoder `opencv-python-headless==4.12.0.88`;
- corrected OpenPose decoder pinned to the same OMZ revision with attribution preserved.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. No new model family, training job, paid resource, self-hosted runner, home/customer media, second framework or duplicate agent is introduced.

Live base at this work item's intake is `main` at `6e6c6121e2f45b3daf61f0dac849a77e7002f98f`; its post-merge Analytics quality run #118 passed on attempt 1. PR #49 is the sole implementation PR. Its first exact head `1b4ecb02bb2701f29cf7715da656bf89462b709b` completed Analytics quality run #119 successfully on attempt 1 with **0 unchanged retries** and **1 CI-triggering PR dispatch**. This evidence-binding state update is the **second and final CI-triggering mutation permitted in this work session**; no further push or retry is permitted in this session if the resulting exact-head run fails.

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

Figshare bounded index probe, exact member admission and broader-source diagnostic (evidence branch only):

```sh
python -m analytics_lab.figshare_probe
python -m analytics_lab.figshare_member_admission --output-dir /path/to/ephemeral-output
python -m analytics_lab.figshare_person_down_diagnostic --output-dir /path/to/ephemeral-diagnostic --candidate-dir /path/to/private-detector-cache
```

## Next executable decision
Run the resulting exact PR #49 head through Linux, Windows and the Analytics quality gate once. If all required checks are green on the unchanged base, merge the bounded broader-source evidence path. The next acceptance-moving work item should not tune this clip or weaken safety rules: establish the smallest commercially clean pose adaptation/replacement decision package. Before any training is launched, pin the exact trainer revision and dependencies, pretrained-weight provenance and transitive commercial rights, export/runtime compatibility, a subject-separated train/validation/holdout split that excludes prior held-outs, and a bounded hosted-CPU/resource ceiling. Continue broadening untouched evidence while that decision is measured; do not train on Subjects 2-4 merely to make prior clips pass.

## Outstanding commercial-release gates
Substantially broader held-out positive/negative real-video evidence across genuinely different subjects, cameras, sites, resolutions, viewpoints, lighting and multi-person scenes; meaningful false-alert/camera-hour and miss measurements; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.
