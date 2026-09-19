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

The first file-disjoint pair was pinned before payload admission:
- negative: `VideoDataset/ADL/SBJ_01_LOC3/ACT25_R_1/20240923130459.mp4`, compressed **277,756**, uncompressed **278,497**, CRC32 `44ac1304`, local-header offset **1,084,380,237**;
- positive: `VideoDataset/Fall/SBJ_10_LOC3/ACT10_R_2/20240915184434.mp4`, compressed **359,774**, uncompressed **360,498**, CRC32 `41961303`, local-header offset **1,738,471,658**.

PR #48 merged exact bounded member admission. Exact first-pair identities are negative SHA-256 `7e6f026e68c280234ac34764a26b7f073e6f1367a259ed756a30a663542d3c92` and positive SHA-256 `8c7c13e1a9a5321e25b4203d35e65e072e2e6fcfd21ed806d7fd17b58b4438fd`; media remained ephemeral.

PR #49 measured the unchanged retained pipeline on that pair. The **ACT25 ADL negative** had 56/56 detector selections, 56/56 safe pose associations, 56/56 upright postures and **0 candidates / 0 false alerts** over **1833 ms = 0.000509 camera-hours**. The **ACT10 Fall-class positive** had 57/57 detector selections and 56/56 linked detector transitions but only **32/57 safe pose associations**, with **25/57** no-associated-pose frames; posture output was **1 down, 4 other, 32 unknown, 20 upright** and no candidate. Because no independent frame-level interval exists, that positive is not scored as a miss and alert delay remains unscored. Across both clips throughput was about **10.02 FPS**.

PR #50 merged the metadata-only selection of a second pair at `27ad4df631bc91734c5158316ca235b0fe1b04da`, before any member payload or model output was inspected:
- negative ACT19 `Laying`: `VideoDataset/ADL/SBJ_06_LOC1/ACT19_R_1/20240920140827.mp4`, Subject 06 / Location 1, compressed **592,355**, uncompressed **593,051**, CRC32 `78d8749b`, local-header offset **1,323,417,041**;
- positive ACT4 `Fall on the back`: `VideoDataset/Fall/SBJ_03_LOC2/ACT4_R_1/20240912_111106.mp4`, Subject 03 / Location 2, compressed **598,064**, uncompressed **598,720**, CRC32 `c8e45df7`, local-header offset **2,000,888,480**.
The pair is subject- and location-disjoint from each other and moves off Subjects 01/10 and Location 3 used by the first Figshare pair.

PR #51 merged the exact bounded admission and unchanged-baseline measurement of that second pair into `main` at `4c9a85486356d34e40dd6c602171efdb1fa44237`. Post-merge Analytics quality run #126 passed on attempt 1. Exact uncompressed-media SHA-256 identities are:
- ACT19 negative: `1f9b3f44b67576c93a61921311830286b4a9eebe45277b90d0c9eb2625fa2a24`;
- ACT4 positive: `a54f715f3ad7d8fc2fe64390842f2c5c16ee03ace2e70f6a785cbeef6ff5f54c`.

The unchanged second-pair measurement is another strong stage-attribution result, not a commercial accuracy estimate:
- **ACT19 `Laying` ADL negative:** 56/56 detector selections, 55/55 linked detector transitions, 56/56 safe pose associations, posture counts **42 upright / 8 other / 6 unknown / 0 down**, **0 candidates / 0 false alerts**, longest qualified down run **0 ms**, decoded duration **1833 ms = 0.000509 camera-hours**, about **9.85 FPS**. Detector inference was about **441 ms**, pose inference **4888 ms**, pose decode **210 ms**, total evidence elapsed **5683 ms**.
- **ACT4 `Fall on the back` Fall-class positive:** 56/56 detector selections and 54/55 linked transitions, but only **47/56 safe pose associations**; 9 frames were `no_candidate_within_bound` / `no_associated_pose`. Posture counts were **39 upright / 9 unknown / 4 other / 4 down**. The longest raw and qualified down run was only **100 ms / 4 samples**, with 9 bridged unknown frames and a maximum bridged unknown gap of **300 ms**; **0 candidate events**. Unmatched nearest poses were not borderline associations: measured selection IoU was 0 and edge-gap-normalized p50 was about **1.76**, so widening association is not justified by this evidence. Detector inference was about **447 ms**, pose inference **4875 ms**, pose decode **198 ms**, total evidence elapsed **5633 ms**.
- Across both clips the unchanged stack processed **112 frames at about 9.90 FPS**. The detector remained strong; the positive again fragmented downstream in pose association/posture representation. The positive is **not scored as a miss and has no scored alert delay** because no independent frame-level event interval exists.

This repeats the same downstream failure boundary across an independent source, different subjects and different locations while a difficult lying negative remains clean. It strengthens the evidence-based case for a bounded commercially clean pose representation/adaptation decision package. It does **not** justify weakening association bounds, keypoint-confidence floors, the 3000 ms persistence requirement, the 750 ms unknown-gap ceiling, or changing the retained detector. Two short Figshare pairs remain engineering evidence only, not commercial accuracy.

PR #52 merged the metadata-only selection of the third pair into `main` at `0f2f50d60fdecc8b20f3afff080724aca3c7ac12`. Exact-head run #128 passed all required Linux/Windows/regression/quality-gate checks after one preserved deterministic fixture-only failure on the first head. The selected pair was fixed before any member payload or model output was inspected:
- negative ACT16 `Sitting`: `VideoDataset/ADL/SBJ_02_LOC5/ACT16_R_1/20240923151305.mp4`, Subject 02 / Location 5, compressed **949,755**, uncompressed **950,297**, CRC32 `39c30723`, local-header offset **1,232,209,160**;
- positive ACT11 `Try to sit on chair, fall`: `VideoDataset/Fall/SBJ_09_LOC3/ACT11_R_1/20240915181458.mp4`, Subject 09 / Location 3, compressed **386,050**, uncompressed **386,799**, CRC32 `7033abe3`, local-header offset **2,371,833,171**.
The pair is subject- and location-disjoint, uses untouched Subjects 02/09 relative to the first two Figshare pairs, and adds Location 5.

PR #53 is the current single implementation PR and continues the same acceptance item. Its first head adds only exact third-pair admission through the existing bounded range/CRC path, keeps output ephemeral, and emits SHA-256 identities before any diagnostic model work. The branch-specific evidence lane deliberately performs admission only on this first head; the second and final CI-triggering change may pin those measured hashes and add the unchanged retained baseline diagnostic. No analytics threshold, detector, pose representation, association bound, 3000 ms persistence rule or 750 ms unknown-gap ceiling changes in this cycle.

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
- first-pair exact ephemeral identities: negative SHA-256 `7e6f026e68c280234ac34764a26b7f073e6f1367a259ed756a30a663542d3c92`, positive SHA-256 `8c7c13e1a9a5321e25b4203d35e65e072e2e6fcfd21ed806d7fd17b58b4438fd`;
- second-pair exact ephemeral identities: ACT19 negative SHA-256 `1f9b3f44b67576c93a61921311830286b4a9eebe45277b90d0c9eb2625fa2a24`, ACT4 positive SHA-256 `a54f715f3ad7d8fc2fe64390842f2c5c16ee03ace2e70f6a785cbeef6ff5f54c`;
- third-pair archive identities are pinned above; uncompressed media SHA-256 values are intentionally pending the first bounded admission run;
- media remains outside public GitHub; Fall clips have only source clip/activity classes unless an independent frame-level interval is established.

### Runtime/model provenance
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0;
- retained pose model `human-pose-estimation-0001` FP16;
- retained evidence detector `person-detection-0200` FP16;
- OpenVINO Runtime `2026.3.1`;
- evidence video decoder `opencv-python-headless==4.12.0.88`;
- corrected OpenPose decoder pinned to the same OMZ revision with attribution preserved.

## Efficiency ledger
One worker, one acceptance-moving work item, one implementation PR. No new model family, training job, paid resource, self-hosted runner, home/customer media, second framework or duplicate agent is introduced.

Live base for PR #53 is `main` at `0f2f50d60fdecc8b20f3afff080724aca3c7ac12`; post-merge Analytics quality run #129 was green on attempt 1. Intake had **0 open implementation PRs** and **0 active runs for main**. Local policy preflight for implementation was allowed with **0 unchanged retries**, **0 CI dispatches this session**, and **0 sessions without tested progress**. PR #53 is the sole implementation PR. Its first exact-head run is the first of at most two CI-triggering branch updates in this session.

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

Figshare evidence paths (evidence branches only):

```sh
python -m analytics_lab.figshare_probe
python -m analytics_lab.figshare_member_admission --output-dir /path/to/ephemeral-output
python -m analytics_lab.figshare_person_down_diagnostic --output-dir /path/to/ephemeral-diagnostic --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.figshare_generalization2_admission --output-dir /path/to/ephemeral-output-2
python -m analytics_lab.figshare_generalization2_person_down_diagnostic --output-dir /path/to/ephemeral-diagnostic-2 --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.figshare_generalization3_admission --output-dir /path/to/ephemeral-output-3
```

## Next executable decision
Run the exact first PR #53 head once through Linux, Windows, the Analytics quality gate, bounded Figshare archive-index probe and exact third-pair admission. If green, record the two emitted SHA-256 identities, make one coherent second/final branch update that pins those hashes and adds the unchanged retained baseline diagnostic, and run the same evidence lane once more. Only a measurable unchanged-baseline result may justify the next algorithm decision. No training may begin until exact trainer revision/dependencies, pretrained weights and transitive commercial rights, export/runtime compatibility, subject-separated train/validation/holdout splits and CPU/resource ceilings are pinned; Subjects 2-4 remain held out and must not be trained on merely to make prior misses pass.

## Outstanding commercial-release gates
Substantially broader held-out positive/negative real-video evidence across genuinely different subjects, cameras, sites, resolutions, viewpoints, lighting and multi-person scenes; meaningful false-alert/camera-hour and miss measurements; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.