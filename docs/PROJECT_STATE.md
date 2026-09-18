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

## Current acceptance-moving work — bounded Figshare archive map
The broader-source path uses the already-reviewed real-world registry entry:
- source: `figshare-fall-2017-activities`;
- title: `Video-Based Fall Detection Dataset with 2017 Activities from 29 Subjects`;
- pinned version: `figshare:28596332:v2:2025-03-14`;
- reviewed license: `CC-BY-4.0`;
- provenance: `figshare:28596332:version-2`.

PR #46 merged the fail-closed Figshare acquisition adapter. It binds article **28596332 version 2** and file id **52990358**, requires exact HTTP `206` byte ranges, limits article metadata to 512 KiB and ZIP tail reads to 128 KiB, rejects unsupported archive structures, and never downloads member payloads during index inspection.

PR #47 is the sole open implementation/evidence PR. Its first exact head failed deterministically because the published archive exceeded the original **10,000-entry / 2 MiB** central-directory admission ceiling. The corrective exact head `03db110674bfa322dfcdcb3cc713ace4ee38953c` passed Linux, Windows and the Analytics quality gate on attempt 1 and measured the archive without fetching its central directory or any member payload:
- file: `VideoDataset.zip`;
- published size: **2,529,520,868 bytes**;
- provider MD5: `c784167d08f2fa94e3afd36cec758e1f`;
- classic-ZIP entries: **22,397**;
- central-directory offset: **2,526,579,061**;
- central-directory size: **2,941,785 bytes** (~2.81 MiB);
- `central_directory_fetched=false`;
- `member_payload_fetched=false`.

The measured mismatch is modest enough to justify one narrowly bounded archive-map step without changing the normal acquisition adapter's **2 MiB per-request** limit. The active correction therefore admits only a probe envelope of **25,000 entries / 4 MiB aggregate central-directory bytes**, splits the measured directory into at most **two exact <=2 MiB range requests**, reuses the existing fail-closed central-directory parser, emits only public member metadata, and still downloads **no video member payload**. This is a source-index measurement boundary, not media admission and not a commercial-accuracy claim.

If the exact-head live probe parses the directory cleanly, use its smallest bounded video-member summary to identify the smallest genuinely disjoint positive/negative pair. Before any member download, record exact member path, compressed/uncompressed size, CRC32, local-header offset, source/version/license/provenance/attribution and an exact cryptographic identity plan. Keep the existing **16 MiB per-item ceiling**. If the source cannot produce a clearly labeled disjoint pair within the bounded member envelope, stop this acquisition path rather than widening it or downloading the 2.36 GiB monolith.

Only after a member pair is admitted should the unchanged retained detector -> orientation recovery -> OpenPose decode -> bounded association -> posture -> temporal evaluator run on it.

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
- exact media members are **not admitted yet**.

### Runtime/model provenance
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0;
- retained pose model `human-pose-estimation-0001` FP16;
- retained evidence detector `person-detection-0200` FP16;
- OpenVINO Runtime `2026.3.1`;
- evidence video decoder `opencv-python-headless==4.12.0.88`;
- corrected OpenPose decoder pinned to the same OMZ revision with attribution preserved.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. No new model family, training job, paid resource, self-hosted runner, home/customer media, second framework or duplicate agent is introduced.

Live base is `main` at `18661bfe25d5e60e961e863684267d8199fc049d`. PR #47 is the only open implementation PR. At this session's intake its exact head `03db110674bfa322dfcdcb3cc713ace4ee38953c` had completed Analytics quality run #113 successfully, with **0 active runs for the head**, **0 unchanged retries in this session**, **0 CI-triggering pushes/dispatches in this session**, and **1 prior session without tested acceptance improvement**. The archive-map correction is one coherent implementation push, not a duplicate lane.

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

Figshare bounded index probe (evidence branch only):

```sh
python -m analytics_lab.figshare_probe
```

## Next executable decision
Run the new exact PR #47 head through Linux, Windows and the Analytics quality gate once. If the bounded live index step passes, inspect only its aggregate/public member-map evidence and select the smallest clearly labeled disjoint positive/negative candidate pair before any payload admission. If the index parser fails deterministically, preserve that failure and diagnose the exact archive-record cause; do not broaden the 25,000-entry / 4 MiB aggregate probe envelope in the same cycle.

## Outstanding commercial-release gates
Substantially broader held-out positive/negative real-video evidence across genuinely different subjects, cameras, sites, resolutions, viewpoints, lighting and multi-person scenes; meaningful false-alert/camera-hour and miss measurements; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.
