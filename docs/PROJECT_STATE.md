# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed model-artifact identities, runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video. Media is re-hashed after execution so changed media cannot inherit stale evidence.

## Current acceptance-moving work
PR #27 / `evidence/0200-continuity-filter` is the single implementation vehicle, based on green `main` revision `6b432adeecec3454ac10c443e575f5528582991c`. It is the required approach change after the three-detector shootout and fixed-threshold tuning both failed to produce a promotable detector configuration.

Exact-head run `35276058586` on revision `fdd2949c8d83479094076e62405d4ad07e238cd9` passed the repository guardrail preflight, **186 Linux repository tests** (one expected optional-OpenCV skip before runtime installation), synthetic replay, rights-bound real-video validation, detector diagnostics, the threshold/continuity evidence command, Windows regression and the final Analytics quality gate on attempt 1. Runtime evidence used OpenVINO `2026.3.1-22476-759c5a6ab8c-releases/2026/3`, OpenCV `4.12.0`, NumPy `2.2.6`, the exact two GMDCSA-24 clips, and the exact hash-pinned `person-detection-0200` artifacts. No media/model workflow artifacts were uploaded.

### Measured detector threshold result
`person-detection-0200` was run once per frame and the same raw outputs were evaluated at confidence thresholds 0.50, 0.40, 0.30, 0.20 and 0.10. The threshold-only evidence bars were >=55% positive fall-interval coverage, >=90% hard-negative person coverage and <=5% frames with multiple person detections on each clip. The measured result remains `recommended_threshold: null`.

- **0.50:** fall-interval coverage 29/96 = **30.2%**; positive duplicate-frame rate **19.0%**; hard-negative coverage **100%**; negative duplicate-frame rate **6.6%**.
- **0.40:** fall-interval coverage 40/96 = **41.7%**; positive duplicate-frame rate **32.2%**; hard-negative coverage **100%**; negative duplicate-frame rate **9.9%**.
- **0.30:** fall-interval coverage 61/96 = **63.5%**; positive duplicate-frame rate **38.5%**; hard-negative coverage **100%**; negative duplicate-frame rate **11.3%**.
- **0.20:** fall-interval coverage 82/96 = **85.4%**; positive duplicate-frame rate **44.3%**; hard-negative coverage **100%**; negative duplicate-frame rate **12.7%**.
- **0.10:** fall-interval coverage 96/96 = **100%**; positive duplicate-frame rate **80.5%**; hard-negative coverage **100%**; negative duplicate-frame rate **19.3%**.

Threshold lowering alone is still rejected. It recovers prone-person signal but admits too many boxes.

### Measured low-threshold continuity result
The next bounded hypothesis reused the exact same raw `person-detection-0200` output at confidence **0.10** but selected exactly one person box per frame by spatial continuity. With a previous selection present, candidates clearing IoU **0.05** were ranked by IoU and then confidence; otherwise the highest-confidence person candidate was reacquired. This is an evidence-only single-person diagnostic and is **not** integrated into the production candidate backend.

Predeclared bars were: >=85% positive fall-interval coverage, >=70% positive fall-interval link rate, >=95% hard-negative coverage, >=90% hard-negative link rate, <=30% positive overall reset rate and <=10% hard-negative reset rate. The result was **accepted: true** and cleared every bar:

- Positive fall interval: **96/96 frames = 100% coverage**.
- Positive fall-interval continuity: **94/95 transitions linked = 98.95% link rate**.
- Positive overall: **173/174 frames = 99.43% coverage**; **170/171 transitions linked = 99.42%**; **1 reset / 171 transitions = 0.58% reset rate**.
- Positive selected confidence: mean **0.5384**, minimum **0.1044**.
- Hard negative: **212/212 frames = 100% coverage**; **211/211 transitions linked = 100% link rate**; **0 resets**.
- Hard-negative selected confidence: mean **0.9701**, minimum **0.7999**.

The evidence-only selector therefore recovered the low-confidence prone-person trajectory on this deliberately tiny single-person staged-real seed while rejecting duplicate boxes by construction and preserving stable normal-person continuity. This is a material detector-stage improvement on the same evidence, but it is **not yet a general detector promotion**: the seed has only one visible subject and does not prove multi-person identity separation, crowded-scene behavior or commercial accuracy.

Detector-only throughput for this run was about **133.6 FPS** on the hosted runner. That value is retained only as a run-local resource observation because hosted hardware varies.

The full production-candidate pipeline still produces zero events, one missed positive episode and zero false alerts on this seed because it still uses the prior detector/pose path. The independently measured pose/posture defect also remains: every real pose candidate from the current pipeline classified `unknown`. Tracking and temporal persistence remain downstream of detector and pose/posture quality.

## Evidence/data baseline
### GMDCSA-24 — bounded staged-real validation source
- Primary repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`.
- Repository license at that revision: MIT. Associated Data in Brief paper: CC BY 4.0, DOI `10.5281/zenodo.13354453`, describing 81 fall and 79 ADL clips from four subjects in three home setups and use for fall-detection training/testing.
- Positive seed: `Subject 1/Fall/05.mp4`, Git blob `4e13ed24f6c2af7062b64b1920ef706c51efe94b`, 5,872,655 bytes, local SHA-256 `1aad4e2ee93b5498283fdc4a0478a5913dffc2ac8e2538574cd821ffb1b485f1`, annotated fall 1.8-5.0 s.
- Hard negative: `Subject 1/ADL/15.mp4`, Git blob `903da9245132cf70c10124edd0625c958f702cb8`, 7,233,079 bytes, local SHA-256 `2cf0d421cfd8e34280bf02afc67a4f1d1abee3cd87a23f11e69ef393fedc6fdb`.
- Exact media identity is verified before and after measured execution; media remains outside public GitHub.

### Other reviewed sources
- UE4 Fall Detection Dataset pinned at `55041766dea68eaddc1df1c06aabd0a51931a22a`, CC BY 4.0, synthetic only; eligible for commercial training/evaluation with attribution but never real-video accuracy evidence.
- Figshare article `28596332` version 2, posted 2025-03-14, CC BY 4.0, real-world secondary source; the ~2.36 GB corpus remains outside the current bounded-resource path.
- UR Fall remains excluded because its official source states non-commercial terms. Roboflow mirrors with unclear upstream provenance remain hold-only.

## Runtime / detector provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Current production-candidate baseline artifacts: `person-detection-retail-0013` FP16 + `human-pose-estimation-0001` FP16, exact size/SHA-384 verified.
- Evidence candidate `person-detection-0200` FP16: XML 254,619 bytes / SHA-384 `654a515935f6dffc0440cffdeeb6a889bc25bf5d64e425ce2337070cdbcce1b1fbbcf42e4b2761265eb65ec07c1cd6f7`; BIN 3,634,654 / `10b6f79b495ad1ef13748938c452379c6b6cd825d11426ffa68be3eeed6c48a0af04e0b913c843e6e7431aa1c13f5471`.
- OpenVINO Runtime baseline: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- No shootout detector or low-threshold continuity selector is integrated into the production-candidate backend.

## Efficiency / execution ledger
One worker, one acceptance-moving work item, one implementation PR. Intake verification: `main` at `6b432adeecec3454ac10c443e575f5528582991c`; post-merge Analytics quality run `35271516888` passed on attempt 1; open implementation PRs were 0 and active runs for the live main head were 0. The repository offline preflight was executed against those live counts and allowed implementation.

PR #27 exact-head run `35276058586` is CI-triggering request **1/2** and produced a tested acceptance improvement on attempt 1; unchanged retries remain **0**. This measured evidence-state update is the second and final CI-triggering mutation permitted for this session. No extra worker, donor model, dataset, framework, runner, paid resource, home/customer footage, training run or licensing shortcut was introduced.

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
python -m analytics_lab.detector_thresholds \
  --manifest /path/to/private-validation/validation-manifest.json \
  --candidate-dir /path/to/private-detector-cache
```

## Next executable step
Finish exact-head verification for PR #27 after this measured state update and merge only if Linux real-video evidence, Windows regression and the final Analytics quality gate remain green on the unchanged tested base.

The low-threshold continuity result materially improves detector-stage recall on the current single-person staged-real seed, so do **not** launch detector fine-tuning yet. Preserve training as a fallback if broader real evidence disproves the continuity approach.

Next attack the already-proven pose/posture failure on the same rights-bound evidence. Build the smallest evidence-only diagnostic that feeds the continuity-selected `person-detection-0200` box into the existing reviewed `human-pose-estimation-0001` path and records aggregate keypoint availability/confidence and posture-classification reasons before/during/after the fall. Determine whether the all-`unknown` result is caused by crop geometry, keypoint confidence/coverage, decoder semantics or posture thresholds before changing the pose model. Do not alter tracking or temporal persistence until pose/posture produces discriminative real-video output.

Separately, before any detector promotion beyond this single-person seed, require a bounded multi-person/crowded-negative check so a one-box continuity heuristic cannot be mistaken for a general multi-person detector solution.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; functional pose/posture discrimination on real video; false-alert and missed-event measurements at meaningful scale; alert-latency distribution once events fire; latency/resource envelope; platform/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Synthetic/stub tests and this two-clip staged-real seed do not establish commercial accuracy or commercial readiness.