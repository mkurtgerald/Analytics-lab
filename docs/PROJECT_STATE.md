# Project state — 2026-09-20

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

### Permanent priority
1. Detection + tracking.
2. LPR/OCR.
3. Face — including required selectable face blurring with policy-driven default blur, permission-gated unblur/reblur, server-side enforcement, auditability and preservation of original evidence.
4. Weapons.
5. Appearance search.

Person-down and slip/fall remain required deliverables and are secondary only in sequencing.

## Detection + tracking — frozen at the first real-video acceptance gate
The tracking implementation boundary remains intact and unchanged on `main` `0fdea197909bac0190cb61e6b4f3d3276cf748d0`, the merge of PR #71. Main Analytics quality run #172 passed on attempt 1.

Landed tracking boundaries include the detector-neutral contract, portable ByteTrack slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` (MIT), bounded evaluator, rights/evidence admission, pre-registered Wikimedia CC0 frames 150-174, canonical RGB24 frame identities and ground-truth package binding, the frozen simple-IoU control, and the first-attempt head-to-head runner. The fixed frame-manifest SHA-256 is `52f00a4fc1e013d9c0cae9647cb386369441f005a970d6d42c449cbd0269a031`; the benchmark-plan SHA-256 is `eb7995a389a22f3528b9bfca97d64f9d8e3c515ebf1c764cf25b68fbdcc3479c`.

The remaining indispensable evidence is still an independently authored exhaustive person/track annotation package for frames 150-174, bound to those canonical frame hashes, followed by one immutable detector-observation sequence fed identically to frozen simple-IoU and portable ByteTrack. Required first-attempt outputs remain ID switches, fragmentation, continuity, matched/missed observations, false-track observations, matched IoU, throughput/FPS, latency, elapsed time, CPU time and bounded resource cost. No tuning is permitted before measurement.

### Deterministic blocker and stop decision
This path has now reached the policy's three-session stop threshold without tested acceptance improvement. The blocker is concrete: the approved GitHub-hosted evidence lane intentionally may hash the fixed decoded frames but may not export frames or author labels; the current execution environment cannot materialize the Wikimedia binary/decoded frames to the independent annotation worker, and its local runtime has no network path to the canonical media host. Using detector/tracker output as annotation truth or weakening the evidence lane would violate the acceptance rules.

Therefore the tracking path is stopped at this gate rather than being polished or represented as complete. No additional tracker framework, Kalman/LAP/native machinery, tuning or synthetic substitute is authorized while this blocker remains. If a future execution environment can safely expose the exact authorized frames for independent annotation, resume directly at the existing head-to-head gate and preserve the untouched first attempt.

## Current critical path — LPR/OCR
The safe alternative is now the highest-priority executable work item. Branch `feature/lpr-ocr-first-runnable` introduces the first replaceable runnable plate/text engineering baseline without adding automatic network/model acquisition:

- plate detection: Open Model Zoo `vehicle-license-plate-detection-barrier-0106` FP16 at exact OMZ commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, with XML/BIN size + SHA-384 verification before OpenVINO opens either artifact;
- runtime: existing reviewed OpenVINO `2026.3.1` CPU family; the adapter rejects a changed model input shape or runtime release;
- OCR engine: Tesseract `5.5.3` at exact commit `db0ec62f81b0737fbbe184d8fea40af5738f8eef`, invoked as an optional local CLI rather than vendored;
- OCR model: `tesseract-ocr/tessdata_fast@87416418657359cb625c412a48b6e1d6d41c29bd`, `eng.traineddata` exactly 4,113,088 bytes with immutable Git blob `bbef4675053b5b468cdb477053e28b1c698ba08e`; local verification also records SHA-256;
- frame boundary: normalized plate boxes -> bounded in-memory BGR crop -> dependency-free RGB PPM -> Tesseract TSV -> normalized 1-16 character A-Z/0-9 text observation;
- no image/media retention, identity behavior, training, network access or accuracy claim is introduced by the library.

Focused local regressions for the new boundary passed 7/7 before repository CI. They cover OMZ row admission, malformed input, exact traineddata binding, BGR-to-PPM crop encoding, TSV normalization/confidence and detector->OCR composition. These deterministic fixtures establish interface behavior only, not LPR accuracy.

### Commercial/release boundary
This first LPR/OCR slice is an engineering integration baseline, not a commercially validated plate reader. The OMZ plate detector is barrier/front-facing and its broader training-data provenance and geographic/plate-style generalization still require release review. Tesseract and tessdata_fast are Apache-2.0, but native Tesseract/Leptonica/image-codec packaging, platform-specific dependency notices, exact training-corpus/font provenance and product redistribution review remain release gates. No donor/model is admitted as a commercial-accuracy claim merely because the code path is runnable.

### Next executable LPR/OCR acceptance step
1. Get this exact branch green on Linux, Windows and the Analytics quality gate without adding downloads to normal CI.
2. After merge, define the smallest rights-cleared plate/text evidence package and first real runnable sample with explicit plate/text ground truth; do not use OCR output as labels.
3. Measure plate detection misses/false positives, exact/character read accuracy where ground truth permits, latency/FPS and CPU/resource cost on identical immutable evidence.
4. If the current detector or OCR engine fails a measured requirement, change only the demonstrated weak component; do not add a second OCR/detector framework speculatively.

## Retained person-down / slip-fall path
The required secondary path remains:
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`.

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. Existing detector/temporal behavior and hard negatives remain protected while higher-priority shared perception/evaluation work advances.

## Runtime/model provenance retained
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- OpenVINO Runtime `2026.3.1`.
- Retained person-down detector `person-detection-0200` FP16 and pose model `human-pose-estimation-0001` FP16.
- Portable ByteTrack donor `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, MIT, association slice only.
- Tesseract `5.5.3` commit `db0ec62f81b0737fbbe184d8fea40af5738f8eef`, Apache-2.0, optional local CLI baseline.
- tessdata_fast English model commit `87416418657359cb625c412a48b6e1d6d41c29bd`, Apache-2.0, exact blob pinned above.

## Evidence/data provenance retained
- Wikimedia CC0 pedestrian source: `Video Codec Test pedestrian area 1080p25.y4m.webm`, author Taurus Media Technik, CC0-1.0, 11,215,394 bytes, SHA-1 `51e89a672896e45cca17aa46cd223630a6266e26`, SHA-256 `bfadaa62cccb42db875d50bb842aa0964fbf72040432e4097c1df59e043e0c26`.
- GMDCSA-24: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos@5abac7693229900cf80f722e878fbb119211fc1c`, reviewed repository license MIT, Zenodo DOI `10.5281/zenodo.13354453`.
- Figshare 2017 activity source: article `28596332`, version 2, CC-BY-4.0; media remains outside public GitHub.

## Efficiency ledger
The LPR/OCR work item started from verified `main` `0fdea197909bac0190cb61e6b4f3d3276cf748d0` with zero open implementation PRs and zero queued/running runs for that exact head. Main run #172 was completed successfully on attempt 1. Tracking separately reached three consecutive sessions without its real-video acceptance improvement and was stopped as required. For the new safe-alternative LPR/OCR item, the fresh local preflight snapshot used zero open implementation PRs, zero active runs, zero unchanged retries, zero CI dispatches and zero stalled LPR sessions; `python tools/guardrails.py preflight --snapshot ...` returned `{"allowed": true, "reasons": []}`. Opening the single implementation PR is CI-triggering action 1/2. No paid resource, self-hosted runner, private/home/customer media, training, release, license change or accuracy claim is introduced.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
python -m unittest tests.test_lpr_ocr -v
```

## Merge rule
Merge the current LPR/OCR acceptance item only if the exact PR head passes Linux, Windows and the Analytics quality gate on unchanged base `0fdea197909bac0190cb61e6b4f3d3276cf748d0`. Verify base/head immediately before merge. One unchanged retry is permitted only for a diagnosed transient infrastructure failure; deterministic failure requires a fix before another run.

## Outstanding commercial-release gates
Detection/tracking still requires substantially broader held-out real-video evidence across people/vehicles/objects, crowded scenes, crossings, occlusions, low light, viewpoints and resolutions; defensible precision/recall where labels permit; track fragmentation/ID-switch measurements; throughput/latency/resource envelopes; privacy/security/provenance; versioned integration; packaging; and explicit owner commercial-release approval.

LPR/OCR now has a runnable baseline but still requires rights-cleared real plate/text evidence, held-out detection/OCR evaluation, broader plate-style/geography validation, runtime/package dependency closure and release approval. Face (including selectable face blurring), Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before any commercial claim.
