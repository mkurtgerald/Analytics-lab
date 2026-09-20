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
Tracking remains stopped at the policy's three-session threshold. The implementation boundary is intact: detector-neutral tracking contract, portable ByteTrack slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` (MIT), bounded evaluator, rights/evidence admission, pre-registered Wikimedia CC0 frames 150-174, canonical RGB24 frame identities, exhaustive ground-truth package binding, frozen simple-IoU control, and first-attempt head-to-head runner.

The fixed frame-manifest SHA-256 is `52f00a4fc1e013d9c0cae9647cb386369441f005a970d6d42c449cbd0269a031`; the benchmark-plan SHA-256 is `eb7995a389a22f3528b9bfca97d64f9d8e3c515ebf1c764cf25b68fbdcc3479c`.

The remaining indispensable tracking evidence is still an independently authored exhaustive person/track annotation package for frames 150-174, bound to those frame hashes, followed by one immutable detector-observation sequence fed identically to frozen simple-IoU and portable ByteTrack. Required untouched first-attempt outputs remain ID switches, fragmentation, continuity, matched/missed observations, false-track observations, matched IoU, throughput/FPS, latency, elapsed time, CPU time and bounded resource cost. No tuning is permitted before measurement.

The deterministic blocker is unchanged: the approved hosted lane may hash but not export/label those decoded frames, while the current independent annotation environment cannot materialize the exact source pixels. No detector/tracker output may substitute for ground truth. Do not add tracker machinery until the exact independent-label path becomes executable.

## LPR/OCR — current critical path
PR #72 merged the first replaceable runnable engineering baseline into `main` at `2244bd2eee1f99fdfbc32f1ba636c288b7bb38a0`. PR run #173 and post-merge run #174 both passed Linux, Windows and the Analytics quality gate on attempt 1.

Current baseline:
- plate detector: Open Model Zoo `vehicle-license-plate-detection-barrier-0106` FP16 at exact OMZ commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0, exact size + SHA-384 verification before OpenVINO opens artifacts;
- runtime: reviewed OpenVINO `2026.3.1` CPU family;
- OCR engine: Tesseract `5.5.3` at exact commit `db0ec62f81b0737fbbe184d8fea40af5738f8eef`, Apache-2.0, optional local CLI;
- OCR model: `tesseract-ocr/tessdata_fast@87416418657359cb625c412a48b6e1d6d41c29bd`, `eng.traineddata` exactly 4,113,088 bytes with immutable Git blob `bbef4675053b5b468cdb477053e28b1c698ba08e`;
- data path: normalized plate box -> bounded in-memory BGR crop -> dependency-free RGB PPM -> Tesseract TSV -> normalized A-Z/0-9 observation;
- no automatic model/media download, image retention, identity behavior, training, or accuracy claim in the library.

### Current acceptance item — preserved first staged plate smoke
Selected source:
- Wikimedia Commons `China license plate-Chongqing 渝A 92518.png`;
- rights page pinned to `oldid=856647637`;
- author: Sdee at Chinese Wikipedia;
- rights: author-released public domain / permission for use for any purpose;
- canonical upload host: `upload.wikimedia.org`;
- dimensions: 680x144;
- byte size: 61,465;
- published SHA-1: `d365a117631a8fa5a2a0fb7a8d2a03fe2e9b73bc`;
- pinned source SHA-256 discovered on the admission-only first head: `47ea02127b3c22856ac548164c822b104c74f7afb711a0cb43458080a11d5433`;
- independently declared normalized alphanumeric text: `A92518` (the Chinese province glyph is deliberately outside the current A-Z/0-9 baseline contract).

PR #74 adds the narrowly scoped hosted LPR evidence lane and excludes that branch from the unrelated person-down/real-video evidence step. Run #177 on first head `5a5a4a83cc49ff70beae47cecfeb501333b29584` passed Linux, Windows and the Analytics quality gate on attempt 1, verified the published byte size/SHA-1, discovered the SHA-256 above, and performed no inference. Run #178 on changed head `f46895d71f963d7460421dde8a08609658c31e6f` also passed Linux, Windows and the Analytics quality gate on attempt 1 and executed the untouched CPU plate-detector smoke with the SHA-256 pinned.

Preserved first-attempt detector smoke result:
- plate detections: **0**;
- best confidence: null;
- full-frame IoU: `0.0`;
- elapsed: `0.007265238 s`;
- CPU: `0.012123617 s`;
- latency: `7.265238 ms`;
- one-image throughput observation: `137.64174 FPS`;
- OpenVINO runtime: `2026.3.1-22476-759c5a6ab8c-releases/2026/3`.

This is a demonstrated detector miss on a staged plate-only graphic. It is not roadway/camera evidence, not a precision/recall estimate, and not commercial accuracy/generalization evidence. No detector threshold or parameter was changed after inspection. Because the detector emitted no plate box, this smoke does not yet measure OCR behavior; the next smallest diagnostic is to exercise the already-reviewed OCR crop contract directly on the exact admitted plate image before considering any detector replacement or second OCR framework.

### Next executable LPR/OCR steps
1. Merge PR #74 only after the documentation-complete exact head again passes all applicable required checks on unchanged base `ac5cc1e61d3b7566fa78eecd14b0976cce22e709`.
2. Localize detector-versus-OCR fitness without tuning: run the exact admitted plate graphic through the existing bounded plate-crop/OCR contract directly, preserving the declared `A92518` engineering-smoke target and recording raw OCR text plus latency/CPU/resource cost. Do not call this accuracy evidence.
3. If the direct OCR path works but the detector still misses, move to the smallest rights-cleared vehicle-scene plate sample before changing detectors; if OCR itself fails, close the exact Tesseract/runtime portability or recognition gap first.
4. Expand only after the smoke path is localized to the smallest rights-cleared, independently labeled real plate/text set with positives and negatives; measure plate misses/false positives, exact/character reads, latency/FPS and CPU/resource cost.
5. Do not introduce a second detector/OCR framework, training, or tuning unless measured evidence identifies a concrete unmet requirement.

## Retained person-down / slip-fall path
The required secondary path remains:
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`.

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. Existing detector/temporal behavior and hard negatives remain protected while higher-priority shared perception/evaluation work advances.

## Runtime/model provenance retained
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- OpenVINO Runtime `2026.3.1`.
- Retained person-down detector `person-detection-0200` FP16 and pose model `human-pose-estimation-0001` FP16.
- Portable ByteTrack donor `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, MIT, association slice only.
- Tesseract `5.5.3` commit `db0ec62f81b0737fbbe184d8fea40af5738f8eef`, Apache-2.0.
- tessdata_fast English model commit `87416418657359cb625c412a48b6e1d6d41c29bd`, Apache-2.0.

## Outstanding commercial-release gates
Detection/tracking still requires broader held-out real-video evidence across people/vehicles/objects, crowded scenes, crossings, occlusions, low light, viewpoints and resolutions; defensible precision/recall where labels permit; track fragmentation/ID-switch measurements; throughput/latency/resource envelopes; privacy/security/provenance; versioned integration; packaging; and explicit owner commercial-release approval.

LPR/OCR has a runnable engineering baseline but still requires rights-cleared held-out real plate/text evidence, broader plate-style/geography validation, exact native-runtime/package dependency closure, notices/provenance review and release approval. The staged public-domain smoke source above does not reduce those commercial gates.

Face (including selectable face blurring), Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before any commercial claim.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
python -m unittest tests.test_lpr_ocr tests.test_lpr_wikimedia_evidence -v
```
