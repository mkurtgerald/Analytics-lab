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

## Detection + tracking — frozen at first real-video acceptance gate
Tracking remains stopped at the policy's three-session threshold. The implementation boundary is intact: detector-neutral tracking contract, portable ByteTrack slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` (MIT), bounded evaluator, rights/evidence admission, pre-registered Wikimedia CC0 frames 150-174, canonical RGB24 frame identities, exhaustive ground-truth package binding, frozen simple-IoU control, and first-attempt head-to-head runner.

The fixed frame-manifest SHA-256 is `52f00a4fc1e013d9c0cae9647cb386369441f005a970d6d42c449cbd0269a031`; benchmark-plan SHA-256 is `eb7995a389a22f3528b9bfca97d64f9d8e3c515ebf1c764cf25b68fbdcc3479c`.

The remaining indispensable evidence is an independently authored exhaustive person/track annotation package for frames 150-174, bound to those hashes, followed by one immutable detector-observation sequence fed identically to frozen simple-IoU and portable ByteTrack. Required untouched outputs remain ID switches, fragmentation, continuity, matched/missed observations, false-track observations, matched IoU, throughput/FPS, latency, elapsed time, CPU time and bounded resource cost. No tuning is permitted before measurement.

The deterministic blocker is unchanged: the approved hosted lane may hash but not export/label those decoded frames, while the current independent annotation environment cannot materialize the exact source pixels. No detector/tracker output may substitute for ground truth. Do not add tracker machinery until the exact independent-label path becomes executable.

## LPR/OCR — current critical path
PR #72 merged the first replaceable runnable baseline. PR #74 preserved the staged public-domain plate smoke. PR #75 preserved the first untouched real vehicle-scene miss. PR #76 merged the documented-envelope front-view smoke into `main` at `49f450e6d97857cd6afc9502284eca21adda7f2f`; post-merge Analytics quality run #187 passed on attempt 1.

Current retained baseline:
- plate detector under replacement review: Open Model Zoo `vehicle-license-plate-detection-barrier-0106` FP16 at exact OMZ commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0, exact artifact size + SHA-384 verified before OpenVINO opens it;
- runtime: reviewed OpenVINO `2026.3.1` CPU family;
- OCR engine: Tesseract `5.5.3` at exact commit `db0ec62f81b0737fbbe184d8fea40af5738f8eef`, Apache-2.0;
- OCR model: `tesseract-ocr/tessdata_fast@87416418657359cb625c412a48b6e1d6d41c29bd`, `eng.traineddata` 4,113,088 bytes, immutable Git blob `bbef4675053b5b468cdb477053e28b1c698ba08e`;
- no automatic model/media download, image retention, identity behavior, training, or accuracy claim in the library.

### Preserved detector evidence
The untouched OMZ detector returned zero plate detections on all three bounded smoke sources, including the preregistered front-facing CC0 Land Rover image with a conservatively declared >=1000 px front plate versus the detector's documented 96 px minimum. The exact source is `Land Rover Defender 110 (L316), front view.jpg`, source SHA-256 `32e5637e39b54c26192c011c1cc5516bd6d35573582b8e09d7bc9aae90ef1db4`, visible normalized text `MPR318`. This single image is engineering-smoke evidence only, not precision/recall or commercial accuracy, but the documented-envelope miss is sufficient measured justification to stop polishing the OMZ detector and screen replacements.

### Bounded replacement screen — three donor candidates maximum
The donor screen is now closed at the policy maximum of three candidates:
1. `ankandrew/open-image-models@f22000e02b30642f317cdba7755c0631638b109e`: repository code is MIT and publishes ONNX plate detectors, but the exact release model asset license, training-dataset provenance/rights and immutable artifact identity are not established strongly enough for the shippable path. Reject for now rather than infer rights from the repository license.
2. `LdDl/license_plate_recognition@ccc2a4ca82845d13070326b2b7104785c484603f`: repository code is Apache-2.0 and publishes ONNX weights trained on Russian plates, but the training dataset and released weight rights/provenance are not pinned well enough for commercial redistribution. Reject for now.
3. `PaddlePaddle/PaddleOCR@dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`: repository code is Apache-2.0, but the plate-detector use would introduce a larger Paddle runtime/model path whose exact selected model artifact, training-data rights and redistribution/package closure are not yet established. Do not add that framework without stronger measured need.

Because all three donor candidates fail the current fail-closed commercial admission bar, the smallest safe alternative is a model-free OpenCV proposal baseline using the already-reviewed OpenCV 4.12.0 morphology/runtime family. This introduces no trained weights or training-data rights. OpenCV source tag `4.12.0` resolves to commit `49486f61fb25722cbcf586b7f4320921d46fb38e` under Apache-2.0; the `opencv-python` packaging tag `88` resolves to commit `fa742a47d3993e45502dff54d96af6a4efb65153` under MIT. Product wheel/native dependency notices remain a release gate.

### Current acceptance item — frozen model-free proposal comparison
The first OpenCV proposal defaults are frozen before real-image output inspection. They use a bounded working width, horizontal-gradient morphology, geometry/rectangularity filtering and deterministic NMS. Proposal scores are engineering heuristics, not probabilities.

The exact next measurement is the already-pinned CC0 front-envelope image used by PR #76. Run the frozen model-free proposal detector and the preserved OMZ detector on the identical decoded source and record aggregate candidate count, best score/confidence, latency, elapsed time, throughput/FPS and CPU cost. No parameter tuning is permitted after the first result.

Decision rule:
- if the model-free baseline produces a plausible bounded proposal without unacceptable runtime cost, retain it only as a provisional proposal stage and next exercise the existing OCR path on independently bounded plate evidence;
- if it produces no useful proposal, preserve the first attempt and change approach rather than tuning against this one image;
- this single-image comparison remains engineering smoke only and never commercial accuracy evidence.

## Retained person-down / slip-fall path
The required secondary path remains:
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`.

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. Existing detector/temporal behavior and hard negatives remain protected while higher-priority shared perception/evaluation work advances.

## Outstanding commercial-release gates
Detection/tracking still needs broader held-out real-video evidence across people/vehicles/objects, crowded scenes, crossings, occlusions, low light, viewpoints and resolutions; defensible precision/recall where labels permit; track fragmentation/ID-switch measurements; throughput/latency/resource envelopes; privacy/security/provenance; versioned integration; packaging; and explicit owner commercial-release approval.

LPR/OCR still needs rights-cleared held-out real plate/text positives and negatives, broader plate-style/geography validation, exact native-runtime/package dependency closure, notices/provenance review, and explicit release approval. All current smoke sources remain non-commercial-accuracy evidence.

Face (including selectable face blurring), Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before commercial claims.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
python -m unittest tests.test_lpr_ocr tests.test_lpr_plate_proposals tests.test_lpr_wikimedia_evidence tests.test_lpr_vehicle_scene_evidence tests.test_lpr_front_envelope_evidence -v
```
