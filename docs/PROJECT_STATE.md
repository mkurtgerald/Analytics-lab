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
PR #72 merged the first replaceable runnable baseline. PR #74 preserved the staged public-domain plate smoke. PR #75 preserved the first untouched real vehicle-scene miss. PR #76 preserved the documented-envelope front-view miss. PR #77 merged the frozen model-free OpenCV proposal comparison into `main` at `59c04ea9af76641c25b567b68a2f38c853dacd5f`. PR #78 then bound the independently authored fixed crop to canonical RGB24 SHA-256 `0d89606f174889fdeab9fd969c3cfbe0a8e033c88eac6c0a4d0385fd45f56599`; post-merge Analytics quality run #192 passed on attempt 1. PR #79 merged the exact official Tesseract Windows execution path after preserving the pre-OCR package-version failure; post-merge Analytics quality run #195 passed on the unchanged merged head. PR #81 then merged the preregistered grayscale/Otsu comparison at `dd4b68974c61a6792ddd1a4873c7e4bc9ec27766`; exact-head run #202 and post-merge run #203 passed.

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

### Preserved untouched OCR control
The detector/proposal selection boundary is removed from OCR measurement. Independent visual inspection predeclared normalized text `MPR318` and the fixed pixel rectangle `(960, 2020, 2740, 2470)` on the 4032x3024 CC0 source before any OCR. PR #78 froze that crop at 1780x450 pixels, 2,403,000 canonical RGB24 bytes and SHA-256 `0d89606f174889fdeab9fd969c3cfbe0a8e033c88eac6c0a4d0385fd45f56599`.

The bounded evidence step uses the official upstream Tesseract 5.5.3 Windows release installer `tesseract-ocr-w64-setup-5.5.3.20260724.exe`, 26,573,224 bytes, SHA-256 `bee9e3434bd94fd65387d9be28cd467a41f61b1275383b55b0f59a1331270ae4`. It runs only on the existing GitHub-hosted `windows-2022` job under a deliberately named `evidence/lpr-ocr-exact-*` PR branch. The lane also downloads only the pinned `tessdata_fast` English artifact and the same reviewed CC0 source, verifies all immutable identities, re-verifies the pre-bound crop hash, installs below `RUNNER_TEMP`, performs exactly one OCR attempt through the existing exact-version adapter, uploads no media/model/runtime artifact, and reports only normalized OCR output plus bounded timing/CPU evidence.

This exact installer is admitted for ephemeral engineering execution only. Its bundled Leptonica/image-codec dependency notices and redistribution rights remain an explicit product-packaging gate; this lane does not approve redistributing the installer or native bundle.

Analytics quality run #193 preserved the first deterministic execution failure before OCR: the official Windows package reported `tesseract v5.5.3.20260724`, while the platform-neutral adapter intentionally admits canonical semantic version line `tesseract 5.5.3`. No OCR process was invoked, so run #193 was not an OCR measurement. The evidence-local correction admits only that one exact reviewed package line, records it, canonicalizes it to semantic version `5.5.3` for the unchanged core adapter, and rejects every other package build string before OCR.

Analytics quality run #194 then preserved the designated first untouched OCR observation on that exact fixed crop. Tesseract returned no normalized text and no confidence (`observed_text=null`, `observed_confidence=null`, `exact_match=false`). OCR elapsed time was 0.466289 s with 0.015625 s process CPU; the complete bounded evidence operation elapsed 8.871279 s with 0.796875 s process CPU. Linux regression, Windows regression and the aggregate gate all passed. This is one independently bound CC0 image and therefore an engineering smoke result only, never a commercial-accuracy estimate.

### Current acceptance item — one preregistered generic preprocessing comparison
The raw miss now justifies one minimal comparison before considering a heavier OCR engine or additional framework. Branch family `evidence/lpr-ocr-exact-preprocess-*` preregisters exactly one generic transform on the identical canonical RGB24 crop: convert RGB to grayscale, apply OpenCV global Otsu binary thresholding at native 1780x450 resolution, convert the binary result back to RGB24 for the unchanged P6/Tesseract adapter, and make exactly one OCR call. There is no resize, inversion, morphology, CLAHE, plate-specific threshold, alternate page-segmentation mode, image-specific parameter, training, or tuning sweep.

The comparison must bind and report the original crop SHA-256, transformed RGB24 SHA-256, measured Otsu threshold, normalized OCR result/confidence, exact-match flag, OCR elapsed/CPU time and total elapsed/CPU time. The first attempt is preserved whether it succeeds or misses. A useful result may retain this preprocessing slice and move to the smallest additional rights-cleared plate/text evidence set; a miss must not trigger iterative preprocessing polish and instead advances donor/runtime reassessment under the existing commercial-rights gates.

Runs #196 and #197 both reached the exact Windows OCR evidence step after green Linux and synthetic regressions, then failed before an OCR observation could be recorded because `parse_tesseract_tsv` required the TSV header on stdout line 1. The exact Tesseract package emitted bounded informational preamble text before the TSV table on this transformed input. The parser fix is deliberately narrow: search only the first 32 stdout lines for a real tab-delimited header containing `level`, `conf`, and `text`; if no such header exists, fail closed exactly as before. The crop, Otsu transform, PSM 7, Tesseract/tessdata identities, expected text and single-attempt rule are unchanged.

Run #202 identified the deeper deterministic issue and cleared it: the minimal `--tessdata-dir` intentionally contained only `eng.traineddata`, so naming the upstream `tsv` config file could not work. Pinned Tesseract 5.5.3 shows that config is exactly `tessedit_create_tsv 1`; the adapter now requests the equivalent output directly with `-c tessedit_create_tsv=1`. The frozen Otsu comparison then completed and returned `IFMPR318J` at confidence 0.4213968133 versus expected `MPR318` (`exact_match=false`), Otsu threshold 129, OCR elapsed 0.469441 s, CPU 0.015625 s. This proves the recognition core is present but with extra surrounding characters; do not tune against this one image.

### Preserved second rights-cleared plate/text sample
The South Carolina sample is now measured without tuning. Admission run #204 bound source SHA-256 `ace70508c959f5c0e8f915d6cd07aa81ad6f7f92e12d8a72e01cc50a2d40750c` plus canonical 1175x310 RGB24 crop SHA-256 `c13f5cb796cbe2d6a6505e41bf75d69d3dc1c9877bd4c8ef0a55a3389b3ced4b`. Exact-head run #206 passed Linux, Windows, bounded OCR evidence and the Analytics quality gate; post-merge run #207 also passed. The unchanged retained path returned `354AVIV` at confidence 0.403112795 versus expected `354AVV` (`exact_match=false`), Otsu threshold 115, OCR elapsed 0.299616 s. This is engineering evidence only and does not justify image-specific tuning.

### Current acceptance item — third rights-cleared plate/text sample
Continue breadth before adding a normalization layer. The next source is Wikimedia Commons `CPU4704 Unembossed Washington License Plate.jpg`, copyright-holder CC0, permanent rights page oldid `1269104394`. Published identity: 5,532,415 bytes, 3549x1779, SHA-1 `39a9d530a1d40d960043d51f7f5c67c906f0450f`. Independent visual inspection predeclares normalized serial `CPU4704` and crop `(180, 600, 3380, 1510)`, which excludes the state name/top registration tabs and bottom motto while preserving the complete serial.

The first head is admission-only: verify size/SHA-1, discover source SHA-256, decode ephemerally and bind the canonical RGB24 crop hash. No OCR occurs until those identities are pinned. A later head may run the already-retained native-size grayscale -> global Otsu -> RGB24 transform, exact Tesseract 5.5.3/tessdata identities and PSM 7 unchanged. This remains engineering evidence only, not commercial accuracy.

Admission run #208 passed Linux, Windows and the Analytics quality gate and bound source SHA-256 `265046b1e374d1a457658308e75f713aa3796c44092f3f8173930d3a7aa17a09` plus canonical 3200x910 RGB24 crop SHA-256 `0e4655be2fe57d47c890855cf22371ff2b7ab764cfc9c1084a692a9ee67d8bca`. No OCR ran on that head. The current head pins both identities and performs exactly one retained Otsu/Tesseract observation with no parameter changes.


## Retained person-down / slip-fall path
The required secondary path remains:
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`.

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. Existing detector/temporal behavior and hard negatives remain protected while higher-priority shared perception/evaluation work advances.

## Outstanding commercial-release gates
Detection/tracking still needs broader held-out real-video evidence across people/vehicles/objects, crowded scenes, crossings, occlusions, low light, viewpoints and resolutions; defensible precision/recall where labels permit; track fragmentation/ID-switch measurements; throughput/latency/resource envelopes; privacy/security/provenance; versioned integration; packaging; and explicit owner commercial-release approval.

LPR/OCR still needs rights-cleared held-out real plate/text positives and negatives, broader plate-style/geography validation, exact native-runtime/package dependency closure, notices/provenance review, and explicit release approval. All current smoke sources remain non-commercial-accuracy evidence.

## Face detection + automated/selectable blurring — owner-directed coordinated sprint
On September 22, 2026 the owner explicitly redirected the active implementation lane to Face detection and automated blurring in tandem. LPR/OCR PR #84 is preserved closed/unmerged for later continuation; this face sprint owns the single implementation slot.

The first acceptance unit deliberately separates detector integration from model admission. It adds normalized face boxes, immutable local face-model verification, a replaceable OpenCV FaceDetectorYN-compatible adapter with no built-in model URL/hash, default-on Gaussian face blurring with bounded box expansion, permission-gated unblur, unauthorized-unblur denial, input-frame immutability, bounded face counts, and privacy audit state that contains no biometric identity.

Detector donors are fail-closed:
1. OpenCV Zoo YuNet at `47534e27c9851bb1128ccc0102f1145e27f23f98`; `face_detection_yunet_2023mar.onnx` LFS SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`, 232,589 bytes, model-directory MIT. Upstream training repo `ShiqiYu/libfacedetection.train@a61a428929148171b488f024b5d6774f93cdbc13` explicitly trains on WIDER FACE, but commercial dataset/training-rights closure is not strong enough for the shippable path.
2. Open Model Zoo `face-detection-retail-0004` at `6697dead54ed1cdd664b0313189c2cb52ee6335e`, FP16 XML SHA-384 `a7f8d1d41998503c4f3cdd8c12275f04f1736e5142127edcb4c76c3e17188499390574095a5b2a9dd78d3d0f77d02034`, FP16 BIN SHA-384 `394185d3e42c34d7f9d43229ec8f5755c07e19fd6469d23883e71707fdd8eb66d90ff3ba1c94adac599b`, Apache-2.0. Published model docs do not establish the training-data provenance required by this repository.
3. MediaPipe BlazeFace short-range at `google-ai-edge/mediapipe@8ac5a39c659578c2595a54ef5277608173c217fe`, Apache-2.0 repository. Exact selected model training-data/weight provenance is not closed strongly enough for commercial shipping.

No face model is bundled or downloaded by this sprint. The detector adapter is ready for a separately admitted artifact without changing privacy architecture.

### Selected pretrained face donor — Google FaceSSD MobileNetV2 Open Images V4
The bounded donor screen found a materially stronger pretrained option in the official TensorFlow Object Detection model zoo: `facessd_mobilenet_v2_quantized_open_image_v4`, archive `facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz`. TensorFlow models source at `8b12ae202a3ccf8f965c730a4e7617204e32000b` is Apache-2.0. Google's model-zoo table reports ~20 ms reference speed and 73 mAP@0.5 for faces, and its footnote states non-face boxes were dropped during training and non-face ground truth ignored during evaluation. Training provenance is Open Images V4 face boxes, materially stronger than the previously rejected WIDER/undisclosed paths.

Open Images annotations are CC BY 4.0 and source images are listed as CC BY 2.0, with Google's explicit caveat that consumers should verify source-image license status. Treat that caveat as a release/legal-review item rather than hiding it. The pretrained model itself is an official Google/TensorFlow artifact under the TensorFlow Apache-2.0 distribution.

Artifact admission is now closed fail-closed at the archive/member-identity boundary. Run #218 passed Linux, Windows and the Analytics quality gate while admitting the exact 130,655,026-byte Google archive and witnessing archive SHA-256 `9ae49a245caddbe7d7bbc82a35da0191a2f2e210161df19be357a1c7f49118d5`, with no extraction, retained model artifact, inference or media. Run #219 then pinned that archive SHA and hashed every regular tar member in memory while again passing Linux, Windows and the Analytics quality gate. The seven regular member identities are now fixed in code: `face_label_map.pbtxt` 56 bytes SHA-256 `87f1e97ff18442302ca7686276e59beac925126b91b14cb15dee710de2ee9c60`; `model.ckpt.data-00000-of-00001` 86,462,816 bytes `693c0eb84b8d9349391d66c38d2ac03b3d628029eb9f9ad7417302f6ac355599`; `model.ckpt.index` 68,838 bytes `e00ccd57873134cc9a8a24cf2f39a16ee59a6037012755564639e06197c15729`; `model.ckpt.meta` 21,373,388 bytes `65346dc8d8f11297df102c6e134120d3a248dd3b7e38ec0fcf3799f49d137c28`; `pipeline.config` 4,829 bytes `f51b55181cf8a614c75ffe716b5c5a6f253ca5037199f37440775a925482f224`; `tflite_graph.pb` 22,222,216 bytes `dc8e2c9e21407b2f6d35f1eb655ba8a0c9c73094e5987231a1a8de2edae74978`; `tflite_graph.pbtxt` 62,525,550 bytes `e1232ff66eedd5676bfa31e78aba28ee6244b0322a5646d1a1f89efa02cb781b`. Any archive-size, archive-hash, member-order, member-size or member-hash change now fails closed before the donor can advance.

This closes immutable artifact identity and engineering provenance only. Runtime compatibility/execution is not yet approved: the currently merged product adapter is OpenCV `FaceDetectorYN`-compatible, while this donor supplies TensorFlow checkpoint/TFLite-graph artifacts. The next face acceptance step must review the smallest compatible execution/conversion path and its transitive runtime dependencies without adding a second framework or licensing shortcut unless measured need justifies it. Product packaging/redistribution notices, Google's per-image Open Images license-verification caveat, held-out rights-cleared real-person evidence and any accuracy claim remain separate gates.

### Preserved face runtime classification — direct OpenCV importer closed
PR #87 merged at `7177d881c13931bfa4e2b8e80ea1b756945f6cb4`; exact-head run #222 and post-merge run #223 passed. The exact admitted FaceSSD archive/member identities were reverified before runtime action. OpenCV 4.12.0 could not construct the quantized TensorFlow graph either binary-only or with the shipped `tflite_graph.pbtxt`; the failures occurred inside OpenCV's TensorFlow importer on quantized/folded constant-node patterns. No inference or media occurred. This closes the direct `readNetFromTensorflow` route without rejecting the donor.

### Current face acceptance item — OpenCV 4.12 SSD graph rewrite classification
Before introducing TensorFlow Lite or another runtime, test OpenCV's own Apache-2.0 TensorFlow Object Detection SSD rewrite helper at exact OpenCV commit `49486f61fb25722cbcf586b7f4320921d46fb38e`. Exact helper identities are pinned: `samples/dnn/tf_text_graph_ssd.py`, 18,314 bytes, Git blob `d27fd0d384f509789bf64f069203d3f9964db576`; and `samples/dnn/tf_text_graph_common.py`, 10,055 bytes, Git blob `c82053b4fb05aa2fd40c45b1cefa550431608847`.

The hosted evidence step must reverify the exact Google archive and all seven member identities, download only those two exact OpenCV helper sources, verify their Git-object identities, materialize only the pinned `tflite_graph.pb` and `pipeline.config` below `RUNNER_TEMP`, run the exact SSD helper to produce an ephemeral derived pbtxt, hash that derived pbtxt if produced, and ask the same pinned OpenCV 4.12.0 runtime to construct the rewritten graph. No `forward`, inference, media, retained/uploaded model, TensorFlow runtime, TFLite runtime or extra converter dependency is permitted.

A helper/rewrite/import failure is a valid negative classification and closes this zero-new-framework route; successful network construction proves runtime compatibility only, not face-detection accuracy. Preserve the first result without tuning the donor or helper.

CI for `evidence/face-privacy-*` installs only the already-reviewed pinned OpenCV wheel and runs synthetic detector-to-blur integration on Linux and Windows. It downloads no face model or real-person media and makes no detection-accuracy claim. The acceptance target is integration correctness: default blur changes bounded face regions, unauthorized unblur remains blurred, authorized unblur returns an unchanged copy, model artifacts fail closed on identity mismatch, and detector output is normalized consistently across platforms.

Face (including selectable face blurring), Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before commercial claims.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
python -m unittest tests.test_lpr_ocr tests.test_lpr_plate_proposals tests.test_lpr_wikimedia_evidence tests.test_lpr_vehicle_scene_evidence tests.test_lpr_front_envelope_evidence tests.test_lpr_ocr_fixed_crop_evidence tests.test_lpr_ocr_exact_evidence tests.test_lpr_ocr_sc_evidence tests.test_lpr_ocr_wa_evidence -v
```
