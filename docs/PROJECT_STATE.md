# Project state — 2026-09-23

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

### Permanent priority
1. Detection + tracking.
2. LPR/OCR.
3. Face — including required selectable face blurring with policy-driven default blur, permission-gated unblur/reblur, server-side enforcement, auditability and preservation of original evidence.
4. Weapons.
5. Appearance search.

Person-down and slip/fall remain required deliverables and are secondary only in sequencing. On September 22, 2026 the owner explicitly moved Face detection + automated/selectable blurring into the active single implementation lane. LPR/OCR PR #84 remains closed/unmerged and must stay preserved until the owner changes direction.

## Detection + tracking — frozen at first real-video acceptance gate
Tracking remains stopped at the policy's three-session threshold. The retained boundary includes the detector-neutral tracking contract, portable ByteTrack slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` (MIT), bounded evaluator, rights/evidence admission, pre-registered Wikimedia CC0 frames 150-174, canonical RGB24 frame identities, exhaustive ground-truth package binding, frozen simple-IoU control, and first-attempt head-to-head runner.

The fixed frame-manifest SHA-256 is `52f00a4fc1e013d9c0cae9647cb386369441f005a970d6d42c449cbd0269a031`; benchmark-plan SHA-256 is `eb7995a389a22f3528b9bfca97d64f9d8e3c515ebf1c764cf25b68fbdcc3479c`.

The remaining blocker is unchanged: an independently authored exhaustive person/track annotation package for those exact frames cannot currently be materialized in the independent annotation environment without violating the approved evidence lane. No detector/tracker output may substitute for ground truth and no further tracker machinery should be added until that exact label path becomes executable.

## LPR/OCR — preserved while Face owns the lane
The retained baseline remains Open Model Zoo `vehicle-license-plate-detection-barrier-0106` FP16 behind a replaceable detector boundary, OpenVINO `2026.3.1` CPU, Tesseract `5.5.3`, pinned `tessdata_fast` English data, and the model-free OpenCV 4.12.0 proposal fallback. All current plate sources/results remain engineering evidence only, not commercial-accuracy evidence.

The detector donor screen reached the three-candidate policy ceiling and did not admit a stronger pretrained plate detector because weight/data redistribution provenance did not close cleanly. The model-free OpenCV proposal baseline therefore remains the smallest safe fallback. Exact Tesseract Windows execution and fixed-crop evidence are preserved, including the Land Rover, South Carolina, and Washington rights-cleared sample identities already recorded in repository tests and third-party/evidence documentation.

PR #84 is intentionally closed/unmerged. Do not revive, reopen, or supersede it while Face owns the single implementation slot.

## Retained person-down / slip-fall path
The required secondary path remains:
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`.

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. Existing detector/temporal behavior and hard negatives remain protected while higher-priority shared perception/evaluation work advances.

## Face detection + automated/selectable blurring — active owner priority
The privacy/product boundary is already in place independently of model admission: normalized face boxes, immutable local-model verification, replaceable detector adapter, default-on Gaussian blur with bounded box expansion, permission-gated unblur/reblur, unauthorized-unblur denial, input-frame immutability, bounded face counts, and privacy audit state containing no biometric identity. Face recognition, embeddings, ReID and identity matching remain explicitly out of scope.

### Donor-screen outcome
The first face donor screen remained fail-closed:
1. OpenCV Zoo YuNet was held out because its WIDER FACE training-data path did not meet the repository's commercial provenance standard.
2. Open Model Zoo `face-detection-retail-0004` was held out because model/training-data provenance was not disclosed strongly enough for the required code/weight/data separation.
3. MediaPipe BlazeFace was held out because the selected pretrained model's training-data/weight provenance did not close strongly enough for commercial shipping.

The official Google/TensorFlow FaceSSD MobileNetV2 Open Images V4 artifact was subsequently provenance-admitted and pinned, but its practical execution/conversion hypothesis is now **closed for this sprint**. Direct OpenCV import, OpenCV SSD rewrite, direct LiteRT, TensorFlow 1 legacy conversion, Coral preconverted lineage, LiteRT C++ SDK, stable `litert-converter`, `ai-edge-litert` Python, TensorFlow 2.21 pip/libclang, TensorFlow 2.21 TOCO source build, and OpenVINO full/cut GraphDef conversion have all been classified and must not be retried unchanged.

PR #104 merged the final negative OpenVINO evidence into `main` at `d6c790661c28126ec17e2745844c3704a63d8ac1`. Both full FaceSSD GraphDef conversion and the documented cut at `raw_outputs/box_encodings`, `raw_outputs/class_predictions`, and `anchors` failed in OpenVINO `2026.3.1` before CPU compilation. No inference, media, input tensor, or retained derived artifact occurred. This closes the FaceSSD/OpenVINO path and the broader FaceSSD practical execution hypothesis for the current sprint.

### Changed donor approach — standard OIDv4 SSD MobileNetV2
The active candidate is the official Google/TensorFlow Object Detection API standard Open Images V4 model `ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz`, not the quantized FaceSSD custom-postprocess graph. The model-zoo lineage is pinned to `tensorflow/models@0558408514dacf2fe2860cd72ac56cbdf62a24c0`, Apache-2.0, with LICENSE Git blob `43fcf7bf1f1f9f824a1debf05d6ced45bf5810aa`.

The corrected OIDv4 label map is pinned to `research/object_detection/data/oid_v4_label_map.pbtxt`, Git blob `643b9e8ed5d9239a3248b895fb32f3b51caa92f3`. `Human face` maps to class id `502`, MID `/m/0dzct`. The candidate archive contains no label-map member, so this exact corrected external label map is the authoritative mapping. Open Images V4 remains subject to Google's caveat: annotations are CC BY 4.0, images are listed CC BY 2.0, and each image license must be independently verified before any image is used for future training/evaluation.

Analytics quality run #259 was the bounded first discovery head and passed Linux, Windows and the aggregate gate. It downloaded the exact official Google archive only in memory, performed no extraction, inference, network construction, media processing or artifact retention, and measured:

- archive size `158,851,107` bytes;
- archive SHA-256 `8dd82cc52625eb9b43c6ed8050868fe1e78f878c6f028519fc731b3c7e9035f7`;
- seven regular files in exact tar order:
  1. `model.ckpt.meta` — 11,915,505 bytes — `541aebeef5c674609f4b2cc229c265b053dc7df3226143b01037c0a42d715f7f`;
  2. `checkpoint` — 77 bytes — `dd1b025d2e155283f5e300ce95bf6d5b6bc0f7fe010db73daa6975eb896ab9cb`;
  3. `frozen_inference_graph.pb` — 66,606,111 bytes — `150f3eb77b741ed24e1a19559589205ccd616b059fc976c8b8d5cefc126bf86b`;
  4. `saved_model/saved_model.pb` — 67,889,548 bytes — `f5453b9c2bb73be4d21eef9eb37a8fce0a2d09903ea1aecd44ff49fa6efe258f`;
  5. `model.ckpt.index` — 14,175 bytes — `0557da0b4b7d555fc1483d3ba3a89732c7606e7176c30839335333f48b9d81f4`;
  6. `pipeline.config` — 4,267 bytes — `cf424b06dabcc7acd6bf71ffd941c5a9890b975f8036444e03a2290391a24614`;
  7. `model.ckpt.data-00000-of-00001` — 57,841,536 bytes — `61bc5931d1cc44cd83b80ebee3cb75757e84c1a49ac5cea71149e36a93da4044`.

PR #105 then hard-pinned those archive/member identities, explicitly required the frozen graph, SavedModel graph and pipeline config, and failed closed if a label-map member unexpectedly appeared. Exact-head Analytics quality run #260 passed Linux, Windows, the bounded standard-donor admission evidence and the aggregate gate. PR #105 merged into `main` at `8267dd00bb645dbc41f76f2ee2e9542473b2bf14`; automatic main verification run #261 also passed. The standard OIDv4 artifact/provenance admission gate is therefore closed green. No model inference, media use, artifact retention or accuracy claim occurred in that admission.

### Standard OIDv4 runtime construction — closed green
PR #106 performed the smallest post-admission runtime-construction attempt with the exact admitted `frozen_inference_graph.pb` and OpenVINO `2026.3.1`. Exact-head Analytics quality run #262 passed Linux, Windows and the aggregate gate. OpenVINO's TensorFlow frontend loaded the graph and the CPU plugin compiled it successfully without inference, an input tensor, media use or retained derived artifact.

The verified model ports were:
- input `image_tensor:0` — uint8, `[?,?,?,3]`;
- boxes `Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/TensorArrayGatherV3:0` — float32, `[?,100,4]`;
- classes `add:0` — float32, `[?,100]`;
- scores `Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/TensorArrayGatherV3:0` — float32, `[?,100]`;
- count `Postprocessor/ToFloat_3:0` — float32, `[?]`.

PR #106 merged to `main` at `bb069a1899ce51a85b90d7a0ee6a7cc2cb1662a6`, and automatic main verification run #263 also passed. This is compatibility evidence only; it establishes neither face-detection accuracy nor commercial readiness.

### Standard OIDv4 synthetic execution — closed green
PR #107 added the class-502-only adapter and one no-media execution through the default-on privacy-blur path. Exact-head Analytics quality run #264 passed repository guardrails, Linux, Windows, the bounded synthetic face evidence and the aggregate gate on unchanged base `bb069a1899ce51a85b90d7a0ee6a7cc2cb1662a6`. PR #107 merged to `main` at `2829df3bb0dc5ac50eccc460715de8f95cb81b77`; automatic main verification run #265 also passed.

That smoke supplied one deterministic generated in-memory uint8 tensor to the admitted standard OIDv4 graph, admitted only class `502` (`Human face`) into the existing normalized `FaceDetection` contract, and exercised the default-on blur policy. No image/video source, real person, retained model/pixel artifact, tuning, recognition/ReID/identity matching, or face-detection accuracy/commercial-readiness claim occurred. The fixed pre-measurement face threshold remains `0.50`.

### Active face gate — rights-cleared real-person source admission
The single current work item is now admission of one separately rights-cleared real-person source before any real-person face inference. The selected first admission candidate is Wikimedia Commons `Face portrait (Unsplash).jpg`, source page `https://commons.wikimedia.org/wiki/File:Face_portrait_(Unsplash).jpg` and canonical upload `https://upload.wikimedia.org/wikipedia/commons/0/04/Face_portrait_%28Unsplash%29.jpg`.

Commons records the image as a William Stitt photograph dated 19 October 2016, published on Unsplash before the 5 June 2017 license change under CC0-1.0. Commons publishes byte length `9,023,019` and SHA-1 `d7ac58077d135b823e71e7b38e763f082e2053bc`. The first evidence head is discovery-only: stream at most 10,000,000 bytes in memory, verify those published properties, compute SHA-256, and emit only source/license identity. It must not decode the image, install a model/runtime, run inference, count faces, retain/upload media, or make an accuracy claim.

Only after exact SHA-256 is hard-pinned, reverified, and that source-admission head merges green may a separate evidence lane perform one untouched face-detection -> automatic-blur measurement with the already-admitted model and fixed `0.50` threshold. That first real-person measurement must preserve misses and false positives and must not tune before the result.

If the standard Google donor later fails on actual rights-cleared evidence, do not donor-shop indefinitely. Move to the project-owned training fallback only after the current donor path is conclusively closed: a permissive detector architecture such as the already-reviewed Apache-2.0 YOLOX-Nano style family plus individually rights-cleared Open Images Human-face training/validation data, with exact trainer/dependency/data/export lineage and no-spend/resource ceilings. Expensive training remains prohibited until the donor path is conclusively closed and the plan fits policy.

## Commercial-release gates
Nothing in the current face evidence supports a commercial accuracy claim. Artifact provenance, runtime compatibility, synthetic integration correctness, rights-cleared real-person accuracy evidence, privacy/security review, package/native dependency closure, notices, versioned integration and explicit owner commercial-release approval remain separate gates.

No home/customer media, credentials, biometric references, embeddings or retained face/model artifacts are permitted in CI. All evidence execution stays on GitHub-hosted runners inside the bounded branch families defined by `AGENTS.md`.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```
