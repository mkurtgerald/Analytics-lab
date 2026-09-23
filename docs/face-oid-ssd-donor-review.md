# Standard Open Images V4 SSD MobileNetV2 donor review

Status: **artifact/provenance identity pinned; exact-head hosted re-verification required before merge**. This is the changed donor approach after the FaceSSD execution/conversion hypothesis was closed. No inference, model construction, media, extraction, retention, or accuracy claim is authorized by this admission step.

## Official model-zoo lineage

The candidate is Google's/TensorFlow Object Detection API standard Open Images V4 detector `ssd_mobilenetv2_oidv4`, archive `ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz`. The official model-zoo entry is pinned to `tensorflow/models@0558408514dacf2fe2860cd72ac56cbdf62a24c0`. That revision's repository license is Apache-2.0; its `LICENSE` Git blob is `43fcf7bf1f1f9f824a1debf05d6ced45bf5810aa`.

This is deliberately distinct from the quantized FaceSSD donor. The standard model-zoo entry reports Boxes output and an upstream Open Images benchmark, but that upstream benchmark is lineage/context only and is not Analytics Lab accuracy evidence.

## Corrected OIDv4 class mapping

The corrected Open Images V4 label map is pinned at the same TensorFlow Models lineage: `research/object_detection/data/oid_v4_label_map.pbtxt`, Git blob `643b9e8ed5d9239a3248b895fb32f3b51caa92f3`. The exact face class mapping is:

- display name: `Human face`
- class id: `502`
- MID: `/m/0dzct`

The Google archive contains **no label-map member**. That absence is explicit and fail-closed in the admission probe; the corrected external TensorFlow Models label-map identity above is therefore the authoritative class mapping. This matters because TensorFlow Models history includes an earlier OIDv4 label-map correction, so class numbers are not inferred from an older snapshot or another Open Images map.

## Open Images rights caveat

Google's Open Images V4 documentation states that annotations are CC BY 4.0 and images are listed as CC BY 2.0, while also making no representations or warranties about the license status of each image and instructing users to verify each image license themselves. That per-image verification requirement remains a release/legal and any future training/evaluation-data gate. No Open Images image is downloaded or used by this admission step.

## First bounded discovery result

Analytics quality run #259 used the exact official Google archive and passed Linux regression, Windows regression, and the aggregate Analytics quality gate. The discovery step performed no extraction, inference, network construction, media processing, model retention, or artifact upload. It measured:

- archive size: `158,851,107` bytes
- archive SHA-256: `8dd82cc52625eb9b43c6ed8050868fe1e78f878c6f028519fc731b3c7e9035f7`
- total tar members: 10 (three directories, seven regular files)

The seven regular files, in tar order, are now pinned exactly:

1. `ssd_mobilenet_v2_oid_v4_2018_12_12/model.ckpt.meta` — 11,915,505 bytes — SHA-256 `541aebeef5c674609f4b2cc229c265b053dc7df3226143b01037c0a42d715f7f`
2. `ssd_mobilenet_v2_oid_v4_2018_12_12/checkpoint` — 77 bytes — SHA-256 `dd1b025d2e155283f5e300ce95bf6d5b6bc0f7fe010db73daa6975eb896ab9cb`
3. `ssd_mobilenet_v2_oid_v4_2018_12_12/frozen_inference_graph.pb` — 66,606,111 bytes — SHA-256 `150f3eb77b741ed24e1a19559589205ccd616b059fc976c8b8d5cefc126bf86b`
4. `ssd_mobilenet_v2_oid_v4_2018_12_12/saved_model/saved_model.pb` — 67,889,548 bytes — SHA-256 `f5453b9c2bb73be4d21eef9eb37a8fce0a2d09903ea1aecd44ff49fa6efe258f`
5. `ssd_mobilenet_v2_oid_v4_2018_12_12/model.ckpt.index` — 14,175 bytes — SHA-256 `0557da0b4b7d555fc1483d3ba3a89732c7606e7176c30839335333f48b9d81f4`
6. `ssd_mobilenet_v2_oid_v4_2018_12_12/pipeline.config` — 4,267 bytes — SHA-256 `cf424b06dabcc7acd6bf71ffd941c5a9890b975f8036444e03a2290391a24614`
7. `ssd_mobilenet_v2_oid_v4_2018_12_12/model.ckpt.data-00000-of-00001` — 57,841,536 bytes — SHA-256 `61bc5931d1cc44cd83b80ebee3cb75757e84c1a49ac5cea71149e36a93da4044`

The three directories are `ssd_mobilenet_v2_oid_v4_2018_12_12`, `ssd_mobilenet_v2_oid_v4_2018_12_12/saved_model`, and `ssd_mobilenet_v2_oid_v4_2018_12_12/saved_model/variables`.

## Admission gate

The current head hard-pins the archive size/SHA-256 and complete regular-member `(path, size, SHA-256)` sequence above. It also requires the frozen graph, SavedModel graph, and pipeline config to be present and explicitly fails if a label-map member unexpectedly appears. Admission is valid only if the unchanged exact head re-downloads the official archive in the bounded hosted lane and re-verifies every identity.

Only after that exact-head admission is green may a later work item attempt the smallest existing-runtime construction path: the standard frozen graph with already-reviewed OpenVINO `2026.3.1` and/or OpenCV `4.12.0`, bounded to one justified first construction attempt each. No inference is authorized merely by artifact admission. If construction succeeds, the first execution must be a no-media synthetic tensor smoke before any separately rights-cleared real-person evidence is considered.
