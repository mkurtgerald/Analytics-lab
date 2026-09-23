# Standard Open Images V4 SSD MobileNetV2 donor review

Status: **discovery only — not admitted**. This is the changed donor approach after the FaceSSD execution/conversion hypothesis was closed. No inference, model construction, media, extraction, retention, or accuracy claim is authorized by this review.

## Official model-zoo lineage

The candidate is Google's/TensorFlow Object Detection API standard Open Images V4 detector `ssd_mobilenetv2_oidv4`, archive `ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz`. The official model-zoo entry is pinned to `tensorflow/models@0558408514dacf2fe2860cd72ac56cbdf62a24c0`. That revision's repository license is Apache-2.0; its `LICENSE` Git blob is `43fcf7bf1f1f9f824a1debf05d6ced45bf5810aa`.

This is deliberately distinct from the quantized FaceSSD donor. The standard model-zoo entry reports Boxes output and Open Images Challenge mAP, but those upstream benchmark numbers are provenance/context only and are not Analytics Lab accuracy evidence.

## Corrected OIDv4 class mapping

The corrected Open Images V4 label map is pinned at the same TensorFlow Models lineage: `research/object_detection/data/oid_v4_label_map.pbtxt`, Git blob `643b9e8ed5d9239a3248b895fb32f3b51caa92f3`. The exact face class mapping is:

- display name: `Human face`
- class id: `502`
- MID: `/m/0dzct`

This corrected mapping matters because the TensorFlow Models history contains a later OIDv4 label-map correction. Do not infer class numbers from an earlier model-zoo snapshot or another Open Images label map.

## Open Images rights caveat

Google's Open Images V4 documentation states that annotations are CC BY 4.0 and images are listed as CC BY 2.0, while also making no representations or warranties about the license status of each image and instructing users to verify each image license themselves. That per-image verification requirement remains a release/legal and any future training/evaluation-data gate. No Open Images image is downloaded or used by this admission step.

## Bounded first evidence head

Branch family `evidence/face-donor-facessd-standard-oid-*` is a temporary routing exception that reuses the existing FaceSSD workflow exclusion from generic real-video evidence while explicitly prohibiting any FaceSSD download on this changed approach. The first head may only:

1. download the exact Google-hosted standard OIDv4 SSD archive in memory under a 250,000,000-byte hard ceiling;
2. compute archive byte length and SHA-256;
3. inspect tar metadata safely in memory and SHA-256 every regular member;
4. emit complete member identities plus the pinned provenance above.

It may not extract files, construct a network, install a runtime, call inference, provide an input tensor, process any image/video, retain/upload any model artifact, or call the donor admitted.

## Admission gate

A later head in the same single PR may mark the artifact admitted only after the exact archive size/SHA-256 and complete regular-member `(path, size, SHA-256)` set discovered from the official archive are hard-pinned and reverified unchanged. Every relevant frozen-graph/config/label-map identity must be explicit; if the archive contains no label-map member, the corrected external TensorFlow Models label-map identity above remains the authoritative class mapping and that absence must be recorded rather than silently inferred.

Only after immutable artifact admission may a separately reviewed head attempt the smallest existing-runtime construction path, with OpenVINO 2026.3.1 and/or OpenCV 4.12.0 bounded to one justified first attempt each. No runtime or inference is approved by this document.
