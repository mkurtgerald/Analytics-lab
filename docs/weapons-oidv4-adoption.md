# Weapons OIDv4 donor reuse boundary

Weapons reuses the already admitted Google/TensorFlow standard OIDv4 SSD MobileNetV2 artifact and OpenVINO 2026.3.1 runtime. The pinned TensorFlow Models revision is `0558408514dacf2fe2860cd72ac56cbdf62a24c0` under Apache-2.0, and the corrected OIDv4 label map is pinned by Git blob `643b9e8ed5d9239a3248b895fb32f3b51caa92f3`.

The first K5-owned adapter is limited to these label-map classes: Knife 285, Kitchen knife 325, Rifle 351, Shotgun 361, Sword 365, Weapon 408, and Handgun 533. This review does not authorize a second detector/runtime, new model or media acquisition, threshold tuning from synthetic fixtures, or any commercial accuracy/readiness claim.

The implementation must stay behind a replaceable project-owned interface, reuse the compile-once standard-OID/OpenVINO lifecycle already established for Face, validate exact output ports and normalized boxes fail-closed, and preserve upstream attribution. Real-world accuracy evidence, native dependency redistribution review, representative soak/performance work, and explicit commercial-release approval remain separate gates.
