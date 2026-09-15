# OpenVINO Open Model Zoo admission review — 2026-09-15

## Decision

**Admit the exact FP16 artifact manifests below as a local-only engineering baseline; do not auto-download or package the binaries yet.** The model files themselves were not downloaded or executed in this change. Commercial release still requires final dependency/provenance/privacy review and real-video evaluation.

## Reviewed upstream

- Project: `openvinotoolkit/open_model_zoo`
- Reviewed commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`
- Repository license at that commit: Apache-2.0
- Open Model Zoo checksum implementation at that commit identifies model manifest checksums as SHA-384.

### Person detector candidate

`models/intel/person-detection-retail-0013/model.yml` at the reviewed commit identifies an Intel pedestrian detector and explicitly points to the Open Model Zoo license. The selected FP16 pair is:

| Artifact | Bytes | SHA-384 |
|---|---:|---|
| `person-detection-retail-0013/FP16/person-detection-retail-0013.xml` | 571233 | `99ad3d4580a0123bef05ff77b6f46ccec16de974d1f5699fb94cd842e3242c6aa641f4977f9a5bb2f0fab42fe51cbb63` |
| `person-detection-retail-0013/FP16/person-detection-retail-0013.bin` | 1445734 | `a67422e3b5ec76057651d2a0237eab862de00e968c7eef1e5f333849ae64f91900bcd30a23e1b7dbaa07313e358759b9` |

### Pose candidate

`models/intel/human-pose-estimation-0001/model.yml` at the reviewed commit identifies a multi-person 2D pose model and explicitly points to the Open Model Zoo license. The selected FP16 pair is:

| Artifact | Bytes | SHA-384 |
|---|---:|---|
| `human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml` | 218215 | `cffe8df7d053b9cbf858a21faa32e30cb8a645416e9ee4ce3fbcc3106094505477d66a06b7e46d6ddb9c2de4b0cee319` |
| `human-pose-estimation-0001/FP16/human-pose-estimation-0001.bin` | 8197354 | `dabb7be42c5be008de354c6670aa8291c2d18e59f18a1c138df9b5200929a03c1e323930a1d21ecff69d3f06007de67c` |

## Admission boundary

`analytics_lab.artifacts` records these exact identities and verifies locally provisioned files by size and SHA-384 before use. It deliberately has no download function. CI does not fetch model files. Paths are bounded beneath a caller-supplied artifact root and symlink/fingerprint mismatches fail closed.

`analytics_lab.perception` is independent of OpenVINO. It defines a pose-candidate contract, deterministic temporary IoU association, and conservative pose-to-posture geometry. That keeps proprietary temporal/event logic and model-specific inference separable. Track IDs are session-local association labels, not identities or ReID.

## What is not yet approved or claimed

- OpenVINO Runtime has not yet been added as a product dependency in this change.
- The four model files have not been downloaded, executed, benchmarked, or redistributed by Analytics Lab in this change.
- No real-video detector, pose, tracking, fall, or person-down accuracy is claimed.
- The current IoU association and posture thresholds are engineering baselines, not validated operating points.
- Apache-2.0 engineering provenance review is not a legal opinion; final commercial packaging remains subject to release review.

## Next executable step

Implement an optional OpenVINO backend that accepts only the four successfully verified local artifacts, performs no network access, and emits `PoseCandidate` values. Reuse/adapt the reviewed Apache-2.0 Open Model Zoo OpenPose post-processing where practical with required notices. Then run authorized labeled positive and normal-negative clips through the complete video -> pose -> temporary track -> posture -> temporal event path and record misses, false alerts, latency and hardware settings.
