# RTMLib admission review — 2026-09-15

## Decision

**Code candidate only; default model path is not admitted to the commercial runtime.** No RTMLib code, model archive, checkpoint, or dataset is vendored or downloaded by Analytics Lab in this change.

## Reviewed source

- Project: `Tau-J/rtmlib`
- Reviewed commit: `03a1693e59e4f7cd84582c0fb30459b3bf18ad42` (2026-08-03)
- Repository code license: Apache-2.0
- Declared Python dependencies at that commit: NumPy, ONNX Runtime, OpenCV contrib, OpenCV, tqdm

RTMLib's `Body` solution defaults to a YOLOX detector archive trained on HumanArt and a separate RTMPose archive. The default balanced detector URL is `yolox_m_8xb8-300e_humanart-c2c7a14a.zip`.

## Commercial-rights blocker

HumanArt's official dataset instructions require authorization for **non-commercial purposes**. On 2026-08-19, OpenMMLab MMPose issue #3271 requested artifact-specific clarification for commercial use and redistribution of this exact HumanArt-trained YOLOX-M checkpoint. That clarification is unresolved in the evidence reviewed for this decision.

Because code licensing does not by itself clear model weights or their training-data restrictions, the HumanArt-trained detector archives are **hold / do-not-ship** for Analytics Lab until an authoritative right to commercial use and redistribution is documented.

## Integration blocker

RTMLib's current `PoseTracker` maintains internal IoU track identifiers but returns keypoints and scores rather than a stable external track-ID contract. Analytics Lab requires explicit session-local track IDs so temporal evidence cannot silently attach observations to the wrong person. A production adapter therefore needs its own reviewed tracker/association boundary even if RTMLib is used for pose inference.

## Allowed next use

RTMLib may be evaluated behind the Analytics Lab adapter only with:

1. the exact code revision pinned;
2. detector and pose artifacts supplied as local, hash-pinned files with separately reviewed commercial rights;
3. automatic model download disabled by construction;
4. an explicit stable track-ID adapter;
5. upstream notices preserved.

The next donor decision is to identify a commercially rights-cleared detector/pose artifact pair or to document the exact missing right. Do not fall back to the default HumanArt archives merely because they are convenient.
