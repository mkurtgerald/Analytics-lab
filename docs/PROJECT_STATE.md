# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed model-artifact identities, runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video. Media is re-hashed after execution so changed media cannot inherit stale evidence.

## Current acceptance-moving work
PR #25 / `evidence/detector-shootout` is the single implementation vehicle, based on live `main` revision `e5678429ebc313d3a6255fcd175f5c2167616646`. It follows merged PR #24, which established that detector recall is the first real-video blocker: the existing `person-detection-retail-0013` detected 53/53 frames before the annotated fall interval, 21/96 during it (21.9%), and 0/25 afterward, while the hard ADL negative remained detected in 212/212 frames.

Exact-head run `35261934536` on revision `7c2cc58e631f7ba4df4238903305bd53c1cf5123` passed the guardrail preflight, 177 repository tests on Linux (one expected optional-OpenCV skip before runtime installation), the synthetic replay, the rights-bound real-video validation/diagnostic/shootout, Windows regression, and the final Analytics quality gate on the first attempt. Runtime evidence used OpenVINO `2026.3.1-22476-759c5a6ab8c-releases/2026/3`, OpenCV `4.12.0`, NumPy `2.2.6`, the exact two GMDCSA-24 clips, and exact hash-pinned OMZ artifacts. No media/model workflow artifacts were uploaded.

The detector shootout compared the unchanged baseline against exactly three reviewed Open Model Zoo candidates at confidence threshold 0.50 on the same clips and CPU envelope:

- **Baseline `person-detection-retail-0013`**: fall-interval coverage 21/96 = **21.9%**; post-fall 0/25; hard-negative person coverage **100%**; detector-only **136.3 FPS**; detector artifacts 2,016,967 bytes.
- **`person-detection-0200`** (256x256 MobileNetV2 SSD): fall-interval coverage 29/96 = **30.2%**; post-fall 2/25 = 8%; hard-negative coverage **100%**; detector-only **295.1 FPS**; detector artifacts 3,889,273 bytes. This is the best measured alternative, but improves fall-interval coverage by only **8.3 percentage points**, far below the predeclared +25-point promotion bar.
- **`person-detection-0202`** (512x512 MobileNetV2 SSD): fall-interval coverage 14/96 = **14.6%**; post-fall 0/25; hard-negative coverage **100%**; detector-only **122.0 FPS**; artifacts 3,889,543 bytes. It regressed prone/fall recall versus baseline.
- **`person-detection-0203`** (480x864 MobileNetV2 ATSS): **0 detections** across both clips at the common 0.50 threshold under this bounded adapter/runtime path; detector-only **68.7 FPS**; artifacts 5,118,709 bytes. It is not viable in this comparison configuration.

**No detector qualified for promotion.** The evidence command therefore returned `recommended_candidate: null`. The policy maximum of three donor candidates for this component is now exhausted; do not add a fourth detector or blind-retry the same comparison.

The full person-down candidate still produces zero events, one missed positive episode and zero false alerts on this deliberately tiny staged-real seed. This is measured engineering evidence, not a commercial accuracy claim. The independently measured pose/posture defect also remains: every real pose candidate classified `unknown` (positive 75/75 and hard negative 221/221 on this exact run). Tracking remains downstream of both blockers.

## Evidence/data baseline
### GMDCSA-24 — bounded staged-real validation source
- Primary repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`.
- Repository license at that revision: MIT. Associated Data in Brief paper: CC BY 4.0, DOI `10.5281/zenodo.13354453`, describing 81 fall and 79 ADL clips from four subjects in three home setups and use for fall-detection training/testing.
- Positive seed: `Subject 1/Fall/05.mp4`, Git blob `4e13ed24f6c2af7062b64b1920ef706c51efe94b`, 5,872,655 bytes, local SHA-256 `1aad4e2ee93b5498283fdc4a0478a5913dffc2ac8e2538574cd821ffb1b485f1`, annotated fall 1.8-5.0 s.
- Hard negative: `Subject 1/ADL/15.mp4`, Git blob `903da9245132cf70c10124edd0625c958f702cb8`, 7,233,079 bytes, local SHA-256 `2cf0d421cfd8e34280bf02afc67a4f1d1abee3cd87a23f11e69ef393fedc6fdb`.
- Exact media identity is verified before and after measured execution; media remains outside public GitHub.

### Other reviewed sources
- UE4 Fall Detection Dataset pinned at `55041766dea68eaddc1df1c06aabd0a51931a22a`, CC BY 4.0, synthetic only; eligible for commercial training/evaluation with attribution but never real-video accuracy evidence.
- Figshare article `28596332` version 2, posted 2025-03-14, CC BY 4.0, real-world secondary source; the ~2.36 GB corpus remains outside the current bounded-resource path.
- UR Fall remains excluded because its official source states non-commercial terms. Roboflow mirrors with unclear upstream provenance remain hold-only.

## Runtime / detector provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Current baseline artifacts: `person-detection-retail-0013` FP16 + `human-pose-estimation-0001` FP16, exact size/SHA-384 verified.
- `person-detection-0200` FP16: XML 254,619 bytes / SHA-384 `654a515935f6dffc0440cffdeeb6a889bc25bf5d64e425ce2337070cdbcce1b1fbbcf42e4b2761265eb65ec07c1cd6f7`; BIN 3,634,654 / `10b6f79b495ad1ef13748938c452379c6b6cd825d11426ffa68be3eeed6c48a0af04e0b913c843e6e7431aa1c13f5471`.
- `person-detection-0202` FP16: XML 254,889 / `7746ed6534bb59c9d16e7af4dbcbc768772288d48aca7e081c059eb1924ea956c67f23381a860a0bf0e187d66b7f1955`; BIN 3,634,654 / `2a149bc8c2f02965c59a998d7ca7868e4ba537c223ae492a8a177386f98997d8c15204f1c7fe847a51c660d9d4116d10`.
- `person-detection-0203` FP16: XML 1,216,181 / `086b17b4fc8b5454e4c892bbc26cdd120b517f89a86eab7ba6bb2a0e4ed5437cd011e65c299b89346f62538cbcc7b735`; BIN 3,902,528 / `906e38b168001ab43117c6cc5a737e5d596d4aae441ca93dc5ea22411d6808fb63c61666bc626411c6cc60fd603ebab8`.
- OpenVINO Runtime baseline: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- None of the three shootout detectors is integrated into the production-candidate backend.

## Efficiency / execution ledger
One worker, one acceptance-moving item, one implementation PR. This item began from green `main` with 0 open implementation PRs, 0 active runs, 0 unchanged retries, 0 CI-triggering requests and 0 no-progress sessions. PR creation was CI request 1/2 and produced new measured evidence on the first attempt; it was not a retry. This state update is the second and final CI-triggering mutation permitted for this session. No extra branch/worker/job/runner, paid resource, new dataset/framework, home/customer footage, model training, or licensing shortcut was introduced.

## Reproduce
Repository checks:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Rights-bound evidence path in an authorized internet-connected no-spend environment:

```sh
python -m analytics_lab.validation_seed --output-dir /path/to/private-validation
python -m analytics_lab.validation_cli --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.detector_diagnostics --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.detector_shootout \
  --manifest /path/to/private-validation/validation-manifest.json \
  --candidate-dir /path/to/private-detector-cache
```

## Next executable step
Finish exact-head verification for PR #25 after this evidence-state update and merge only if Linux real-video evidence, Windows regression and the final Analytics quality gate remain green on the unchanged tested base.

Then **change approach rather than adding detectors**. `person-detection-0200` is the only alternative that improved fall-interval recall and it is substantially faster, but the gain at threshold 0.50 is insufficient. The next detector-recall experiment should therefore be a bounded confidence/threshold-sensitivity diagnostic on the already-measured best path (no fourth donor): quantify whether missed prone frames carry useful sub-threshold person confidence and whether a lower threshold causes duplicate/noisy detections on the same positive and hard-negative clips. If that cannot materially recover recall under a defensible detection-noise bound, the measured donor evidence is sufficient to begin a separate rights-cleared fine-tuning/training plan within the no-spend resource envelope.

Once detector recall is materially improved, attack the already-proven pose/posture failure (`unknown` on every real candidate) before changing tracking or temporal persistence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across cameras/sites; materially improved detector recall; functional pose/posture discrimination on real video; false-alert and missed-event measurements at meaningful scale; alert-latency distribution once events fire; latency/resource envelope; platform/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Synthetic/stub tests and this two-clip staged-real seed do not establish commercial accuracy or commercial readiness.
