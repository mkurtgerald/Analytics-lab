# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed model-artifact identities, runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video. Media is re-hashed after execution so changed media cannot inherit stale evidence.

## Current acceptance-moving work
The single implementation vehicle is `evidence/detector-shootout`, based on live `main` revision `e5678429ebc313d3a6255fcd175f5c2167616646`. It follows merged PR #24, which quantitatively established the first real-video blocker before any model change.

The exact two-clip GMDCSA-24 seed remains the only real-video evidence input in this step. The current `person-detection-retail-0013` baseline detects the positive person in 53/53 frames before the annotated fall interval, only 21/96 frames during the 1.8-5.0 s fall interval (21.9%), and 0/25 frames afterward; the hard ADL negative remains detected in 212/212 frames. The full candidate still produces zero person-down events, one missed positive episode and zero false alerts on this deliberately tiny staged-real seed. This is measured engineering evidence, not a commercial accuracy claim.

The same diagnostic also established that the simplified pose/posture baseline is independently non-functional on this seed: positive 75/75 pose candidates and negative 223/223 candidates classified `unknown`. Tracking is downstream and is not the present attack surface.

The current branch adds one bounded detector comparison command and focused regressions. It compares the unchanged baseline against exactly three previously reviewed Open Model Zoo alternatives on the same clips, threshold and CPU runtime envelope:

- `person-detection-0200` FP16 — MobileNetV2 SSD, 256x256 input, Apache-2.0 OMZ lineage; pinned XML 254,619 bytes / SHA-384 `654a515935f6dffc0440cffdeeb6a889bc25bf5d64e425ce2337070cdbcce1b1fbbcf42e4b2761265eb65ec07c1cd6f7`; BIN 3,634,654 bytes / SHA-384 `10b6f79b495ad1ef13748938c452379c6b6cd825d11426ffa68be3eeed6c48a0af04e0b913c843e6e7431aa1c13f5471`.
- `person-detection-0202` FP16 — MobileNetV2 SSD, 512x512 input, Apache-2.0 OMZ lineage; pinned XML 254,889 bytes / SHA-384 `7746ed6534bb59c9d16e7af4dbcbc768772288d48aca7e081c059eb1924ea956c67f23381a860a0bf0e187d66b7f1955`; BIN 3,634,654 bytes / SHA-384 `2a149bc8c2f02965c59a998d7ca7868e4ba537c223ae492a8a177386f98997d8c15204f1c7fe847a51c660d9d4116d10`.
- `person-detection-0203` FP16 — MobileNetV2 ATSS, 480x864 input, Apache-2.0 OMZ lineage; pinned XML 1,216,181 bytes / SHA-384 `086b17b4fc8b5454e4c892bbc26cdd120b517f89a86eab7ba6bb2a0e4ed5437cd011e65c299b89346f62538cbcc7b735`; BIN 3,902,528 bytes / SHA-384 `906e38b168001ab43117c6cc5a737e5d596d4aae441ca93dc5ea22411d6808fb63c61666bc626411c6cc60fd603ebab8`.

No candidate is promoted by documentation or donor benchmark numbers. The hosted evidence lane must measure each model's positive before/during/after detection coverage, hard-negative detection coverage, detector-only CPU FPS, preparation time and artifact bytes on the exact same seed. The evidence-only recommendation requires at least a 25 percentage-point gain in fall-interval coverage over the measured baseline while preserving at least 90% person coverage on the hard negative; among models that clear that bar, it prefers the smallest verified artifact set and then measured speed. Production integration remains a separate later change.

## Evidence/data baseline
### GMDCSA-24 — bounded staged-real validation source
- Primary repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`.
- Repository license at that revision: MIT. Associated Data in Brief paper: CC BY 4.0, DOI `10.5281/zenodo.13354453`, describing 81 fall and 79 ADL clips from four subjects in three home setups and use for fall-detection training/testing.
- Positive seed: `Subject 1/Fall/05.mp4`, Git blob `4e13ed24f6c2af7062b64b1920ef706c51efe94b`, 5,872,655 bytes, local SHA-256 `1aad4e2ee93b5498283fdc4a0478a5913dffc2ac8e2538574cd821ffb1b485f1`, annotated fall 1.8-5.0 s.
- Hard negative seed: `Subject 1/ADL/15.mp4`, Git blob `903da9245132cf70c10124edd0625c958f702cb8`, 7,233,079 bytes, local SHA-256 `2cf0d421cfd8e34280bf02afc67a4f1d1abee3cd87a23f11e69ef393fedc6fdb`.
- Media stays outside the repository and exists only in ephemeral evidence execution. Exact identity is verified before and after measured work.

### Other reviewed sources
- UE4 Fall Detection Dataset pinned at `55041766dea68eaddc1df1c06aabd0a51931a22a`, CC BY 4.0, synthetic only; useful for commercial training/evaluation with attribution but never real-video accuracy evidence.
- Figshare article `28596332` version 2, posted 2025-03-14, CC BY 4.0, real-world secondary source. The ~2.36 GB corpus remains outside the current bounded-resource path.
- UR Fall remains excluded from the commercial path because its official source states non-commercial terms. Roboflow mirrors with unclear upstream provenance remain hold-only.

## Runtime/perception baseline
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`.
- Current pinned production-candidate evidence artifacts: `person-detection-retail-0013` FP16 + `human-pose-estimation-0001` FP16, exact size/SHA-384 verified before runtime use.
- OpenVINO Runtime: `2026.3.1`, Apache-2.0 source lineage.
- Evidence decoder: `opencv-python-headless==4.12.0.88`, used only for ephemeral hosted validation execution.
- The three shootout detectors remain evidence candidates only until measured on the held seed and separately integrated. No new framework is introduced.
- RTMLib remains code-only; its default HumanArt-trained weights remain hold/do-not-ship.

## Efficiency / execution ledger
One worker, one acceptance-moving item, at most one implementation PR. At the start of this item live `main` was `e5678429ebc313d3a6255fcd175f5c2167616646`, its post-merge Analytics workflow was green, open implementation PRs were 0, queued/running runs for the new head were 0, unchanged retries were 0, CI-triggering requests were 0, and consecutive no-progress sessions were 0 because PR #24 produced new measured failure attribution.

All implementation, tests, workflow wiring and this state update are batched on `evidence/detector-shootout` before opening the one implementation PR. No paid resource, self-hosted runner, new dataset, new framework, home/customer footage, model training, threshold relaxation or commercial claim is introduced. The comparison adds only the three reviewed detector artifact pairs and keeps each downloaded item below 8 MiB.

## Reproduce
Dependency-free repository checks:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Prepare the exact bounded evidence seed in an authorized internet-connected no-spend environment:

```sh
python -m analytics_lab.validation_seed --output-dir /path/to/private-validation
```

With OpenVINO Runtime `2026.3.1` and the pinned decoder provisioned:

```sh
python -m analytics_lab.validation_cli --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.detector_diagnostics --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.detector_shootout \
  --manifest /path/to/private-validation/validation-manifest.json \
  --candidate-dir /path/to/private-detector-cache
```

The `evidence/*` pull-request lane performs the bounded acquisition and commands on a GitHub-hosted Linux runner with read-only repository permission, emits aggregate JSON only, and uploads no media or model artifact.

## Next executable step
Open the single detector-shootout PR and execute exact-head hosted evidence. If one or more candidates materially restore fall-interval recall while preserving the hard-negative boundary, select the smallest measured winner and integrate only that detector in the next bounded work item. Do not promote a candidate based on upstream AP alone.

If all three fail the measured gate, the donor limit is exhausted for this component: record the exact failure and replan rather than scanning more models or blind-retrying. Training/fine-tuning becomes eligible for consideration only because measured reviewed donors were inadequate, and still requires rights-cleared training data plus a bounded resource plan.

After the detector boundary is materially improved, immediately attack the already-measured pose/posture defect (`unknown` on every real candidate) before changing tracking or temporal persistence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across cameras/sites; materially improved detector recall; functional pose/posture discrimination on real video; false-alert and missed-event measurements at meaningful scale; alert-latency distribution once events fire; comparable latency/resource evidence; platform wheel/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Synthetic/stub tests and this two-clip staged-real seed do not establish commercial accuracy or commercial readiness.
