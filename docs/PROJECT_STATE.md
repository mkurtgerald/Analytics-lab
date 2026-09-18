# Project state — 2026-09-18

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Retained person-down candidate path
`authorized local video -> reviewed detector -> spatial continuity/orientation recovery -> reviewed pose -> bounded pose association -> conservative posture -> temporal persistence -> evidence-linked candidate -> labeled evaluation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Media remains outside public GitHub and is re-hashed before measured execution.

## Measured real-video frontier
- **Subject 1:** first matched engineering seed in PR #33: 1 positive, 1 candidate, 1 match, 0 misses, 0 hard-negative false alerts, about 0.0036 decoded camera-hours, 4024 ms alert delay. Engineering evidence only.
- **Subject 2:** same-detector orientation recovery raised labeled-fall detector coverage to 126/138 = 91.30%. A later bounded near-diagonal ambiguity correction improved the longest qualified down run to 2544 ms while preserving 3000 ms persistence and the 750 ms unknown-gap budget. The remaining interruption is genuinely upright, so Subject 2 is not tuned further.
- **Subject 3:** positive remained a miss while its sitting-to-sleeping-on-floor negative stayed at 0 false alerts. A selected-person crop recovered more poses but regressed longest qualified persistence 1184 ms -> 512 ms and increased CPU cost; PR #38 was closed unmerged.
- **Subject 4:** exact retained path produced 0 candidates / 1 miss on the positive and 0 candidates / 0 false alerts on the dynamic prone/push-up negative across 11,632 ms / 0.0032311111 camera-hours. The labeled positive window had 129/129 detector coverage and 129/129 safe pose association, 55 qualified down observations, and a 1312 ms longest qualified run. This localized the dominant wall to pose/posture fragmentation rather than detector recall or association.

Exact Subject-4 media identities:
- positive `Subject 4/Fall/03.mp4` SHA-256 `4f178dea77bf9abb9036bcd7fedfbe345358fc2d63e1704dc49b7ec1a621e2f7`;
- negative `Subject 4/ADL/07.mp4` SHA-256 `6e410abbc6d4ef7bf52ac835b952a74a94311d0977fdfddb21885477602b7b69`.

After three consecutive sessions without tested acceptance improvement on the existing OpenPose/posture path, further holdout-specific tuning stopped.

## Alternative pose candidate
PR #40 merged to `main` at `20f52a58c0df4134752be4b78370f4385f25fec3`, pinning Open Model Zoo `human-pose-estimation-0005` FP16 at OMZ commit `6697dead54ed1cdd664b0313189c2cb52ee6335e` under Apache-2.0, its Apache-2.0 Associative Embedding reference decoder, and bounded evidence dependency `scipy==1.17.1`.

Pinned `0005` artifacts:
- XML: 1,063,570 bytes, SHA-384 `37595cec7cb044266eb7cb934fcf596d5b0382b12a03f5462f046a52aba3f9c96097377773618ca957d7b6941a12334b`;
- BIN: 19,039,904 bytes, SHA-384 `ef4ab20cd0695a4b86789607acc6eb636d07fbb6786f30300ea713bb94a9110ed1ecec5db5577398ddc13663cc2ce690`.

Donor metadata only: `0005` is EfficientHRNet/Associative Embedding, 17 COCO keypoints, published 45.6% COCO AP / 5.9206 GFLOPs. The retained `human-pose-estimation-0001` is published 42.8% / 15.435 GFLOPs. These figures do not establish Analytics Lab accuracy or runtime performance.

## Active acceptance item — PR #41 same-video 0001 vs 0005 A/B
PR #41 is the sole implementation/evidence PR. Both arms use the exact Subject-4 pair, `person-detection-0200`, orientation recovery, continuity selection, safe association policy, posture semantics, 3000 ms persistence, 750 ms unknown-gap budget, evaluator and CPU device. Only pose model/decoder changes.

### Earlier deterministic blockers
Head `60d23fea98766a76375f30e3b2b3b226391c7075` passed guardrails, the full synthetic suite and retained Subject-4 evidence, then rejected candidate acquisition because the generic validation downloader has a deliberate 16 MiB item ceiling while the separately reviewed `0005` FP16 BIN is 19,039,904 bytes. Head `f07c5452a762989a9ec49c7a13970301c79b60fb` corrected only that mismatch with an exact-artifact-only 20 MiB bound; it again passed the ordinary regressions and retained evidence but still failed before A/B metrics were emitted. No unchanged retry was used for either deterministic failure.

### Current-session stage localization
Head `bb4421764a5af1158135272d4ab9fe701c93f54d` added only fixed, non-sensitive failure-stage attribution. Its exact-head Linux run passed policy classification, all 43 guardrail regressions, all 226 synthetic tests (1 optional OpenCV skip), synthetic replay, runtime provisioning and the retained Subject-4 baseline. The A/B then failed specifically at **`candidate_decode`**, proving candidate provisioning, OpenVINO model load/runtime contract and candidate inference had already succeeded. Windows was correctly skipped and the quality gate failed.

### Demonstrated decoder defect and bounded correction
Read-only comparison against the exact pinned OMZ Associative Embedding implementation exposed the deterministic decoder defect. The local adaptation computed `diff = tags[:, None] - pose_tags[None, :]`; with the reviewed one-dimensional embedding this tensor is shaped `(new_points, grouped_poses, tag_size)`. The local code incorrectly attempted `num_added, num_grouped = diff.shape`, which raises because `diff` has three axes. The pinned upstream implementation explicitly uses `diff.shape[0]` and `diff.shape[1]`.

The correction changes only those two count assignments to the pinned upstream semantics. A focused regression constructs two tag-compatible joints and exercises `_match_by_tag` with the assignment solver stubbed, so the ordinary test suite verifies that the embedding axis is not mistaken for a grouping axis without adding SciPy to the normal test environment. No model, threshold, detector, association policy, posture rule or temporal rule changes.

## A/B retention rule
Keep `0005` only if it materially improves the positive acceptance target against the same-run `0001` baseline **and** the push-up negative remains 0 candidate events / 0 false alerts. Record positive/negative candidates, matches, misses, false alerts, detector/association/posture evidence, longest qualified run, alert delay if any, pose inference/decode time, total elapsed time and FPS. Green unit tests or donor-paper metrics alone are insufficient. Do not weaken posture or temporal rules to make `0005` pass.

## Provenance
### GMDCSA-24
- repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`;
- pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`, aligned with v2.1;
- repository license at that revision: MIT;
- Zenodo v2.1 DOI `10.5281/zenodo.13354453`; paper DOI `10.1016/j.dib.2024.110892`.

### Runtime/models
- OMZ commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0;
- retained pose `human-pose-estimation-0001` FP16, exact reviewed artifacts;
- candidate pose `human-pose-estimation-0005` FP16, exact identities above;
- evidence detector `person-detection-0200` FP16, exact reviewed artifacts;
- OpenVINO `2026.3.1`; OpenCV `4.12.0.88`; candidate decoder dependency `scipy==1.17.1`.

## Efficiency ledger
One worker, one acceptance-moving item, one PR. No training, fourth detector, paid resource, self-hosted runner, home/customer media, duplicate agent, or second implementation lane. In the current work session the stage-localization mutation used CI-trigger 1/2. This decoder correction plus regression is CI-trigger 2/2. No unchanged retry has been used. Further deterministic failure in this session must be diagnosed read-only/local without another CI-triggering mutation.

The connector-only worker cannot execute a repository-local clone/preflight, so no local machine preflight is claimed. Exact-head hosted CI/evidence is the execution authority.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Authorized no-spend evidence environment:
```sh
python -m pip install --no-cache-dir 'openvino==2026.3.1' 'opencv-python-headless==4.12.0.88' 'scipy==1.17.1'
python -m analytics_lab.validation_seed --output-dir /path/to/private-validation
python -m analytics_lab.person_down_pose_ab --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
```

## Next executable decision
Let the corrected exact head run once. If `0005` materially improves the positive while the hard negative remains zero-candidate/zero-false-alert and CPU cost remains acceptable, retain the evidence adapter and merge only if Linux, Windows and Analytics quality are green on the unchanged tested head. If `0005` fails the measured retention rule, close PR #41 unmerged. If another deterministic defect prevents a valid A/B measurement, preserve this session's 2/2 CI budget and diagnose it without another triggering mutation. Commercial promotion still requires substantially broader held-out positive/negative evidence across cameras/sites, meaningful false-alert/miss measurements, latency/resource envelope, privacy/security/provenance review, versioned integration and explicit owner release approval.
