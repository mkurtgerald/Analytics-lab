# Alternative pose-path admission review — human-pose-estimation-0005

## Decision

**Admit `human-pose-estimation-0005` FP16 as a bounded engineering A/B candidate only.** It is not promoted to the retained person-down path, packaged, redistributed, or approved for commercial release by this review. The next executable step is a same-video A/B measurement against the current retained `human-pose-estimation-0001` path using the already rights-cleared Subject-4 positive/push-up-negative pair.

This review exists because three consecutive measured sessions did not improve end-to-end Subject-3/4 acceptance on the current OpenPose/posture path. Further threshold tuning around those holdouts is stopped. The alternate pose path must earn retention by measured evidence rather than by model-paper metrics.

## Pinned upstream identities and rights

Primary upstream remains Open Model Zoo at commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, repository license Apache-2.0.

`models/intel/human-pose-estimation-0005/model.yml` at that exact revision identifies the model as EfficientHRNet / Associative Embedding and points to the Open Model Zoo license. The selected FP16 pair is:

| Artifact | Bytes | SHA-384 |
|---|---:|---|
| `human-pose-estimation-0005/FP16/human-pose-estimation-0005.xml` | 1,063,570 | `37595cec7cb044266eb7cb934fcf596d5b0382b12a03f5462f046a52aba3f9c96097377773618ca957d7b6941a12334b` |
| `human-pose-estimation-0005/FP16/human-pose-estimation-0005.bin` | 19,039,904 | `ef4ab20cd0695a4b86789607acc6eb636d07fbb6786f30300ea713bb94a9110ed1ecec5db5577398ddc13663cc2ce690` |

The upstream reference decoder `demos/common/python/model_zoo/model_api/models/hpe_associative_embedding.py` is also Apache-2.0 at the same commit. It uses `scipy.optimize.linear_sum_assignment` for keypoint grouping.

For the bounded Python 3.11 evidence lane, the reviewed SciPy pin is `scipy==1.17.1`. PyPI identifies SciPy 1.17.1 as BSD-licensed and compatible with Python >=3.11. No optional SciPy extras are required by the decoder path. Commercial packaging still requires normal dependency/notices review; this engineering review is not legal advice.

## Integration characteristics

Current retained pose model `human-pose-estimation-0001`:
- OpenPose-style 18-keypoint model;
- input `1x3x256x456` BGR;
- outputs 38-channel PAFs plus 19 heatmaps;
- published COCO AP 42.8%;
- 15.435 GFLOPs;
- reviewed FP16 artifact pair totals 8,415,569 bytes.

Candidate `human-pose-estimation-0005`:
- EfficientHRNet / Associative Embedding, up to 17 COCO keypoints;
- input `1x3x288x288` BGR;
- outputs `heatmaps` `1x17x144x144` plus `embeddings` `1x17x144x144x1`;
- published COCO AP 45.6%;
- 5.9206 GFLOPs, about 61.6% lower theoretical compute than `0001`;
- reviewed FP16 artifact pair totals 20,103,474 bytes, about 2.39x the current model size.

Published AP/GFLOPs are donor metadata only and do not establish person-down accuracy or runtime performance in Analytics Lab.

## Bounded A/B acceptance plan

The first implementation experiment must change only the pose model/decoder. Keep the exact retained `person-detection-0200`, orientation recovery, safe association policy, conservative posture semantics, 3000 ms persistence, 750 ms unknown-gap budget, confidence rules, evaluator, source pair and media hashes unchanged.

Run current `0001` and candidate `0005` on the same rights-cleared Subject-4 evidence and record, separately for each path:
- positive/negative candidate-event counts, matches, misses and false alerts;
- detector coverage and track continuity to confirm the upstream input is unchanged;
- pose association coverage;
- posture/basis counts and decisive reset causes;
- qualified `down` observations and longest qualified run;
- alert delay if a match occurs;
- pose preparation/inference/decode time, total elapsed time and aggregate FPS.

Retention rule: `0005` must materially improve the positive acceptance target versus the same-run `0001` baseline while the push-up negative remains at **0 candidate events / 0 false alerts**. If positive acceptance does not improve, reject the candidate regardless of synthetic/unit-test success. If it improves but introduces a meaningful CPU regression, record the tradeoff and do not silently promote it. Do not weaken temporal/posture rules to accommodate the candidate.

## Explicit non-approvals

This review does **not** authorize training/fine-tuning, a fourth person detector, GPU jobs, home/customer footage, public-webcam use without explicit rights, model redistribution, product dependency promotion, a commercial accuracy claim, or commercial release.

The candidate remains an evidence-only alternative until exact-head Linux, Windows, Analytics quality and the bounded real-video A/B lane are green on one unchanged implementation head and the measured retention rule is satisfied.
