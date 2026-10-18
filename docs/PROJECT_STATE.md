## October 18 bounded public live traffic test

The owner asked to watch public live streams for testing. Select one public
Caltrans-created CWWP2 camera: district 1 index 88, `US-101 : Eureka / 14th -
Looking North (C039)`, with the catalog-exposed 320x240 HLS playlist
`https://wzmedia.dot.ca.gov/D1/eureka_14th_320x240.stream/playlist.m3u8`.
Primary CWWP2/CCTV/ownership terms permit public/commercial use of Caltrans
material subject to contrary/third-party notices and fair use. This public
delivery does not require inventing a direct TMC subscription requirement.
The bounded traffic-only test has no identity inference, plate text, camera
control, training, archive, redistribution or continuing capture service.

The metadata precheck returned HTTP 200 for a single-variant 320x240 HLS v3
master and three consecutive MPEG-TS entries totaling about 30 seconds; it did
not download video or run inference. A camera's record modification timestamp
is not its capture clock. Other reviewed options did not provide an equivalent
immediate test: TfL requires registration and publishes periodic images;
USGS provides public-domain scientific camera images, with separate video
hosting/access considerations. No substitute or private feed is selected.

The narrow hosted Linux lane reuses the existing hash-locked runtime and exact
OMZ 0106 XML/BIN identities (1,612,224 model bytes). Share its existing verified
runtime to expose only vehicle class 1 at 0.50 and one inference per sampled
frame; OCR/plate and face/privacy behavior is unchanged. Reuse the admitted
ByteTrack adapter with fresh session state per segment. The vehicle head is
trained/validated for front-facing barrier scenes: its published AP is not an
accuracy estimate for this traffic camera.

Require the exact in-service catalog record and source directory, HTTPS with
no redirects or credential/query substitution, one master/single-child path,
the latest two consecutive unencrypted TS segments from a verified three-entry
playlist, at most 35 seconds declared media
and 4 MiB media bytes. Reject nested variants, keys/maps/ranges, end lists and
discontinuities; verify TS framing/hash before local decoding. One additional
child read after at most 15 seconds must demonstrate advertised sequence
advancement. That check supports a live-provider observation, not a known
camera-capture UTC timestamp.

Decode at most 2,400 frames, fixed 320x240 uint8 with FPS at most 60, sampling
at most 100 frames around 3 Hz. Record actual decoded spans and media-relative
times separately from EXTINF duration and receipt UTC. Do not fabricate
unobserved duration, camera UTC, labels, cross-segment identity or missing
observations. Reports bind source/rights/model/config/code/runtime/CPU identity,
segment/frame hashes and every count/failure. With no independent labels,
precision, recall and false-alert rates stay null; no accuracy gate is passed.

All acquired models/media and intermediates remain under a new hosted
RUNNER_TEMP child and are deleted on success/failure; no artifacts are uploaded.
The five-minute job stays unchanged, with a cooperative 180-second evidence
budget, remaining-time-capped HTTP inactivity timeouts, phase logs and cleanup
reserve. HTTP opening/header parsing is inactivity-bounded rather than a strict
wall-clock interruption. Metadata and media byte/request ceilings are explicit;
there is no mirror, repeated retry, source sweep or background watcher.

This distinct live compatibility work starts from PR #149's verified merge
`8f5ccf4df0487248266bc04b1c14f01b05cb4e49`; exact-head #369
(`39160332882`) and automatic main #370 (`39160463159`) passed Linux/Windows
and the gate, running 829 tests with existing optional skips. That related
association repair family used two total requests and stays closed. Preserve
parked PR #148's two requests, PR #146's eight and FPN's four. No old OCR/source
or donor experiment is retried. This new live work has zero CI requests at
initial preparation; the first hosted result and repair are recorded below.

Local validation ran 877 tests with three existing optional-native skips;
guardrails, compilation, deterministic replay and diff checks passed. Generated
controls cover bounded acquisition/cleanup and actual tracker confirmation when
decoded indices are sparse. Tracking uses dense sample ordinals while retaining
original decoded indices. Publisher metadata demonstrated changing bandwidth,
attribute ordering and media playlist versions 3/4; parse those reviewed forms
without weakening URI, codec, dimensions or forbidden-feature checks. An initial
segment HEAD returned 404; a fresh metadata request with cache revalidation then
returned HTTP 200 and fixed Content-Length 188,000 for the current segment. No
segment body, model or native runtime was acquired locally. The reader requests
cache revalidation and will preserve any live-source failure without fallback.

PR #150's first head `3c025e69dfef9ad4aae34cae5c9e0fd395de3bbe`, run #371
(`39171665889`), passed all 877 Linux tests (four existing optional skips) and
runtime installation, then failed in capture with an advertised TS HTTP 404.
Cleanup ran; no model acquisition, decode, compilation or inference occurred.
Windows skipped and the quality gate failed. Source diagnosis fetched only fresh
metadata and HEADs: advertised sequence 18318 returned 404, while 18319 and 18320
returned 200 with Content-Length 210,560 and 150,212 in the same session. The
oldest advertised segment had already expired; this is not evidence of zero
detections or acceptable false positives.

The repaired reader selects the latest two entries directly, never requests the
oldest, performs six total source GETs and still requires playlist advancement
before decoding. This reduces acquisition to roughly 20 seconds and uses two
fresh segment-local tracking sessions. All upper byte/time/frame limits,
source/model/runtime identities, cleanup and null accuracy metrics stay fixed.
CI requests are 1/2 used, unchanged retries 0/1; the changed-head request is the
last allowed attempt for this work item. Its actual outcome remains pending in
this preparation snapshot and will be persisted in the PR body.
Repair validation ran 878 tests with three existing optional-native skips;
guardrails, compilation, deterministic replay and diff checks passed. The new
expiry fixture proves the oldest URL is never requested; a selected segment's
404 still fails immediately. No extra source or automatic retry is added.

## October 18 core perception correctness follow-up

The owner's minimum 90% requirement is an accuracy goal across weapons,
LPR/OCR and detection/tracking; engineering completion is separate. This change
repairs four reproduced normal-input failures using admitted components. No
model, media, training, threshold sweep, new framework or deployment is added.
Generated controls demonstrate correctness changes, not field accuracy.

The portable ByteTrack slice still used unweighted association after PR #147
repaired the separate perception tracker. At default settings, two 0.90 person
detections overlap at IoU 3/13. When A exits, stationary B is assigned A's handle
and B's own track is marked lost. Actual OMZ-row parsing into TrackingSession
reproduces one ID switch over three correctly localized observations. Reuse
the existing stdlib weighted assignment to maximize valid cardinality first,
then exact total stage similarity. Retain all donor stages, category gates,
confidence cutoffs, recovery and capacity behavior; this is an adapted portable
slice, not full upstream Kalman/LAP accuracy parity.

Tracking evaluation also needs weighted matching: on two overlapping objects
with stable supplied IDs and a valid jittered box, its former augmenting-path
choice reports two spurious ID switches despite a strictly better total-IoU
matching with zero switches. Canonical weighted frame matching preserves
maximum cardinality, misses and false-track accounting while correcting that
measurement. This remains the repository's documented diagnostic protocol,
not MOTChallenge, HOTA or IDF1; the registered 90% IDF1 target is unqualified.

The weapon parser silently returned the first 64 accepted proposals. A valid
65-proposal SSD output (64 knives and a rifle, all confidence 0.90) dropped the
rifle while reporting successful inference. Preserve the 64-output capacity,
but reject excess admitted detections instead of returning an incomplete result.
Exactly 64 still succeeds; below-cutoff, nonweapon and zero-area proposals do
not consume accepted capacity. This prevents a silent omission, not the model's
historical zero-output rifle failure or an independently measured recall gain.

The plate proposal path also round-tripped integer pixel coordinates through
normalized floats. A contour (245,50,160,40) in a 960x200 generated image scales
back to left 244.99999999999997; floor includes a foreign column and emits a
161x40 crop. Recover a scaled integer only when dividing it by the source
extent reproduces the original normalized float exactly. This avoids a broad
epsilon: distinct fractional neighbor floats retain floor/ceil behavior, as
does the existing PPM bound.
The composed proposal/crop/recognizer regression uses generated pixels and
fake native TSV output; it does not establish real OCR improvement.

PR #147 is merged at `83197b1a86555c0062a23d55aac7013d8581745b`;
exact-head #365 (`39156878925`) and automatic main #366 (`39157027152`) passed
all required jobs. Conservatively carry its one CI request into this related
association follow-up: one additional request remains in the two-request family
budget, with zero unchanged retries. Before publication this follow-up has used
zero new requests. Preserve parked PR #146's eight requests, FPN's four and
PR #148's two. No stalled hypothesis is reopened or reset.

PR #148 remains closed unmerged at `1e1b3812158fb47ce2f9956be933b24bf90d4393`.
Run #367 failed on generated Windows temporary-path aliases; the correction
passed all 827 tests on Linux/Windows in #368, then the third canonical Wikimedia
plate-image GET returned HTTP 429. The first two sources were verified; cleanup
ran without a reported cleanup error. Installer/model acquisition and native OCR
never started: no PSM comparison result or gain exists. The prepared post-run
review commit is `850ecafbe695319df96be7892094896676778d7b`; do not retry under
a renamed lane or silently substitute images. This follow-up contains no PSM
change, source acquisition or workflow change.

Representative independently labeled positives and hard negatives remain the
main accuracy gap. Face stays at its existing engineering checkpoint. The
owner-supplied JAIJANYANI surveillance repository was reviewed at
`4f4c988891347290156a789061af00e92d858cb8`: it supplies whole-frame label-change
timestamps, not the missing core detectors, OCR, tracks or accuracy data; no
code or weights were adopted.

Nineteen new generated tests cover the four repairs. Before implementation,
ByteTrack/evaluator regressions failed in twelve subcases, weapon overflow
failed three tests and LPR precision failed six checks. After implementation,
the full local suite passed 829 tests with three existing optional OpenCV skips;
guardrails, compilation, deterministic replay and diff checks passed. Targeted
controls retain actual ID switches, maximum match cardinality, lower-confidence
and unconfirmed association stages, failure atomicity, exact-cap weapons,
filtered proposals, fractional crop edges and separated plate observations.
One bounded generated dense-512 probe retained all 512 ByteTrack outputs and
all 512 evaluator matches; it is a configured-capacity check, not a product
latency or throughput claim. Independent crop review found no blocker.
Exact-head hosted checks are pending when this implementation snapshot is authored.

## October 18 core priority and tracking accuracy repair

The owner set weapons, LPR/OCR and detection/tracking as the core deliverables,
with a minimum 90% accuracy goal and a request to materially advance completion.
Face detection/privacy remains at its existing engineering checkpoint. These
are delivery targets, not achieved accuracy scores. Retain donor-first reuse:
the admitted OIDv4 weapons baseline, Tesseract/OMZ LPR boundaries and portable
MIT ByteTrack association slice remain available. No new model is adopted based
only on upstream benchmark scores, source licenses or successful compilation.

This focused change repairs a newly demonstrated default-setting association
failure in the reusable OMZ/pose/IoU/temporal pipeline. Two stationary person
boxes overlap at IoU 3/13, above the unchanged 0.15 threshold. Person A is down
at 0..2,250 ms then exits; B is other at 0..2,250 ms and down at 2,500..3,000 ms.
Neither has three seconds down, but unweighted matching gives A's handle to B
and emits one candidate spanning 0..3,000 ms. The inverse continuous-positive
control loses B's valid episode when A exits. Both are reproduced using generated
SSD rows and pose heatmaps through the actual core composition, not native media.

A deterministic stdlib rectangular Hungarian assignment now preserves maximum
valid association cardinality, then maximizes exact total IoU. Exact binary-float
integer scaling prevents the cardinality bonus from erasing small differences.
Canonical track/box order, configured capacity, expiration, retryable failure
state, confidence and temporal cutoffs are unchanged. The generated false
candidate falls from one to zero; the continuous positive is restored from zero
to one. Both two-person fixtures retain exactly 23 actual observations. Missing
observations are not fabricated. Geometric handles remain session-local and do
not establish person identity or solve every real-world tracking ambiguity.

Fifteen new tests cover both composed cases, the previous maximum-cardinality
trap, weight/greedy counterexamples, 21,279 exhaustive tiny weighted matrices,
subnormal precision, permutations, capacity/atomicity and dense default256
assignment. An independent read-only review passed 1,792 additional exact-oracle
checks and twelve tracker permutations without finding a blocker. No model,
media, new dependency, training or deployment is introduced. Full local verification passed 810 tests with three existing optional OpenCV
skips; guardrails, compilation, deterministic replay and diff checks passed.
Exact-head #365 and automatic main #366 subsequently passed, as recorded above.
Synthetic evidence does not establish field accuracy.

This work starts from main `dac3c6372c1ff77dad90e0f920062eabe899446a` and contains
only the new weighted association, perception integration, its tests and this
substantive state update. It received one hosted CI request. It is distinct
from PR #146's unmerged participant-source/readiness/parser work and its separate
zero-IoU configuration correction. Preserve that PR's eight runs (#357..#364;
six passing heads and two deterministic failures), prepared review commit
`0bf0321f4fe3648a8315571c16e22420ae370185` and local patch. That exhausted
proposal was parked before PR #147 opened; no failed source
parser test or old donor experiment is retried or renamed by this tracking fix.

GMDCSA24/Figshare participant-media commercial permissions remain operationally
on hold pending independent review. Their proposed code enforcement in PR #146
is unmerged, and is not claimed as shipped by this change. Acquire no held media.
Existing parked OCR glyph, FaceSSD and FPN hypothesis counters remain preserved.

Next core evidence boundaries: independently authored boxes/track IDs for the
already reviewed CC0 tracking window; a frozen positive/hard-negative weapons
comparison using exact model/source/annotation identities; and whole-plate exact
OCR correctness plus false accepted reads and abstentions on a reserved labeled
set. The historical three OCR reads were wrong, and no stronger weapons donor
has yet demonstrated a measured gain. These gaps must close before reporting
high accuracy or raising completion estimates to 90%.

Preregister these initial high-accuracy targets before selecting fresh held-out
evidence: weapon precision >=95% and recall >=90% for each declared supported
class; LPR exact whole-plate accuracy >=95% on the frozen readable/in-scope
positive set (abstentions count as misses), reporting false accepted reads separately; object
detection precision >=95%, recall >=90%, and tracking IDF1 >=90% on the declared
tracking set. Alarm-producing paths still require zero observed false alerts
on declared held-out negative exposure, with exposure, misses and uncertainty
reported. These are future qualification targets, not achieved scores or a
universal no-error guarantee. Exposed historical examples are regressions and
cannot become fresh held-out evidence by renaming them.

# Project state — 2026-09-24

## October 14 accuracy repair and guarded donor recovery

PR #143 merged at `709dc5d5376d8fd0d27138dc9ff62b5adb15528d` after
exact-head run #351 and main #352 passed. Its paired native comparison kept
all thresholds fixed: retail-0013 versus optional 0200 produced 4 versus 50
observations on the 200-frame fall clip, and 9 versus 12 on the 149-frame ADL
clip. Both arms produced zero candidate events, zero activity matches and one
missed falling interval. More detector admissions did not establish recovery
of the falling event. Actual decoded spans were 6,672 and 4,960 ms; the ADL
clip's 960 ms unannotated tail leaves annotated-negative counts/rates and
clinical false-positive rates unqualified. PR #144's separate PPM allocation
hardening merged at `b4d1c97f3abb8c4ba6abf592fba9be9945755aa4`; exact-head
#353 and automatic main #354 passed. It was memory safety work, not accuracy.

A generated composition of the real reusable detector/pose/tracker/temporal
pipeline demonstrated an accuracy defect in both detector profiles. Thirteen
250 ms frames with confident down poses produce no candidate when one detector
score is 0.60: the admitted low-confidence observation breaks the 0.70 temporal
run. Lowering only that score to 0.49 filtered the observation, retained its IoU
handle across the absent frame, and incorrectly produced a 0.90 candidate at
3,000 ms despite `max_unknown_gap_ms=0`. Known absence now reaches the existing
unknown-evidence policy before the frame's actual observations. Missing markers
never create tracks, extend observed lifetime or count as observations. A
bounded frame buffer preserves per-person isolation; uninterrupted positives
and explicitly configured short unknown bridging remain supported. Empty
frames advance expiration, avoiding expired-handle ghosts at track capacity.
This is a demonstrated false-acceptance repair, not a real-video accuracy rate.

The existing reviewed 0200 detector now has an opt-in guarded orientation path.
Primary detections take precedence. Only a primary miss immediately after one
admitted person may try +/-90-degree views of the same detector, with unchanged
0.50 detection confidence and existing IoU/temporal gap limits. At most three
detector passes yield at most one spatially linked fallback box; original-frame
cropped pose and the corrected integer coordinate mapping stay common. Fresh
per-clip recovery state is isolated from a reused warm backend, and resets on
missing/multiple people, frame/time/dimension gaps or exceptions. The default
remains off; retail plus recovery rejects before constructing a runtime. No
threshold was loosened and no additional model/framework was introduced.

Dataset provenance is corrected independently of the software notice. Primary
author Zenodo record `13354453`, DOI `10.5281/zenodo.13354453`, identifies the
v2.1 archive as CC-BY-4.0 (2024-08-21; Ekram Alam), with the GitHub v2.1 tag
resolving exactly to the pinned `5abac7693229900cf80f722e878fbb119211fc1c`.
The data authorization reference and credit now bind that license and version;
the repository's separate MIT software copyright/notice is retained. The
selected clips are unchanged; subset/manifest activity mapping is disclosed.
These engineering provenance facts do not authorize a commercial release.

The narrow `evidence/person-down-accuracy-*` lane prepares only that same
retrospective S4 pair (7,996,782 bytes) and the reviewed four 0200/pose artifacts
(12,304,842 bytes), once under hosted Linux RUNNER_TEMP. Publisher annotation
identity is checked before acquisition. Exactly one orientation-OFF suite and
one ON suite share the same host, decoder, 1,000-frame ceiling, confidence,
posture, IoU, temporal and scoring settings; both include the missing-evidence
repair. Reports retain all fall misses and normal candidates, bind immutable
source/model/config/runtime/hardware identities, require equal decoded spans
and frame counts, and keep clinical/gate/promotion claims false. Both other
person-down lanes and the generic seven-pipeline lane are excluded. Assets are
deleted on success/failure; no media/model/frame artifact is retained/uploaded.

Local verification passed 795 full tests with three existing optional OpenCV
skips, plus guardrails, deterministic replay, compilation and diff checks. The
15 new missing-evidence tests, 13 guarded-recovery tests and 11 generated-byte
OFF/ON wrapper tests pass without skips. A separate read-only review found no
blocker in the bounded acquisition, unchanged settings, cleanup or metric
preservation. No model/media/runtime was acquired locally.

Native OFF/ON results and exact-head Linux/Windows checks are pending when this
implementation snapshot is authored. Generated checks cannot establish a
real-video gain. This new work has no CI requests yet; completed/parked PR #140,
#142, #143 and #144 counters are preserved and are not reset by this work.

## October 14 LPR crop preallocation budget

The PPM encoder previously iterated/allocated plate crop bytes before its
downstream OCR adapter enforced a 16 MiB limit. A 4096x4096 generated image
reproduces pixel access before rejection. The encoder now preflights the exact
PPM header and RGB pixel budget before reading pixels, leaving small crops
byte-identical. Four isolated synthetic regressions pass (including the exact
boundary); exact-head run #353 and main #354 subsequently passed. No model, media, cutoff,
dependency, rights, or release policy changes. This is memory-safety hardening,
not a plate-reading accuracy improvement.

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

### Permanent priority
1. Detection + tracking.
2. LPR/OCR.
3. Face — including required selectable face blurring with policy-driven default blur, permission-gated unblur/reblur, server-side enforcement, auditability and preservation of original evidence.
4. Weapons.
5. Forensic search — deterministic metadata/time/camera/LPR/track search first, then rights-cleared appearance/vector ranking.

Person-down and slip/fall remain required deliverables and are secondary only in sequencing. On September 24, 2026 the owner-directed Face hardening finish line was reached and the single active implementation lane moved back to LPR/OCR. PR #84 remains closed/unmerged as preserved historical work; the current successor is `evidence/lpr-ocr-exact-glyph-2`. Face feature expansion is stopped except for regression, packaging/provenance, and later explicitly authorized evidence needed to preserve the accepted boundary.

## October 13 perception integration phase

PR #142 merged as `4e37fdfd1c7f0f9fa24da834d6e85d50a4204aaa`; corrected
exact-head run #349 and automatic main run #350 passed every required job.
The owner requested the next phase after its measured positive miss.

The four observations over 200 fall-clip frames are detector admissions:
each accepted SSD person box yields a pose candidate and an observation even
when posture is unknown. Generated end-to-end checks retain 200 observations
for 200 high-confidence detections with zero or 0.69 pose peaks; changing the
detector score to 0.49 removes them at the fixed 0.50 cutoff. These checks
localize the sparse input without blaming or loosening temporal thresholds.

A separate coordinate defect is repaired before donor integration: native pose
inference rounded/clamped a fractional detector box to integer pixel bounds,
but keypoints mapped back through the fractional extent. A shared crop-bounds
helper now supplies the exact inferred crop transform; the original detector
box and confidence remain unchanged for tracking. Three generated fractional/
boundary subcases failed before this repair and pass after it; the integral
case remains unchanged. The repair does not itself recover missing detections.

The already reviewed `person-detection-0200` donor is now an explicit option in
the reusable OMZ backend. The exact XML/BIN pair is shared with the existing
catalogue; pinned upstream README/model.yml confirm 1x3x256x256 BGR input and
person class 0. Model selection binds artifact identity, input shape and class
semantics together. Unknown profiles reject before file/runtime work. The
retail-0013 profile remains the default, and every detector/posture/tracker/
temporal cutoff stays fixed. The simplified cropped-heatmap pose path is common
to both arms; this does not substitute the retained 0.10/orientation/PAF path.
Validation records the actually selected detector artifacts instead of always
claiming the retail pair.

The isolated `evidence/person-down-detector-compare-*` hosted Linux lane prepares
the same reviewed Subject-4 sources and publisher annotations once, verifies
six unique existing model files (14,321,809 bytes total), and executes one
baseline suite plus one 0200 suite on the same host/settings. Both use the
corrected crop transform, 0.50 detector cutoff and 1,000-frame ceiling. Each
suite shares its compiled backend across the two clips and creates fresh
tracking/temporal state. The paired report binds media/model/annotation/config/
runtime/hardware identities, frame counts and actual decoded exposure, plus
observation, event, activity-match and miss deltas. It discloses that a decoded
frame digest is unavailable. Unannotated tails cannot supply scored negative
event counts/rates or a clinical false-alert gate; all assets are deleted on
success/failure, with aggregate metadata only. Both other real-video lanes are
excluded; no model shopping, training, new framework or parameter sweep is added.

This is an optional engineering integration and retrospective activity
comparison. It cannot promote a default or prove deployment accuracy. Native
results are pending when this implementation snapshot is authored; preserved
PR #142 measurements use the older crop mapping and are not the paired control.
PR #142's two-request budget and PR #140's four-request budget remain exhausted
for their completed/parked work; this new crop/profile integration has no CI
requests yet and does not reset either counter.

Local verification passed 751 full tests with three existing optional OpenCV
skips. The nine profile regressions and 18 generated-byte comparison regressions
pass without skips; the paired configs differ only in the detector selector,
and generated tail/result/source drift remains fail-closed. Guardrails,
compilation, deterministic replay and diff checks pass. The pinned upstream
0200 README and model manifest were re-read to verify class, shape, bytes and
SHA-384; no model/media/runtime was acquired or executed locally. Exact-head
hosted Linux/Windows and the paired native result remain required before merge.

## October 13 real-data regression phase

The preceding false-acceptance and temporal-continuity repair merged as PR #141
at `df1b2492ef4d7b6f71a7ef17fd1daa6c220dbcf9`; exact-head run #346 and
automatic main run #347 passed Linux, Windows and the required gate. The owner
then directed the next phase toward real evidence and no false positives.

Before scoring more clips, the retained end-to-end and orientation scanners
were found to expand decoded coverage to declared endpoints. Generated scan
fixtures reproduced one second being reported as one hour, an uncovered
positive being counted as missed, and one candidate over three seconds being
reported as one per camera-hour instead of 1,200. Both scanners now use actual
first/last decoded timestamps, require the declared start to match, reject a
single-timestamp span, and pass original labels to the evaluator for coverage
checking. Fourteen failing subcases now pass; this repairs measurement integrity,
not the detector's real-video accuracy. Covered decoding beyond a rounded
nominal endpoint remains accepted.

The pinned publisher Subject-4 Fall.csv and ADL.csv tables independently contain
the historical Fall/03 falling interval 1,700..6,000 ms and ADL/07 exercise
interval 0..4,000 ms. Whole-table size/Git-blob/SHA-256 identities and exact
selected rows are checked before media/model preparation. This resolves the
missing activity-label provenance for a retrospective regression; these are
publisher falling/exercise classes, not clinical person-down or injury truth.
Subject 4 was used earlier, so it cannot supply fresh held-out generalization.

The isolated `evidence/person-down-regression-*` lane replays only that already
reviewed pair (7,996,782 media bytes) and the existing four OMZ artifacts
(10,432,536 bytes) on hosted Linux CPU. One shared backend and one suite call
use retail-0013, cropped pose-0001 heatmap peaks, IoU tracking and unchanged
posture/temporal thresholds; the frame ceiling is 1,000 per clip. This differs
from the retained 0200/orientation diagnostic. The hash-locked reviewed runtime
is reused, all assets stay under RUNNER_TEMP and are deleted even on failure,
and the generic seven-pipeline evidence step is excluded. No parameter sweep,
training, model shopping or additional media is introduced.

The aggregate report binds source, annotations, media, models, protocol and
settings, and preserves matches/misses alongside normal-clip candidates and
actual decoded ADL-selected exposure. The nested evaluation is explicitly a
whole-clip activity proxy. Any tail beyond the publisher's rounded normal
endpoint is unannotated; if it exists, annotated negative event counts and
camera-hour rates remain unscored, and the replay never qualifies a false-alert
acceptance gate. Original positive endpoints cannot
be clipped or extended to pass; an uncovered label rejects the run. Silent
positives remain misses. A comparable historical result bound to this new plan
is unavailable, so baseline delta remains unmeasured. No deployment accuracy,
improvement or automatic promotion is claimed.

Local verification passed 722 tests with three existing optional OpenCV skips,
including 16 offline evidence-contract tests. Guardrails, syntax checks,
deterministic candidate replay and diff checking passed. The two exact upstream
CSV texts were independently rehashed against all declared identities and
accepted by the annotation verifier without downloading or decoding media.
Hosted run #348 (`38431713938`) executed the exact frozen plan
`3a4b5ca9746acee274a8e9f760fc43e904d80ab3a9a717a01dc6ae914571e74f`
on head `8180d2b358b52c79797d1f1529699f01e7a41def`. Linux passed all 722
tests with four existing optional skips and completed one native suite call.
Fall/03 decoded 200 frames spanning 6,672 ms, produced only four posture
observations, emitted no candidates and missed the one source falling interval.
ADL/07 decoded 149 frames spanning 4,960 ms and produced nine posture
observations with no candidates. Its 960 ms unannotated tail correctly leaves
annotated false-alert counts/rates null; zero full-clip candidates does not
qualify the owner's no-false-positives goal. The suite took 2,502.277 ms including
235.044 ms shared-backend preparation on hosted Linux CPU, with OpenVINO
`2026.3.1-22476-759c5a6ab8c-releases/2026/3`, OpenCV `4.12.0` and NumPy
`2.2.6`. Exact CPU model was not captured, so this observation cannot establish
a comparable hardware performance claim. The aggregate result/log is preserved
in PR #142; no media/model/frame artifact was uploaded or retained.

Run #348 failed Windows: three assertions and one private-helper test mixed
Windows TEMP's short alias with its canonical resolved path (722 tests,
three failures, one error, five existing optional skips). Test paths now resolve
before comparison/helper calls, matching existing canonical-root behavior.
Product code, labels, model identities and thresholds are unchanged by that
correction. A corrected exact-head hosted run is required before merge; the
work item has used one of two CI requests, with zero unchanged retries.

Next acceptance boundary: recover independently labeled positive activity
through perception while preserving separately labeled normal-activity negatives.
Four posture observations over 200 positive frames do not establish that
changing persistence thresholds would solve the miss. This historical pair is
fixed regression evidence; do not tune it into success or describe it as new
held-out generalization. The admitted baseline remains unpromoted.

## October 13 precision-first correction

The owner redirected the work toward no false positives and getting the
analytics back on track. Active work now targets demonstrated false accepts
before further donor/model expansion. Zero observed false alerts on a declared,
independently annotated held-out negative set is the goal; negative camera-hours,
missed incidents and per-class recall must be reported together. Emitting no
results is not sufficient, and finite tests cannot guarantee zero deployment
errors. Raw detections and person-down events remain review candidates, without
automatic confirmation or action authority. Face privacy retains its separate
default-blur and fail-closed controls.

A source audit reproduced five OCR false accepts on current main using bounded
TSV fixtures: word confidence `1000` was clipped to `1.0`; `ABC` at score `99`
plus `123` at score `5`, with cutoff `0.95`, became `ABC` at confidence `0.99`;
a malformed-confidence suffix was discarded the same way; a 17-character
value became a different 16-character prefix; `ABC?123` lost its uncertain glyph
and became `ABC123`. These are false accepts into normalized,
searchable plate metadata; no automatic LPR alarm emitter is present and no
real-world false-alert reduction is claimed from the fixtures.

The repair rejects invalid/non-finite/out-of-range word confidence, unsupported
characters and overlength identifiers. It abstains on incomplete/uncertain word
sequences rather than dropping a word, preserves only deliberate ASCII
space/hyphen separators, and reports the minimum word score for the complete
plate. Valid complete reads remain covered. This deliberately rejects more
ambiguous reads and may increase missed reads; future held-out evaluation must
measure that cost. Existing model/runtime/configuration and frozen native
thresholds are unchanged. No model/media acquisition or training is added.

The same audit reproduced a person-down false candidate through the actual
evidence scan orchestration with generated detections and poses: person A is
down at 0/500/1000 ms, then spatially disjoint person B at
1500/2000/2500/3000 ms. Neither segment lasts three seconds, yet both the
end-to-end and retained orientation scanners emitted one candidate/false alert
because they kept a constant temporal track despite detection reacquisition.
All five Figshare diagnostics import the retained orientation scanner, so the
repair belongs in both shared scan paths rather than a new model hypothesis.

The detection association must expose loss/new-segment decisions, including
reacquisition after an empty detection frame. A fresh bounded temporal episode
and session prevent one segment from lending duration to the next; trace
persistence resets too. Unknown pose while the detection remains spatially
linked still follows the existing explicitly configured bridge. Regression
fixtures must retain the uninterrupted positive and fresh full-length episode,
reject direct spatial jumps and short lost/reacquired runs, and exercise many
segments without accumulating stale temporal tracks. This is a demonstrated
logic repair under generated inputs, not a new real-video accuracy claim or a
restart of the stopped native experiments.

Verification of this repair: 695 full local regression tests passed with three
existing optional OpenCV skips; guardrails, compilation, deterministic replay
and diffcheck passed. LPR/forensic focused verification passed 42 tests with
no skips; independent review also exercised 65 additional parser checks.
Continuity verification passed 115 focused synthetic tests with no skips,
including both real scan-orchestration functions driven by generated metadata.
Those fixtures reproduced eight failing module subcases before the repair and
now reject the disconnected short episodes while preserving continuous positives.
The 1,100-segment stress fixture retained at most one live engine and one active
track. No real media, donor/runtime/model or training was acquired or executed.
Exact-head hosted Linux/Windows qualification is still required before merge.

The ResNet101-FPN exploration in PR #140 is parked with its native source and
results preserved. Run #344 passed discovery; run #345 passed the Linux native
path (one compile, three immutable-input observations, graph cleanup) but failed
Windows due to LF-to-CRLF conversion of hash-pinned JSON. All generated/rifle/
chair observations returned zero weapon candidates; no gain over the baseline
was observed. Prepared LF-repair commit
`7672f369f8bc0ee4441349070eac45e934bea6cc` remains unmerged and unverified on
hosted Windows. Its four-request ceiling is exhausted and stays recorded; this
new main-based OCR false-acceptance repair is a distinct owner-priority work
item, not a renamed donor retry or a reset of those counters. Keep one active
implementation PR, preserve main until exact-head checks pass, and stop the
no-gain model lane before publishing the new repair.

The current evidence gaps are still substantive: OCR's three retained real
positive measurements were wrong (MPR318->IFMPR318J, 354AVV->354AVIV,
CPU4704->CPU404), and no representative independently labeled no-plate negative
suite is admitted. Weapons/face engineering examples lack independently authored
box annotations; model execution success cannot supply their false-positive or
recall metrics. The next acceptance work is a frozen held-out positive/hard-
negative evaluation, not image-specific threshold tuning or declaring success
from zero outputs.

## October 11 donor-first acceleration — Weapons comparison

The owner selected **stronger weapons detection** as the active donor-first
priority. Retain the already-admitted OIDv4 SSD/OpenVINO stack as the runnable
baseline and reuse its neutral `DetectionCandidate` contract and seven-class
taxonomy. The earlier rifle-selected CC0 observation produced zero candidates;
it remains unscored because source selection metadata is not detector ground
truth. The admitted RF-DETR Nano path contains architecture source only, with
`pretrain_weights=None`; weightless construction cannot establish stronger
weapon detection. This direction prioritizes a usable pretrained replacement
and measured comparison before project-owned training.

The new `analytics_lab.weapons_evaluation` boundary binds actual normalized
annotation boxes to an existing held-out `WeaponTrainingAsset`, its exact media
identity/dimensions, annotation identity/hash, class/count summary and declared
evaluation rights. Train rows, identifiable people, uncleared rights and
inconsistent annotation declarations are rejected. Inputs are defensively copied,
with at most 256 cases and 64 labels/predictions per case. Each asset retains
the manifest's one declared positive class or explicit true-negative summary;
mixed-class assets require a separately extended manifest. No source image,
video, learned weights or native runtime is acquired or executed.
Annotation hashes prove internal binding, not independent authorship or the
truth of caller-supplied labels/rights. Those remain an evidence-review obligation.

The frozen engineering protocol uses confidence >= 0.50 and class-exact,
one-to-one maximum-cardinality IoU >= 0.50 matching. It reports per-class and
aggregate TP/FP/FN, precision/recall with undefined denominators represented
explicitly, positive-image misses and negative-image false positives. This is
not COCO/Open Images AP, a threat assessment or an automatic promotion decision.
The dataset digest binds held-out media, annotations and rights independently
of predictions, enabling a candidate/baseline comparison that rejects changed
data or protocol and preserves both donors' code/model/config/runtime identities.

`python -m analytics_lab.weapons_evaluation --synthetic-demo` exercises the
annotation-to-comparison path with generated metadata. The baseline misses one
rifle and emits no false positives; the candidate recovers that fixture rifle
but emits one handgun false positive on a true negative. The report preserves
both changes so a recall gain cannot hide a negative-image regression. This
fixture is not a measured improvement from an executed model.

All 22 focused generated-metadata tests passed without skips. The full suite
ran 676 tests successfully with three existing optional OpenCV skips; repository
guardrails, compilation, the existing person-down replay and the comparison CLI
also passed. Independent review verified all 512 possible 3x3 matching graphs,
tiny/extreme-aspect IoU, defensive mutation revalidation and repaired aggregate
count invariants, with no remaining concrete blockers. Hosted exact-head
Linux/Windows qualification remains the merge gate. No model, media, runtime
dependency, workflow or control file changed.

One additional pretrained candidate was inspected from the already-pinned
TensorFlow model-zoo lineage: OIDv4 SSD ResNet101-FPN 512x512. The official Google
storage server returned a Content-Length of 638,212,759 bytes for
`ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20.tar.gz`
on October 11. Only HTTP HEAD metadata was read; no archive, model or media was
downloaded. This exceeds the existing 250,000,000-byte bounded discovery lane.
The candidate is not admitted, and its upstream whole-dataset score is not
Weapons accuracy evidence. The donor screen remains limited to the retained
OIDv4 baseline, existing RF-DETR architecture and this additional candidate.

After exact-head hosted qualification and merge, the next executable gate is
independently authored held-out Weapon labels
and a separately bounded exact-weight/runtime admission for the replacement.
The existing compile-once runtime, admitted CC0 identities and inference output
contract can be reused for that comparison. No stopped tracking, OCR glyph or
FaceSSD experiment is restarted, and no commercial or downstream deployment is
made by this work.

## October 11 forensic batch-ingestion bound repair

`ForensicIndex.extend` previously exhausted its input with `tuple(records)` before
checking the configured batch bound. A finite oversized producer was therefore
fully consumed, and producer length hints were consulted before iteration. The
collector now retains at most `max_records` items and consumes only one additional
item to reject overflow, without consulting a producer length hint.

Collection still finishes before any record is added. Batch overflow retains the
existing error and precedes per-record validation. Producer failures before the
bound leave existing records unchanged; `TypeError` retains its public `ValueError`
translation. The later add phase retains exact duplicate idempotency, conflicting
duplicate rejection and its existing partial-add behavior. The bound is on item
consumption/reference storage, not the execution time of an individual producer
call, object size, concurrent mutation or backend persistence.

Finite generated-metadata regressions reproduced seven failing baseline subcases.
The repaired source passed 140 forensic tests and 642 permitted source tests under
a process/network/native-load guard, with zero skipped tests or guard violations.
Nine subprocess tests and five optional native/media tests were explicitly
deferred locally. Exact-head hosted Linux/Windows qualification and independent
review remain required. Consumer pins, packages, schemas, authorization and
operator UI are unchanged; this is component resource-safety progress, not
installed or integrated forensic-search acceptance.

## Generated-pixel appearance producer — source boundary prepared

`forensic_histogram.extract_bgr24_histogram` computes the existing model-free
appearance descriptor from actual caller-supplied BGR24 pixel bytes and the exact
evidence-linked record ROI. It reads no camera, file or network source. Input must
be immutable `bytes`, at most 16 MiB, with dimensions in 1..8192, an exact
top-down row stride at least `width * 3`, and an independently supplied full-buffer
SHA-256 including padding. Source ID, event ID, UTC time and normalized bbox must
match the original `ForensicRecord`; frame index is explicit and bounded. Missing
record evidence/bbox, mutable or mis-sized buffers, changed pixels, invalid
coordinates, rounded-empty pixel crops and face/license-plate records fail closed.
Even positive normalized intervals can collapse under floating-point pixel
rounding; these are rejected rather than silently expanded.

The fixed algorithm uses 4 bins per RGB channel and index
`16*(R >> 6) + 4*(G >> 6) + (B >> 6)`. Crop bounds are floor(left/top) and
ceil(right/bottom), with exclusive upper pixel bounds. Each axis samples
`min(extent, 64)` cell centers using integer arithmetic, so no more than 4096
pixels contribute. Only actual selected pixels count; row padding does not.
Counts divided by sample count yield the 64-dimensional L1 descriptor. The exact
canonical configuration JSON and its computed SHA-256 are available separately.
No learned model, new dependency, face recognition or identity inference is added.

`HistogramExtraction` retains metadata only: a defensive copy of the original
record/descriptor, exact frame/layout/ROI metadata, crop/sample facts, extraction
config SHA-256, and a digest binding all of those fields. Original detector/source
event, producer/version/revision, configuration/model hashes, bbox and attributes
are unchanged. The extraction hash never replaces a detector configuration hash.
Source attribute counts and normalized-text lengths are bounded before copying
or whitespace normalization, including manually mutated input objects.
`descriptor_for_ranking()` revalidates the metadata binding before returning an
independent descriptor. That digest detects inconsistent/tampered claims; it is
not an authenticity signature. The caller must independently establish the frame
and event association; a fabricated but internally consistent caller declaration
cannot be authenticated by this pixel function.

Generated-pixel tests cover bin/channel boundaries, crop rounding, stride/padding,
sampling bounds, determinism, metadata/source tampering, preserved provenance,
ranking and the existing forensic envelope. These are engineering fixtures, not
detector/appearance accuracy or production authentication evidence. Work is bounded
by 16 MiB of hashing plus 4096 sampled pixels; no runtime latency SLA is claimed.
Existing wire envelopes remain unchanged and do not carry the separate extraction
binding. Product work must preserve it in an explicitly versioned atomic
event/descriptor store before authenticated two-event comparison is wired.
Edge worker staging, paired authorization, Neural source packaging and the
operator Compare UI remain open; no real-media execution or product repin occurs.

## Appearance scale probe — vector-need measurement prepared

The next appearance-search gate measures the current model-free in-memory descriptor scan before any vector infrastructure is adopted. The probe uses deterministic synthetic metadata descriptors only, measures ranking-call elapsed time separately from descriptor generation, supports 1..10000 candidates, 2..512 dimensions and 1..5 iterations, and records median/max runner timing plus result count. Probe output is runner-specific engineering evidence only with `identity_claim=false`, `authorizes_action=false` and `performance_claim=false`; it is not an SLA or release-performance claim. Vector infrastructure must not be adopted until this measured baseline shows a real need.

## Appearance result handoff — K5 serialization active

Appearance ranking results use the already-accepted `analytics.forensic-result.v1` envelope rather than a parallel transport. An appearance payload identifies the probe observation, fixed descriptor schema/kind, deterministic ordered similarity scores, and evidence-linked operator hits/playback pointers for each candidate. The payload is bounded to at most 1000 matches, requires one shared probe observation, unique candidate observation IDs, exact deterministic ranking order, and preserves `identity_claim=false` / `authorizes_action=false` at the envelope, match and hit levels.

### October 2 serialized-handoff regression repair

The consumer parser previously validated only the envelope's outer fields and
flags. Generated valid producer payloads could therefore be changed to contain a
nested authority/identity claim, an unversioned field, or playback pointing at
another source, timestamp or evidence event and still parse successfully. This
is a validation gap; no unauthorized product execution has been demonstrated.

The handoff now validates the closed hit/trail/appearance shapes using the
existing metadata, evidence, playback and trail contracts. It rejects duplicate
JSON fields, preserves source/evidence binding and appearance ranking bounds,
rechecks mutable payloads before serialization, and applies the same 1,000,000
byte ceiling to serialized output and input. Existing valid producer bytes and
the versioned schema remain unchanged. No dependencies, workflows, authority
rules, models or media paths changed.

Local generated-metadata verification reproduced 88 failing subcases on the
unchanged base. The repaired suite ran 605 tests with three existing optional
OpenCV skips; guardrails, the synthetic person-down replay and bytecode
compilation passed. Independent review passed 108 focused forensic tests and
confirmed 32 additional valid producer boundary cases remain byte-identical.
Next executable step: review and exact-head hosted Linux/Windows qualification.
This closes neither the product-owned authenticated consumer adapter nor actual
VMS/Neural UI integration; commercial and real-media acceptance remain separate.

### October 3 appearance-score handoff regression repair

Valid color-histogram descriptors allow L1 normalization error up to `1e-9`,
but matching two such descriptors could produce a similarity just above `1.0`
and fail before the existing operator handoff. The scorer now bounds only that
accepted rounding excess to `1.0`; descriptor admission, explicit wire-score
validation, ordinary scores, and non-authorizing/non-identity flags are unchanged.
Saturated scores use the existing deterministic tie order.

Generated-metadata regressions reproduced four failing cases on unchanged main
and now exercise descriptor -> ranking -> forensic envelope -> parser with full
evidence/time retention, plus invalid-input rejection. Local verification ran
610 tests with three existing optional OpenCV skips; guardrails, synthetic replay
and compilation passed. Next step is independent review and exact-head hosted
Linux/Windows qualification. This repairs a library producer-to-operator boundary;
it does not establish installed Neural/VMS UI integration or real-media accuracy.

## Appearance benchmark — deterministic engineering metrics active

The first appearance benchmark is project-authored fixture evaluation only, not person/object identity ground truth and not a commercial accuracy claim. It measures deterministic model-free ranking with precision@K, recall@K, reciprocal rank and top-1 fixture relevance over evidence-linked synthetic/model-free descriptor cases. Cases require exact category/schema/dimension compatibility, unique candidate IDs, explicit fixture-relevant candidate IDs, bounded K and bounded case counts. Benchmark results and summaries remain `identity_claim=false` and `authorizes_action=false`.

## Appearance / visual similarity — model-free ranking active

The first appearance-search gate is deliberately non-biometric and model-free. It uses evidence-linked, L1-normalized color-histogram descriptors behind schema `analytics.appearance-descriptor.v1` and kind `model_free.color_histogram.v1`. Face and license-plate categories are rejected. Learned/ReID descriptor kinds are rejected. Ranking uses deterministic histogram intersection with an inclusive similarity threshold, bounded candidate/result counts, exact category/dimension compatibility, deterministic tie ordering, and `identity_claim=false` / `authorizes_action=false`.

The donor screen remains capped at three: OpenCV histogram algorithms are acceptable as an algorithmic reference without a new pretrained runtime dependency; DINOv2 ViT-S/14 remains on HOLD because pretrained training-data commercial provenance is not yet strong enough; OpenCLIP ViT-B/32 LAION-2B remains on HOLD for the same fail-closed commercial provenance reason. No learned weights are admitted by this gate.

## Forensic search — foundation active

Forensic search is now a first-class Analytics Lab capability rather than a synonym for appearance/ReID search. The first engineering slice is metadata-first and model-independent: normalized analytic observations can be indexed and searched by bounded UTC time range, source/camera, category, session-local track id, exact/prefix plate/OCR text, minimum confidence, and exact generic analytic attributes. Existing Tracking and LPR/OCR outputs have explicit adapters into this record contract.

PR #121 exact-head run #301 passed Linux, Windows and the Analytics quality gate, merged to main as `4305dfca6ddb794d96d7d51cb4c9773a0e7eeae2`, and post-merge run #302 passed. The deterministic metadata-search foundation is therefore closed green.

The reference index is intentionally bounded and deterministic. It stores normalized analytic metadata only, not video/image pixels. Track ids remain session-local association identifiers and do not create biometric identity or cross-camera identity claims. Generic attributes carry their own confidence and provenance so future reviewed producers can add color, garment, vehicle, direction, zone, or similar facets without changing the query contract.

PR #122 exact-head run #304 and post-merge main run #305 are green, closing immutable evidence provenance for every searchable forensic record: source event id, producer + exact version, configuration SHA-256, optional model SHA-256 and optional source revision. The active next forensic gate is a durable stdlib-SQLite backend that must preserve the reference ForensicIndex semantics exactly across close/reopen, bound capacity under concurrent writers, reject corrupt/unknown schema state, preserve evidence/attribute fidelity, safely reopen paths containing `#`, `?`, spaces or `%`, and store normalized analytic metadata only. No raw media, biometric identity, learned embedding, ReID or vector donor is introduced by this gate.

PR #126 exact-head run #313 and post-merge main run #314 are green, closing operator-safe forensic result projections with metadata-only playback references, immutable evidence binding, bounded pre/post-roll, `identity_claim=false`, and `authorizes_action=false`. The active next forensic gate is a versioned serialized handoff for K5/VMS consumption. The first schema is `analytics.forensic-result.v1`; serialization must be deterministic JSON, bounded to 1 MB on parse, contain only the already-approved metadata/evidence/playback-pointer fields, preserve non-authorizing/non-identity semantics, and reject unknown schema versions or unexpected top-level fields.

## Detection + tracking — frozen at first real-video acceptance gate

### October 7 tracking-session iterator-bound regression repair

`TrackingSession` previously materialized detector inputs and backend outputs
with `tuple` before enforcing their configured counts. An overlong producer
could therefore run indefinitely or allocate beyond the configured frame limit;
an untrusted `__len__` or `__length_hint__` could also trigger premature allocation
or an unrelated conversion error. Both boundaries now collect explicitly, retain
at most the configured limit, and reject on the first excess item without reading
another item or consulting producer length hints. Overflow still raises instead
of truncating, before item/duplicate validation; accepted values preserve order
and the tuple contract. Existing iterable-error conversion and session frame/time
commit-after-validation semantics remain unchanged. Backend-owned state rollback
is still the backend's responsibility.

Ten new generated-only regression methods reproduced 14 failing and 12 error
subcases against unchanged source. An additional compatibility regression covers
arbitrarily large valid configured bounds. All 16 tracking tests and all 632
permitted source-suite tests now pass under an isolated execution guard. Nine subprocess
tests and five optional OpenCV/NumPy tests were explicitly excluded from this
source-only run, not reported as passing. Forty stdlib SQLite opens used only
disposable generated fixtures; no prohibited execution events occurred.
Repository guardrails, the in-process synthetic person-down replay and compilation
of all 180 analytics/test source files passed. Next step: independent exact-candidate
review and full applicable hosted Linux/Windows qualification. No model, media,
dependency, workflow or installed-product claim is added, and the stopped
real-video evidence hypothesis remains unchanged.

### October 6 ByteTrack capacity-transition regression repair

Generated detection sequences demonstrated two failures in the existing portable
ByteTrack backend. A rejected over-capacity frame changed retained track state,
so a valid same-frame low-confidence retry could lose its prior association.
Also, an expired lost track still occupied capacity until after new-track
admission, repeatedly rejecting a replacement even beyond the lost-frame budget.

The backend now stages independent track objects and counters, preserves every
existing association/recovery opportunity, expires still-lost over-age tracks
before new-track admission, and commits only after output validation succeeds.
Genuine capacity excess still raises; configured bounds, confidence thresholds,
association ordering, confirmation behavior and session-local ID allocation are
unchanged. No detections are silently dropped to bypass capacity.

Six new generated-only regression methods reproduced five failures and three
expiry-subcase errors against unchanged source. All 13 ByteTrack tests now pass,
covering high/low-confidence mutation, unconfirmed removal, staged ID rollback,
same-frame retry, expired replacement, preserved recovery and output failure.
The full local suite ran 633 tests with three existing optional OpenCV/NumPy
skips; guardrails, synthetic person-down replay and source compilation passed.
Next step: independent exact-candidate review and hosted Linux/Windows
qualification. This repairs the supported source contract only; it adds no
tracker machinery, model/media path or installed-product claim and does not
reopen the stopped real-video evidence hypothesis below.

Tracking remains stopped at the policy's three-session threshold. The retained boundary includes the detector-neutral tracking contract, portable ByteTrack slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` (MIT), bounded evaluator, rights/evidence admission, pre-registered Wikimedia CC0 frames 150-174, canonical RGB24 frame identities, exhaustive ground-truth package binding, frozen simple-IoU control, and first-attempt head-to-head runner.

The fixed frame-manifest SHA-256 is `52f00a4fc1e013d9c0cae9647cb386369441f005a970d6d42c449cbd0269a031`; benchmark-plan SHA-256 is `eb7995a389a22f3528b9bfca97d64f9d8e3c515ebf1c764cf25b68fbdcc3479c`.

The remaining blocker is unchanged: an independently authored exhaustive person/track annotation package for those exact frames cannot currently be materialized in the independent annotation environment without violating the approved evidence lane. No detector/tracker output may substitute for ground truth and no further tracker machinery should be added until that exact label path becomes executable.

## LPR/OCR — smallest runnable engineering baseline closed
The retained runnable baseline is Open Model Zoo `vehicle-license-plate-detection-barrier-0106` FP16 behind a replaceable detector boundary, OpenVINO `2026.3.1` CPU, Tesseract `5.5.3` with pinned `tessdata_fast` English data, and the model-free OpenCV 4.12.0 proposal fallback. This is an engineering baseline only; it is not a commercial-accuracy or release-readiness claim.

Exact rights-cleared CC0 OCR evidence is already preserved from successful hosted-Windows runs. The preregistered gray/Otsu Land Rover run #202 observed `MPR318 -> IFMPR318J`; South Carolina run #206 observed `354AVV -> 354AVIV`; Washington run #209 observed `CPU4704 -> CPU404`. The frozen aggregate Levenshtein edit distance is therefore `5` across three samples. These misses are retained as measured evidence and must not be normalized away or used for image-specific tuning.

The donor screen previously reached its three-candidate ceiling without admitting replacement OCR weights because training-data/weight commercial provenance did not close strongly enough. The project-authored OpenCV glyph fallback was then attempted as a no-weights alternative. Runs #211 and #212 exposed pre-measurement harness defects; the crop-interface mismatch was diagnosed. After Face completed, successor PR #111 resumed that hypothesis. Run #276 and changed-approach Linux run #277 both failed before any aggregate glyph comparison result because the pinned Wikimedia acquisition path returned HTTP 429. Synthetic glyph regressions passed, but no rights-cleared real glyph observation was produced. Under the three-session stop rule, this network-dependent glyph-evidence hypothesis is now closed without acceptance or rejection of the recognizer; do not retry it unchanged, do not tune it from synthetic fixtures, and do not substitute unpinned media.

The smallest runnable LPR/OCR engineering baseline is therefore frozen at the already-verified Tesseract 5.5.3 path plus the existing replaceable detector/proposal boundaries. Remaining non-baseline gates are broader independently labeled rights-cleared held-out evidence, a stronger OCR donor only if code/weight/data rights close cleanly, native dependency/redistribution notices, representative soak/performance work, and explicit commercial-release approval. PR #111 exact-head run #278 and post-merge main run #279 are green. The single active implementation lane is now Weapons. PR #112 exact-head run #280 and post-merge main run #281 are green, closing the first fixed seven-class OIDv4 weapon allowlist/parser baseline behind the existing neutral `DetectionCandidate` boundary. PR #113 exact-head run #282 and post-merge main run #283 are also green, closing the compile-once warm-reuse lifecycle around the same admitted graph/runtime. PR #114 exact-head run #285 and post-merge main run #286 are green, closing the hash-locked exact-model Weapons execution-compatibility gate: the admitted archive/graph reverified, the model compiled once, exactly one generated 300x300 uint8 inference ran at the frozen 0.50 threshold, the input remained immutable, and no media/model/pixel artifact was retained. The generated blank tensor emitted zero allowlisted detections; this is preserved as a synthetic engineering observation only. The active next gate is source admission for two individually rights-cleared CC0 real images—a Winchester Model 1895 rifle positive candidate and a chair non-weapon negative candidate. First-head run #287 passed Linux, Windows and the Analytics quality gate and discovered exact SHA-256 identities without decode or inference: rifle `4c58e411b9e3e7348420915dcb32be0e6dcd0132f6120a6049af36791605c0f2`; chair `73db796087c551375d7d537ca78c81ede7b106cd8d25e1fad729a29a7b59483c`. Pinning head run #288 passed Linux, Windows and the Analytics quality gate, PR #115 merged as `029fbbe71e92265d957285b3dfd08f6deeaf3d2d`, and post-merge run #289 passed. The source-admission gate is therefore closed green. The next prepared gate is one untouched two-source Weapons measurement: stream/reverify the admitted graph, compile once, decode each pinned CC0 source in memory with EXIF auto-orientation disabled, execute exactly one frozen-0.50 inference per source, preserve every allowlisted detection, verify input immutability, and keep misses/false positives unscored because source roles are selection metadata rather than independent detector ground truth. PR #116 exact-head run #290 and post-merge main run #291 are green, closing the first untouched two-source real Weapons engineering observation. At the frozen 0.50 threshold, the admitted OIDv4 model emitted zero allowlisted detections on both the Winchester Model 1895 rifle-selected CC0 source and the Chair.JPG non-weapon CC0 source; both inputs remained immutable, compile_count=1 and inference_count=2, and no media/model artifact was retained. Because independently authored detector ground truth is unavailable, misses and false positives remain unscored and this is not an accuracy claim. The zero output on the rifle-selected source is preserved as a material engineering signal and may justify evaluating a stronger separately admitted detector/model path after the current runtime-hardening gate. PR #117 changed-head run #293 passed Linux, Windows and the Analytics quality gate after the Windows-only `resource` import was narrowed to Linux execution scope; it merged to `main` as `8c6ee3b07f429722491d90aba324dd8e19298987`, and post-merge run #294 passed. The bounded hardening evidence preserved one compile, three warm inferences, fixed threshold `0.50`, immutable input, per-call hosted-CI timing and bounded RSS observations with no media/artifact retention or performance/accuracy claim. OIDv4 Weapons runtime hardening is therefore closed. Because the first real rifle-selected source produced zero allowlisted detections, the project-owned stronger-detector fallback is now active. PR #118 exact-head run #295 and post-merge main run #296 are green, closing the fail-closed training-rights/compute contract around `roboflow/rf-detr@d96d6ff9303ec5af7bb8f73e83f0f04ecb6ef728` (Apache-2.0 architecture only), `RFDETRNano`, `pretrain_weights=None`, offline/CPU-only first proof, frozen seven-class taxonomy, exact commercial rights per media asset, project-owned annotation identities, disjoint train/validation splits, and the existing five-minute hosted ceiling. PR #119 exact-head run #297 and post-merge main run #298 are green, closing source admission for the official RF-DETR 1.11.0 release commit `d96d6ff9303ec5af7bb8f73e83f0f04ecb6ef728`. The admitted source path verifies the exact Apache-2.0 license/version, Nano configuration, pretrained-weight early return and non-14-patch DINOv2 local-backbone semantics without installing RF-DETR, downloading weights/media, constructing a model, or making accuracy/readiness claims. The active next gate repairs the training manifest so true negative/background assets are represented explicitly as `source_class=None`, zero boxes and no annotation classes, while positive rows require at least one box and an immutable class summary exactly matching the frozen source class. This prevents negatives from being mislabeled as generic `weapon` before any training execution. Tracking remains frozen at its independent-label blocker.

## Retained person-down / slip-fall path
The required secondary path remains:
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`.

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. Existing detector/temporal behavior and hard negatives remain protected while higher-priority shared perception/evaluation work advances.

## Face detection + automated/selectable blurring — ~90% ENGINEERING maturity; feature expansion stopped
The privacy/product boundary is already in place independently of model admission: normalized face boxes, immutable local-model verification, replaceable detector adapter, default-on Gaussian blur with bounded box expansion, permission-gated unblur/reblur, unauthorized-unblur denial, input-frame immutability, bounded face counts, and privacy audit state containing no biometric identity. Face recognition, embeddings, ReID and identity matching remain explicitly out of scope.

### October 1 privacy-cap regression repair
The OIDv4 and YuNet parsers previously discarded admitted detections above `max_faces`, hiding overflow from downstream privacy enforcement. They now raise `RuntimeError` on the first excess admitted face instead of returning a partial detection set. At the cap, all admitted boxes remain available for blur in deterministic order. Only faces surviving the existing class/threshold/geometry filters count toward the cap; those filters and the frozen OIDv4 threshold are unchanged.

Synthetic adapter-to-privacy regressions cover caps 1, 2 and 64 with separate non-overlapping boxes. Cap+1 fails before blur, frame copying, or a success audit under default blur, denied unblur, authorized unblur and policy passthrough. Callers must withhold unassessed frames when detection/privacy fails; falling back to displaying the original is not safe. This fixes loss of already detected faces and makes no claim about detector false negatives or face accuracy.

Local verification: the new overflow regression reproduced 24 failing subcases on the base, then passed after the repair; the full suite ran 592 tests with three existing optional OpenCV skips. Repository guardrails, the synthetic person-down replay and bytecode compilation passed. Native preprocessing is stubbed in the dependency-free adapter boundary tests; no model, media or runtime was acquired. Next executable step: normal exact-head hosted Linux/Windows CI before accepting the patch. Actual consumer-side failure handling remains an integration gate.

### September 24 engineering-maturity checkpoint
PR #110 hardened the reusable standard-OID/OpenVINO runtime so model conversion/CPU compilation occurs once per runtime instance and subsequent detections reuse the compiled model. Exact head `436854718f2fbc261bc74dc715ad74713473cb30` passed run #274 on unchanged base `d4a76770690171656a789c025ad88aab4511dd3a`; it merged to `main` as `8d42e7110419900a9e8d8ef9f97df80fb0774fdd`, and post-merge run #275 passed. The bounded hardening evidence used only deterministic synthetic input for three sequential calls and recorded compile count, per-call engineering timing, peak process RSS and input immutability. Existing regression coverage includes zero-face output, multiple class-502 filtering, non-face rejection, malformed-output fail-closed behavior, default blur, denied-unblur remains blurred, authorized-unblur passthrough, input immutability, max-face bounds, and bounded large-region Gaussian planning. The earlier admitted CC0 real portrait remains the sole real-person execution sample and is not a basis for an accuracy claim.

Face is therefore recorded as approximately **90% ENGINEERING maturity only**. This is not commercial accuracy, release approval, biometric identity functionality, or a product-performance claim. The remaining gates are: broader rights-cleared held-out evidence across face scale, pose, lighting, multiple faces and no-face negatives; packaging/native-dependency notices and redistribution review; long-run soak/resource/performance validation on representative supported hardware; and explicit commercial-release approval. Do not tune the frozen `0.50` threshold from the one admitted real source and do not add recognition, embeddings, ReID or identity matching.

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

### Rights-cleared real-person source admission — closed green
PR #108 selected Wikimedia Commons `Face portrait (Unsplash).jpg`, source page `https://commons.wikimedia.org/wiki/File:Face_portrait_(Unsplash).jpg` and canonical upload `https://upload.wikimedia.org/wikipedia/commons/0/04/Face_portrait_%28Unsplash%29.jpg`, solely as the first real-person evidence source. Commons records the image as a William Stitt photograph dated 19 October 2016, published on Unsplash before the 5 June 2017 license change under CC0-1.0, with byte length `9,023,019` and SHA-1 `d7ac58077d135b823e71e7b38e763f082e2053bc`.

Discovery run #266 passed the repository guardrails, Linux regression, the bounded source-admission step, Windows regression and the aggregate quality gate. The source was streamed only in memory, never decoded, and produced SHA-256 `7356daa8fd4ad53b946ce0036f06b014431dc89b7ae29ecd8ef18fc54edce6b5`. The pinning head then hard-pinned and reverified that exact size/SHA-1/SHA-256; exact-head run #267 passed Linux, Windows, the bounded admission evidence and the aggregate gate. PR #108 merged into `main` at `13c3f9f7ee4e3b7a23e029557251e6bbce71a76e`, and automatic main verification run #268 also passed.

No image decode, face inference, face count, model execution, media retention, or detection-accuracy/commercial-readiness claim occurred during admission. The rights-cleared source is now admitted for the separate one-attempt face-detection -> automatic-blur measurement.

### First real-person face -> automatic-blur measurement — active one-attempt gate
The current acceptance-moving work item is one untouched execution of the admitted standard OIDv4 graph against the exact admitted CC0 portrait, through the class-502-only adapter and existing default-on privacy-blur path. The threshold is frozen at `0.50` before execution. The evidence lane must reverify both model and source identities, decode pixels only in memory, run exactly one CPU inference, preserve every returned class-502 score/normalized box, verify input-frame immutability, and record the privacy action and whether the output pixels changed.

There is no independently authored face ground-truth annotation for this source, so misses and false positives remain explicitly unscored rather than inferred from the source title or detector output. Zero detections, one detection, or multiple detections must all be preserved as obtained; none is grounds for threshold tuning or an unchanged rerun. No recognition, embeddings, ReID, identity matching, model/media retention, or commercial-accuracy claim is permitted. This gate is a single-source engineering observation only.

Runs #269 and #270 both passed the normal Linux regressions and reached the dedicated measurement step, then were cancelled by the repository's existing five-minute Linux job ceiling before any face result was emitted. Run #270 already changed the detector input to the model's pinned 300x300 inference size, so it was not eligible for an unchanged retry. Run #271 changed the acquisition path and made the previously opaque budget measurable: exact graph stream/verification completed in `1.549758s`, source stream/verification in `0.417205s`, image decode in `0.145742s`, and OpenVINO conversion+CPU compilation in `189.092199s`. The job then exhausted the remaining budget inside the combined inference+blur phase without emitting a result.

The next repaired head keeps the exact admitted model/source, fixed `0.50` threshold, all provenance/security checks and the five-minute ceiling. It preserves the run-#271 stream-to-exact-pinned-graph optimization, times the single detector inference separately from privacy processing, reuses that exact detection tuple so privacy enforcement cannot trigger a second inference, and bounds pathological full-resolution Gaussian kernel work by downscaling the detected face region before Gaussian blur and restoring it afterward while preserving the requested Gaussian strength in source-pixel units. This is a runtime-budget repair, not a detection result or accuracy claim. No model/media artifact is retained.

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
