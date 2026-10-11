# Analytics Lab Roadmap

This roadmap prioritizes analytics that create differentiated situational understanding rather than commodity object detection.

## Current delivery direction — October 11, 2026

The owner selected stronger weapons detection as the first donor-first acceleration target. Reuse the existing OpenVINO runtime, neutral detection contract and seven-class taxonomy. Retain the admitted OIDv4 SSD as the comparison baseline; its zero output on the rifle-selected engineering source remains an unscored observation.

Qualify replacements on the same independently authored, rights-cleared held-out annotations with frozen confidence and IoU thresholds. Report per-class misses and false positives, including negative-image alerts, alongside the exact donor/data identities. An architecture source admission or weightless RF-DETR construction is not a trained weapons detector. Prefer admitted pretrained weights before considering project-owned training.

The first executable slice is the bounded annotation-to-donor comparison. Next, acquire independent annotations and close exact weight/runtime/resource admission for a replacement, then run a preregistered baseline comparison. The reviewed OIDv4 ResNet101-FPN archive is 638,212,759 bytes by an official-server metadata response; it has not been downloaded or admitted and exceeds the existing bounded model-discovery lane. No broader donor search or training run starts from this change.

The locally tested scene-change experiment remains unpublished while Weapons is the active work item. Historical stopped tracking, OCR glyph and FaceSSD hypotheses remain stopped.

## Phase 0 — Foundation

- Standard event schema
- Donor-code provenance policy
- Video-file and RTSP ingestion adapters
- Reusable detector / tracker / pose interfaces
- Benchmark harness
- Golden test clips and expected-event fixtures
- Confidence, abstention, and failure-reporting conventions

## Phase 1 — First analytic family

### A1 — Slip / Fall Causality
Detect a fall and reason about relevant precursors such as a spill, obstruction, surface condition, or interaction immediately before the event.

### A2 — Man Down / Human Distress
Detect collapse, prolonged prone state, inability to recover, crawling, stumbling, or other combinations that may indicate distress rather than merely a person being horizontally oriented.

### A3 — Near Miss
Detect dangerous proximity and trajectory interactions between people, vehicles, forklifts, mobile equipment, or other tracked objects even when no collision occurs.

### A4 — Scene Integrity
Detect meaningful persistent scene changes such as blocked views, displaced safety equipment, propped doors, new obstructions, removed objects, or other state changes.

## Phase 2 — Temporal and behavioral reasoning

### A5 — Behavioral Escalation
Model sequences such as confrontation, aggressive approach, repeated gestures, pursuit, pushing, and fighting as an escalating event rather than waiting for a single violent frame.

### A6 — Blind-Spot Risk
Determine when PTZ movement, occlusion, obstruction, or scene changes create loss of coverage over important areas.

### A7 — Incident Genesis
Correlate multiple low-level observations into an event chain explaining how a higher-order incident developed.

### A8 — Liability Reconstruction
Automatically assemble a before / during / after chronology with linked observations and evidence references for review.

## Phase 3 — Multi-camera intelligence

- Cross-camera event correlation
- Multi-camera incident reconstruction
- Coordinated-behavior analysis
- Camera coverage optimization
- Risk heatmaps based on incidents and near misses rather than simple occupancy

## Promotion gates

An analytic is not considered ready for downstream integration until it has:

1. a defined event contract;
2. reproducible test data;
3. measured precision / recall or task-appropriate metrics;
4. documented failure cases;
5. dependency and license review;
6. deterministic regression tests where practical;
7. latency and compute measurements;
8. confidence / abstention behavior;
9. a versioned integration adapter.
