# Standard Analytic Event Schema

Every mature Analytics Lab analytic should emit a normalized event envelope. Implementations may attach analytic-specific metadata, but the common fields should remain stable.

## Draft v0.1

```json
{
  "schema_version": "0.1",
  "event_id": "uuid",
  "analytic_id": "slip_fall_causality",
  "event_type": "fall_with_precursor",
  "source_id": "camera-001",
  "site_id": "optional-site-id",
  "start_time": "2026-09-14T22:31:08Z",
  "end_time": "2026-09-14T22:37:31Z",
  "confidence": 0.96,
  "severity": "high",
  "status": "confirmed",
  "objects": [],
  "zones": [],
  "observations": [],
  "evidence": [],
  "relationships": [],
  "recommended_actions": [],
  "metadata": {}
}
```

## Required concepts

### Confidence
A normalized score from 0.0 to 1.0 representing the analytic's confidence in the event conclusion. Confidence should not be presented as calibrated probability unless calibration has actually been demonstrated.

### Status
Suggested values:
- `candidate`
- `confirmed`
- `cleared`
- `abstain`

`abstain` is important: the analytic may determine that available evidence is insufficient for a reliable conclusion.

### Observations
Atomic facts used to support higher-order reasoning, for example:

```json
{
  "type": "person_posture_change",
  "timestamp": "2026-09-14T22:37:19Z",
  "confidence": 0.94
}
```

### Relationships
Links between observations or objects, for example:

```json
{
  "subject": "person-17",
  "relation": "intersected",
  "object": "spill-region-2",
  "timestamp": "2026-09-14T22:37:18Z"
}
```

### Evidence
Pointers to supporting video windows, frames, tracks, masks, or derived artifacts. Evidence records should reference media rather than embed sensitive media directly into event messages.

## Principle

The system should preserve the difference between **what was observed** and **what was inferred**. Higher-order analytics should be able to explain which observations produced an inference.
