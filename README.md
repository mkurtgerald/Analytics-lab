# Analytics Lab

Public research and development laboratory for next-generation video analytics. 
Its purpose is to prototype, benchmark, and mature new analytics using clean modular interfaces and carefully tracked third-party components.

## Goals

- Develop genuinely new video analytics rather than duplicate commodity rules.
- Start with reviewed donor detectors, trackers, runtimes and weights, then add project-owned analytic and integration logic.
- Keep donor-code provenance and licensing visible and auditable.
- Standardize analytic outputs so proven modules can later integrate with K5 Vision.
- Maintain reproducible benchmarks, test clips, and evaluation criteria.
- Keep experimental work isolated from the production VMS codebase.

## Initial analytic tracks

1. Slip / fall causality and liability reconstruction
2. Man-down and human-distress detection
3. Near-miss detection
4. Behavioral escalation
5. Scene-integrity and blind-spot risk
6. Incident genesis / temporal event reasoning

## Repository status

The library contains detection/tracking, LPR/OCR, face detection/privacy, seven-class weapons detection, person-down candidates and forensic metadata/appearance search. These are engineering components with separate real-media and product-integration acceptance gates.

The current priority is reducing false accepts and false alerts. Keep uncertain results as review candidates or abstain. Evaluate changes on independently labeled positives and hard negatives; require zero observed false alerts on the declared held-out negative exposure while reporting recall and missed incidents. A silent detector that misses every positive does not meet the goal. Zero observed errors in a finite test is not a guarantee of zero errors in deployment.

The first repairs make OCR abstain on incomplete or ambiguous plate reads and restart persistence in the shared person-down diagnostic scans when detection continuity is lost. Plate confidence must not exceed its weakest recognized word; disconnected people cannot contribute time toward one person-down candidate. Donor perception stays replaceable; runtime smoke tests and larger models do not establish accuracy. The retained Open Images V4 SSD remains the weapons baseline, and RF-DETR supplies architecture source only. The tested ResNet101-FPN experiment showed no candidate gain on its two admitted samples and is parked.

Run the generated-metadata comparison:

```bash
python -m analytics_lab.weapons_evaluation --synthetic-demo
```

The demo deliberately trades a recovered rifle detection for a false positive on a negative fixture. It verifies the evaluation path and makes no real-world accuracy claim.

## Licensing

Unless a file or directory explicitly states otherwise, original project code is proprietary and source-available for evaluation only. Commercial use, redistribution, production deployment, sublicensing, and incorporation into commercial products require a separate commercial license from the copyright holder.

Third-party components remain subject to their original licenses and are documented separately.
