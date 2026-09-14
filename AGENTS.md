# Analytics Lab development instructions

## Mission and scope
Develop original analytics for paid distribution inside the owner's platforms. This is an independent source-available commercial R&D repository, not a free-use product or a fork of the VMS. Preserve LICENSE and all upstream notices. Do not modify other repositories, production deployments, camera settings, home-camera access, self-hosted runners, or other agents' schedules.

The owner delegates routine research, implementation, testing, and narrow validated Analytics-lab merges. No spending, external contracts, licensing changes, physical actuation, or production/commercial release is delegated. An IP/software attorney should review commercial legal terms before release; engineering provenance checks are not a legal opinion.

## Execution loop
Read live main, this file, docs/PROJECT_STATE.md, open PRs, and actual CI. Continue one implementation PR at a time; never create a competing successor. Fix the first demonstrated failure before scope expansion. Use small concrete code and test increments rather than status-only commits or repeated redesigns. Check head freshness before writing; do not force push. Merge only when the exact current head has verified passing required tests and no unresolved regression. An unrun check is unknown, not passed. Do not lower quality gates to get green.

Notify the owner only for meaningful verified milestones, material regressions, or decisions requiring owner access, licensing authority, spending, hardware or commercial-release approval. Do not ask the owner to manage ordinary implementation. A scheduled invocation is bounded work, not a continuously running developer or a guarantee of delivery.

## Commercial and data controls
Prefer mature commercially compatible donor perception components; pin exact code commits and weight hashes, record licenses and transitive dependencies, and preserve notices. Approve code, weights, data, and usage rights separately. Do not introduce GPL/AGPL, noncommercial, research-only, or unknown-license components into the shippable path without approval and a separate rights review. Never assert ownership of donor work. Keep commercial logic isolated behind versioned adapters.

Use authorized intentionally public webcams only where rights support the exact automated analysis, retention, training and distribution. A viewable URL is not permission. No unapproved media download, home/customer footage, credentials, identifiable public footage or secret URLs in commits or CI logs. Synthetic observation fixtures must be identified as synthetic and cannot establish real-world accuracy.

## Acceptance
First milestone: authorized video -> reviewed detector/tracker/pose -> temporal analytic -> evidence-linked event. Current posture replay is only one boundary, not the milestone itself. Do not infer injury, cause, intent or fault from an alert. Require held-out annotated video evaluation (including normal negatives), false alerts per camera-hour, missed-event measurements, latency/resource tests, privacy/security/provenance review and versioned integration before commercial promotion. Public webcams alone do not supply labeled positive events.

Run `python -m unittest discover -s tests -v` and the replay command in docs/PROJECT_STATE.md. Persist concrete results and next executable step with each implementation change. Keep tests off all private/home/self-hosted runners.
