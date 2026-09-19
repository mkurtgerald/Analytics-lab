# BDD100K tracking evidence review — rejected for current commercial acceptance path

## Decision

BDD100K is **not admitted** as real multi-object tracking acceptance evidence for Analytics Lab's commercial product-development path under the currently published data terms. Do not download or benchmark the dataset for this acceptance item unless an owner-authorized commercial license or qualifying BAIR Commons membership/affiliate right is established separately.

This is a data-rights decision, not a judgment on technical quality. BDD100K is technically relevant: its official repository describes multi-object detection tracking among the supported tasks. The blocker is the rights grant for the underlying downloaded data and labels.

## Pinned authoritative source

- Repository: `bdd100k/bdd100k`
- Reviewed revision: `9ac17c6c7c51d2fc83065fccd707cd5b1882a293`
- Data-license source: `doc/source/license.rst` at that revision
- Code repository license: BSD-3-Clause, which does **not** override the separate data/label terms

The official license document separates repository code/resources from the downloaded data and labels. It permits educational, research and not-for-profit use generally, while commercial use is granted to BDD and BAIR Commons members and their affiliates; it directs other commercial users to UC Berkeley's Office of Technology Licensing for commercial licensing opportunities.

## Why this matters to the evaluator

A permissive code-repository license, public download URL, or annotation-format license is not sufficient evidence that the underlying image/video bytes may be used for commercial product development. For that reason, `TrackingEvidenceManifest` now requires both:

1. an authoritative `rights_source`, and
2. an explicit `commercial_evaluation_authorized=True` affirmation after review of the rights applying to the actual evidence bytes.

If commercial rights are absent, conditional, unknown, or depend on a membership/license that has not been established for Analytics Lab, admission fails closed before multi-object metrics are accepted.

## Resource decision

No BDD100K media, labels, models, weights, or bulk archive are downloaded by this work item. No accuracy result is produced. This avoids consuming hosted bandwidth/storage on evidence that cannot currently support the commercial acceptance claim.

## Revisit condition

Revisit only if the owner establishes a separate commercial license or other documented right that clearly covers this use. Any future admission must still bind the exact dataset/version, sequence, annotation SHA-256, ordered frame-manifest SHA-256, attribution, exhaustive multi-object scope and the authoritative commercial-rights source.
