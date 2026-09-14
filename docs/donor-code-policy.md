# Donor Code Policy

Analytics Lab may use external open-source or source-available components to accelerate research, but donor code must remain traceable, legally compatible, and architecturally replaceable.

## Preferred donor-code profile

Prefer components with:
- Apache-2.0, MIT, BSD-2-Clause, or BSD-3-Clause licensing;
- active maintenance;
- reproducible releases or pinned commits;
- documented model / dataset provenance;
- clear redistribution terms;
- modular APIs that can sit behind an adapter.

## Review required before use

Do not merge a donor component into a production candidate until the following are documented:

1. Upstream repository and maintainer
2. Exact commit, release, or model version
3. Source-code license
4. Model-weight license, if different
5. Training-dataset terms, if relevant
6. Required attribution / notices
7. Commercial-use permission
8. Modification / redistribution obligations
9. Security and maintenance risk
10. Replacement strategy

## Restricted-by-default categories

The following are not automatically approved:
- GPL / AGPL components
- LGPL components embedded in ways that may create distribution obligations
- research-only licenses
- non-commercial licenses
- Commons Clause or similar commercial restrictions
- SSPL / BUSL or other source-available terms requiring review
- unlicensed repositories
- code copied from articles, gists, forums, or generated snippets with unclear provenance
- model weights whose commercial-use rights are unclear
- datasets without explicit rights for the intended use

## Clean architecture rule

Third-party inference engines should be wrapped behind project-owned interfaces. Analytic-specific temporal logic, event reasoning, confidence fusion, evidence linking, and standardized event output should remain independent of any single donor implementation wherever practical.

## No license laundering

Public availability does not make code open source. Modification does not erase upstream obligations. The project must not relabel donor code as original proprietary IP.

## Record keeping

Every approved donor component must be entered in `THIRD_PARTY.md` before release or commercial evaluation.
