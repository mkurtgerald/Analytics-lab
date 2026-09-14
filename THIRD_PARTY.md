# Third-Party Component Register

Every external library, repository, model, weight file, dataset, code fragment, or other donor component used by Analytics Lab must be recorded here before it is merged into a production candidate.

## Policy

Preferred licenses:
- Apache-2.0
- MIT
- BSD-2-Clause
- BSD-3-Clause

Components under copyleft, source-reciprocal, non-commercial, research-only, custom, or unclear licenses require explicit review before use. GPL, AGPL, LGPL, SSPL, BUSL, Commons Clause, CC-BY-NC, research-only model terms, and unknown licenses are not automatically approved.

## Required fields

| Component | Upstream URL | Version / Commit | License | Purpose | Modified? | Redistribution Requirements | Approval |
|---|---|---|---|---|---|---|---|
| _None yet_ | | | | | | | |

## Rules

1. Never copy donor code without recording its source and license.
2. Preserve all required copyright and attribution notices.
3. Treat model weights and datasets separately from source-code licenses.
4. Do not assume a GitHub repository is reusable merely because its source is public.
5. Do not commit third-party datasets or weights unless redistribution rights are clear.
6. Prefer replaceable adapters around donor components so the project can swap them later.
