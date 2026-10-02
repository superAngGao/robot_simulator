# C2c4 render-binding review corrections

Initiative: q54-optical-lab-p12-4c4c2c-render-binding-migration
Stage: implementation
Author: codex
Version: v2
Date: 2026-10-03
Status: in_review
Related Files: implementation-note v1; independent review v1; optics/cuda_direct_light.py; optics/warp_native_render.py
Owner Summary: Repairs F1-F4 from the independent review of 8b1d13c. This is still only C2c4; no complete C2/C4 or activation approval is claimed.

## Changes and proof obligations

- F1: initial Warp launch sealing compares actual c_float bytes with expected little-endian binary32 bytes, including signed zero. Integer ABI checks remain exact. The recording matrix now attempts to establish a fresh baseline with wrong signed-zero native storage, as well as testing post-seal mutation. No recorded render command executes in these recording tests.
- F2: descriptive CUDA render evidence is built and frozen at binding, not on timed enqueue. Evidence retrieval returns the frozen object. Existing stage stability additionally covers consumed inline first-hit distance/mask/count fields, shade light count and evidence identity. Real acquire/enqueue/finish tests forbid the full validator and evidence builder after preparation; separate scalar-drift tests require rejection before any sample launch/event.
- F3: only the sacrificial test source manifest adds the exact render/camera transitive modules. A cold import through the real package entry validates admission and still rejects an unlisted module. Product source policy and A/K protocol remain unchanged.
- F4: only three provenance definitions plus one new helper are allowed to evolve from the original move. All other moved AST bodies/re-export identities remain protected. The historical gate fixture is checked against the historical revision; complete v13 preimages and unchanged workload/closed gates are checked by the migration suite. Closed product files remain byte-identical. Case catalog tests distinguish the two declared new catalog fields from unchanged workload content.

The v1 reviewer report is archived unchanged in meaning. Its 20-row independent migration proof, four findings, failed/incomplete test rounds and process hygiene are retained rather than replaced by new pass counts.

## Verification status

The first local correction round had 275 passes and one test-environment failure: isolated Python omitted user-installed NumPy. The cold subprocess now receives only parent installed paths outside the repository, retaining isolated startup and the repository import ledger. No dependency was installed. Final local, real-device and independent re-review evidence will be recorded separately at an exact frozen revision.

The primary183 SSH endpoint still timed out; authorized fallback101 uses H200 first and two real A100 devices for multi-device tests. Each test round has bounded owned-process supervision and a post-round process/GPU check. A deadline is not reported as a passing fault test or automatically labeled a GPU hang.

## Boundaries and cleanup

No gate, row, threshold, fixture preimage, product worker admission or request protocol changes in this repair. C2d/e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host-payload and shipping remain closed. Final product source admission and complete native integration are not delivered here.

The frozen canonical camera/render records remain durable selected-backend infrastructure. Temporary two-lane migration/preimage comparison tests follow the P12.4e cleanup obligation already stated in v1; historical review evidence remains documentation, not a live fallback implementation.
