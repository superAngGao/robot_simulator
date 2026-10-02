# C2c4 versioned render-parameter binding migration

Initiative: q54-optical-lab-p12-4c4c2c-render-binding-migration
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: runtime-integration design v3 at674af14 sections3/3.1;
C2c3 approval c8b86e7 and evidence9c49722; schema.py; provenance.py;
cuda_direct_light.py; warp_native_render.py; render_migration_v12/v13.json
Owner Summary: Implement the approved v13/v8 migration and untimed native scalar
checks. This is a bounded C2c checkpoint, NOT completed C2/C4 or worker activation.

Base: 9c49722. This note and the candidate are claims for independent review.
Do not interpret the version increment as permission to generate results.

## Delivered scope

- Gate/result v13, row semantics v8, C3 one-shot binding v6 and stage evidence v3.
  The four reproduced sequence identities enter the case catalog; S remains
  explicitly unresolved. Original canonical case hashes, scene/replay/fixture
  digests, all20 mandatory row objects and numerical gates are unchanged.
- Every row keeps the complete old semantic preimage and gains the resolved
  render contract. Historical v12 and current v13 complete preimages/hashes are
  test fixtures. The former is readable data only; no executable v12 fallback.
- The independent clean single-lane worker measurement declaration replaces
  same-process paired residency. No paired comparator runs in this change.
- Stage -> aggregate binding -> receipt references form a one-way parameter
  chain. Canonical parsing checks frozen sequence, scene/case, profile, shadows,
  camera/frame/time and exact binary32 bits. Runner checks request sample/pass
  identity before the still-unconditional C3 result rejection.
- CUDA stage evidence observes the actual bound raygen/shade ABI and first-hit
  scalar configuration. Its existing dispatch/plan hash includes these facts;
  the existing exact a2 adapter checks them against the frozen sequence and the
  actual packed scene role table before issuing registered C3 provenance.
  Arena binding checks the same relation. Existing target/template/stage
  preflight remains untimed; acquire does not gain graph traversal or hashing.
- CUDA immutable render-stage sealing now also protects its seal/baseline.
  Full preflight checks every raygen/shade positional argument. Shadow-off has
  no bias argument: evidence retains the inactive bias configuration, not a
  claim that the kernel consumes it. No dummy storage or new kernel math.
- Warp recorded bundles expose an untimed, non-authorizing canonical check
  that first validates actual packed native descriptors, then compares their
  scalars and actual query metadata to the same record. It does NOT create a
  Warp C4 arena or automatically adopt it into the still-closed dispatcher.
  Future C2e admission must require this check, not just its descriptive hash.

These records remain descriptive outside the registered exact-resource adapter.
Python is the accepted trusted process domain, not a hostile-code sandbox.
No hash, parser round-trip or copied dataclass grants execution authority.

## Validation record at initial candidate freeze

Primary183:40031 timed out. Authorized101 fallback connected; H200 NVL physical2
is first in CUDA_VISIBLE_DEVICES. The pre-existing dirty common worktree is
untouched. Testing uses a separate worktree and existing robot_sim/CUDA12.8.

Local163 CPU parameter/migration tests and changed-file Ruff checks passed.
An earlier local broader run was not a valid runtime-hook check because this
Python lacks Torch/Warp; no dependencies were installed to conceal that gap.
First server development round:103 passed,5 failed,35.73s. Failures were three
error-message assertions after earlier rejection, a sample-index fixture
inconsistent with its new record, and an obsolete2x1 output-shape assertion.
Those were corrected without weakening validation; explicit wrong-sample/pass
and actual native-ABI mutation tests were added. No watchdog expiration occurred.
Round-end own process and GPU records were clear. Full final validation and
independent review are still required and will be archived separately.

Tests must cover all20 semantic preimages, every scalar mutation, metadata,
actual CUDA wrong-camera/ambient and same-shape ABI permutation, actual Warp
packed scalar drift across all profile/shadow/diagnostic variants, unchanged
C4b authority attacks and fixed-cost acquire, plus C0/C1/source regressions.
The Warp tests only record/inspect commands; they do not submit recorded work.

## Remaining work and unchanged gates

Final closed-worker source admission remains a separate C2c integration task;
this change does not expand the current manifest import allowlist or protocol.
C2d supervision/guardian adoption, C2e Warp arena and C2f real two-lane closed
worker acceptance remain required. R0 orbit sample selection in a future arena
must bind the actual per-sample camera; the existing one-shot CUDA R0 adapter
here supports its frozen sample0 only and cannot claim a complete orbit lane.

C4 result binding, a3b, baseline/results, product GPU preparation/submission,
host payloads, shipping, depth-two delivery and B3 remain closed. C4c3/c4,
C4d/B3 joint ownership and final activation reviews remain outstanding.
P12.4e removes losing-lane adapters and comparison-only fixtures; selected-lane
canonical parameters, validation and regression infrastructure remain durable.
No shared collab policy, private agent instructions or historical review is changed.
