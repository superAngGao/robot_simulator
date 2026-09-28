# C4c1b2 deterministic kernels / fixed shade checkpoint

Initiative: q54-optical-lab-p12-4c4c1b2-fixed-shade
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-09-28
Status: in_review
Related Files: C4c1 design v1/v2/v3; optics/warp_native_render.py; optics/warp_execution.py
Owner Summary: Review only deterministic kernel writes and fixed shade ABI; not the recorded bundle.

## Scope and implementation

Base is 48bbc20, following the independently accepted aa97f09 storage checkpoint.
This bounded continuation implements the approved per-ray miss initialization,
one-thread scalar resets, and four explicit shadow/diagnostics kernel signatures.
No camera factory, recorded command bundle, worker driver or C4 activation is
included. Those require subsequent implementation and independent review.

First-hit writes hit/range/position/normal/instance/material defaults before any
intersection. Three reset entries write the two BVH counters, two shadow counters,
or two shadow plus nine traversal counters respectively. Resets have no array
allocation, fill call or runtime diagnostics branch.

The four shade entries are production kernel objects with ordered ABI tuples in
one registry. No-shadow entries omit scene traversal inputs and shadow bias;
diagnostics-off entries omit all nine overlay arguments. Shared device functions
implement light sampling, color accumulation and traversal; there is no second
compatibility math implementation. Traversal returns local counters in a fixed
Warp struct, and its caller aggregates into the caller's independent scalar
storage. The former runtime-flag shade kernel is removed, including its fallback.

Compatibility execution selects the fixed entry and constructs its argument tuple
outside any eligible benchmark path. Its existing allocation, completion event
and wait policy remains in place. It does not execute a recorded bundle or claim
the unfinished C4c1c partial-submission lifetime. Legacy reduced-profile hit-mask
initialization is retained, although first-hit now also initializes that field.

## Validation performed by the author

On authorized fallback 101, `/data3/ang.gao/robot_sim_c4_work`, conda `robot_sim`:

```text
python -m pytest tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py tests/gpu/test_optical_gpu_runtime.py -q
75 passed in 5.44s
ruff check [the five changed Python files]
All checks passed
```

The new twelve-row shadow/diagnostics/profile matrix uses actual compatibility
launches on a nondefault stream and independent CPU full outputs. It inspects
physical outputs even when a profile filters them, including a miss after
sentinel contamination, different materials, directional and point lights. It
checks diagnostic channel selection, disjoint counters and count invariants.
Three real GPU tests repeat the one-thread reset kernels from nonzero sentinels.
These are individual-kernel/compatibility tests, not execution or fault testing
of a recorded bundle. AST tests bind the declared ABI to each real kernel and
ensure wrappers use the shared device functions.

The later C4c1c supervised full camera/recorded matrix, including failure paths,
is still required. This checkpoint cannot substitute for it. Full regression
and a fresh-session independent review follow candidate freeze.

## Gates, history and cleanup

No gate, closed-worker request surface, GPU preparation authorization, result,
baseline, a3b, host delivery or shipping capability changes. Existing C4/B3 joint
activation requirements remain in force. The completion ledger records the
owner's instruction to continue through C4 without altering those requirements.

183 remains unreachable; fallback work preserves the original dirty trees.
Recovered b1 content has verified byte provenance, not recovered original commit
ancestry. Reconcile those histories before integration when 183 returns.

These kernels belong to the Warp optical lane: retain its single source if Warp
is selected, or remove it with that lane under the existing final cleanup ledger
if CUDA is selected. No new permanent comparator infrastructure is introduced.
