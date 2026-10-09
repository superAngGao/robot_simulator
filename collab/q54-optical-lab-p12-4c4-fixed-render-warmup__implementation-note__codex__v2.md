# Fixed R0 warmup execution and diagnostic readback

Initiative: Q54 / P12.4 / C4 runtime integration
Stage: repaired code and native positive-driver support, before independent review
Author: codex
Version: v2
Date: 2026-10-10
Status: in_review
Related Files: fixed-render-warmup implementation-note v1 and code-review v1; runtime-integration design-request codex v3 sections 4, 5, 5.1, 6, 8, 9; remaining-setup gap-scan codex v1
Owner Summary: Freeze the actual Warp execution plan so an empty replacement cannot masquerade as completed rendering. Extend the existing native driver to execute the production warmups and save final-frame RGB. Native output and performance remain unproved.

Base: 04b74aca5a5f6734e1b7d84ff4fe2eb7014ef796.
Whole warmup base: e43a86c8e63a37da7bd84a5a8f5616559c7e12a1.
Candidate: commit containing this note; exact HEAD belongs in the review packet.
Worktree: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design.

## F1 repair

The unchanged independent v1 report is archived alongside this note. Its original
counterexample remains at ../c4_assets/warmup_review_v1/test_warp_plan_drift.py,
SHA256 ee93f4b16576c3b858ccd3eada07ab048a4e96e87486cfa90f323a3d6d2bccba.

_WarpSetupTransaction now uses frozen slots for its original bundle, derived
ordered command tuples, parameters, runtime scope, events and execution cursor.
Ordinary assignment/deletion and reinitialization reject; internal state changes
use object.__setattr__. Reflection/monkeypatch resistance is not promised.
There is no second renderer, plan registry or new validation framework.
The existing bundle validator still checks packed commands before execution.

Tests reject empty, reordered, foreign and cross-frame plans, cursor/state and
scope replacements, deletion and reinitialization. Positive execution still
issues fifty recorded launches and ten terminal waits. A scratch wrapper at
../c4_assets/test_warmup_original_repro_rejected_v2.py executes the unchanged
original counterexample and requires FrozenInstanceError before any leaf or
completion evidence. Copied transactions remain unregistered on both lanes.
The 86-source v16 catalog is refreshed only for the changed Warp source.

## Existing native driver extension

Two fixed selectors exercise production run_fixed_render_warmups under the
original worker/supervisor: runtime_render_warmup_success and
runtime_render_warmup_diagnostics. Existing input, compilation, storage and
graph checks are reused. An observer wraps the existing resource scope, records
native device/stream state and original leaf operations, then restores the hook.
Exceptions retain the original guarded failure; optional diagnostic writes do
not mask it. The returned evidence must be actually registered and adopted.

One subsequent untimed setup guard reads the completed final RGB buffer to
warmup_rgb.f32 and warmup_rgb.json. The original held owner list retains host
returns immediately, before conversion/serialization. Readback is diagnostic,
not a performance sample, R2 delivery or product result. Raw validation checks
upstream input/graph identities, ten exact guard completions, stream/device
scope and restoration, operation order, final frame/dimensions, byte hash and
finite floats. It does not yet compare numerical output with the CPU oracle.

CUDA warmup_launch is declared immediately before the existing native entry;
its observer returned flag describes the declaration context, not native kernel
return. Real completion depends on the actual terminal wait and completed guard.
Warp's operation context surrounds the actual recorded launch.

## Author verification

Python C:/Python314/python.exe -B, pytest cache disabled. Final existing handles
are terminal, and scoped process checks found no matching remaining invocation.

- Session 25620: 64 passed in 92.27s, exit0: native_render_warmup,
  native_warmup_support, native_render_graph_support and the unchanged-repro
  wrapper. The support fixture uses actual production lifecycle/driver code
  with explicit native doubles; it is not GPU rendering evidence.
- Session 7757: 260 passed, 13 skipped in 78.09s, exit0: native source policy,
  sources, scope observer and CUDA artifact support. Skips are platform/asset
  or pinned-runtime checks, not successful native validation.
- Earlier support run had two failures because the CUDA .cpu() double returned
  its already adopted GPU object. Corrected the double to return a distinct host
  owner; duplicate adoption remains rejected in production.
- Catalog SHA256: 946ec6c70a03bce47178f9f7c40b74e8534714b54c2112ec033742a64b35e16e.
- git diff --check passed before this note. No new GPU run was launched for v2.

The separate eight-case graph-only campaign at 6300531 completed successfully;
its native evidence review is still pending. That campaign launched no optical
kernels and cannot prove this revision's warmups or image correctness.

## Remaining work

Fresh independent code review must close F1 and inspect this native-driver
extension. Then real dual-lane Go2 warmup/readback and required failure/hang
cases must run. Image correctness, fair timing, final readiness/origin, final
graphs/reset/C3/seal/barrier, measured Warp arena/timeline, R1 and complete C4
lifecycle integration are still outstanding. Existing product, result and
baseline gates remain closed. The owner's systematic whole-C4 refactor and
fresh whole review remain required; this repair is not that refactor.
