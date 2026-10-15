# C4c1b1 Warp kernel single-source extraction — implementation note

Author: Codex
Status: review request
Scope: first bounded part of approved C4c1b; **not C4c1b completion**

## Change

The Warp optical kernels and their Warp device-function dependencies have moved
from `optics/warp_execution.py` to production module
`optics/warp_native_render.py`. The compatibility executors import the exact
kernel objects from the new module. The equations and legacy invocation paths
are unchanged. A normalized text comparison of the entire `if _HAS_WARP`
kernel/function block up to its CPU-only fallback is byte-identical before and
after the move (53,156 characters). The new module remains importable without
optional Warp.

This deliberately includes the brute-force and device-scene first-hit kernels:
their functions share the existing helper graph with BVH direct-light kernels.
Keeping that graph together avoids duplicate Warp device-function definitions
and is a mechanical relocation rather than a new renderer implementation.

`tests/unit/optics/test_warp_native_kernel_source.py` asserts exact object
identity and rejects duplicate legacy definitions of the frozen render entries.

## Verification

- Target: `183.166.183.33` (`hd03-gpu2-0031`), H200, fresh exact
  `robot_sim` CUDA 12.8 environment.
- Warp GPU optical executor/runtime/C3 command regressions: `60 passed in
  212.08s` on `CUDA_VISIBLE_DEVICES=0,1`.
- New source ownership unit tests: `2 passed in 4.37s`.
- Ruff check and format check: passed.
- `git diff --check`: passed.
- Full non-slow suite: pending before commit.

## Scope boundaries and next review

No caller-owned buffer ABI, deterministic reset, four fixed shade entries,
render evidence, immutable factory, or recorded render bundle is present yet.
C4c1b2 must add these before C4c1b can be considered complete. C4c1c's
supervised sacrificial execution and independent CPU oracle remain later work.
No product GPU preparation, a3b, result, baseline, host payload, or shipping
path has been activated.

Please review only this source-ownership extraction and compatibility parity;
do not interpret approval as C4c1b/C4c1c approval.

## Key reasoning

The alternative was to extract only BVH direct-light kernels and leave helper
functions in the legacy module. Warp kernel compilation depends on those
helpers as module-level device functions, so that approach risks a hidden
second mathematical source or cross-module compilation drift. Moving the
whole dependent kernel block mechanically keeps one definition. The main
debugging check was that CPU-only imports need stubs for raygen and the BVH
mask initializer as well as the previously defined legacy stubs; the ownership
test and GPU regression suite exercise both import and compiled identities.
