# Connect fixed render inputs and output storage

Initiative: Q54 / P12.4 / C4 render integration
Stage: implementation, parent section 5 / remaining-setup G2
Author: Codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: runtime-integration design-request codex v3 sections 4, 5, 8;
  remaining-setup gap-scan codex v1 G2; warp-slot-construction code-review v2
Owner Summary: Reuse the exact uploaded fixed scene/world/BVH arrays and existing
  Warp/CUDA output allocators in one source-session storage preparation. This
  connects real production builders without another uploader or allocator.

Base 83013616e4df5cf6bae13094e42ecae9cb529bf3. The preceding output allocation
mechanics were independently approved at fdb86a03bc771e9422d7b5b04e0179b2a792684f.

CanonicalFixedRenderFixtureHost retains the actual stats object from its one
existing host BVH build. The original array-only derivation API delegates to the
same math and discards stats; frozen fixture/content digests are unchanged.
New exact-prepared wrappers verify the existing storage registry, retain the
original prepared/fixture owners, and construct Warp scene/world/BVH or CUDA
scene/first-hit/shade views without copying device data. CUDA tensor-name lists
are shared with the existing tensor_items accessors. R0 does not gain refit
authority. No statistics are fabricated from placeholder values.

SourceSession.prepare_fixed_render_storage is single-use after this session's
existing Warp module or CUDA factory preparation. It accepts only fixed native
input and an exact diagnostic bool. One existing setup guard validates the
original upload/runtime binding and compiled set, wraps its input views and
allocates one complete output bundle. The existing custody path retains views,
every returned array and the aggregate. Explicit target/stream scope precedes
the CUDA availability preamble and all allocation calls. Failure escapes the
session's existing irreversible guarded failure path. Literal source admission
only extends the existing lane-specific closure into two storage phases.

This storage uses frame-zero query scalars in its initial CUDA input view. Later
camera/frame binding must construct the appropriate final per-frame stage; this
is not a recorded frame graph. There are no command recordings, terminal or
diagnostic events, launches, warmups, readiness/origin, arena activation, timing
or result exports. R1 mutable world/acceleration and the complete slot graph
remain work under the parent design; this checkpoint is not their replacement.

Author tests: 291 passed in 18.76s with Python 3.14, -B, pytest -q -p no:cacheprovider:
test_prepared_fixed_render_inputs, test_native_contract_extraction,
test_render_parameters, test_native_output_allocation, test_cuda_artifact_runtime,
test_cuda_scene_native, test_warp_module_preparation. Tests run actual fixed
materialization/digest checks, prepared registry, view/output builders, source
session method, lifecycle and custody; native arrays/scopes and the already
reviewed intake/compilation facts are CPU doubles. Native execution is unproved.

Initial verification found the old relocation AST test already stale for the
two previously extended CUDA factory functions (verified against base a27ce7b).
It now explicitly identifies those and this delta's four new/two changed fixed
input definitions while preserving all other baseline bodies and legacy alias
identities. Existing cold-import and frozen render-content checks still pass.
An initial new test had the CUDA dependency-double flag unset and incorrectly
expected five traversal counters (there are nine; diagnostic event phases are
different). Both fixture/assertion errors were corrected; production behavior
was unchanged. All 84 source files pass unchanged policy; refreshed catalog
SHA256 cf71b91af50a026636691656b094430091337720d813421e29dc3c0cf8da6124.

Fresh independent review and real allocation/scope/failure validation remain
required. The server's separate compilation regressions remain at 4fe6b3a and
have not executed this delta. Full rendering/C4, systematic refactor, performance
and product activation remain incomplete.
