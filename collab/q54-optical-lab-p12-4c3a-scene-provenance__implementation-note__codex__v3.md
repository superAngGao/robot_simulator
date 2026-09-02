# Q54 P12.4 C3a provenance implementation note v3

Status: implementation update for review. This note supersedes v2's
`submission_capability` design; no benchmark result or baseline is produced.

## C3/C4 authority boundary

- The public `c4_result_capable` string and its export are removed.
- `build_c3_one_shot_native_execution_binding()` has no capability parameter
  and returns only `C3OneShotNativeExecutionBindingEvidence`.
- The C3 type and schema are explicitly one-shot and non-result-capable.
- Even if the frozen activation object is manually changed to `result_capable
  = true`, `SingleLaneSampleRunner` canonicalizes and validates the C3 object,
  then rejects it before entering the timed context or calling `submit_sample`.
- `validate_result_record()` rejects any result receipt containing the C3
  type. Current result schema v9 therefore has no successful producer.
- C4 must add a different evidence type and constructor. Its acceptance must
  mechanically bind actual arena, slot, generation, output lease, and ring
  evidence; changing a string or gate value cannot activate the current type.

## Exact dynamic storage ABI

`q54.p12.4.dynamic-dataflow.v1` freezes complete ordered allocation manifests
for the current Warp and CUDA operations:

1. Scene update reads the required local geometry/instance arrays and the
   selected replay rotation/translation allocations, and produces the six
   world-geometry arrays plus both triangle AABB arrays.
2. Acceleration reads exactly both AABB arrays and exposes the seven complete
   BVH result arrays used by traversal.
3. Render reads exactly the world geometry, primitive metadata, all traversal
   BVH arrays, and material/light arrays. This includes `render_only`, because
   the current direct-light renderer still enqueues its shade entry.

The outer binding additionally freezes lane renderer ID, acceleration ID,
output profile, shadow mode, and a dataflow-contract identity. Render-stage
evidence independently records output profile and shadow mode; the outer
builder can only verify those facts, not supply a different identity.

Validation compares the full ordered `NativeStorageSignature` tuple, including
storage and owner tokens. Missing, extra, reordered, same-shaped substitute,
wrong-operation, and wrong-profile evidence fail during canonical construction,
before the runner can enter a timed boundary.

## Verification

- Focused benchmark/provenance unit tests: 78 passed.
- Full benchmark plus optical unit tests: 378 passed.
- Optical benchmark/runtime/timeline/Warp GPU tests: 62 passed (440 total).
- Ruff targeted checks and `git diff --check`: passed.
- Frozen Go2 assets remain unavailable on this server; the existing asset
  preflight still fails closed and no Go2 digest or baseline was fabricated.
