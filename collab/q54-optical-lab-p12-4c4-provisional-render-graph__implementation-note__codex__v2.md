# Bind provisional CUDA commands to their retained factory

Initiative: Q54 / P12.4 / C4 runtime integration
Stage: provisional render graph correction
Author: codex
Version: v2
Date: 2026-10-10
Status: in_review
Related Files: provisional-render-graph implementation-note v1 and code-review v1;
  runtime-integration design-request v3; warp-module-preparation design-request v2
Owner Summary: Address review F1 using the existing guardian, extension-set and
  runtime-binding checks. No new registry, execution path or gate is introduced.

Previous candidate: 275095edb035135855da73e81b2a43f55543a4b1. Original whole graph
base: 15f64a96224a2ce27dabacf57c9aa6f2f1e288bc. The complete fresh-session review
is preserved verbatim in code-review v1; it requested changes for F1.

The graph now resolves the CUDA renderer from the already adopted immutable
factory tuple, requires custody of its original context/renderer, checks the
renderer context against that tuple, and calls the existing _require_cuda_extension
validator. That validator checks the issued module set and the exact indexed
device, stream object and stream handle. The renderer module must be that issued
module. Before initializing events, resolve those owners again and validate every
retained stage against that context, factory and its actual captured entrypoints.
Thus module/context/factory replacement rejects before binding; inconsistent
returned stages reject before events and stay retained on failure.

The CPU fixture now uses an actual CudaOpticalExecutionContext instance and the
existing extension-set registry/context validator. Native construction and source
intake remain doubles, including the upstream require_cuda_extension lookup.
Ten added counterexamples cover replaced module/context/factory/tuple, stream
object, handle, device, and actual immutable stages constructed with foreign
entries or another renderer, plus a context change during binding. The tests
observe the original error, no initialized events, quarantine and retained owners.
Initial stage fault tests attempted assignment to immutable fields and failed
at that earlier protection (2 failed, 70 passed); corrected tests construct actual
foreign stages through the original renderer method without changing frozen fields.

Author validation: native_render_preparation, native_output_allocation,
prepared_fixed_render_inputs and cuda_artifact_runtime: 199 passed in 41.46s,
tool session 7204 exited 0. The prior candidate's cold bootstrap/source run
19782 completed 281 passed, 13 skipped in 160.35s; that is predecessor evidence,
not revised native acceptance. Source regeneration validates the same 84 LF
sources and unchanged policy/inventory; only native_setup's source digest changed.
Catalog SHA256: d155264973687acd910924c0aa9907bb8a0970bc55bc917e2486e5ff50574845.
Unchanged original reviewer counterexamples remain outside the tree, with a
separate v2 wrapper asserting their rejection. That wrapper and the revised
native_source_policy/native_sources suites: 208 passed, 4 skipped in 44.62s,
tool session 16462 exited 0 (observed live PID 30340, parent 37996).

The storage-only native campaign remains at isolated revision 3d6692b and is not
graph validation. Native recording/events, optical execution, observed warmups,
final readiness, final graph, R1, full C4 validation/refactor and later product
gates remain incomplete and closed as recorded in v1. This correction does not
claim a rendered image or performance result.
