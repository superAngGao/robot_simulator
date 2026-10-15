# Q54 Optical Lab P12.4c2 Native CUDA Execution Timeline

Author: Codex

Date: 2026-08-15

Status: P1 fixed-diagnostics-ABI revision implemented for Claude re-review; no
a3b lane, sample, or baseline is authorized

Approved C2 design:
`ae7cdcb60ac799567988ffb6b7435253b0f07d00`

## Outcome

P12.4c2 implements the independently reviewable native CUDA render slice from
the approved v2 design. A prepared renderer binds stable Torch-owned CUDA
inputs and caller-owned output buffers to one indexed Torch device/stream. The
render stage enqueues raygen, first-hit, and optional direct-light shade without
allocation, bridge creation, extension loading, or host/device synchronization.
The complete transaction owner records the sole optical terminal event with a
separate `finish()` call after the final registered stage.

This commit does not add Torch-native scene update, acceleration, reusable
arenas, output leases, RGB8 packing, D2H delivery, a benchmark submit API, or a
shipping CUDA path. C3, C4, and B3 must adopt the same transaction before a3b
can become eligible. No performance result was generated or inspected.

## Shared transaction timeline

`optics/cuda_timeline.py` now owns:

- `CudaOpticalExecutionContext`, which requires an indexed CUDA device and a
  strongly owned Torch stream on that exact device;
- scoped target-device/current-stream activation with ambient device and
  current-stream restoration;
- a frozen, non-empty, identity-unique and name-unique ordered stage list;
- validation of pre-created, initialized, target-device terminal and optional
  diagnostic events;
- ordered single-use `enqueue_stage()` transitions and one owner-controlled
  `finish()`;
- fail-closed poisoning after an enqueue/protocol/event-record failure;
- immutable completion evidence containing device, stream handle, ordered
  stage names, stage-specific execution evidence, strong resource owners, and
  optional deferred timing pairs.

Render `enqueue()` never records or exports terminal completion. `finish()` is
rejected before all stages enqueue, after a prior finish, or after poison. It
records the caller-prepared event once on the selected stream and returns
`recorded=True`; it does not claim physical completion or wait on the host.
Diagnostic events remain separate and are resolved only by an external caller
after terminal completion.

## Bound native render contract

`optics/cuda_direct_light.py` adds explicit native input/output contracts and
the prepared `CudaDirectLightNativeRenderer` / bound
`CudaDirectLightRenderStage` API. Binding validates:

- exact device, dtype, rank, shape, contiguity, count, capacity and kernel index
  range for every input and output;
- complete-or-absent traversal diagnostic overlays;
- disjoint input/output and output/output storage;
- camera, lighting, role-mask, shadow and profile scalars;
- presence of the required prepared extension entries.

The binding freezes both tensor storage identity and execution identity. It
therefore rejects tensor resize/replacement and changes to renderer/module,
input bundles, output bundle, camera, output profile, shadow state, RGB
constants, or shadow bias at transaction construction and again immediately
before enqueue. `CudaDirectLightStageEvidence` binds the terminal ticket to the
actual operation, ray source, output profile, shadow state, ray count, and
diagnostic-overlay state so a later receipt cannot reinterpret one render as a
different gate row.

The small `allocate_cuda_direct_light_buffers()` helper exists only outside
native enqueue for tests and the blocking compatibility facade. It is not an
arena and is not used as evidence of C4 completion.

## Out-only extension surface

The extension version is bumped to `robot_sim_cuda_direct_light_v8` and exports
only fixed-signature entries:

- `pinhole_camera_rays_out`;
- `first_hit_rays_out`;
- `shade_direct_light_no_shadow_out`;
- `shade_direct_light_no_shadow_diagnostics_out`;
- `shade_direct_light_with_shadows_out`.
- `shade_direct_light_with_shadows_diagnostics_out`.

The two base shade entries accept no traversal counter argument. The two
diagnostic entries accept nine fixed-position tensor arguments. Binding freezes
the exact callable selected by shadow and diagnostics state together with one
immutable positional-argument tuple. Enqueue only invokes that prebound pair;
it constructs no Python list/tuple or C++ container. The C++ implementation
uses only a fixed nine-element stack pointer array. It contains no
`std::vector<torch::Tensor>` or `<vector>` include. The diagnostics-off entries
therefore need neither dummy tensors nor a fallback container.

All four accept caller-owned contiguous CUDA outputs, validate their related
tensor contracts, install `c10::cuda::CUDAGuard`, and launch on
`at::cuda::getCurrentCUDAStream(anchor.get_device())`. The source contains no
Torch tensor allocation. Counter initialization uses same-stream
`cudaMemsetAsync`, including empty-ray submissions. First-hit explicitly writes
every miss field; shade overwrites every RGB/intensity value; each enabled
diagnostic has distinct storage.

No first-hit/shade barrier remains. Stream order supplies the dependency, and
the only primary optical event is recorded by transaction `finish()` after
shade (or after first-hit for geometry-only work).

## Compatibility and eligibility boundary

The existing `CudaDeviceBvhOpticalExecutor` and
`CudaDeviceBvhDirectLightOpticalExecutor` now adapt their Warp-owned inputs and
allocate outputs before constructing a render-only transaction. They call the
same native `enqueue()` plus `finish()` implementation, wait only on the
returned terminal event, and preserve current output profiles, diagnostic
overlay, and blocking `ready_event=None` behavior.

They remain benchmark-ineligible because their preparation still includes
Warp readiness waits, Warp/Torch views, uploads, allocation, extension
preparation, and a terminal host wait. The a3a audit now resolves and wraps the
real native render-enqueue and transaction-finish callables. A real GPU test
runs those callables inside the timing guard and observes both high-level
operations with no forbidden bottom-level contamination. Phase authorization
remains empty and no lane gains `submit_sample()`.

## Failure and ownership boundary

Validation before transaction construction enqueues no work. Once a stage
begins enqueue, immediate launch failure, illegal bound-resource mutation,
ordering violation, diagnostic event failure, or terminal record failure
poisons the transaction and exposes no completion ticket. C2 deliberately does
not recycle a poisoned output/event slot; C4 must provide its final lease,
drain, and reuse disposition.

C0 physics publication completion remains a separate input-lifetime event. It
is not accepted by this API as the optical terminal event and is not inserted
into an optical completion ticket.

## Cleanup contract

The `cuda_optical_executor` P12.4e cleanup entry now includes the native
timeline, renderer, dedicated unit/GPU tests, and their symbols. If Warp wins,
these surfaces and exports are deleted with the CUDA optical lane. If CUDA
wins, the compatibility-only bridge/wait path must still be collapsed after
C3/C4/B3 establish the selected native product path. This implementation does
not create a permanent dual-backend abstraction.

## Verification

Commands run in the existing `robot_sim` environment:

```text
PYTHONPATH=. pytest -q \
  tests/unit/optics/test_cuda_timeline.py \
  tests/unit/benchmarks/test_optical_benchmark_a3a.py \
  tests/unit/benchmarks/test_optical_benchmark_contracts.py \
  tests/unit/benchmarks/test_optical_benchmark_core.py \
  tests/gpu/test_cuda_optical_timeline.py \
  tests/gpu/test_optical_benchmark_native.py \
  tests/gpu/test_optical_gpu_runtime.py

PYTHONPATH=. pytest -q tests/unit
PYTHONPATH=. pytest -q tests/gpu

python -m ruff check \
  optics/cuda_timeline.py optics/cuda_direct_light.py optics/__init__.py \
  benchmarks/optical/lanes/audit.py \
  tests/unit/optics/test_cuda_timeline.py \
  tests/gpu/test_cuda_optical_timeline.py \
  tests/gpu/test_optical_gpu_runtime.py
git diff --check
```

Observed before this implementation commit:

- P1-focused timeline, a3a, benchmark, raygen, profile, shadow and
  compatibility regressions: 118 passed;
- exact optical GPU files after the fixed-ABI revision: 56 passed;
- full unit suite after the fixed-ABI revision: 1095 passed, 20 skipped;
- parent C2 full GPU suite before the ABI-only revision: 324 passed;
- Ruff and the extension rebuild: passed.

No benchmark sample, timing artifact, baseline, threshold decision, backend
selection, product-default change, or dependency installation occurred.

## Review request

Please review whether:

1. registered stages cannot finish early, enqueue out of order/after finish, or
   expose a ticket before the single terminal event record;
2. device/stream selection and restoration hold for native extension work, not
   only a synthetic stage;
3. binding validation and frozen stage evidence prevent buffer/scalar/profile
   mutation or row reinterpretation after preflight;
4. all extension entries are truly caller-owned/out-only, same-stream, and
   overwrite every required per-ray/scalar output without an intermediate
   synchronization;
5. compatibility callers preserve behavior while remaining visibly
   benchmark-ineligible;
6. runtime audit evidence comes from the wrapped real native callables and the
   a3a executable boundary remains closed;
7. failure handling avoids false completion/reuse claims and leaves C4, C3,
   B3, a3b, and shipping ownership outside this slice;
8. the cleanup ledger produces a clean repository under either terminal backend
   decision.

The requested P1 re-review should additionally confirm that the timed shade ABI
has no dynamic tensor container, diagnostics-on binds nine fixed ordered
arguments, diagnostics-off has no counter/dummy/fallback argument, and both
paths use the frozen callable selected before transaction enqueue.
