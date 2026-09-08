# Q54 Optical Lab P12.4 C4b CUDA Arena and Transaction Templates

- Initiative: Q54 optical computation workflow / P12.4 backend selection
- Stage: implementation note
- Author: Codex
- Version: v1
- Date: 2026-09-08
- Status: ready for review
- Approved design: `q54-optical-lab-p12-4c4-native-output-arena-leases__design-request__codex__v1.md`

## Owner summary

C4b is implemented for CUDA R0/R1 and R2 `none/device_only`. The candidate now
has an exact, preallocated scene/AABB/LBVH/workspace/render/event graph and a
per-slot/frame reusable validation template whose activated transactions remain
single-use. C3 evidence is built from and checked against the actual CUDA
objects; it is no longer accepted as caller-supplied digest strings.

This commit does **not** open a3b, result validation, baseline generation,
host-payload delivery, S, or backend selection. C4c, C4d, B3, and their joint
activation review remain mandatory.

## Open questions addressed

- C4b CUDA storage ownership and per-generation transaction activation.
- Exact C3-to-C4 provenance binding for the CUDA candidate.
- R0 preservation of the complete a2 packed-scene plus derived-fixture input.
- R1/R2 acceleration-class identity despite their current shared CUDA LBVH
  implementation.
- One terminal event per physical slot and explicit R2 device-only endpoint.

## What changed

### Reusable validation, single-use execution

`CudaOpticalTransactionTemplate` freezes and validates the ordered stages,
stage evidence, execution context, terminal event, and optional diagnostic
events. Activating it creates fresh C2 state; it does not reset a used or
poisoned transaction.

The C4 CUDA target keeps the template private. Callers can inspect canonical
plan/resource evidence but cannot obtain a public template activation entry.
Multiple speculative prepared samples remain legal; only the plan that wins
the arena's FIFO acquire receives the live lease and can enqueue.

### Exact allocation graph

`CudaNativeOpticalArena` allocates before measurement:

- session-owned CUDA local-scene arrays, replay views, and one LBVH workspace;
- per-slot world geometry/AABBs, CUDA LBVH outputs, C2 ray/hit/shade outputs,
  primary counters, optional nine-counter overlay, one terminal event, and
  optional phase event pairs;
- R0's complete packed-scene allocation set and complete derived fixed-fixture
  allocation set.

The physical resource record stores every logical view separately, including
dtype, shape, stride, active byte offset, and active byte count. Shared replay
storage is counted once without losing the identities of its frame views.
Mutable slot allocations are disjoint, and all terminal/diagnostic events are
identity-unique.

### Mechanical C3 provenance

`benchmarks.optical.cuda_arena` constructs the canonical
`C3OneShotNativeExecutionBindingEvidence` from the target's real stage objects
and tensors. The product module does not import benchmark code.

Before issuing C4 provenance, the arena verifies:

- row/case/boundary/lane/profile/shadow/input/readiness identities;
- R0 fixed, R1 dynamic, and R2 static acceleration classes and IDs;
- exact stage order and product operation IDs;
- actual C2/C3 dispatch evidence;
- every storage pointer, dtype, shape, allocation size, active range, and owner;
- the complete ordered C2 render-output ABI.

Provenance is arena-issued, cached once per target, and cannot be reconstructed
from public strings. A foreign same-shape arena binding is rejected.

### Submission and result access

The measured portion performs only C4 state/identity activation plus the
prebound stage calls. The tested timeline is:

```text
R0: render -> finish
R1: acquire -> scene -> CUDA LBVH rebuild -> render -> finish
R2 none/device_only:
    acquire -> scene -> CUDA LBVH rebuild -> render
    -> device-only endpoint -> finish
```

There is no intermediate completion event. `resolve()` waits on the exact
terminal event outside the guarded enqueue. Device channels are reachable only
through a generation-scoped view after resolution; view access fails after the
device-result hold is released.

Any stage/finish/resolve or post-submit binding-stability failure poisons the
C4 control arena without manufacturing completion or reclaim evidence.

## Files touched

- `optics/cuda_timeline.py`
- `optics/cuda_arena.py` (new)
- `optics/cuda_direct_light.py`
- `optics/native_arena.py`
- `optics/__init__.py`
- `benchmarks/optical/cuda_arena.py` (new)
- `benchmarks/optical/__init__.py`
- `tests/unit/optics/test_cuda_arena_contract.py` (new)
- `tests/unit/optics/test_cuda_timeline.py`
- `tests/unit/optics/test_native_optical_arena.py`
- `tests/gpu/test_cuda_native_arena.py` (new)
- `tests/gpu/test_cuda_optical_timeline.py`
- `OPEN_QUESTIONS.md`
- `PROGRESS.md`
- `REFLECTIONS.md`

## Verification

On `192.168.80.101`, in the existing `robot_sim` environment:

- focused C4b/C2/C4a unit and GPU set: 66 passed;
- combined C2/C3/C4 unit and GPU set: 111 passed;
- broad `tests/unit/optics` plus `tests/unit/benchmarks`: 446 passed;
- repository-wide pytest: 1850 passed, 115 skipped, with one pre-existing
  pytest fixture deprecation warning. This run also collected an accidental
  duplicate copy of the C4a unit test under `tests/gpu`; the original canonical
  test was part of the same passing run, and the duplicate was removed before
  commit, so the committed repository test set is a strict subset of the set
  that passed;
- Ruff lint: passed for every touched Python file.

## Known limitations and next boundary

- C4c Warp native render/arena is not implemented here.
- C4d normal drain, poisoned shutdown, quarantine, and final guardian handoff
  are not implemented here.
- B3 host conversion, pinned staging, D2H, ordered async delivery, and external
  host holds are not implemented here.
- R2 host payload rows and all S rows remain unsealable/unsubmittable.
- `gate.json` activation remains false; no benchmark result or baseline was
  generated or inspected.
- No new external architectural source was required, so `REFERENCES.md` is
  unchanged.

## Review request

Please review especially:

1. whether hiding template activation behind arena-issued prepared/live objects
   fully preserves generation/event reuse authority;
2. whether the physical-allocation plus logical-view manifest closes shared
   replay-view ownership without double-counting memory;
3. whether the benchmark-to-product C3 adapter direction and exact storage
   cross-check prevent receipt/runtime divergence;
4. whether R0's full prepared scene/fixture retention matches the approved C3
   provenance contract;
5. whether C4b correctly stops short of C4d/B3/a3b activation.
