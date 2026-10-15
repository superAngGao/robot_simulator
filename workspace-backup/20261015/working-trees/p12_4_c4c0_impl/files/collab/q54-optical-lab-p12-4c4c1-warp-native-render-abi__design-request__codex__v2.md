# Q54 Optical Lab P12.4 C4c1 Warp Native Render ABI — Review Corrections

- Initiative: Q54 optical computation workflow / P12.4 backend selection
- Stage: C4c1 design request
- Author: Codex
- Version: v2
- Date: 2026-09-20
- Base: `16bc10b76cb08c9154edc01d72657c9e394a93b4`
- Status: revised design request for Claude review

This document supersedes the conflicting parts of C4c1 v1. Unchanged v1
contracts remain in force, including the caller-owned buffer ABI,
deterministic reset, four fixed shade entries, immutable command recording,
profile filtering, C3 provenance, C4c0 activation gates, and the prohibition
on a3b/results/baselines. If wording differs, v2 controls.

The revision closes three review findings:

1. readiness becomes an exact production type with a mechanically enforceable
   identity chain;
2. C4c1 no longer exposes a multi-command compatibility path whose partial
   submission can outlive ordinary Python unwinding; and
3. numerical correctness is judged by an independent CPU canonical executor,
   not by another path through the migrated Warp kernels.

No runtime code is authorized by this document. This is still the C4c1 design
review before implementation.

## 1. Outcome and unchanged scope

C4c1 still produces exactly one production definition of the Warp optical
math and a caller-owned, recorded render command bundle. It proves that the
bundle has the frozen storage, dispatch, provenance, reset, and numerical
semantics required by C4c2.

C4c1 still does **not** provide:

- a public Warp transaction or terminal completion ticket;
- a C4 slot/generation/output lease;
- normal product drain, reclaim, or poison recovery;
- a closed-worker GPU preparation operation;
- host payload/readback or B3 storage;
- a result-capable native lane; or
- any baseline/result artifact.

The following gates remain false and are not edited by C4c1:

```text
closed_worker_runtime_contract.gpu_preparation_enabled = false
native_result_activation.result_capable                = false
a3b result outlet                                       = closed
baseline read/write outlet                              = closed
host-payload arena seal                                 = closed
```

The v1 statement that an ordinary public compatibility driver may execute the
new recorded bundle is withdrawn. The production compatibility executors may
share the migrated kernel definitions, but they do not adopt the recorded
multi-command bundle in C4c1.

## 2. Exact readiness contract moves into production

### 2.1 Ownership

`NativeReadinessEvidence` and `NATIVE_INPUT_SCHEMA_VERSION` move from
`benchmarks/optical/native.py` to the CPU-safe, worker-allowlisted production
module:

```text
optical_worker/native_contract.py
```

That module already owns prepared-input identity registration. Keeping the
evidence type beside the registry makes exact-type and exact-identity checks
available to both candidate backends without importing the benchmark package
or `optics` package initialization in the closed worker.

`benchmarks.optical.native` imports and re-exports the exact production class
and schema constant for compatibility. It must not subclass, wrap, mirror, or
redeclare them. The package export must satisfy:

```python
benchmarks.optical.native.NativeReadinessEvidence \
    is optical_worker.native_contract.NativeReadinessEvidence
```

Existing JSON field names and canonical SHA-256 encoding remain unchanged.
Moving the class must not invalidate previously frozen case/manifest digests.

### 2.2 Exact authority chain

The production module provides backend-specific exact validators over one
common authority registry. A valid prepared input requires all of the
following before any command is recorded or enqueued:

1. the prepared object has the exact approved backend type;
2. `type(readiness) is NativeReadinessEvidence` from
   `optical_worker.native_contract`;
3. the prepared object identity is registered for the current process and
   backend;
4. the readiness object identity is the exact registered readiness object for
   that prepared input;
5. canonical round-trip is performed by directly calling the known exact
   class, not a method discovered on the supplied object:

   ```python
   round_tripped = NativeReadinessEvidence.from_payload(
       readiness.to_payload()
   )
   assert round_tripped == readiness
   assert round_tripped.sha256 == readiness.sha256
   ```

6. the exact prepared manifest identity and its frozen content/topology/
   scene/replay/fixture fields match the readiness evidence;
7. backend, device ordinal, compilation evidence, warmup evidence, and input
   contract match the registered authority; and
8. the current candidate's exact native allocations remain the ones recorded
   by the prepared-input storage authority.

Canonical round-trip establishes payload integrity; the registry establishes
origin and object identity. Neither substitutes for the other.

The approved validators are explicit functions, for example:

```python
assert_warp_prepared_authority(prepared, readiness)
assert_cuda_prepared_authority(prepared, readiness)
```

They use direct imports and exact types. They do not use:

- `__module__` or `__name__` strings;
- `getattr(..., "from_payload")` or `getattr(..., "to_payload")`;
- method-existence or attribute-shape checks;
- subclasses;
- protocols as authenticity proofs; or
- caller-supplied hashes as identity authority.

`optics/cuda_arena.py` is migrated in the same C4c1 slice so the repository has
one rule. Its module/name and duck-type readiness checks are deleted, not left
as a second accepted path. The new Warp factory calls the same production
authority layer before recording commands.

The scene contract schema remains owned by its existing production contract.
The readiness parser may validate its canonical scalar fields, but the exact
scene schema/digest authority is established by comparison with the registered
prepared manifest. C4c1 does not create a duplicate scene-schema truth in the
worker contract.

### 2.3 Required negative coverage

Both backends must reject before command recording and before the first GPU
enqueue:

- a class with the same module/name text;
- a subclass of the real readiness class;
- a duck-typed object with matching methods and fields;
- a copied exact-type evidence object with identical payload but unregistered
  identity;
- an exact registered readiness paired with a different prepared object;
- a copied/re-uploaded storage graph with correct digest strings;
- a mutated payload followed by a recomputed public hash; and
- a benchmark-local redeclaration of the old class.

Regression tests cover current CUDA arena construction as well as the Warp
factory. This migration is not allowed to weaken C4b provenance.

## 3. C4c1 execution containment and partial-submit safety

### 3.1 Public compatibility path remains legacy-lifecycle-shaped

The new module remains the single source of Warp device functions and kernels.
`optics/warp_execution.py` imports those exact definitions so equation changes
cannot drift. However, during C4c1 the public `GpuDeviceBvh*` compatibility
executors retain their existing blocking allocation/event/lifetime structure.
They do **not** call the new multi-command recorded bundle.

Consequently:

- C4c1 adds no public `WarpDirectLightCompatibilityDriver` for the bundle;
- an ordinary caller cannot obtain a bundle and partially submit it through a
  supported compatibility API;
- product code does not claim C4 slot, terminal-event, drain, or poison
  semantics before C4c2/C4d; and
- compatibility execution remains benchmark-ineligible for the already
  documented allocation, bridge, event, and blocking reasons.

Sharing kernel objects means sharing mathematics, not sharing the unfinished
recorded lifetime protocol.

### 3.2 Where the recorded bundle may execute in C4c1

All C4c1 executions of the new recorded bundle—successful validation and
failure injection alike—run in a fixed, supervisor-managed sacrificial child
process. They never execute in the ordinary pytest process or through a
production executor.

The child program is repository source named in an immutable supervisor
manifest. It accepts only a fixed canonical test request. It does not import a
callback, plugin, pickle, dynamic module name, or caller-provided executable.
No raw Warp object crosses the process boundary.

Before the first recorded launch, the child constructs a sealed test guardian
that strongly owns:

- the indexed Warp device and stream owner;
- every prepared native input allocation and stable view;
- local scene, AABB, BVH, ray, geometry, shade, and diagnostic storage;
- every recorded `wp.Launch` owner and module/kernel owner;
- readiness, storage, dispatch, and command evidence;
- any event used solely for the test drain; and
- any host output copied only after successful drain.

The guardian exists before the first command and cannot release individual
members while execution is active or failed.

### 3.3 Success disposition

The sacrificial child executes the fixed command tuple in order. On success it
performs an explicit target-stream/device drain outside benchmark timing. Only
after that drain succeeds may it inspect/copy results, serialize the fixed
test response, release the resource graph, and exit normally.

This synchronization is validation-only. It does not become the C4 product
drain and is not evidence for R0/R1/R2 eligibility.

### 3.4 Failure after any partial submission

The child maintains a monotonic `recorded_launch_count`. A failure before the
first GPU submission may cancel normally. Once the count is nonzero, any
exception, response failure, pipe loss, injected command failure, or drain
failure poisons the whole test graph.

The only legal post-submit outcomes are:

1. supervisor watchdog is armed through the approved abort-start/ACK
   handshake, emergency synchronization of the exact known target device
   succeeds, and the disposable child exits; or
2. watchdog arming fails, device scope is foreign/unknown, synchronization
   raises, or synchronization hangs; the full graph remains quarantined and
   the supervisor terminates the disposable process without ordinary Python
   teardown.

The child never returns a recoverable Python exception to an ordinary caller
after partial submission. It never releases the guardian merely because the
bundle is marked poisoned. It never reuses the process or session.

This is a validation containment mechanism, not an alternative C4 lifecycle.
C4c2 must bind the bundle to the reviewed C4 lease/transaction authority before
any production or closed-worker execution. C4d remains responsible for normal
joint drain/reclaim/poison behavior.

### 3.5 Failure matrix

The fixed bundle has one command for each frozen dispatch position (three for
R0 and up to five for R1/R2 when scene/acceleration are composed later). C4c1
tests its own render command positions and preserves a generic harness capable
of the final five-position tuple.

For every executable position, tests inject:

- failure immediately before the command;
- failure immediately after the command has submitted;
- failure after one or more earlier commands submitted;
- successful commands followed by drain failure; and
- a deliberately hung emergency sync handled by the supervisor watchdog.

Assertions distinguish pre-submit and post-submit paths. After partial
submission they require poison, nonzero submitted-command evidence, retained
guardian ownership, no normal result, and one process-terminal disposition.
A weakref/finalizer marker for representative input/output/launch owners must
remain unset until successful drain authorization or process death. A bundle
poison flag alone is never accepted as a resource-safety proof.

## 4. Independent correctness oracle

### 4.1 Primary numerical authority

`CpuDirectLightOpticalExecutor` is the primary numerical oracle. The test
constructs it from the frozen canonical registry and canonical host pinhole
rays. It does not import or invoke the migrated Warp device functions or Warp
recorded commands.

The asymmetric correctness case uses:

- a non-square camera;
- off-center principal point;
- `fx != fy`;
- non-identity camera rotation;
- asymmetric geometry and primitive order;
- multiple numeric instance and material IDs;
- distinct material/color/light responses; and
- both hit and miss pixels with shadowed and unshadowed hits.

The recorded Warp result is compared directly with the CPU result for:

- row-major ray origins and directions;
- `hit_mask`;
- `range_m`;
- `position_world`;
- `normal_world`;
- `numeric_instance_id`;
- `material_index`;
- `rgb`; and
- `intensity`.

Integer/boolean fields require exact equality. Floating fields use one frozen
absolute/relative tolerance table appropriate to the existing CPU-vs-GPU
precision contract. Tests may not derive or loosen tolerance after observing
the C4c1 result.

### 4.2 Traversal-counter invariants

Traversal counters are implementation diagnostics, so CPU pixel outputs are
not used to invent identical traversal counts. They are validated by explicit
invariants:

- all counters are `int32`, shape `[1]`, independently stored, and
  nonnegative;
- both stack-overflow counts are zero for canonical and boundary fixtures;
- maximum depths are within the frozen legal stack bound and are zero when no
  corresponding traversal occurs;
- with shadows disabled, all shadow primary/overlay counters are reset to
  zero even after sentinel reuse;
- when traversal diagnostics are enabled,
  `shadow_ray_count == directional_shadow_ray_count + point_shadow_ray_count`;
- `shadow_ray_count == occluded_shadow_ray_count +
  unoccluded_shadow_ray_count`;
- visit/test totals are zero when ray count is zero and otherwise satisfy the
  documented nonnegative lower/upper relations; and
- small hand-authored BVH/light microfixtures freeze exact counts only where
  the topology and expected traversal sequence can be independently derived.

Counter invariants are defined in production/test contract constants, not
copied from the observed output.

### 4.3 Secondary wiring comparison

Compatibility-executor versus recorded-bundle comparison remains useful, but
only as a secondary wiring regression. It can prove that the compatibility
executor imported the same kernel definitions, chose the same fixed shade ABI,
and preserved profile filtering. It is not a numerical oracle and cannot by
itself satisfy C4c1 correctness acceptance.

The primary CPU comparison must pass independently even if the compatibility
comparison is skipped.

## 5. Revised implementation slices

### C4c1a — production readiness migration

- move `NativeReadinessEvidence` and its schema constant to
  `optical_worker/native_contract.py`;
- re-export the exact object from benchmark modules;
- implement direct exact-type/canonical/registry validators for Warp and
  CUDA;
- remove CUDA arena module/name and duck-type authenticity checks; and
- pass the cross-backend forgery and identity regression matrix.

This lands before the Warp bundle can be prepared.

### C4c1b — single-source render ABI

- move the reusable Warp optical device/kernel definitions to
  `optics/warp_native_render.py`;
- implement the exact caller-owned buffer ABI, deterministic reset, four
  shade entries, evidence, immutable factory, and command bundle from v1;
- point existing compatibility executors at the exact shared kernel objects
  without routing them through the recorded bundle; and
- preserve all C4c0/product activation gates.

### C4c1c — contained execution and correctness

- add the fixed sacrificial child and immutable supervisor test manifest;
- prove guardian retention and emergency-sync/quarantine dispositions for
  every partial-submit position;
- compare recorded Warp outputs to the independent CPU canonical executor;
- validate counter invariants and sentinel reuse;
- retain compatibility-vs-recorded only as a secondary wiring test; and
- produce the C4c1 implementation note and evidence inventory.

No slice may enable `PREPARE_FROZEN_ROW`, create C4 result capability, or emit
a benchmark result.

## 6. Expected file surface

Expected production edits are limited to:

- `optical_worker/native_contract.py`;
- `benchmarks/optical/native.py` and its exact re-export surface;
- `optics/cuda_arena.py` for readiness-validator convergence;
- new `optics/warp_native_render.py`;
- `optics/warp_execution.py` for single-source imports; and
- narrow package exports required by existing callers.

Expected validation-only edits include focused unit/GPU tests and one fixed
sacrificial helper launched under the existing supervisor. The helper is not a
product worker command and is not added to the product closed-worker source
allowlist.

Out of scope remain gate activation, worker GPU request dispatch, C4 arena
integration, host delivery, a3b, baselines, backend selection, and deletion of
either optical backend.

## 7. C4c1 acceptance checklist

C4c1 may be approved only when all of the following are true:

- [ ] benchmark readiness is the exact re-export of the production type;
- [ ] Warp and CUDA authority checks use exact type, direct canonical parsing,
      and registered object identity;
- [ ] no production readiness check uses module/name strings or duck typing;
- [ ] copied, subclassed, same-name, and unregistered readiness attacks fail
      before recording/enqueue;
- [ ] CUDA C4b provenance regressions remain green;
- [ ] reusable Warp optical math has one production definition;
- [ ] compatibility executors share that definition but do not execute the
      recorded bundle;
- [ ] the recorded bundle has no ordinary public compatibility driver;
- [ ] every recorded-bundle GPU execution occurs in the fixed disposable
      supervised child;
- [ ] successful test execution drains before graph release;
- [ ] every partial-submit failure retains the graph until sync authorization
      or process-terminal quarantine;
- [ ] hung sync is killed by the acknowledged supervisor watchdog;
- [ ] independent CPU oracle parity passes for rays, geometry, IDs, material,
      RGB, and intensity;
- [ ] traversal diagnostics pass explicit invariants and reset/reuse tests;
- [ ] compatibility parity is treated only as secondary wiring evidence;
- [ ] product GPU preparation, a3b, results, baselines, host payload, and
      shipping paths remain closed; and
- [ ] no performance claim or backend-selection evidence is produced.

## 8. Review decision requested

Please review whether v2 now provides:

1. a mechanically implementable exact readiness authority shared by both
   backends;
2. a safe C4c1 boundary for partial recorded submissions without pretending
   C4c2/C4d lifecycle already exists; and
3. a genuinely independent correctness oracle.

Approval authorizes only the three C4c1 implementation slices above. C4c2,
C4c3, C4c4, C4d/B3, product worker GPU preparation, a3b, result generation,
and baseline access remain separate review points.
