# C4c2a: one production owner for native input and provenance contracts

Initiative: q54-optical-lab-p12-4c4c2a-native-contract-extraction
Stage: design-request
Author: codex
Version: v1
Date: 2026-09-29
Status: in_review
Related Files: C4c arena design v7; C4c1c independent review v8;
optical_worker/native_contract.py; benchmarks/optical/native.py;
benchmarks/optical/receipts.py; benchmarks/optical/timing.py;
optics/cuda_arena.py; optics/device_bvh.py; optics/__init__.py
Owner Summary: First bounded C4c2 prerequisite: move existing contracts once,
preserve their behavior and legacy imports, and keep every execution gate shut.
This is not the complete Warp timeline/worker design or its implementation.

Base: `fd18b87` (evidence-only archive following independently approved C4c1c
code `6d57fa789cb50e4191239a4d00f628f25824bb46`). The C4c1c report approves
only that validation checkpoint; earlier C1a/b approvals retain their own
scope. The owner's continuing C4 initiative permits proposing the next slice,
not skipping its independent design/code review.

## 1. Why this prerequisite exists

The closed worker cannot load `benchmarks.optical` package initialization,
instrumentation, coordinator or runner. Exact prepared-input/readiness types
already live in production `optical_worker/native_contract.py`, but their
canonical host source, manifest, validation/certification, C3 dataflow receipt
and timing context still live in the benchmark package.

Specific current dependencies:

- `native.py` imports benchmark cases, eligibility, receipts and timing;
  eligibility imports runtime instrumentation. Merely re-exporting a class in
  another file does not remove this dependency.
- Its host fixture builder imports `optics.device_bvh`, which imports Warp.
- `optics/__init__.py` eagerly imports both optical implementations even when a
  caller requests only CPU scene contracts or one Warp module.
- CUDA C4b's `_canonical_c3_binding` checks benchmark module/type-name strings;
  `_validate_c3_adapter_authority` invokes a method obtained from the caller's
  object. Relocation must update the real consumer, not preserve the old string
  through a fabricated `__module__` value.
- Runtime preparation and recording helpers have genuine timed-boundary
  assertions. Moving their implementation must not create a second ContextVar
  or turn those assertions into no-ops.

The deliverable is a benchmark-independent **contract entry surface**, not a
complete closed-worker GPU import closure. Go2 materialization still depends
on the example MJCF importer; camera construction, final warmup ordering,
guarded normal waits and the full worker graph need the subsequent C4c2
integration design. They are explicitly not certified by this extraction.

## 2. Bounded delivery and exclusions

This slice delivers:

1. Single production definitions for current native input/manifest helpers,
   C3 receipts and their identity registry, and the existing CPU timing context.
2. Single production definitions for the pure frozen case/content identities
   and input/acceleration identity tables those contracts require.
3. A CPU-only owner of the existing host median BVH algorithm and its stats.
4. Deferred, statically enumerated `optics` package exports, preserving its
   explicit public imports without eagerly importing the losing optical lane.
5. Exact C3 type and direct registry checks in the existing CUDA arena.
6. Migration, import-isolation, provenance, timing and affected GPU regression
   evidence, with all previous runtime gates unchanged.

No Warp timeline, arena, worker request, new authority-minting entry, normal-wait
protocol, warmup policy, publication handoff, event behavior, kernel, storage
layout, delivery, native result binding or baseline is added here. No source
allowlist/AST policy relaxation, timeout change, product preparation enablement,
schema/gate threshold change or selection decision is authorized.

## 3. Definition ownership after extraction

| Existing owner | Sole definition owner after this slice | Legacy location |
| --- | --- | --- |
| `benchmarks/optical/native.py` | `optical_worker/native_input.py` | explicit re-exports |
| `benchmarks/optical/cases.py` | `optical_worker/cases.py` | explicit re-exports |
| `benchmarks/optical/workload_contract.py` | `optical_worker/workload_contract.py` | explicit re-exports |
| input/acceleration identity constants in `eligibility.py` | `optical_worker/lane_contract.py` | imports the same constants; eligibility/audit logic stays put |
| `benchmarks/optical/receipts.py` | `optical_worker/provenance.py` | explicit re-exports |
| `benchmarks/optical/timing.py` | `optical_worker/timing.py` | explicit re-exports |
| CPU BVH builder, CPU node/stats types and topology hashing in `device_bvh.py` | `optics/host_bvh.py` | imports/re-exports the same definitions |

`optical_worker/native_contract.py` remains the sole owner of already-approved
prepared types, readiness type and physical-storage/readiness authority. Do not
redefine or relocate those types again. Its registries remain process-local.

The native-input migration includes the existing explicit backend preparation
and wrapper helpers, whose imports are function-local. **CPU-safe describes
import and host-only construction**, not a promise that explicitly calling a
legacy GPU-preparation function performs no GPU work. The closed worker still
uses its separately guarded `native_setup.py`, and its product PREPARE remains
disabled. No unguarded legacy preparation is adopted into the worker.

Move implementations rather than copying them. Old paths resolve to identical
function/class/constant objects. Inventory public exports and private symbols
actually imported by repository consumers before editing; preserve those
aliases explicitly. Do not substitute modules through `sys.modules`, forge
type module names, implement a generic dynamic loader, or retain a second body
behind a compatibility wrapper. Python `__module__` changes caused by the real
move are expected; serialized contract content is not allowed to change.

No runtime-hook/eligibility/coordinator imports are permitted from the new
production owners. `lane_contract.py` takes only pure identity definitions;
`EntryPointAudit`, hook discovery, eligibility evaluation and instrumentation
stay in the benchmark package. New imports must not introduce a reverse cycle.

## 4. CPU BVH and package import behavior

Extract `DeviceBvhBuildStats`, `_HostNode`, `build_host_median_bvh`,
`host_bvh_topology_sha256`, their CPU-only constants and mathematical helpers.
Keep the public stats name for compatibility. The device buffer, recorded
command factories, Warp kernels and upload/refit helpers remain in
`device_bvh.py` and refer to the extracted objects.

The builder's primitive ordering, tie-break, leaf rules, sort/partition
behavior, normalized level layout, dtype, bounds, quality calculation and
topology serialization must not change. Compare normalized AST bodies for the
moved mathematical implementation with the base revision, and rerun frozen
fixture digests. Reuse or move the CPU canonical-hash helper necessary for
topology serialization without importing `device_scene` to hash host data.
There remains one host median algorithm, not a worker-specific implementation.

Replace eager `optics` exports with a literal, source-enumerated `__getattr__`
dispatch whose branches use static relative imports. Preserve the existing
`__all__` names, explicit `from optics import Name`, repeated attribute identity
and `dir()` visibility. A missing name raises `AttributeError`. Cache only the
resolved known exports; no caller-provided module/path/callable is interpreted.
No `importlib`, `eval`, `exec`, `runpy`, filesystem module discovery, wildcard
dynamic import or reflection fallback is introduced. `from optics import *`
may intentionally request all public exports; it is not a permitted worker
entry and does not count as a single-lane cold import.

The exact package initializer is source-frozen like any other code. Lazy
exports are an import dependency mechanism, not a security sandbox. Existing
physics/sensing initializers are not rewritten preemptively. Their actual
transitive CPU import graph must be observed; if satisfying the guarantees
below requires another package/runtime redesign, return for a bounded design
amendment rather than silently enlarging this slice.

## 5. Identity, evidence and timing invariants

The evidence direction remains:

```text
frozen case/content -> one packed source -> manifest/exact allocations
-> local stage/storage facts -> C3 binding -> registered live binding authority
```

Neither relocation nor canonical round-trip mints live authority. The C3
registry has exactly one definition, with the same weak-reference cleanup and
strong target/prepared/readiness identity tuple. Re-exporting it cannot create
a second registry. Parsed, copied, replaced or same-named receipt objects are
still unregistered unless produced by the existing exact adapter path.

Change CUDA arena validation to import the real production
`C3OneShotNativeExecutionBindingEvidence` and require `type(binding) is` that
class. Canonical parsing calls the known class's implementation, not a method
resolved from a supplied type. Invoke the production registry validator
directly with the actual binding, target, prepared input and readiness. Do not
trust a caller-provided authority method. Preserve every existing content,
ordered ABI, acceleration-class, device, storage/view and C4 plan check.

The benchmark adapter remains responsible for constructing its C3 binding from
real resources; it merely imports the shared types/registry. This slice does
not create a Warp binding adapter or a C4 result-capable receipt. The public C3
builder stays one-shot and non-result-capable.

Move the entire existing pure-CPU timing implementation once, including its
ContextVars, runtime-scope identity registry, types and assertion functions.
Both import paths reference the same objects. Entering a boundary through
either path must reject forbidden work invoked through the other; exception
unwinding restores exactly the prior context and scopes. Do not install two
guards, proxy to a disabled guard, or infer scope from a boolean/hash.

Preserve constructor/function signatures, schema strings, field order,
canonical JSON, row meanings and hashes for equal logical input. All five
frozen case hashes/content contracts and applicable replay/fixture hashes must
remain unchanged. Physical identity hashes remain runtime-specific, as before.
No device data is copied/uploaded merely to migrate a contract.

Instrumentation entry resolution must point to the actual implementation where
an entry moved. Retain the supported external alias inventory and verify actual
hook observations from both call routes; no stale source token or old alias
alone proves coverage. Source/coverage digests may naturally change with source
relocation; that does not authorize changing result semantics or activation.

## 6. Import boundary and existing supervised tests

Fresh subprocess tests import each new production contract entry and exercise
host-only scene/fixture/replay construction. They must observe:

- no repository `benchmarks.*`, `examples.*` or `tools.*` module;
- no attempted `torch`/`warp` import or CUDA extension load;
- no optical CUDA/Warp implementation module loaded as a package side effect;
- no GPU allocation, stream/event creation or initialization used to decide
  whether an import was CPU-only.

Use an import-attempt recorder that records rejected attempts even if a
`try/except ImportError` would swallow them; checking only final `sys.modules`
is insufficient. The recorder is test code, not a new product import policy.
Canonical host scenes used here are built with the same production packer and
frozen data; no test-only replacement packer is allowed.

Runtime-specific helper calls are tested separately in the existing real GPU
environment. A Warp-only leaf import must not import CUDA optical modules or
Torch merely because of the package initializer; CUDA-specific explicit calls
retain their actual dependencies and are not relabeled Warp-only.

The C1c test child intentionally has its own frozen test source closure. Extend
that enumerated closure for the new single-definition files actually imported,
while retaining byte-hash admission and no source/bytecode fallback. Update
tests that instrument a defining module to address its actual moved owner.
Do not weaken negative assertions to make an old monkeypatch target pass.
This is not permission to add these files to C4c0's product allowlist or import
the C1c test loader/driver from production.

## 7. Acceptance matrix

Implementation review must inspect the actual moves/call graph and require:

1. **Unique definitions and API identity.** Every migrated class/function is
   identical through legacy and new imports; no duplicate implementation or
   registry remains. Unknown lazy export fails without loading a backend.
   Explicit exports preserve their types/callables; both import orders work.
2. **CPU closure.** Fresh subprocess import-attempt and loaded-module checks
   above. Negative controls prove the detector catches both a runtime import
   swallowed by `try/except` and a benchmark-package import. No GPU required
   for this CPU property; mocks do not stand in for later device tests.
3. **Frozen content.** All five case definitions/content maps unchanged; real
   asymmetric, synthetic replay and physics host contracts reproduce the
   frozen hashes. Recovered exact Go2 assets reproduce both scene/fixture
   digests through the existing materializer. Missing assets fail, never skip
   a mandatory identity check or use a substitute. Go2 example imports in this
   separate legacy materialization test do not satisfy the new CPU-entry test.
4. **BVH identity.** AST move comparison, empty/single/multi-primitive, degenerate
   centroids, equal keys, sort/partition and topology hash tests. Existing
   leaf-4/sort acceleration enforcement remains in the runtime factories.
5. **Shared timing state.** Both cross-import directions, nested scopes,
   exceptions, and real forbidden-entry observations. Original C3a source and
   runtime-hook eligibility tests still fail closed.
6. **C3/a2 authority.** Positive real CUDA adapter binding; same-name forged
   receipt, canonical but unregistered receipt, copied receipt, wrong target,
   wrong prepared/readiness pair, ordered storage omission/replacement and
   same-object Torch `set_()`/Warp pointer drift are rejected pre-enqueue.
   The arena's direct registry check must not invoke an injected object's
   claimed authority callback.
7. **Runtime regressions.** Existing C3a/C3b/C3c/C3d/C3e, C4a/C4b, readiness,
   Warp recording/buffer/compatibility and C1c supervised-source/guard tests
   proportional to the affected imports. Real dual-device cases run in
   `robot_sim` with the recorded CUDA 12.8 toolchain; preserve exact device/
   stream restoration and C4 lease uniqueness. No benchmark runs.
8. **Closed gates.** Byte-identical frozen gate/runtime-contract JSON, unchanged
   product worker/launcher/AST/request surface, continued host seal rejection,
   and a3b/result denial before preflight/submit. No C2 timeline or worker GPU
   preparation is reachable merely by importing the new modules.

Report independent results, original regression closures, limitations and
actual clean base/HEAD. A move is not accepted only because two new aliases
agree with one another; retained frozen digests and existing real consumers
are the external anchors.

## 8. Follow-on checkpoints and cleanup

This extraction is C4c2a, a prerequisite within the already planned C4c2 scope.
Approval authorizes only these moves/checks. Before any subsequent runtime
implementation, submit the C4c2 integration design covering:

- fixed worker-side workload/asset/camera materialization without benchmark or
  example package initialization;
- narrow source-policy treatment of existing read-only packed-ABI inspection,
  not a global `ctypes` exemption;
- final readiness/warmup/guardian seal ordering without reopening frozen facts;
- versioned ACK-before-call supervision of setup/resolve waits without silently
  altering C4c0 A/K or retrying failed guarded drains;
- native Warp timeline, 3/5 diagnostic phase pairs and one terminal event;
- exact R0/R1 arena/lease authority and O(1) acquire;
- same-slice transport-failure custody and exact post-submission idle close.

Those decisions cannot be inferred from this CPU extraction. C4c3 static
interop/R2, C4c4 closure, C4d/B3 and the joint activation review retain their
existing order and scopes. No new performance/memory threshold is introduced.

Cleanup classification is explicit: shared selected-backend input/receipt/
timing contracts and host BVH remain only while the selected runtime or durable
regression core consumes them; the frozen case catalog belongs to the durable
canonical workload core. Existing two-lane tables and comparison-only branches
are specialized/deleted at P12.4e with the losing lane. No plugin registry or
permanent dual-backend selector is added. Thin legacy benchmark re-exports are
compatibility paths to the same implementation, not a second data plane; remove
unconsumed aliases in the existing cleanup ledger. The C1c-only helpers retain
their existing deletion classification. C4 completion still cannot be claimed
before its own remaining review and cleanup obligations.
