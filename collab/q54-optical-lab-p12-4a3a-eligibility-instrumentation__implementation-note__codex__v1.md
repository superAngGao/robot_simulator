# Q54 Optical Lab P12.4a3a Eligibility and Instrumentation

Author: Codex

Date: 2026-08-13

Status: P1 runtime-instrumentation revision implemented for Claude re-review

Approved dependency-order contract:
`57073b8cabe39d7e4104d8cef77fce70fc17956c`

## Outcome

P12.4a3a is implemented as a fail-closed audit/schema slice. It does not wrap
the current allocating, synchronizing, or bridging product calls in a fake
benchmark lane. It does not execute a performance row, write a sample, inspect
a baseline, or claim shipping coverage.

The current Warp and CUDA audit adapters intentionally expose only
`preflight_sample(...)`. They do not expose `submit_sample(...)`. Both current
product paths are ineligible under the frozen contracts and remain so until the
approved C0/C2/C3/C4/B3 prerequisites replace the audited invalid entry points.

## Frozen identities and phase authority

Gate schema v6 binds the approved exact acceleration identities and the
runtime-hook registry digest:

```text
q54.p12.4.runtime-hook-registry.v1
3edfec9d9bb11c16b2bd2af17a472f0928f4372c11c5e2ae13cd614089591985
```

```text
Warp production_dynamic / production_physics:
  cpu_median_leaf4_topology_warp_refit_v1

CUDA production_dynamic / production_physics:
  cuda_morton_karras_lbvh_rebuild_v1

Warp and CUDA production_static:
  cuda_morton_karras_lbvh_rebuild_v1
```

The input contracts are separate:

```text
R0: fixed_native_scene_v1
R1/R2: canonical_native_replay_v1
S: gpu_engine_published_frame_v1
```

The a2 native manifest for S covers only static optical inputs. It explicitly
excludes the measured physics transforms, whose identity must later be proven
by the publication receipt.

Phase authorization is executable, not documentation-only:

- P12.4a3a may execute no result boundary and may emit no result;
- P12.4a3b may execute R0/R1/R2 and may emit only sealed native artifacts;
- P12.4a4 may inspect the native common subset but execute no new rows;
- P12.4b4/d is the first phase authorized to execute S.

`SingleLaneSampleRunner` requires an explicit execution phase and checks this
authorization before prepared-input or lane preflight access, preventing a
replay-backed S sample from being attempted under a3b or an implicit default
phase from granting execution authority.

## Preflight and operation trace

Every future lane must return immutable `LaneEligibilityEvidence` before the
timer begins. It binds:

- row/case/boundary/lane;
- input contract;
- renderer and exact acceleration identity;
- required ordered runtime operations;
- versioned concrete entry-point audits;
- exact external-asset manifest evidence;
- eligibility decision and reason codes.

The runner rejects ineligible evidence before calling `submit_sample(...)`.
It then installs the frozen runtime-hook registry outside the timed boundary;
the lane does not own this step. High-level product entry wrappers write to one
runner-owned `RuntimeOperationTrace`, whose records now bind `capture_source`
and `runtime_hook_id` in addition to entry point, operation, counts, and bytes.
A lane-authored record cannot impersonate a runtime wrapper.

The same scope wraps concrete NumPy, Torch, Warp, DLPack, product staging, CUDA
extension-loader, Tensor materialization/upload, allocation-family, and
host/global synchronization APIs. If one is reached inside timing, the wrapper
raises `TimedBoundaryViolation` before calling the original function. Thus a
lane that reports only `scene_update` or `render` cannot hide a lower-level
`torch.empty`, `wp.to_torch`, host staging, or synchronize operation.
Backend-specific alias hooks are entry-point scoped, so auditing a Warp call
does not import the CUDA optical implementation merely to install its aliases.

`instrumented=True` is not trusted by itself. Eligibility recomputes the exact
registered/verified hook set and manifest digest, while the runner resolves and
installs the same set again. Missing, stale, manually asserted, or unavailable
hooks fail closed before submission.

The trace must begin at and contain the frozen high-level transaction in order:

```text
R0: render
R1: scene update -> acceleration -> render -> reclaim
R2: fresh scene update -> acceleration -> render -> optional pack
    -> delivery -> reclaim
S:  physics step -> publication -> stream wait -> zero-copy binding
    -> scene update -> consumer completion -> acceleration -> render
    -> optional pack -> delivery -> reclaim
```

Missing, reordered, reused, or context-leaking traces are rejected. A lane
cannot fabricate a replacement trace during resolve because the final receipt
must equal the runner-captured trace and digest.

## Source coverage versus runtime observation

The pre-a3b source manifest now hashes exact callable bodies, including helper
methods on the reviewed call graph. It is only a coverage checklist and returns
no verified runtime hooks. It cannot set `instrumented=True` or grant lane
eligibility.

Runtime evidence is separate. Negative GPU tests construct the actual scene,
BVH, executors, render result, and delivery facade outside timing, install the
registry, then invoke the actual product callable inside a timed context. No
high-level lane code reports contamination. The bottom-level wrapper observes
and rejects the first real invalid operation:

- CUDA LBVH: `warp.to_torch` (`warp_to_torch_bridge`);
- Warp direct-light camera executor: `numpy.asarray`
  (`host_materialization`);
- CUDA direct-light camera executor: `warp.synchronize_event`
  (`host_synchronize`);
- synchronous FULL delivery: the real `stage_optical_channels` alias
  (`host_materialization`).

The unit suite separately exercises allocation, upload, host materialization,
global sync, host sync, Warp-to-Torch view, and extension-load hook classes and
proves each fails before its original callable runs.

Confirmed current blockers include:

- Warp camera render allocates ray/result/diagnostic arrays;
- dynamic Lab scene preparation performs allocation and host event waits;
- current sync/readback and RGB8 delivery allocate or materialize host data;
- CUDA optical render creates Torch views of Warp optical buffers, allocates
  outputs, and performs target-wide synchronizations;
- current CUDA LBVH allocates, synchronizes, converts Warp AABBs to Torch, and
  wraps Torch output back into Warp.

These are negative instrumentation tests only. No current product call is
executed as a performance sample in a3a.

## Receipt contracts

`ExecutionReceipt` now binds four lane-owned evidence objects:

- eligibility evidence;
- runner-captured operation trace;
- `AccelerationReceipt`;
- `ShippingPublicationReceipt`, required only for S.

`AccelerationReceipt` freezes topology origin/lifetime, scene and acceleration
operation identities, refit/rebuild classification and counts, primitive/node/
leaf coverage, kernel-launch counts, forbidden allocation/materialization/
upload/D2H/global-sync/bridge counts, completion-event recording, and the exact
operation-trace digest.

`ShippingPublicationReceipt` separately freezes physics run/step/frame/slot,
publish and done events, consumer binding/stream wait, zero-copy binding, last
physics-buffer reader, logical reclaim, physical reuse-safety evidence, drain/
reclaim endpoints, and zero replay/host-poll/host-synchronize counts. A
numerically identical replay frame cannot validate as S because
`input_contract` and `input_provenance_sha256` include the publication receipt.

The coordinator may add only run/repeat/sample/timing metadata. It cannot
populate or rewrite eligibility, execution, native input, acceleration,
publication, operation trace, payload, lifecycle, counters, or memory facts.

## Exact Go2 asset preflight

Go2 rows hard-fail before timing unless all of the following hold:

- the model remains at a reviewed `unitree_go2` path inside a Git checkout;
- checkout origin is the canonical `google-deepmind/mujoco_menagerie` remote;
- manifest revision equals checkout `HEAD`;
- every and only regular model files are covered by sorted size/SHA-256 records;
- those files exactly equal the commit-tracked `unitree_go2` files;
- tracked model files are unmodified and no untracked model file exists other
  than the run manifest itself;
- `scene.xml` is the entrypoint and is included in the digest.

The raw manifest file SHA-256 is simultaneously bound into lane eligibility,
`input_provenance_sha256`, and the result's `artifact_hashes.asset_manifest`.
Missing, copied-from-gate, or mismatched artifact evidence is rejected.

The current clean test host has no Go2 checkout, so Go2 eligibility correctly
hard-fails. A surrogate directory with a self-consistent manifest cannot pass.

## Cleanup ledger

The ledger now names the temporary Warp and CUDA comparison-lane adapters.
P12.4e must delete the losing adapter and collapse the selected adapter into one
thin durable regression lane. The reusable eligibility, asset, receipt,
operation-trace, runtime-hook registry, runner, case, validation, and artifact
contracts remain in the durable core only while the selected-backend
regression remains active. The current source manifest and comparison-only
tests remain explicitly temporary.

## Verification

```text
conda run -n robot_sim ruff check benchmarks/optical \
  tests/unit/benchmarks/test_optical_benchmark_contracts.py \
  tests/unit/benchmarks/test_optical_benchmark_core.py \
  tests/unit/benchmarks/test_optical_benchmark_a3a.py

conda run -n robot_sim ruff format --check benchmarks/optical \
  tests/unit/benchmarks/test_optical_benchmark_contracts.py \
  tests/unit/benchmarks/test_optical_benchmark_core.py \
  tests/unit/benchmarks/test_optical_benchmark_a3a.py

conda run -n robot_sim python -m pytest tests/unit/benchmarks/ -q
conda run -n robot_sim python -m pytest tests/unit -q
conda run -n robot_sim python -m pytest \
  tests/gpu/test_optical_benchmark_native.py \
  tests/gpu/test_optical_gpu_runtime.py \
  tests/gpu/test_optical_warp_executor.py -q
```

Observed before this implementation commit:

- a3a/benchmark contract tests: 58 passed;
- full unit suite: 1067 passed, 20 skipped;
- benchmark native GPU smoke: 2 passed;
- related optical GPU regression: 51 passed;
- Ruff check and format check: passed.

No benchmark sample or baseline artifact was generated.

## Review request

Please review whether:

1. exact input/acceleration identities and phase authority match approved v3;
2. preflight fails before timed submission and cannot emit an ineligible result;
3. runner-owned runtime wrappers, hook provenance, and real product-call
   negative tests close the P1 instrumentation objection;
4. execution/acceleration/publication provenance cannot be coordinator-filled;
5. Go2 Git and file-manifest provenance prevents surrogate substitution;
6. current adapters correctly remain audit-only and ineligible;
7. no a3b, baseline inspection, C0 implementation, or product migration leaked
   into this slice.
