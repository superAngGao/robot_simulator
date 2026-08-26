# Q54 Optical Lab P12.4c3a Frozen Workload and Result-Closure Follow-up

Author: Codex

Date: 2026-08-26

Status: implemented for Claude re-review; C3b-C3e, C4/B3, a3b results,
baseline, and performance inspection remain closed

Reviewed implementation: `cadf7bb243adafbfd521d460a1640dab9204bb33`

## Outcome

This follow-up closes the three P1 findings against `cadf7bb` before any
benchmark result exists:

1. case labels no longer confer canonical provenance; the gate freezes the
   exact packed-scene, replay, and applicable R0-fixture digests;
2. a versioned activation gate now prevents construction of a P12.4a3b runner
   while C4/B3 approval is absent, and C3 bindings carry an explicit
   non-result-capable capability;
3. row/input provenance and a canonical binding round-trip are validated
   before lane preflight or the first timed enqueue.

No sample, baseline, timing, or performance result was generated or inspected.

## Frozen workload content

`benchmarks/optical/workload_contract.py` is a dependency-light registry of
the exact logical content allowed into a result-capable lane. Each gate case
now contains:

- `scene_contract_sha256`;
- `transform_replay_sha256` when an R1/R2 replay exists;
- `fixed_render_fixture_sha256` when an R0 fixture exists;
- the exact external revision and aggregate digest when the case uses Go2.

`benchmarks/optical/workloads.py` is the reproducible materializer. It builds:

- the 7 x 3 asymmetric scene as 21 source-ordered triangles with distinct
  geometry/material inputs and two asymmetric lights;
- the 120-frame body-triangle z-cycle exactly from the case definition;
- the physics body-triangle logical scene without replay provenance;
- the Go2 scene from the official importer after exact asset preflight;
- normalized fixed world/BVH fixtures through the existing formal builder.

The frozen Go2 source is official `google-deepmind/mujoco_menagerie` commit
`da76818e269b82289eba39808e2fb91d679d6994`. Its 25 tracked `unitree_go2`
files have canonical aggregate
`479dcbdb52c64ca86243d6b1d3b7167bb7d3820af1e60e355168b3c17ddf3ed0`.
Asset preflight now rejects a different clean HEAD as well as modified,
untracked, incomplete, or surrogate content.

The generic `PackedOpticalSceneHost.from_registry()`,
`CanonicalTransformReplayHost.from_arrays()`, and diagnostic
`CanonicalNativeInput.from_arrays()` remain available. They do not acquire
result capability from caller-supplied case labels. The only result-capable
constructor, `CanonicalNativeInput.from_scene_contract()`, compares actual
scene/replay/fixture digests with the frozen case content before creating its
manifest. Correct labels paired with a different registry, all-`z=123`
replay, or a different fixed transform fail there.

## C3 versus C4 capability

Native binding schema v2 adds `submission_capability` with two values:

- `c3_one_shot_non_result_capable`, the safe default returned by the C3
  factory and one-shot sealing tests;
- `c4_result_capable`, reserved for final C4 slot/generation/lease-bound
  bindings after C4/B3 review.

The frozen `native_result_activation` contract is v1 and currently has
`result_capable=false`, `activation_commit=null`, required reviews `[C4, B3]`,
and blocks both result-emitting phases `P12.4a3b` and `P12.4b4/d`.
`SingleLaneSampleRunner` rejects either phase at construction, before any row
preflight, binding, submit, operation trace, or result-record path. A future
approval must change this contract in a versioned owner decision; changing
only the phase authorization list is not sufficient.

Even after activation, the runner accepts only a `c4_result_capable` binding.
The result validator also rejects a receipt carrying the C3 one-shot
capability. Thus C3 evidence cannot be repurposed as a final C4 submission
object.

## Fail-before-enqueue validation

For an activated future lane, the runner validates the following before lane
preflight:

- manifest `input_contract` equals the frozen row contract;
- manifest case and case hash equal the row and gate;
- scene digest equals the case's frozen scene digest;
- R0 has exactly the frozen fixture and no replay;
- R1/R2 have exactly the frozen replay and no fixture;
- S has neither replay nor fixed-fixture provenance.

After lane binding but still before the timing context, the runner performs
`NativeExecutionBindingEvidence.from_payload(binding.to_payload())`. This
canonical round-trip re-runs schema, stage order, exact storage, and dataflow
validation rather than trusting an already-constructed dataclass. The runner
then compares the canonical object with prepared storage and requires the C4
capability. Any failure leaves `submit_sample()` uncalled and the operation
trace empty.

Gate/result schemas are v8, row semantics are v5, and the native binding
schema is v2. Row semantics include the complete case content contract and
the activation contract, so a later reviewed activation necessarily changes
the gate and row hashes before results can be emitted.

## Verification

Commands are run on `192.168.80.101` in `robot_sim`, with `PYTHONPATH=.`:

```text
pytest -q tests/unit/benchmarks/test_optical_benchmark_contracts.py \
  tests/unit/benchmarks/test_optical_benchmark_c3a.py \
  tests/unit/benchmarks/test_optical_benchmark_core.py \
  tests/unit/benchmarks/test_optical_benchmark_a3a.py

pytest -q tests/gpu/test_optical_benchmark_native.py \
  tests/unit/optics/test_scene_contract.py

ruff check benchmarks/optical \
  tests/unit/benchmarks/test_optical_benchmark_a3a.py \
  tests/unit/benchmarks/test_optical_benchmark_c3a.py \
  tests/unit/benchmarks/test_optical_benchmark_contracts.py \
  tests/unit/benchmarks/test_optical_benchmark_core.py \
  tests/gpu/test_optical_benchmark_native.py
```

Observed so far:

- focused contract/C3/runner/asset unit suite: 80 passed;
- full optics plus benchmark unit suites: 369 passed;
- native-input GPU plus scene-contract suite: 6 passed;
- affected optical GPU suites: 62 passed;
- Ruff: passed.

## Re-review request

Please verify that:

1. every result-capable input is now bound to frozen content rather than case
   labels, including exact Go2 source identity;
2. the frozen a3b activation rejects before any lane callback;
3. C3 one-shot evidence cannot satisfy a C4 result submission;
4. row/input mutual exclusion and exact digest checks occur before preflight;
5. a directly constructed invalid binding cannot bypass the canonical
   pre-enqueue round-trip;
6. the changes still create no result or baseline artifact and leave C3b-C3e,
   C4/B3, and a3b execution closed.
