# C4c2a single-owner contract extraction

Initiative: q54-optical-lab-p12-4c4c2a-native-contract-extraction
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-09-29
Status: in_review
Related Files: design-request v1; independent design review v1;
optical_worker/native_input.py; optical_worker/provenance.py;
optical_worker/timing.py; optics/host_bvh.py; optics/__init__.py
Owner Summary: Bounded extraction implemented; not a runtime activation.
Independent code review and the remaining author validation are still required.

## Revision and actual scope

Implementation base: `a1f6c8c87752559c7b25994d450a36c56c952b32`, the evidence-only
archive of the design approval at `6084245689ae50976f772daa269cc92d19605568`.
Executable/test commits: `953cbe3` and
`0dcbf2a790adf3d695b9501e0ff01be564c8c06a`. This note adds no executable changes.

Moved the existing native-input, case/content, C3 receipt/registry and timing
implementations into the production owners listed by the approved design.
Legacy modules explicitly re-export the same objects. Pure identity tables
live in `optical_worker/lane_contract.py`; benchmark eligibility and runtime
instrumentation remain in the benchmark package. The already-approved
`native_contract.py` types, storage/readiness registries and physical checks
are unchanged.

The CPU median BVH, stats, CPU nodes, topology hash and canonical hash helper
now live in `optics/host_bvh.py`; device modules import those exact objects.
Normalized class/function AST tests compare moved bodies against the immutable
pre-move base. The package initializer uses literal static-import branches,
preserves the export inventory and caches only known requested values. No
arbitrary module/path loader or module substitution was added.

CUDA arena C3 validation now uses the exact production class, its known
serialization implementation and the shared live registry directly. It does
not call an authority callback supplied by a receipt. Existing a2 target,
physical-storage, ordered ABI, slot/generation and acquire checks are retained.
New real CUDA tests exercise a copied receipt with a forged authority callback
and an issued receipt with an injected callback; neither callback may run.

The C1c test-only source list includes the newly moved owners. No product
allowlist, AST policy, worker request, guardian, watchdog or runtime-contract
code changed. Cleanup ledger symbol locations follow the move; shared code is
retained only when consumed, and losing-lane branches/unconsumed aliases still
have the existing P12.4e deletion requirement.

## Author validation observed so far

Primary SSH retry timed out; the owner-authorized fallback 101 was used.
Fresh detached runtime tree:
`/data3/ang.gao/robot_sim_c4_c2a_impl_0dcbf2a`, exact HEAD `0dcbf2a...`.
Environment: `robot_sim`, Python 3.12, Torch 2.9.1+cu128, Warp 1.12,
`CUDA_HOME=/data3/ang.gao/anaconda3/envs/robot_sim` (NVCC 12.8).
No installs, preexisting worktree changes, push/merge or benchmark runs.

Use `/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim
env CUDA_HOME=/data3/ang.gao/anaconda3/envs/robot_sim python -m pytest` in that
tree for the following observed groups:

- **413 passed**, 21.22 s: all `tests/unit/benchmarks`, plus
  `tests/unit/optics/test_native_contract_extraction.py`,
  `test_warp_native_command_contract.py`, `test_native_optical_arena.py`,
  `test_cuda_timeline.py`, `test_cuda_scene_native.py`,
  `test_cuda_lbvh_native.py`, `test_warp_render_buffer_contract.py`,
  `test_warp_render_command_contract.py`, `test_warp_fixed_shade_abi.py`,
  `test_warp_native_kernel_source.py`, and the four
  `test_sacrificial_sources/watchdog/lifecycle/leaf_descriptors.py` files.
  Log: `/data3/ang.gao/c4-c2a-0dcbf2a-cpu.log`.
- **198 passed**, 82.31 s: `tests/gpu/test_optical_benchmark_native.py`,
  `test_optical_warp_native_commands.py`, `test_optical_cuda_scene.py`,
  `test_optical_cuda_lbvh_native.py`,
  `test_optical_warp_cuda_lbvh_comparison.py`, `test_cuda_native_arena.py`,
  `test_native_contract_runtime_imports.py`, `test_warp_render_buffers.py`,
  `test_warp_render_recording.py`, `test_warp_shade_microfixtures.py`,
  `test_optical_gpu_runtime.py`.
  Log: `/data3/ang.gao/c4-c2a-0dcbf2a-gpu.log`.
- Local CPU-only Python 3.14: new extraction test **32 passed**. Ruff check
  and format check pass for all 22 changed Python files; diff check passes.

The GPU group includes exact a2 pointer/storage drift, C3 binding/authority,
C4 lease/event identity, real import attempts and shared low-level runtime-hook
observations. Real dual-device cases are not replaced by ordinal mocks.
Functional groups overlapped; these elapsed durations are not performance
measurements or candidate-selection evidence.

At note creation the complete 135-case sacrificial GPU suite, separate C0 GPU
setup/timeline regression, and all-five-case frozen content probe are still
running. Their logs/probe are outside the candidate:

- `/data3/ang.gao/c4-c2a-0dcbf2a-supervised.log`;
- `/data3/ang.gao/c4-c2a-0dcbf2a-setup-timeline.log`;
- `/data3/ang.gao/c4-c2a-content-probe.py`, with explicit candidate root and
  recovered asset root `/data3/ang.gao/c4_asset_frozen`.

No pending check is claimed as passing. The reviewer must receive their final
outcomes before accepting this checkpoint. The content probe uses real frozen
Go2 assets and preflight; missing assets or digest mismatch is a failure.
Final results will be archived as evidence, not used to waive requirements.

## Draft failures and limitations

An initial broad local benchmark run had 83 passes, 8 platform skips and 45
failures: 44 failed at the unavailable Torch runtime-hook import, and one
revealed stale cleanup-ledger symbol paths. The paths were corrected, and
the full server CPU group above passed without skips. Local environment
failures were not counted as successful runtime validation.

Early new-test failures exposed test serializer handling of mapping proxies
and SpatialTransform, and an incorrect test assumption that a ContextVar's
default was None. Tests now serialize logical registry data through its actual
types and assert restoration to the exact previous context object. No product
behavior was changed to satisfy those tests. Two long lines in the cold probe
were fixed by the second commit; final lint passes.

The import detector records attempted imports even when ImportError is
swallowed, and is calibrated against Torch and benchmark attempts. It is test
instrumentation, not a hostile-Python sandbox or a product policy change.
The CPU cold test reconstructs logical registry data, calls the sole real
packer, and checks frozen scene/fixture/replay hashes. Separate legacy Go2
materialization is not claimed to be the future worker's clean source closure.

## Gates and next design

Frozen gate and closed-worker runtime-contract JSON are unchanged in Git.
Public product worker, launcher, AST, request and preparation gates remain
unchanged. C3 builders remain one-shot; no C4 result-capable binding is added.

This is only C4c2a. Warp timeline/R0-R1 arena/worker integration requires its
next independent design review. C4c3/c4, C4d/B3, product GPU preparation,
a3b/result/baseline, host payload/shipping and backend selection remain closed.
Completing these moves does not approve C4c2, C4c or C4 in aggregate.
