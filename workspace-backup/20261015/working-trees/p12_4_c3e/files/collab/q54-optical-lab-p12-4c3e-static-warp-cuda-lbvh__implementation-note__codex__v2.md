# Q54 Optical Lab P12.4c3e Warp Scene Command Freeze Fix (Codex v2)

Date: 2026-09-07
Base: `f5c1c610141b998ef75876443d76f5f19e0bae08`
Scope: close the C3e Warp scene-command binding review finding only; no C4/B3,
a3b activation, benchmark result, baseline, or backend selection

## Review finding

The aggregate C3e bundle previously bound the identity and evidence digest of
`WarpSceneUpdateCommandBundle`, but the scene bundle itself did not retain an
independent snapshot of its recorded command tuple. A caller using
`object.__setattr__()` could replace `_commands` after sealing while leaving
the stage evidence unchanged. In particular, replacing the tuple with `()`
allowed the state machine to claim that scene update completed without
launching the recorded update command.

## Contract change

`WarpSceneUpdateCommandBundle` now has a one-way seal and an immutable binding
signature. The frozen state binds:

- the exact command tuple object and the identity/order of every command;
- the exact local scene, transform input, world output, and Warp stream owners;
- device ordinal and native stream handle;
- manifest/input-contract provenance;
- dispatch count and canonical dispatch-plan digest;
- the original evidence object and digest.

The public `validate_bound()` performs the full setup-time validation. It
revalidates the stream/device, transform and output ABI, input/output storage
disjointness, recomputes the current input/output storage manifests and scene
dispatch plan, rebuilds `WarpSceneUpdateCommandEvidence`, and compares the
current binding signature with the construction-time baseline.

The seal cannot be reopened or deleted through ordinary assignment, and its
signature baseline cannot be rewritten through the public object interface.
The C3e comparison bundle calls `scene_bundle.validate_bound()` from `arm()`
before the timed boundary.

`enqueue()` additionally executes an allocation-free lightweight identity
check before launching any command. This catches command/owner/stream drift
introduced after `arm()` without rebuilding manifests or evidence inside the
timed region. A failure poisons the enclosing C3e comparison state through its
existing fail-closed exception path.

## Negative coverage

GPU tests now prove pre-enqueue rejection for:

- replacing the recorded command tuple with an empty tuple;
- replacing it with a different same-length command tuple;
- reversing a two-command triangle-plus-plane bundle;
- reopening `_sealed` through ordinary assignment;
- mutating the command tuple after C3e `arm()` but before scene enqueue.

The last case is rejected by the lightweight enqueue check and leaves the C3e
bundle poisoned. The normal Warp scene-update to CUDA-LBVH path remains
allocation/bridge free inside timing.

## Verification

```text
python -m pytest -q \
  tests/unit/optics/test_warp_native_command_contract.py \
  tests/gpu/test_optical_warp_native_commands.py \
  tests/gpu/test_optical_warp_cuda_lbvh_comparison.py
18 passed

python -m pytest -q tests/unit
1149 passed, 20 skipped

python -m pytest -q \
  tests/gpu/test_cuda_publication_binding.py \
  tests/gpu/test_optical_benchmark_native.py \
  tests/gpu/test_optical_gpu_runtime.py \
  tests/gpu/test_optical_warp_executor.py \
  tests/gpu/test_optical_warp_native_commands.py \
  tests/gpu/test_optical_cuda_scene.py \
  tests/gpu/test_optical_cuda_lbvh_native.py \
  tests/gpu/test_optical_warp_cuda_lbvh_comparison.py \
  tests/gpu/test_cuda_optical_timeline.py
108 passed
```

Targeted `ruff check` and `git diff --check` pass. The frozen a3b activation
remains `false`, its activation commit remains null, and no benchmark artifact
or baseline was produced.
