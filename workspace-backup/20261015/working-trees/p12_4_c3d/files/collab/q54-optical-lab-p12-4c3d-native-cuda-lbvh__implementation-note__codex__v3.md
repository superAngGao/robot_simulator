# Q54 Optical Lab P12.4c3d Native CUDA LBVH — Grid Range Closure (Codex v3)

Date: 2026-09-03
Base fix: `b77bd79a4abc005059eb7b57f842b73dd455c1df`
Scope: close the remaining maximum-grid P1; no C3e/C4/B3/a3b activation or baseline results

## Review issue and resolution

The v2 range fix correctly limited flattened bounds storage to signed 32-bit
indexing, but the bounds initialization launch still used the overflow-prone
ceil-div expression:

```text
(bound_value_count + kThreads - 1) / kThreads
```

At the legal maximum, adding `255` exceeds `INT32_MAX` before the division.

The production CUDA source now uses one overflow-free helper for every grid
calculation:

```text
value / divisor + (value % divisor != 0)
```

This helper is used for primitive, internal-node, and flattened-bound launch
sizes. The old `+ kThreads - 1` form no longer exists in the source.

The maximum flattened-bound arithmetic is frozen with compile-time assertions:

```text
primitive_count   = 357,913,941
bound_value_count = 2,147,483,643
bound_blocks      = 8,388,608
```

The unit contract independently recomputes and asserts the same values and
checks the actual production source selects the safe helper. As in v2, the
runtime boundary test uses CUB's null-workspace query and allocates no giant
primitive/node tensors.

Because executable CUDA source changed, the extension/cache identity is now:

```text
q54.p12.4.cuda-lbvh-extension.v3
robot_sim_cuda_lbvh_native_v3
```

An older cached extension therefore cannot satisfy this contract.

## Verification

```text
python -m pytest -q \
  tests/unit/optics/test_cuda_lbvh_native.py \
  tests/gpu/test_optical_cuda_lbvh_native.py
20 passed in 68.60s (cold extension v3 build)

python -m pytest -q tests/unit
1148 passed, 20 skipped in 54.42s

python -m pytest -q \
  tests/gpu/test_cuda_optical_timeline.py \
  tests/gpu/test_cuda_publication_binding.py \
  tests/gpu/test_optical_benchmark_native.py \
  tests/gpu/test_optical_cuda_scene.py \
  tests/gpu/test_optical_cuda_lbvh_native.py \
  tests/gpu/test_optical_gpu_runtime.py \
  tests/gpu/test_optical_warp_executor.py \
  tests/gpu/test_optical_warp_native_commands.py
104 passed in 8.43s
```

Targeted `ruff check` and `git diff --check` pass. The a3b activation remains
`false`/`null`; no benchmark result or baseline was produced.
