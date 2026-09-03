# Q54 Optical Lab P12.4c3d Native CUDA LBVH — P1 Range Closure (Codex v2)

Date: 2026-09-03
Base implementation: `dda030cb04784f233d4cfd08ee9f81d66cd1bf40`
Scope: close the review P1 only; no C3e/C4/B3/a3b activation or baseline results

## Review issue

The first C3d implementation accepted up to `1,073,741,824` primitives based
only on the signed 32-bit node count. The CUDA source also flattens the
`[node_count, 3]` bounds arrays with `int` indices, so the actual limiting
condition is:

```text
(2 * primitive_count - 1) * 3 <= INT32_MAX
```

The first invalid primitive count is therefore `357,913,942`, not the former
node-only limit.

## Resolution

The public and native limits are now exactly:

```text
CUDA_LBVH_MAX_PRIMITIVES = 357,913,941
kMaxPrimitiveCount       = 357913941
```

At the maximum legal value:

```text
node_count * 3 = 2,147,483,643 <= 2,147,483,647
```

At the first illegal value:

```text
node_count * 3 = 2,147,483,649 > 2,147,483,647
```

Python rejects counts above the public constant before loading/querying the
extension. Both C++ entry points independently enforce the same limit:

- `cuda_lbvh_workspace_requirements(...)`;
- `cuda_lbvh_rebuild_out(...)`.

The CUDA source has compile-time assertions proving the maximum legal and
first illegal arithmetic boundaries. The extension ABI/source identity was
bumped to `q54.p12.4.cuda-lbvh-extension.v2`, and its JIT module name was bumped
to `robot_sim_cuda_lbvh_native_v2`, so an old cached binary cannot satisfy the
new range contract.

No address arithmetic was widened in this slice. All relevant kernel indices
remain consistently 32-bit and are now protected by the common bound.

## Boundary verification without giant allocations

The new tests do not allocate primitive, node, or bounds arrays at the limit.
They verify:

- Python accepts `357,913,941` and rejects `357,913,942`;
- the arithmetic invariant on both sides of the boundary;
- the production CUDA source uses the same constant in both public entries;
- the real C++ null-workspace query accepts the maximum legal count;
- the real C++ query rejects the first illegal count;
- the accepted query reports the exact derived node capacity.

Results:

```text
python -m pytest -q \
  tests/unit/optics/test_cuda_lbvh_native.py \
  tests/gpu/test_optical_cuda_lbvh_native.py
19 passed in 3.04s

python -m pytest -q tests/unit
1147 passed, 20 skipped in 54.67s

python -m pytest -q \
  tests/gpu/test_cuda_optical_timeline.py \
  tests/gpu/test_cuda_publication_binding.py \
  tests/gpu/test_optical_benchmark_native.py \
  tests/gpu/test_optical_cuda_scene.py \
  tests/gpu/test_optical_cuda_lbvh_native.py \
  tests/gpu/test_optical_gpu_runtime.py \
  tests/gpu/test_optical_warp_executor.py \
  tests/gpu/test_optical_warp_native_commands.py
104 passed in 9.19s
```

Targeted `ruff check` and `git diff --check` pass. The a3b activation fields
remain unchanged (`false` / `null`), and no benchmark result or baseline was
produced.
