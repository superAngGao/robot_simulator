# Q54 GPU Optical L5C.3c Benchmark Cleanup Implementation Note

Date: 2026-05-05

## Scope

Added a same-process direct-light comparison mode to
`benchmarks/bench_optical_device_scene.py`.

New CLI:

```bash
--compare-direct-light
--fail-on-overflow
--verbose-warp
```

This mode measures, in one process and on the same sequence of snapshots/BVHs:

- `bvh` or `bvh_refit` first-hit only;
- `bvh_direct` or `bvh_refit_direct`;
- `bvh_shadow` or `bvh_refit_shadow`.

This is intended to replace the earlier practice of running three separate
processes for first-hit / no-shadow / shadow comparisons. Separate processes
were too noisy for fine-grained conclusions because setup, Warp module loading,
GPU clock state, and Python host jitter differed between rows.

## Output Changes

The benchmark now keeps stdout as parseable CSV by default. Warp initialization
and module-load logs are suppressed through `wp.config.quiet = True`; use
`--verbose-warp` only for debugging module loading.

The benchmark CSV now reports warmup/repeat, distribution statistics, and
traversal diagnostics:

```text
warmup
repeat
*_ms_mean
*_ms_p50
*_ms_p90
*_ms_std
bvh_stack_overflow
bvh_max_stack_observed
shadow_stack_overflow
shadow_max_stack_observed
```

These diagnostics are read after the timed GPU execution window, so they do not
inflate the reported traversal/shading timing. They make benchmark output usable
as a pre-preview safety check for silent stack overflow.

`--fail-on-overflow` exits non-zero if either primary or shadow stack overflow is
reported. This should be used for preview/README image generation and other runs
where an invalid-looking image would cost more time to debug than failing early.

## Same-Process Results

Commands run:

```bash
conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_dense_single --warmup 2 --repeat 5 --refit-bvh --compare-direct-light

conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
  --case robot_dense_pack --warmup 2 --repeat 5 --refit-bvh --compare-direct-light
```

Observed p50 results:

```text
robot_dense_single, 65,536 rays, 161,280 triangles
  update p50: 0.246 ms
  refit p50:  0.344 ms
  first-hit:  2.364 ms
  direct:     2.876 ms
  shadow:     3.267 ms
  stack:      primary max 11, shadow max 10, overflow 0

robot_dense_pack, 65,536 rays, 645,120 triangles
  update p50: 0.321 ms
  refit p50:  0.323 ms
  first-hit:  1.199 ms
  direct:     1.347 ms
  shadow:     1.876 ms
  stack:      primary max 11, shadow max 8, overflow 0
```

The previous cross-process anomaly where `robot_dense_pack` direct no-shadow
appeared faster than first-hit did not reproduce under same-process comparison.
The better interpretation is that the earlier row ordering was measurement
noise, not a semantic or kernel correctness signal.

## Interpretation

Same-process comparison gives a cleaner cost model:

- `robot_dense_single`:
  - direct shading overhead over first-hit: about 0.51 ms p50;
  - shadow overhead over no-shadow direct: about 0.39 ms p50.
- `robot_dense_pack`:
  - direct shading overhead over first-hit: about 0.15 ms p50;
  - shadow overhead over no-shadow direct: about 0.53 ms p50.

Inline shadow any-hit remains under the earlier 3x trigger threshold on these
robot scenes. Split shadow-ray buffers/kernels are still not justified.

The pack scene being faster than the single scene in first-hit timing is not
impossible despite having more triangles: BVH traversal cost is driven by ray
distribution, hit/miss pattern, node visits, and primitive tests, not only total
triangle count. The benchmark still lacks node-visit/primitive-test counters, so
this should be treated as a traversal-distribution observation rather than a
fully explained performance model.

## Benchmark Infrastructure Decision

The issue was not just a bad one-off run. The harness needed stronger
experimental boundaries. Going forward:

- Use same-process comparison when comparing kernels that share setup/BVH data.
- Treat single-mode runs as smoke or broad trend checks, not fine-grained
  kernel A/B evidence.
- Report distribution shape (`p90`, `std`) alongside `p50`; noisy host-side
  components should not be interpreted from p50 alone.
- Keep stdout machine-parseable; environment/log metadata should not be mixed
  into CSV rows.
- Use `--fail-on-overflow` for preview and decision runs.
- Keep host wall-time columns as end-to-end component timing. Add CUDA-event
  timing as a separate backend/column later rather than silently changing the
  meaning of existing columns.

## Remaining Benchmark Work

- Add optional node-visit and primitive-test instrumentation kernels when we
  need to explain BVH quality beyond wall time.
- Add high-resolution Menagerie Go2 GPU direct-light preview timing once the
  preview path exists.
- Add optional CUDA event timing for GPU-only windows. Warp exposes
  `wp.Event(enable_timing=True)` and `wp.get_event_elapsed_time(...)`, but this
  should be introduced as an explicit timing backend/extra column so existing
  host wall-time columns remain comparable.
