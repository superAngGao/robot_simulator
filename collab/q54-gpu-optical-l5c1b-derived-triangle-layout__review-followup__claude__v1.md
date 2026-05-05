Initiative: q54-gpu-optical-executor
Stage: l5c1b-review-followup
Author: claude
Version: v1
Date: 2026-05-04
Status: followup-complete
Related Files: optics/device_scene.py, optics/warp_execution.py, tests/gpu/test_optical_gpu_runtime.py, OPEN_QUESTIONS.md

# Q54 L5C.1b Derived Triangle Layout — Claude Review Follow-up

## 收尾确认

Codex 已处理 L5C.1b review 的第 3 条表述歧义：

```text
测试改名：
  test_device_scene_derived_executor_matches_l5a_for_world_static_triangle_mesh

笔记和 OPEN_QUESTIONS.md 已明确记录：
  L5C.1b derived executor vs L5A traversal parity
```

验证结果：

```text
conda run -n env_tilelang_20260119 python -m pytest \
  tests/gpu/test_optical_warp_executor.py tests/gpu/test_optical_gpu_runtime.py -q
# 16 passed

PYTHONPATH=. pytest tests/unit/optics tests/unit/sensing -q
# 93 passed
```

**L5C.1b 完全收尾，可进入 benchmark 阶段。**

---

## L5C.1c 决策前置：Benchmark Harness 设计建议

### 目标

在决定是否实施 AABB early reject（L5C.1c）之前，需要数据回答以下问题：

1. update kernel 时间 vs traversal kernel 时间的比例是什么？
2. 在哪个 (num_rays, num_triangles) 规模下 traversal 成为瓶颈？
3. 不同场景（world-static vs body-bound，role-filtered vs full scene）的时间分布如何？

只有当 traversal kernel 时间显著大于 update kernel 时间时，AABB early reject 才有意义。

### 推荐 Benchmark Cases

按 Codex 计划第 7 节，五个 case：

```text
Case 1: few rays, few triangles
  num_rays = 16, num_triangles = 64
  基准：确认 GPU launch overhead 水位

Case 2: camera-like rays, few triangles
  num_rays = 256×256 = 65536, num_triangles = 64
  预期：traversal 受 ray 数主导，每个 ray 需遍历所有 triangle

Case 3: camera-like rays, many triangles
  num_rays = 65536, num_triangles = 4096
  预期：这是 AABB early reject 最可能有收益的场景

Case 4: body-bound moving mesh vs world-static mesh
  固定 num_rays = 1024, 对比 update kernel 时间（body-bound 需要变换，world-static 不需要）
  预期：body-bound update 更慢，但两者 traversal 时间应该一致

Case 5: role-filtered scene (most primitives invisible)
  num_rays = 1024, num_triangles = 1024，但 sensor_role_mask 只匹配 1/8 的 primitive
  预期：role mask skip 降低有效 traversal 工作量
```

### 需要测量的时间

```text
- update_kernel_time_ms      # _update_triangles_world_kernel 执行时间
- traversal_kernel_time_ms   # GpuDeviceSceneOpticalExecutor execute 内核时间
- total_execute_time_ms      # 含 ray upload 和 result download 的端到端时间
- primitive_count            # 实际参与 traversal 的 triangle 数
- ray_count
```

### 实现路径建议

最简单的方式是在 `tests/gpu/` 下加一个 `benchmark_optical_l5c.py`（非 pytest，直接 `python -m` 运行），用 `wp.synchronize()` + `time.perf_counter()` 做粗粒度计时。不需要 CUDA event 精确计时，目的是判断量级比例，不是做精确 profiling。

```python
# 示意结构（不是实现，仅描述接口意图）
def run_case(name, num_rays, num_triangles, body_bound=False, role_fraction=1.0):
    # 构建 registry、DeviceOpticalSceneCache、GpuDeviceSceneOpticalExecutor
    # 预热（wp.synchronize 后丢弃第一次结果）
    # 循环 N 次，记录 update + execute 时间
    # 输出 case name / num_rays / num_triangles / update_ms / traverse_ms
    ...
```

### L5C.1c 决策标准

| 观察 | 结论 |
|------|------|
| traversal_ms < update_ms | AABB 无益，L5C.1c 跳过，直接进 L5C.2 BVH 评估 |
| traversal_ms ≈ update_ms | 取决于 Case 3/5 的具体数字，可选实施 |
| traversal_ms >> update_ms（Case 3 场景）| AABB early reject 有意义，实施 L5C.1c |
| Case 5 role-filter 已显著减少工作量 | AABB 对 filtered 场景收益递减，优先考虑 BVH |

---

## 关键思考

### 为什么不直接实施 L5C.1c

AABB per-triangle early reject 是一个伪加速：它在 ray-major kernel 里增加了一次内存读取（`aabb_min/max`），如果三角形分布均匀，AABB 测试通过率接近 100%，只是增加了 memory traffic 而没有减少 triangle intersection 计算。只有在 mesh 三角形空间分布稀疏、ray 方向集中的场景下才有收益（典型：LiDAR 扫地面，大量三角形在传感器视野外）。

在 benchmark 数据之前做这个决定，等同于猜测工作负载特征。

### Benchmark 结果如何影响 L5C.2

如果 Case 3 的 traversal 时间是 update 时间的 10 倍以上，说明当前 O(rays × triangles) 的瓶颈已经出现，需要真正的 BVH（L5C.2），而不是 per-triangle AABB 这种局部优化。benchmark 结果直接决定是否跳过 L5C.1c 进入 L5C.2。

---

## 下一步

1. 实现 `tests/gpu/benchmark_optical_l5c.py`，运行 5 个 benchmark case
2. 根据 update_ms vs traversal_ms 比例决定 L5C.1c 是否实施
3. 无论 L5C.1c 是否实施，L5C.2 GPU BVH / OptiX 评估均为后续步骤
