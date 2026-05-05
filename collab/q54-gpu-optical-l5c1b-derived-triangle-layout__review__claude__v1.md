Initiative: q54-gpu-optical-executor
Stage: l5c1b-review-claude
Author: claude
Version: v1
Date: 2026-05-04
Status: review-complete
Related Files: optics/device_scene.py, optics/warp_execution.py, tests/gpu/test_optical_gpu_runtime.py

# Q54 L5C.1b Derived Triangle Layout — Claude Review

## 总体判断

实现正确，关键的刚体变换性质（`det(R)=1` → 法向量可直接旋转）使用正确。`triangles_world` 已完全移除，snapshot 字段干净。测试覆盖了顶点重建和法向量校验。**L5C.1b 通过，可进入 L5C.1c benchmark 决策阶段。**

---

## 具体观察

### 1. Update kernel 的法向量计算 ✅

实现笔记 Section 3 写的：

```
normal_world = R_world_geometry @ normalize(cross(e1_local, e2_local))
```

这利用了 `det(R) = 1` 时 `(R @ e1) × (R @ e2) = R @ (e1 × e2)` 的性质，在 local 坐标系做 cross product + normalize，再旋转到 world，比在 world 坐标系做 cross product 数值更稳定（local 坐标下 build-time 已过滤退化三角形，cross product 结果不会是零向量）。

build-time 过滤（`_pack_registry_for_device` 里的 `np.linalg.norm(cross) <= _BUILD_EPS_F32`）保证了 update kernel 不需要再做退化检查，✅。

### 2. Traversal kernel 消费 derived buffers ✅

L5C.1a 的 `GpuDeviceSceneOpticalExecutor` traversal 每个 ray×triangle pair 需要重建 `e1/e2/normal`（约 9 次浮点运算），现在直接读 `triangle_e1/e2/normal_world`，减少了 `num_rays × num_triangles` 的冗余计算。

normal flip（`ndotd > 0 → flip`）的逻辑保留，与 L5A 一致 ✅。

### 3. triangles_world 移除 ✅

`DeviceOpticalSceneSnapshot` 已经不包含 `triangles_world` 字段，update kernel 也不再分配该 buffer。符合 review 建议的"同 PR 移除"策略。

### 4. 测试覆盖评估

L5C.1b 新增的测试（line 346）用 `v0/e1/e2` 重建顶点并校验 `triangle_normal_world`，这是 update kernel 正确性的直接验证 ✅。

L5C.1a/L5A parity 测试保持绿色，说明切换 derived layout 后 traversal 结果没有退化 ✅。

**一个潜在的覆盖缺口**：目前的测试都是 CPU-unit 或单次 GPU smoke test。没有覆盖"body-bound mesh 在多帧移动后，每帧 update kernel 输出的 `triangle_normal_world` 方向随旋转正确变化"的动态 case。当前 L5C 阶段这个缺口可以接受，但建议在 L5C.1c 或 L5C.2 的 benchmark 测试里补上一个旋转体的多帧 parity 验证。

---

## L5C.1c 决策建议

按照 L5C.1 计划，L5C.1c（AABB early reject）应该 **benchmark-gated**。在实施之前需要回答：

1. 当前 traversal kernel 时间 vs update kernel 时间的比例是多少？
2. 典型场景的 triangle 数量（robot-scale）是多少？

如果 traversal 时间远小于 update 时间，AABB 完全没必要（robot-scale 场景通常 < 1000 个三角形，GPU traversal 极快）。

建议的 benchmark 触发条件：
- 如果 `num_triangles > 500` 且 `num_rays > 10000`（camera-like），才考虑 AABB
- 否则直接跳到 L5C.2 GPU BVH 评估

---

## 关键思考

### derived layout 的 memory trade-off

`triangles_world[N, 9]` → `v0/e1/e2/normal[N, 3]×4` = `[N, 12]`，内存增加 33%。但对 GPU traversal 来说，derived layout 的 cache 效率更好：Moller-Trumbore 需要连续读 `e1/e2`，在 `[N, 3]×2` 的 AoS 布局下这两个读是相邻的，比从 `[N, 9]` 里用偏移量读 `v1-v0/v2-v0` 更友好。

对于 L5C.1c 的 AABB，如果加上 `aabb_min/max[N, 3]×2` 则总计 `[N, 18]`，是原始 `triangles_world[N, 9]` 的 2 倍。这个 trade-off 需要 benchmark 支持才能决定。

### update kernel 的编译时间问题

`_update_triangles_world_kernel` 扩展后（新增 4 个输出 buffer）会触发 Warp 重新编译。第一次运行测试时可能需要额外 10-30 秒编译时间，这是正常的，不是 bug。

---

## 下一步

1. 运行 L5C.1c 决策所需的 benchmark（按计划第 7 节的 5 个 case）
2. 根据 benchmark 结果决定是否实施 AABB early reject
3. 如果跳过 L5C.1c，直接进入 L5C.2 GPU BVH / OptiX 评估
4. 补充 body-bound mesh 多帧旋转的动态 parity 测试（建议在 L5C.2 之前完成）
