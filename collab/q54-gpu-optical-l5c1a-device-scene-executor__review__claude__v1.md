Initiative: q54-gpu-optical-executor
Stage: l5c1a-review-claude
Author: claude
Version: v1
Date: 2026-05-04
Status: review-complete
Related Files: optics/device_scene.py, optics/warp_execution.py, tests/gpu/test_optical_gpu_runtime.py
Owner Summary: Review of L5C.1-pre (int64 role mask) + L5C.1a (GpuDeviceSceneOpticalExecutor). Implementation correct. Resource lifetime issue resolved. Four targeted observations for L5C.1b.

# Q54 L5C.1a Device Scene Executor — Claude Review

## 总体判断

实现正确，分阶段策略落地良好。int64 role mask 升级和 executor lifetime 处理都按照 review 建议执行。测试覆盖完整（parity / unknown role / int64 bit>31 / resource retention）。**L5C.1a 可以进入 L5C.1b。**

---

## 具体观察

### 1. int64 role mask 升级（L5C.1-pre）✅

`_MAX_INT32_ROLE_BITS = 31` → `_MAX_INT64_ROLE_BITS = 63` 方向正确。

需要确认一点：`DeviceOpticalRoleTable.from_roles` 在 role 数量超过 63 时应该 raise `ValueError`，这个检查在原来的 `if len(unique_roles) > _MAX_INT32_ROLE_BITS` 里已经有，改为 63 后需要确认 error message 也同步更新为 "63 roles"，不然测试失败时的诊断信息会误导。

### 2. GpuDeviceSceneOpticalExecutor kernel 设计（L5C.1a）✅

从实现笔记看，kernel 继承了 L5A 的 ray-major 布局和 `_is_better_hit` 的 distance + source_order_key 两级 tie-break。这是正确的。

L5C.1a 的 kernel 在 triangle 遍历时仍然在每个 ray×triangle pair 里重算 `e1/e2/normal`，这是预期行为（L5C.1b 才改）。

一个待确认的细节：role mask filtering 是 `primitive_role_mask & sensor_role_mask == 0` 时 skip。对于 `sensor_role_mask = 0`（unknown role），所有 primitive 都应该 skip，结果是全 miss。测试已覆盖这个 case，✅。

### 3. Resource Lifetime（L5C.1a）✅

按 review 建议处理：`resources` 只持有 ray arrays 和 snapshot world primitive buffers，不持有 `DeviceOpticalSceneSnapshot` 或 `GpuPublishedFrame`。

frame slot 释放路径：
```
execute() 返回 → gpu_runtime.py 调用 complete_device_consumer() → done_event 记录
→ result.resources 不含 frame → frame slot GC 不受 result 生命周期阻塞
```

这个路径是正确的。

### 4. 待处理：parity 测试的三角形覆盖

实现笔记 Section 5 列出的 GPU 测试覆盖了：
- body-bound plane parity with L5B.1 ✅
- unknown role all-miss ✅
- int64 role bit > 31 ✅
- resource retention ✅

但还没有**三角形 parity** 测试（body-bound mesh vs L5B.1 / L5A）。目前测试只用 plane 验证了 parity。L5C.1a 的 triangle traversal kernel 和 L5A 的 `_brute_force_first_hit_kernel` 逻辑相同，但消费的数据路径不同（L5C.1a 从 `DeviceOpticalSceneSnapshot` 读，L5A 从 host-packed workload 读）。

建议在 L5C.1b 开始前，补一个 triangle parity test（world-static mesh，验证 L5C.1a 和 L5A 的 hit_mask / range_m / position_world 数值一致），作为 L5C.1b derived layout 切换的回归基准。

---

## L5C.1b 实施建议

### 扩展 `DeviceOpticalSceneSnapshot`

在 `device_scene.py` 的 `DeviceOpticalSceneSnapshot` 里新增四个字段：

```python
triangle_v0_world: object | None = None    # float32[num_triangles, 3]
triangle_e1_world: object | None = None    # float32[num_triangles, 3]
triangle_e2_world: object | None = None    # float32[num_triangles, 3]
triangle_normal_world: object | None = None  # float32[num_triangles, 3]
```

用 `None` 而不是直接分配，方便 L5C.1a 代码路径（和 CPU-only 测试）不受影响。

### update kernel 扩展

`_update_triangles_world_kernel` 目前输出 `triangles_world[N, 9]`。L5C.1b 需要同时输出 `v0/e1/e2/normal`：

- `v0 = R_wg @ v0_local + r_wg`（已有 vertex 循环，`vertex=0` 就是 v0）
- `e1 = R_wg @ (v1_local - v0_local)`（等价于 `v1_world - v0_world`）
- `e2 = R_wg @ (v2_local - v0_local)`
- `normal = normalize(e1 × e2)`（在世界坐标下，旋转不改变法向量方向，只需对 cross product 归一化）

**注意**：cross product 的幂等性——`(R @ e1) × (R @ e2) = det(R) * R @ (e1 × e2)`。对于正交旋转矩阵 `det(R) = 1`，所以世界坐标下的法向量等于 `R_wg @ normalize(e1_local × e2_local)`，不需要在世界坐标下重做 cross product。

build-time 的退化三角形过滤（`np.linalg.norm(cross) <= _BUILD_EPS_F32`）已经在 `_pack_registry_for_device` 里处理，所以 update kernel 不需要再判断退化，直接计算即可。

### parity 验证策略

L5C.1b 的 parity 测试应该在同一个 test case 里同时验证：

```python
# 用 v0/e1/e2 重建 v1, v2，与 triangles_world 比较
v1_reconstructed = v0 + e1
v2_reconstructed = v0 + e2
assert_allclose(v1_reconstructed, triangles_world[:, 3:6], atol=1e-5)
assert_allclose(v2_reconstructed, triangles_world[:, 6:9], atol=1e-5)
```

以及：

```python
# 验证 traversal 结果和 L5C.1a 一致（切换 derived layout 前后 parity）
```

### `triangles_world` 的移除时机

L5C.1b parity 稳定后，在同一个 PR 里：
1. 移除 `DeviceOpticalSceneSnapshot.triangles_world`
2. 把 `GpuDeviceSceneOpticalExecutor` 里读 `triangles_world` 的分支全部切换为 `v0/e1/e2`
3. 更新 `update_device_optical_scene_from_gpu_frame` 不再分配 `triangles_world` buffer

不要把移除拆成独立 PR，否则会有短暂的两套数据并存期，容易引入 executor 用错字段的 bug。

---

## 关键思考

### 为什么 L5C.1a 不直接用 derived layout？

L5C.0 的 `_update_triangles_world_kernel` 已经是约 200 行的 Warp kernel，逻辑密度很高。L5C.1a 如果同时改 update kernel 输出 + traversal kernel 输入，parity 失败时无法区分是 update 出错还是 traversal 出错。分离变更是正确的工程决策。

### int64 role mask 升级的时机选择

在 L5C.1a 之前做升级（L5C.1-pre）是正确的。如果等到 L5C.1a 之后再升级，`GpuDeviceSceneOpticalExecutor` 的 kernel 就要同时处理 int32 和 int64 的迁移，而 Warp kernel 参数类型变更需要重新编译，可能引入额外调试成本。提前做 ABI 升级、验证通过后再开发 executor，降低了耦合。

### resource lifetime 设计的核心约束

Q52 frame pool 的 borrow/complete 协议要求 `complete_device_consumer` 必须在 frame 被再次 borrow 之前调用。如果 result.resources 持有 frame，frame 的 Python 对象生命周期就取决于 result 的 GC 时间，在异步 pipeline 里这是不可控的。只持有 primitive buffers（这些 buffers 是 executor 自己分配的）才能保证 frame slot 及时释放。

---

## 下一步

1. 补 triangle parity test（L5C.1a，world-static mesh，L5C.1a vs L5A）
2. 实现 L5C.1b：扩展 snapshot + update kernel + traversal 切换 + parity 验证 + 移除 triangles_world
3. L5C.1b 完成后再决定是否做 L5C.1c（AABB），需要 benchmark 数据支持
