Initiative: q50-render-backend
Stage: implementation-note
Author: claude
Version: v3
Date: 2026-04-24
Status: implemented
Related Files: collab/q50-render-backend__implementation-note__claude__v2.md
Owner Summary: Q50 Step 3 完成：build_render_scene_from_gpu 桥接函数已实现并测试（4 个测试全部通过）。GPU 训练路径现在可以通过 RenderScene 接口可视化任意 env。

## Open Questions Addressed

- **Q50 Step 3 — GPU 桥接**: `build_render_scene_from_gpu(engine, env_idx)` 已实现。
  GpuEngine 的 Warp 数组（`q_wp`）现在可以通过 CPU FK 转换为 `RenderScene`，
  供任意 `RenderBackend`（MatplotlibBackend / RerunBackend）消费。

## What Changed

`rendering/scene_builder.py` 新增 `build_render_scene_from_gpu`（lines 174–203）：

```python
def build_render_scene_from_gpu(engine, env_idx=0, include_contacts=True) -> RenderScene:
    merged = engine.merged
    q_all = engine.q_wp.numpy()          # GPU → CPU (zero-copy Warp accessor)
    if env_idx >= q_all.shape[0]:
        raise IndexError(...)
    q_np = q_all[env_idx].astype(np.float64)
    X_world = merged.tree.forward_kinematics(q_np)   # CPU FK
    contacts = engine.query_contacts(env_idx) if include_contacts else None
    terrain = getattr(merged, "terrain", None)
    return build_render_scene(merged, X_world, contacts=contacts, terrain=terrain)
```

设计要点：
1. **零拷贝读取**：`engine.q_wp` 是 Warp 数组，`.numpy()` 触发 GPU→CPU DMA，
   不需要额外缓冲区。
2. **CPU FK**：GPU 上的 FK 结果（`x_world_R_wp` / `x_world_r_wp`）也可用，
   但直接用 CPU FK 更简单，且渲染帧率远低于物理步频，CPU 开销可接受。
3. **委托给 build_render_scene**：复用已有的 CPU 路径，避免重复逻辑。
4. **include_contacts 开关**：允许调用方跳过 GPU→CPU 接触数据传输（训练中
   不需要可视化接触时可节省带宽）。

## Files Touched

- `rendering/scene_builder.py` — 新增 `build_render_scene_from_gpu`（~30 行）
- `tests/rendering/test_gpu_bridge.py` — 4 个新测试

## Tests Added

```
tests/rendering/test_gpu_bridge.py::TestBuildRenderSceneFromGpu::test_returns_render_scene          PASSED
tests/rendering/test_gpu_bridge.py::TestBuildRenderSceneFromGpu::test_shape_count_matches_merged    PASSED
tests/rendering/test_gpu_bridge.py::TestBuildRenderSceneFromGpu::test_env_idx_out_of_bounds_raises  PASSED
tests/rendering/test_gpu_bridge.py::TestBuildRenderSceneFromGpu::test_include_contacts_false_gives_empty_contacts  PASSED
```

测试策略：使用 `unittest.mock.MagicMock` 模拟 GpuEngine（避免 Warp/CUDA 依赖），
用真实 `MergedModel`（单 FreeJoint body + BoxShape）验证 FK 和形状提取路径。

全量 fast suite：794 passed, 1 skipped（无回归）。

## Known Limitations / Coverage Gaps

- **真实 GpuEngine 端到端测试缺失**：当前测试用 mock engine，不覆盖真实 Warp
  数组的 `.numpy()` 调用。端到端测试（真实 GpuEngine + RerunBackend）延后到 Q51。
- **CPU FK 而非 GPU FK**：`x_world_R_wp` / `x_world_r_wp` 已有，但未使用。
  如果渲染帧率成为瓶颈，可切换为直接读取 GPU FK 结果（避免重算）。
- **Step 4 阻塞于 Q51**：传感器字段（IMU body vel、joint force）需要先定案
  CPU/GPU 统一 telemetry contract，见 Q51。

## 关键思考

### Non-obvious technical decisions

**为什么用 CPU FK 而不是直接读 `x_world_R_wp`？**

两种方案都可行：
- 方案 A（当前）：`q_wp.numpy()` → CPU FK → `build_render_scene`
- 方案 B：`x_world_R_wp.numpy()` + `x_world_r_wp.numpy()` → 直接构造 `SpatialTransform` 列表

方案 A 更简单：`build_render_scene` 接受 `List[SpatialTransform]`，CPU FK 直接产出。
方案 B 需要从 (N, nb, 3, 3) + (N, nb, 3) 重建 `SpatialTransform` 对象，多一层转换。
渲染帧率（~30 fps）远低于物理步频（5000 Hz），CPU FK 开销（~0.1ms）完全可接受。

**为什么不把 `build_render_scene_from_gpu` 放在 GpuEngine 里？**

依赖方向规则：`physics/` 不能 import `rendering/`。
`build_render_scene_from_gpu` 放在 `rendering/scene_builder.py`，
接受 duck-typed `engine` 参数（不 import GpuEngine 类型），符合依赖方向。

### Debugging difficulties

无。实现路径清晰，mock 测试策略直接。

## Commit

`9cbad1a feat: add RenderBackend ABC, MatplotlibBackend, RerunBackend, GPU bridge (Q50)`
