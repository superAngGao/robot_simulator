---
Initiative: q54-optical-l3-direct-light
Stage: implementation-note
Author: claude
Version: v1
Date: 2026-05-02
Status: complete
Commit: 5ed31e1
Related Files: optics/execution.py, optics/scene.py, sensing/optical.py, tests/unit/optics/test_direct_light_executor.py
Owner Summary: L3 direct-light optical executor 已落地。CpuDirectLightOpticalExecutor 实现了确定性两面 Lambertian 直接光照，支持 point/directional 光源和可选 shadow rays，输出 rgb (float64, unbounded) + intensity (BT.709 luminance)。shadow occlusion 走模块级 helper，直接读 snapshot.acceleration，不调用 executor 私有方法。81 tests passed。
---

# Q54 L3 Direct-Light Implementation Note

## 1. 实现摘要

本次提交在 L2 BVH executor 基础上，新增了 `CpuDirectLightOpticalExecutor`，完成 Q54 L3 阶段。

### 核心设计

```
CpuDirectLightOpticalExecutor
  ├── geometric_executor: CpuBvhOpticalExecutor (default)
  ├── shadows: bool = True
  ├── ambient_rgb: (r, g, b) = (0.0, 0.0, 0.0)
  ├── background_rgb: (r, g, b) = (0.0, 0.0, 0.0)
  └── shadow_bias: float = 1e-6

execute(snapshot, spec) -> OpticalComputeResult
  1. 委托 geometric_executor 做 first-hit geometry
  2. 对每个 hit ray 遍历 snapshot.lights
  3. 每个 enabled light 计算 Lambertian contribution
  4. shadows=True 时用模块级 _is_occluded() 做 shadow ray
  5. 返回 geometry channels + rgb + intensity
```

## 2. 关键技术决策

### 2.1 directional light 方向约定

采用 "从 shaded point 指向光源" 的约定（OpenGL/GLSL 惯例）。
在 `OpticalLightSpec.position_or_direction_world` docstring 中明确说明。
executor 内部统一 normalize，不假设调用方已归一化。

**备选方案**：用 "光线传播方向"（MuJoCo dir 约定），但此方案需要 executor 内部取反，容易混淆。

### 2.2 shadow ray 实现路径

**最终方案**：模块级 `_is_occluded(snapshot, origin, direction, max_distance, sensor_role) -> bool`

直接读 `snapshot.acceleration`（BVH 数据），不调用 geometric executor 的私有方法。

**拒绝的方案**：
- 调用 `executor.execute(...)` 做 shadow ray：每条 shadow ray 分配完整 result，O(n_channels × n_shadow_rays) 内存分配，太贵。
- `CpuDirectLightOpticalExecutor` 持有 `CpuBvhOpticalExecutor` 引用并调用私有方法：紧耦合，executor 替换时会 break。

模块级 helper 的好处：测试时可独立注入 `CpuReferenceOpticalExecutor`（无 acceleration 要求），生产路径默认 BVH。

### 2.3 missing acceleration 行为

`shadows=True` 时，`_is_occluded` 读取 `snapshot.acceleration`。
若 snapshot 没有 cpu_bvh acceleration，raise `MissingAccelerationError`，不 fallback。

测试路径：可通过显式传入 `geometric_executor=CpuReferenceOpticalExecutor` + `shadows=False` 绕过 BVH 要求。

### 2.4 rgb / intensity 语义

- `rgb`：float64, unbounded linear RGB。不裁剪，tone mapping 是消费者责任。
- `intensity`：BT.709 luminance，`dot(rgb, [0.2126, 0.7152, 0.0722])`。
- 两者都是确定性的，与光照参数和 albedo 严格成比例。

### 2.5 OpticalLightSpec.intensity 单位差异

directional light：`intensity` 是无量纲倍率（sun-like），无距离衰减。
point light：`intensity` 是 W/sr（luminous intensity），有 `1/max(d², eps)` 衰减。

第一版接受此不一致，在 `OpticalLightSpec.intensity` docstring 中明确说明两种 kind 的语义差异。

### 2.6 两面 Lambertian

当前 first-hit executor 已把三角面法线对齐 primary ray（朝向 ray 方向），
因此 L3 第一版天然是 two-sided Lambertian。
在 executor docstring 中明确注明，避免以后误认为 bug。

## 3. 测试覆盖

新增 `tests/unit/optics/test_direct_light_executor.py`，覆盖：

| 测试场景 | 验证点 |
|---------|--------|
| directional light 正面照射 | rgb ∝ albedo × n_dot_l × color × intensity |
| directional light 背面（n_dot_l < 0） | rgb = ambient × albedo |
| point light inverse-square attenuation | rgb × d² = const |
| ambient-only（无光源） | rgb = ambient_rgb × albedo |
| miss -> background_rgb, intensity=0 | miss semantics |
| disabled light 被忽略 | enabled=False 不贡献 |
| shadow ray 遮挡 directional light | 遮挡体在 hit 和光源之间 |
| shadow ray 遮挡 point light（遮挡体在光源前） | max_distance = dist_to_light |
| shadow ray 不遮挡（遮挡体在光源后） | point light max_distance 截断 |
| sensor_role 过滤影响 primary + shadow | roles 过滤一致性 |
| camera postprocess 重塑 rgb/intensity | build_pinhole_camera_image_result |
| missing BVH acceleration raise | MissingAccelerationError |
| schema: channel names, dtype, shape | capabilities 声明与实际输出一致 |

最终运行结果：**81 passed in 0.29s**（含 optics + sensing 全量单元测试）。

## 4. 关键思考

### 4.1 非显而易见的技术决策

**shadow ray max_distance 截断**：
point light 的 shadow ray `max_distance = distance_to_light - shadow_bias`。
若 `max_distance <= 0`（点光源非常靠近 hit point 或 shadow_bias 过大），
treat as unoccluded——这是正确的，因为此时 shadow ray 出发点已超过光源。
代码中加了注释说明原因，避免以后有人把这当 bug 修掉。

**zero-length directional light**：
在 `_validate()` 中检查，不在 `OpticalLightSpec.__init__` 中检查。
保持 spec 是 dumb dataclass，validation 在 executor 入口做（与 L0/L1/L2 executor 风格一致）。

**`rgb` 不裁剪的测试设计**：
测试使用 intensity=2.0 的光源，故意让 rgb 超过 1.0，
验证 `rgb.max() > 1.0` 而不是 `assert rgb <= 1.0`，
显式测试 "unbounded" 的语义。

### 4.2 调试难点

**shadow ray visibility role**：
shadow ray 应使用 primary spec 的 `sensor_role`，不是全局可见性。
"对当前 sensor 不可见的物体不应遮挡该 sensor 的光线"——语义上自洽，但容易漏掉。
测试中显式构造了 roles 分离场景（遮挡体对 "depth" 不可见，对 "rgb" 可见），
验证两种 role 下 shadow behavior 不同。

**ambient + hit 的组合语义**：
`accumulated = ambient_rgb × albedo` 作为基础，光源 contribution 叠加。
miss ray 不经过 albedo，直接返回 `background_rgb`（不乘 albedo）。
这两条路径容易混淆；测试中同时覆盖了 hit-with-no-lights 和 miss 两种情形。

## 5. Non-goals（明确不实现）

以下内容在 L3 第一版中明确不实现，详见算法计划 §14：

- PBR / BRDF
- 镜面反射 / 折射 / 间接光
- tone mapping / gamma / 曝光
- 面光源 / 软阴影
- GPU 直接光照
- Mitsuba / OptiX adapter
- texture sampling

## 6. 后续工作

- L4/L5：GPU executor + Q52 device result lifecycle
- L6：Mitsuba offline / high-fidelity adapter
- InterfaceMaterial 完善后，albedo_rgb 从 per-shape material 读取
- 光源 intensity 单位统一（directional 和 point 分离为不同字段或单位）
