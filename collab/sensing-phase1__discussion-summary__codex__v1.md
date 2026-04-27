Initiative: sensing-phase1
Stage: discussion-summary
Author: codex
Version: v1
Date: 2026-04-26
Status: review-request
Related Files: collab/render-physics-pipeline__proposal__codex__v1.md, collab/render-physics-pipeline__implementation-note__codex__v1.md, OPEN_QUESTIONS.md#Q51, OPEN_QUESTIONS.md#Q52, OPEN_QUESTIONS.md#Q53
Owner Summary: 记录 2026-04-26 在 phase-1 published-frame consumer integration 完成之后，关于 `sensing/` 模块边界、`StateSampleView` 第一阶段收敛、以及 `SurfaceQueryView / ImagingView` 仍待进一步审查的问题，供后续与 Claude 继续审查和展开实现。

---

## 1. Current Baseline

到目前为止，`PublishedFrame` 主线已经从控制平面设计推进到了 phase-1 可运行集成：

- publish/control-plane 已落地
- CPU/GPU engine 都接上了统一 publish surface
- 真实消费路径已打通：
  - `PublishedFrame -> host debug snapshot`
  - `PublishedFrame -> RenderScene`
  - `PublishedFrame -> RenderScene -> backend`
  - `PublishedFrame -> TelemetrySnapshot`
- 当前关键测试状态：
  - `29 passed`

实现级详情见：

- `collab/render-physics-pipeline__implementation-note__codex__v1.md`

---

## 2. New Architectural Conclusion

本轮最重要的新结论是：

**sensor / view 不应继续堆进 `physics core`。**

推荐边界：

- `physics/`
  - 负责真值与 publish contract
  - 如：`PublishedFrame`、`ForceState`、`ContactInfo`
- `rendering/`
  - 负责可视化与 render backend
  - 如：`RenderScene`、scene bridge、backend helpers
- `sensing/`
  - 负责 sensor views、sensor readings、观测模型与 builders

也就是说：

`physics -> sensing`
`physics -> rendering`

而不是让 `sensing` 成为 `physics core` 的一部分。

---

## 3. Why `sensing/` Instead of `observation/`

包名倾向已收敛为：

- `sensing/`

原因：

- 比 `observation/` 更清楚
- 不易与 RL observation / policy observation 混淆
- 对 IMU / camera / lidar / force sensor 这些传感器语义更直接

---

## 4. Sensor Views: First-Level Taxonomy

不做枚举式大一统 `SensorView`。

当前仅把“按观测机制分类”视为正确方向，但**并非三类 view 都已进入可实现状态**。

### 4.1 `StateSampleView`

面向：

- IMU
- joint encoder
- joint velocity sensor
- generalized acceleration / force telemetry
- force/contact sensor
- binary contact switch

本质：

- 采样物理状态或力学量
- 不依赖几何可见性
- 不依赖渲染

### 4.2 `SurfaceQueryView`

面向：

- LiDAR
- depth ray sensor
- proximity / range sensors
- raycast clearance sensors

本质：

- 关心“表面在哪里”
- 偏几何查询，不是成像

当前状态：

- 这是一个合理方向
- 但 CPU/GPU query 执行差异尚未收敛
- 已提升为 open question，见 `OPEN_QUESTIONS.md#Q53`

### 4.3 `ImagingView`

面向：

- RGB camera
- segmentation camera
- event camera
- thermal camera
- future imaging sensors

本质：

- 需要“可成像世界”
- 依赖几何 + 语义 + 材质/光照等更重的输入

当前状态：

- 不再视为已定归属
- 它是否应属于 `sensing/` 还是 `rendering/`，已提升为 open question
- 见 `OPEN_QUESTIONS.md#Q53`

---

## 5. First Batch of Sensors

第一批优先实现，不从 camera 开始，而从 numeric / state-like sensors 开始：

- `IMU`
- `joint encoder`
- `force/contact sensor`
- `contact switch`

这些都映射到：

- `StateSampleView`

原因：

- 它们最直接验证 published contract 是否足够
- 不会过早陷进 rendering / material / camera model 细节
- 对 Q51 的收敛价值最高

---

## 6. `StateSampleView` Design Philosophy

`StateSampleView` 的设计哲学已经明确：

1. **只从 `PublishedFrame` 和正式公开 bridge 取值**
   - 不读 engine 私有 scratch

2. **CPU/GPU 不对称时显式返回 `None`**
   - 不伪造“统一值”
   - 不猜测未 publish 的量

3. **先做 host-side、语义稳定的基础 view**
   - 它是 numeric sensors 的公共输入层
   - 不是最终 sensor output

4. **后续 sensor reading 都从它派生**
   - `IMUReading`
   - `JointStateReading`
   - `ForceSensorReading`
   - `ContactStateReading`

---

## 7. `StateSampleView` Minimal Field Set

当前收敛的最小字段集：

```python
@dataclass
class StateSampleView:
    frame_id: int
    step_index: int
    sim_time: float
    env_idx: int

    q: object | None
    qdot: object | None

    X_world: object | None
    v_bodies: object | None
    contact_count: object | None

    telemetry: TelemetrySnapshot | None
```

这里已经不再建议把 `TelemetrySnapshot` 的字段平摊复制进 `StateSampleView`。

当前更倾向的关系是：

- `StateSampleView`
  - 补充运动学与 contact count
- `TelemetrySnapshot`
  - 继续承载 telemetry / force / acceleration 相关字段
- 两者通过**组合**而不是平铺复制的方式关联

### Field-source summary

#### CPU

- `q / qdot`：来自 `CpuPublishedFrame`
- `X_world / v_bodies`：来自 `CpuPublishedFrame`
- `force_sensor`：当前没有，`None`
- `contact_count`：来自 `CpuPublishedFrame.contact_count`
- `telemetry`：来自 `TelemetrySnapshot`（其内部再映射 `ForceState`）

#### GPU

- `q / qdot`：来自 `GpuPublishedFrame.q_wp / qdot_wp`
- `X_world / v_bodies`：来自 published slot buffers
- `contact_count`：来自 `contact_count_wp`
- `telemetry`：来自 `TelemetrySnapshot`（其内部再映射 `telemetry_ref`）

---

## 8. First Reading Schemas

第一批对应的最小 reading schema 草案：

### 8.1 `IMUReading`

```python
@dataclass
class IMUReading:
    frame_id: int
    sim_time: float
    env_idx: int
    body_index: int

    orientation_world_R: object | None
    angular_velocity_body: object | None
    linear_acceleration_body: object | None
```

### 8.2 `JointStateReading`

```python
@dataclass
class JointStateReading:
    frame_id: int
    sim_time: float
    env_idx: int

    joint_pos: object
    joint_vel: object
```

### 8.3 `ForceSensorReading`

```python
@dataclass
class ForceSensorReading:
    frame_id: int
    sim_time: float
    env_idx: int

    qfrc_applied: object | None
    tau_smooth: object | None
    body_force: object | None
    contact_force: object | None
```

### 8.4 `ContactStateReading`

```python
@dataclass
class ContactStateReading:
    frame_id: int
    sim_time: float
    env_idx: int

    contact_count: int | None
```

这些 reading 都从：

- `StateSampleView`

派生。

---

## 9. `physics/telemetry.py` Ownership: Temporary Placement

当前共识：

- `physics/telemetry.py` 先保留
- 它目前更接近 `PublishedFrame` / `ForceState` 的 host-side bridge

但这不是最终模块归属定案。

如果未来：

- `TelemetrySnapshot` 持续长出更强的 sensor-facing 语义

则应考虑迁移到：

- `sensing/`

并把 `physics/` 仅保留为真值层与 publish contract 层。

这一点已经记入：

- `OPEN_QUESTIONS.md#Q51`

短期内更偏向：

- `StateSampleView` 组合 `TelemetrySnapshot`

而不是在 `sensing/` 里重写一份与 `TelemetrySnapshot` 平行的字段集合。

---

## 10. Recommended Next Step

如果继续实现，最自然的入口不是 camera，也不是 async queue，而是先正式立：

- `sensing/state_sample.py`

第一版只包含：

- `StateSampleView`
- `build_state_sample_view(...)`

后续再逐步加：

- `sensing/readings.py`
- `sensing/builders.py`

并优先实现：

- `build_joint_state_reading(...)`
- `build_contact_state_reading(...)`
- 再到 `build_imu_reading(...)`
- `build_force_sensor_reading(...)`

当前**不建议**马上做：

- `ImagingView` 实现
- `SurfaceQueryView` 的 CPU/GPU query runtime

---

## 11. Questions For Claude

当前最值得请 Claude challenge 的点：

1. `sensing/` 的包边界是否合适
2. `StateSampleView` 是否应采用“运动学字段 + `TelemetrySnapshot` 组合”这一结构
3. 第一批 sensor 优先顺序是否合理
4. `TelemetrySnapshot` 留在 `physics/` 作为过渡层是否可接受
5. `SurfaceQueryView` / `ImagingView` 是否应在 `Q53` 收敛前继续停留在设计层，而不是进入实现层
