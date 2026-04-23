Initiative: q50-render-backend
Stage: proposal
Author: claude
Version: v1
Date: 2026-04-23
Status: draft
Related Files: OPEN_QUESTIONS.md#Q50, rendering/render_scene.py, rendering/scene_builder.py, rendering/viewer.py
Owner Summary: 引入 RenderBackend(ABC) 接口，包装现有 matplotlib viewer，实现 RerunBackend，补充 GPU 桥接和传感器字段。Phase 3 渲染层主线，RL 训练监控的前提。

---

## Problem

渲染层目前约 15% 完成，存在四个已确认的架构缺陷：

1. **无 `RenderBackend(ABC)` 接口**：`viewer.py` 是 matplotlib 的具体实现，无抽象契约。
   加新后端只能并排堆放，没有共同接口。

2. **无 GPU → RenderScene 桥接**：`scene_builder.py` 只接受 CPU 类型（`MergedModel` /
   `RobotTreeNumpy`）。`GpuEngine` 有 N 个并行 env 的 GPU 数组，没有
   `build_render_scene_from_gpu(engine, env_idx)` 路径。

3. **无传感器提取路径**：`RenderScene` 缺 IMU（body linear/angular vel）、关节力矩、
   力传感器字段——这些是 RL obs 的必要输入，也是 RL 训练监控的必要可视化元素。

4. **无 multi-env 视图接口**：无法在训练中选取 env #k 查看。

---

## Goal

本 proposal 覆盖 Q50 实施计划的 Step 1–4（Step 5 Vulkan 是中期，不在本轮范围）：

- Step 1：引入 `RenderBackend(ABC)`，`MatplotlibBackend` 包装现有 `viewer.py`
- Step 2：`RerunBackend` 实现（验证接口 + 实际可用）
- Step 3：GPU 桥接 `build_render_scene_from_gpu(engine, env_idx)`
- Step 4：`RenderScene` 加传感器字段（IMU body vel + joint torque）

---

## Scope

**In scope（本轮）**：
- `RenderBackend(ABC)` 接口定义
- `MatplotlibBackend`（包装现有 viewer，不改 viewer 内部逻辑）
- `RerunBackend`（`rerun-sdk` 依赖，可选安装）
- `build_render_scene_from_gpu(engine, env_idx)` — CPU numpy 提取，不做 DLPack 零拷贝
- `RenderScene` 新增 `sensor_data: SensorData | None` 字段（IMU + joint torque）
- `rendering/__init__.py` 导出新公共 API

**Out of scope（本轮）**：
- VulkanBackend（Step 5，中期）
- DLPack 零拷贝 GPU→CPU（Q31 gap item 3，性能优化）
- 多机器人 merged scene（Q20 Scene 重构后再做）
- 域随机化视觉 / USD 导出（Phase 4+）

---

## Affected Files / Layers

```
rendering/
  backends/                    ← 新建目录
    __init__.py
    base.py                    ← RenderBackend(ABC)
    matplotlib_backend.py      ← 包装现有 viewer.py
    rerun_backend.py           ← RerunBackend（可选依赖）
  render_scene.py              ← 新增 SensorData dataclass + RenderScene.sensor_data 字段
  scene_builder.py             ← 新增 build_render_scene_from_gpu()
  __init__.py                  ← 更新导出

rendering/viewer.py            ← 不改（MatplotlibBackend 包装它）
rendering/shape_artists.py     ← 不改

tests/rendering/               ← 新建
  test_render_backend_abc.py   ← ABC 接口契约测试
  test_matplotlib_backend.py   ← MatplotlibBackend offscreen 测试
  test_rerun_backend.py        ← RerunBackend headless save 测试
  test_gpu_bridge.py           ← build_render_scene_from_gpu 测试
  test_sensor_data.py          ← SensorData 字段测试
```

依赖方向不变：`rendering/` → `physics/`，`physics/` 不导入 `rendering/`。

---

## Proposed Design

### 1. `RenderBackend(ABC)` — `rendering/backends/base.py`

```python
from abc import ABC, abstractmethod
from ..render_scene import RenderScene


class RenderBackend(ABC):
    """Abstract contract for all rendering backends.

    Lifecycle: open() → render_frame()* → close()
    Headless/CI: set_output(path) before open().
    """

    @abstractmethod
    def open(self) -> None:
        """Initialize the backend (open window or prepare file sink)."""
        ...

    @abstractmethod
    def render_frame(
        self,
        scene: RenderScene,
        timestamp: float,
        env_index: int = 0,
    ) -> None:
        """Render one frame from scene at the given simulation timestamp.

        Args:
            scene      : Backend-agnostic scene snapshot.
            timestamp  : Simulation time in seconds (required; backends that
                         don't use it may ignore it, but callers must supply it).
            env_index  : Which parallel env to visualise (default 0).
                         Backends that don't support multi-env may ignore it.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Flush and release all backend resources."""
        ...

    def set_output(self, path: str) -> None:
        """Configure headless/CI output path (e.g. .gif, .rrd, .mp4).

        Optional override. Default: no-op (live window mode).
        """

    @property
    @abstractmethod
    def supports_offscreen(self) -> bool:
        """True if this backend can render without a display (CI-safe)."""
        ...
```

设计说明：
- `timestamp` 必填：Rerun 需要，matplotlib 忽略，代价为零，统一调用方签名。
- `env_index` 可选：单 env 调用方不需要关心，多 env 训练监控时传入。
- `set_output` 非抽象：不支持 headless 的后端不需要覆盖（默认 no-op）。
- 无 `render_trajectory()` 方法：轨迹回放是调用方循环调用 `render_frame()`，不是后端职责。

### 2. `MatplotlibBackend` — `rendering/backends/matplotlib_backend.py`

包装现有 `viewer.py` 的 `RobotViewer`，适配 `RenderBackend` 接口。

```python
class MatplotlibBackend(RenderBackend):
    """Matplotlib 3D backend — wraps existing viewer.py.

    Supports offscreen rendering via MPLBACKEND=Agg.
    """

    def __init__(self, figsize=(10, 8), save_path: str | None = None): ...
    def open(self) -> None: ...          # plt.figure() + Axes3D
    def render_frame(self, scene, timestamp, env_index=0) -> None: ...  # draw shapes + contacts
    def close(self) -> None: ...         # save gif if save_path set
    def set_output(self, path: str) -> None: ...
    @property
    def supports_offscreen(self) -> bool: return True  # Agg backend
```

`render_frame` 内部调用 `shape_artists.py` 的绘制逻辑（已有），不重复实现。
现有 `RobotViewer` 保持不变，`MatplotlibBackend` 是薄包装层。

### 3. `RerunBackend` — `rendering/backends/rerun_backend.py`

```python
class RerunBackend(RenderBackend):
    """Rerun SDK backend — headless CI + live training monitor.

    Requires: pip install rerun-sdk
    """

    def __init__(self, app_id: str = "robot_simulator", save_path: str | None = None): ...
    def open(self) -> None: ...          # rr.init() + rr.save() or rr.connect()
    def render_frame(self, scene, timestamp, env_index=0) -> None: ...
    def close(self) -> None: ...
    def set_output(self, path: str) -> None: ...
    @property
    def supports_offscreen(self) -> bool: return True
```

`render_frame` 内部映射：
- `PositionedShape(box)` → `rr.Boxes3D`
- `PositionedShape(sphere)` → `rr.Points3D` + radius
- `PositionedShape(capsule/cylinder)` → `rr.Capsules3D / Cylinders3D`
- `PositionedShape(convex_hull)` → `rr.Mesh3D`（顶点 + 凸包面）
- `ContactPoint` → `rr.Arrows3D`（法向箭头）
- `skeleton_links` → `rr.LineStrips3D`
- `SensorData.body_velocities` → `rr.Arrows3D`（速度向量，可选）

时间轴：`rr.set_time_seconds("sim_time", timestamp)`

### 4. GPU 桥接 — `scene_builder.py` 新增函数

```python
def build_render_scene_from_gpu(
    engine: "GpuEngine",
    env_idx: int = 0,
    include_contacts: bool = True,
) -> RenderScene:
    """Build a RenderScene from GpuEngine state for one env.

    Copies GPU arrays to CPU (numpy). Not zero-copy — intended for
    visualization/debug, not hot-path RL obs extraction.

    Args:
        engine      : GpuEngine after a step.
        env_idx     : Which parallel env to extract (0-indexed).
        include_contacts: Whether to extract contact points.

    Returns:
        RenderScene ready for any rendering backend.
    """
```

实现路径：
1. `engine.q[env_idx].numpy()` → 调用 `merged.tree.forward_kinematics(q)` → `X_world`
2. 用现有 `build_render_scene(merged, X_world, ...)` 构建场景
3. 接触点：`engine.query_contacts(env_idx)` → `ContactInfo` list → `ContactPoint` list
4. 传感器数据：`engine.body_velocities_wp[env_idx].numpy()` → `SensorData`

注意：`GpuEngine` 目前没有 `merged` 属性，需要调用方传入或从 `engine.static_data` 重建。
方案：`build_render_scene_from_gpu(engine, merged, env_idx)` — 调用方传入 `merged`。

### 5. `RenderScene` 传感器字段 — `render_scene.py`

```python
@dataclass(frozen=True)
class SensorData:
    """Sensor readings extracted from physics state for RL obs / visualization.

    All arrays are (n_bodies, ...) numpy arrays.
    """
    body_linear_vel: NDArray[np.float64]   # (n_bodies, 3) world-frame linear velocity
    body_angular_vel: NDArray[np.float64]  # (n_bodies, 3) world-frame angular velocity
    joint_torques: NDArray[np.float64]     # (n_joints,) applied joint torques


@dataclass
class RenderScene:
    shapes: list[PositionedShape]
    contacts: list[ContactPoint]
    terrain: TerrainInfo
    skeleton_links: list[tuple[NDArray, NDArray]]
    body_positions: list[NDArray]
    body_names: list[str]
    sensor_data: SensorData | None = None   # ← 新增，向后兼容（默认 None）
    deformable_meshes: list = field(default_factory=list)
    particles: list = field(default_factory=list)
```

`sensor_data=None` 保持向后兼容，现有调用方不需要修改。

---

## Test Plan

### 单元测试

**`test_render_backend_abc.py`**（5 tests）：
- `MatplotlibBackend` 是 `RenderBackend` 的子类
- `RerunBackend` 是 `RenderBackend` 的子类
- `supports_offscreen` 两者均为 True
- `set_output` 调用不抛异常
- 空 `RenderScene` 不崩溃（open → render_frame → close）

**`test_matplotlib_backend.py`**（4 tests）：
- offscreen 渲染不抛异常（MPLBACKEND=Agg）
- `set_output("out.gif")` → close 后文件存在
- 含 shapes + contacts 的 scene 正常渲染
- `env_index` 参数被接受（不报错）

**`test_rerun_backend.py`**（4 tests）：
- `set_output("debug.rrd")` → close 后 .rrd 文件存在
- 所有 shape 类型（box/sphere/capsule/cylinder/convex_hull）不崩溃
- contacts 渲染为箭头不崩溃
- `timestamp` 正确传入（通过 rrd 文件可验证，或 mock rr.set_time_seconds）

**`test_gpu_bridge.py`**（4 tests）：
- `build_render_scene_from_gpu` 返回 `RenderScene`
- shapes 数量与 merged model 一致
- `env_idx` 越界抛 IndexError
- 含接触时 contacts 非空

**`test_sensor_data.py`**（3 tests）：
- `RenderScene(sensor_data=None)` 向后兼容
- `SensorData` 字段形状正确
- `build_render_scene_from_gpu(include_contacts=False)` 时 contacts 为空

### 集成测试

- 四足 quadruped 跑 50 步 → `build_render_scene_from_gpu` → `RerunBackend.render_frame` → `.rrd` 文件存在且非空
- 现有 `simple_quadruped.py` example 不受影响（`RobotViewer` 路径不变）

---

## Tradeoffs

| 决策 | 选择 | 备选 | 理由 |
|------|------|------|------|
| GPU 桥接是否零拷贝 | 否（CPU numpy copy） | DLPack Warp→Torch | 可视化不在热路径；零拷贝是 Q31 gap item 3，独立优化 |
| `timestamp` 是否必填 | 是 | 可选 | Rerun 需要；统一签名比可选参数更清晰；matplotlib 忽略代价为零 |
| `set_output` 是否抽象 | 否（默认 no-op） | 抽象 | 不支持 headless 的后端不应被强制实现；live window 是合理默认 |
| `SensorData` 是否独立 dataclass | 是 | 直接加字段到 RenderScene | 传感器字段是可选的，独立 dataclass 更清晰；`None` 向后兼容 |
| `RerunBackend` 是否可选依赖 | 是（`pip install .[rerun]`） | 强依赖 | rerun-sdk 约 50MB，不应强制所有用户安装 |
| `build_render_scene_from_gpu` 签名 | `(engine, merged, env_idx)` | `(engine, env_idx)` | GpuEngine 目前无 merged 属性；传入 merged 更显式，避免隐式耦合 |

---

## References

- **Drake `SceneGraph`**：role-based geometry separation + pluggable renderer，最直接参考。
  `SceneGraph` 持有 geometry roles（proximity/illustration/perception），renderer 订阅 illustration role。
  我们的 `RenderScene` 对应 illustration snapshot，`RenderBackend` 对应 renderer。

- **Isaac Lab 3.0 pluggable renderer**：`VisualizationMarkers` + `InteractiveScene` 分离，
  tiled rendering for N envs（`TiledCamera`）。multi-env 视图的参考。

- **Rerun 官方文档**：
  - operating-modes（headless/save）：`rr.save("debug.rrd")` 无需 DISPLAY
  - archetype reference：`Boxes3D / Capsules3D / Cylinders3D / Mesh3D / Arrows3D / LineStrips3D`
  - 0.24 起支持 file + live 双 sink 同时输出

- **Genesis 三后端**：PyRender / LuisaRender / Madrona，后端切换通过 `gs.init(backend=...)` 实现。
  我们的 `RenderBackend` 工厂函数可参考此模式。

- **Q50 OPEN_QUESTIONS.md**：完整背景、缺陷分析、后端路线图。

---

## 关键思考

**为什么先做 Rerun 而不是直接做 Vulkan？**

Rerun 的核心价值不是画质，而是**验证 `RenderBackend` 接口的正确性**，同时提供
RL 训练监控能力（timeline scrubbing、headless CI）。Vulkan 是高性能实时渲染，
但实现复杂度高（需要 swapchain、renderpass、descriptor sets），在接口未稳定前
投入 Vulkan 等于在错误的抽象层上建设。

Rerun 的另一个价值：它的 archetype 系统（`Boxes3D / Capsules3D / Mesh3D`）
与我们的 `PositionedShape` 类型几乎一一对应，实现成本低，验证接口的覆盖面广。

**`env_index` 参数的设计权衡**

`render_frame(scene, timestamp, env_index=0)` 中 `env_index` 是 hint，不是 scene 的一部分。
这意味着 `RenderScene` 本身是单 env 的快照（已经从 GPU 提取了特定 env 的数据），
`env_index` 只是传给后端用于标注（例如 Rerun 的 entity path `env_0/robot/...`）。

备选方案是让 `RenderScene` 包含所有 env 的数据，后端自己选取。
拒绝原因：`RenderScene` 会变成 `(N_envs, ...)` 的批量结构，
与现有 CPU 单 env 路径不兼容，且大多数后端只需要一个 env。
保持 `RenderScene` 为单 env 快照，`build_render_scene_from_gpu(engine, merged, env_idx)` 负责提取。
