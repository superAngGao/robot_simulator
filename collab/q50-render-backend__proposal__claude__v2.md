Initiative: q50-render-backend
Stage: proposal
Author: claude
Version: v2
Date: 2026-04-23
Status: draft
Related Files: OPEN_QUESTIONS.md#Q50, OPEN_QUESTIONS.md#Q51, rendering/render_scene.py, rendering/scene_builder.py, rendering/viewer.py, rendering/shape_artists.py
Owner Summary: 引入 RenderBackend(ABC) 接口，MatplotlibBackend 自持 figure/artists（不委托 RobotViewer），RerunBackend 含 convex_hull face 提取，GPU 桥接用 engine.merged。Step 4 传感器字段延后至 Q51。v2 修正 Codex challenge 三个 findings。

---

## Changes from v1

Codex challenge（commit b60452d）发现三个问题，v2 全部收敛：

1. **Finding 2（高风险）— convex_hull 缺 face indices**
   `scene_builder.py` 目前只把 `vertices` 存入 `params`，没有面拓扑。
   v2 修正：`_shape_to_type_params` 对 `ConvexHullShape` 同时提取
   `faces`（`shape.face_topology().simplices`），存入 `params["faces"]`。
   `RerunBackend` 直接读 `params["faces"]`，不在 backend 内重跑凸包。
   `MatplotlibBackend` 的 `draw_convex_hull` 已有 `scipy.ConvexHull`，
   但现在可以直接用 `params["faces"]` 跳过重算（可选优化，不阻塞）。

2. **Finding 3（中风险）— MatplotlibBackend 不是薄包装**
   `RobotViewer` 只有批量 API（`render_pose` / `animate`），没有
   `open → render_frame* → close` 生命周期。
   v2 修正：`MatplotlibBackend` 自持 `fig / ax`，直接调用
   `shape_artists.py` 的绘制函数（已有），不委托 `RobotViewer`。
   `RobotViewer` 保持不变，`MatplotlibBackend` 是对 `shape_artists` 的薄包装。

3. **Open Question — `build_render_scene_from_gpu` 签名**
   `GpuEngine` 已有 `self.merged`（`engine.py:64`，`PhysicsEngine.__init__`）。
   v2 修正：签名改为 `(engine, env_idx=0, include_contacts=True)`，
   内部用 `engine.merged`，不需要调用方传入。

4. **Finding 1（阻塞）— joint_torques 无干净数据源**
   已在 v1 处理：Step 4 整体延后至 Q51 收敛。
   v2 同步清理 Scope / Affected Files / Test Plan 中残留的 `SensorData` 引用。

---

## Problem

渲染层目前约 15% 完成，存在三个已确认的架构缺陷（Finding 1 的传感器问题归入 Q51）：

1. **无 `RenderBackend(ABC)` 接口**：`viewer.py` 是 matplotlib 的具体实现，无抽象契约。
   加新后端只能并排堆放，没有共同接口。

2. **无 GPU → RenderScene 桥接**：`scene_builder.py` 只接受 CPU 类型（`MergedModel` /
   `RobotTreeNumpy`）。`GpuEngine` 有 N 个并行 env 的 GPU 数组，没有
   `build_render_scene_from_gpu(engine, env_idx)` 路径。

3. **convex_hull 缺 face topology**：`scene_builder.py` 只把 `vertices` 存入
   `PositionedShape.params`，没有面索引。`RerunBackend` 需要 `rr.Mesh3D(vertices, indices)`，
   如果 face 不在 `RenderScene` 里，backend 只能重跑凸包或回头依赖 physics 对象，
   两者都违背 backend-agnostic 设计目标。

---

## Goal

本 proposal 覆盖 Q50 实施计划的 Step 1–3：

- Step 1：引入 `RenderBackend(ABC)`，`MatplotlibBackend` 自持 figure/artists
- Step 2：`RerunBackend` 实现（含 convex_hull face 提取）
- Step 3：GPU 桥接 `build_render_scene_from_gpu(engine, env_idx)`，内部用 `engine.merged`
- ~~Step 4：传感器字段~~ → 延后，见 Q51
- ~~Step 5：VulkanBackend~~ → 中期，不在本轮

---

## Scope

**In scope（本轮）**：
- `RenderBackend(ABC)` 接口定义
- `MatplotlibBackend`（自持 fig/ax，调用 `shape_artists.py`，不改 `viewer.py`）
- `RerunBackend`（`rerun-sdk` 可选依赖）
- `scene_builder._shape_to_type_params` 对 `ConvexHullShape` 补充 `faces` 字段
- `build_render_scene_from_gpu(engine, env_idx)` — CPU numpy 提取，用 `engine.merged`
- `rendering/__init__.py` 导出新公共 API

**Out of scope（本轮）**：
- `SensorData` / `RenderScene.sensor_data` — 延后至 Q51
- VulkanBackend（Step 5，中期）
- DLPack 零拷贝 GPU→CPU（Q31 gap item 3）
- 多机器人 merged scene（Q20 Scene 重构后再做）
- `MeshShape` 支持 — 显式标注为 unsupported，backend 跳过（见下文）

---

## Affected Files / Layers

```
rendering/
  backends/                    ← 新建目录
    __init__.py
    base.py                    ← RenderBackend(ABC)
    matplotlib_backend.py      ← 自持 fig/ax，调用 shape_artists
    rerun_backend.py           ← RerunBackend（可选依赖）
  render_scene.py              ← 不改（sensor_data 延后）
  scene_builder.py             ← _shape_to_type_params 补充 convex_hull faces；
                                  新增 build_render_scene_from_gpu()
  __init__.py                  ← 更新导出

rendering/viewer.py            ← 不改
rendering/shape_artists.py     ← 不改

tests/rendering/               ← 新建
  test_render_backend_abc.py
  test_matplotlib_backend.py
  test_rerun_backend.py
  test_gpu_bridge.py
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
            timestamp  : Simulation time in seconds.
            env_index  : Which parallel env to visualise (default 0).
                         Used for entity path labelling; scene is already
                         a single-env snapshot.
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
- `timestamp` 必填：Rerun 需要，matplotlib 忽略，统一调用方签名。
- `env_index` 是 entity path hint，不是 scene 的一部分（scene 已是单 env 快照）。
- `set_output` 非抽象：不支持 headless 的后端不需要覆盖。
- 无 `render_trajectory()`：轨迹回放是调用方循环 `render_frame()`，不是后端职责。

### 2. `MatplotlibBackend` — `rendering/backends/matplotlib_backend.py`

v2 修正：自持 `fig / ax`，直接调用 `shape_artists.py`，不委托 `RobotViewer`。

```python
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from rendering.shape_artists import (
    draw_box, draw_sphere, draw_cylinder, draw_capsule,
    draw_convex_hull, draw_contacts, draw_terrain, SHAPE_DRAWERS,
)
from .base import RenderBackend
from ..render_scene import RenderScene


class MatplotlibBackend(RenderBackend):
    """Matplotlib 3D backend.

    Manages its own figure/axes. Calls shape_artists directly.
    Supports offscreen rendering via MPLBACKEND=Agg.

    Lifecycle:
        backend = MatplotlibBackend(save_path="out.gif")
        backend.open()
        for scene, t in frames:
            backend.render_frame(scene, t)
        backend.close()  # saves gif if save_path set
    """

    def __init__(self, figsize=(10, 8), save_path: str | None = None): ...
    def set_output(self, path: str) -> None: ...

    def open(self) -> None:
        # plt.figure() + add_subplot(111, projection="3d")
        # store self._fig, self._ax
        ...

    def render_frame(self, scene: RenderScene, timestamp: float, env_index: int = 0) -> None:
        # clear ax, draw shapes via SHAPE_DRAWERS dispatch, draw contacts, draw terrain
        # store frame for gif if save_path set
        ...

    def close(self) -> None:
        # if save_path: FuncAnimation → save gif
        # plt.close(self._fig)
        ...

    @property
    def supports_offscreen(self) -> bool:
        return True  # Agg backend
```

`render_frame` 内部：
1. `ax.cla()` 清除上一帧
2. 遍历 `scene.shapes`，按 `shape_type` 调用 `SHAPE_DRAWERS[shape_type](ax, pos, rot, **params)`
3. `draw_contacts(ax, scene.contacts)`
4. `draw_terrain(ax, scene.terrain)`
5. 如果 `save_path` 设置，把当前 figure 存入帧列表

`close()` 时如果有帧列表，用 `matplotlib.animation.ArtistAnimation` 保存 gif。

### 3. `RerunBackend` — `rendering/backends/rerun_backend.py`

```python
class RerunBackend(RenderBackend):
    """Rerun SDK backend — headless CI + live training monitor.

    Requires: pip install rerun-sdk
    """

    def __init__(self, app_id: str = "robot_simulator", save_path: str | None = None): ...
    def set_output(self, path: str) -> None: ...

    def open(self) -> None:
        # rr.init(self._app_id)
        # if save_path: rr.save(save_path)
        # else: rr.connect()
        ...

    def render_frame(self, scene: RenderScene, timestamp: float, env_index: int = 0) -> None:
        # rr.set_time_seconds("sim_time", timestamp)
        # entity prefix: f"env_{env_index}"
        # dispatch shapes, contacts, skeleton
        ...

    def close(self) -> None: ...

    @property
    def supports_offscreen(self) -> bool:
        return True
```

`render_frame` shape 映射：

| `shape_type` | Rerun archetype | params 字段 |
|---|---|---|
| `box` | `rr.Boxes3D` | `size` |
| `sphere` | `rr.Points3D` + `radii` | `radius` |
| `cylinder` | `rr.Cylinders3D` | `radius`, `length` |
| `capsule` | `rr.Capsules3D` | `radius`, `length` |
| `convex_hull` | `rr.Mesh3D` | `vertices`, **`faces`**（v2 新增） |
| `mesh` | skip（unsupported，log warning） | — |

`convex_hull` 映射（v2 修正）：
```python
verts = params["vertices"]   # (N, 3) float64
faces = params["faces"]      # (F, 3) int32  ← scene_builder 已提取，不在 backend 重算
rr.log(f"{prefix}/shape_{i}", rr.Mesh3D(vertex_positions=verts, triangle_indices=faces))
```

`contacts` → `rr.Arrows3D`（法向箭头）
`skeleton_links` → `rr.LineStrips3D`

### 4. `scene_builder.py` 修改 — convex_hull faces + GPU 桥接

#### 4a. `_shape_to_type_params` 补充 faces

```python
elif isinstance(shape, ConvexHullShape):
    topo = shape.face_topology()          # FaceTopology，已有 .simplices (F,3) int
    return "convex_hull", {
        "vertices": topo.vertices.copy(),  # (N, 3) float64，已是 hull 顶点
        "faces": topo.simplices.copy(),    # (F, 3) int32
    }
```

`ConvexHullShape.face_topology()` 已存在（`geometry.py:586`），
`FaceTopology` 已有 `.simplices`（scipy ConvexHull 在构造时计算，`_build_convexhull_face_topology`）。
这是纯读取，不新增计算。

#### 4b. `build_render_scene_from_gpu`

```python
def build_render_scene_from_gpu(
    engine: "GpuEngine",
    env_idx: int = 0,
    include_contacts: bool = True,
) -> RenderScene:
    """Build a RenderScene from GpuEngine state for one env.

    Copies GPU arrays to CPU (numpy). Not zero-copy — intended for
    visualization/debug, not hot-path RL obs extraction.

    Uses engine.merged (available on all PhysicsEngine subclasses).

    Args:
        engine          : GpuEngine after a step.
        env_idx         : Which parallel env to extract (0-indexed).
        include_contacts: Whether to extract contact points.

    Returns:
        RenderScene ready for any rendering backend.
    """
    merged = engine.merged  # PhysicsEngine.__init__ sets self.merged

    # 1. Extract q for this env, run FK on CPU
    q_np = engine._scratch.q[env_idx].numpy().astype(np.float64)
    X_world = merged.tree.forward_kinematics(q_np)

    # 2. Build scene via existing CPU builder
    contacts = engine.query_contacts(env_idx) if include_contacts else None
    return build_render_scene(merged, X_world, contacts=contacts, terrain=merged.terrain)
```

`engine.merged` 来自 `PhysicsEngine.__init__(self, merged)` → `self.merged = merged`（`engine.py:64-65`）。
`GpuEngine` 继承 `PhysicsEngine`，所以 `engine.merged` 始终可用。

### 5. Unsupported shape policy

`MeshShape` 在 `scene_builder` 里已产出 `shape_type="mesh"`，但 backend 均不支持。
v2 明确策略：

- `MatplotlibBackend.render_frame`：遇到 `mesh` 跳过，不报错。
- `RerunBackend.render_frame`：遇到 `mesh` 调用 `logging.warning("RerunBackend: mesh shape not supported, skipping")`，不崩溃。
- 测试：`test_render_backend_abc.py` 加一个 `mesh` shape 的 scene，验证两个 backend 均不抛异常。

---

## Test Plan

### 单元测试

**`test_render_backend_abc.py`**（6 tests）：
- `MatplotlibBackend` 是 `RenderBackend` 的子类
- `RerunBackend` 是 `RenderBackend` 的子类
- `supports_offscreen` 两者均为 True
- `set_output` 调用不抛异常
- 空 `RenderScene` 不崩溃（open → render_frame → close）
- 含 `mesh` shape 的 scene 不崩溃（unsupported shape policy）

**`test_matplotlib_backend.py`**（4 tests）：
- offscreen 渲染不抛异常（MPLBACKEND=Agg）
- `set_output("out.gif")` → close 后文件存在
- 含 shapes（box/sphere/capsule/cylinder/convex_hull）+ contacts 的 scene 正常渲染
- `env_index` 参数被接受（不报错）

**`test_rerun_backend.py`**（4 tests）：
- `set_output("debug.rrd")` → close 后 .rrd 文件存在
- 所有支持的 shape 类型（box/sphere/capsule/cylinder/convex_hull）不崩溃
- `convex_hull` 使用 `params["faces"]`，不重跑凸包（mock 验证 `rr.Mesh3D` 被调用）
- contacts 渲染为箭头不崩溃

**`test_gpu_bridge.py`**（4 tests）：
- `build_render_scene_from_gpu` 返回 `RenderScene`
- shapes 数量与 merged model 一致
- `env_idx` 越界抛 IndexError
- `include_contacts=False` 时 contacts 为空

**`test_scene_builder_convex_hull.py`**（2 tests）：
- `_shape_to_type_params(ConvexHullShape(...))` 返回的 params 含 `"faces"` 键
- `faces` 形状为 `(F, 3)` int，`vertices` 形状为 `(N, 3)` float

### 集成测试

- 四足 quadruped 跑 50 步 → `build_render_scene_from_gpu` → `RerunBackend.render_frame` → `.rrd` 文件存在且非空
- 现有 `simple_quadruped.py` example 不受影响（`RobotViewer` 路径不变）

---

## Tradeoffs

| 决策 | 选择 | 备选 | 理由 |
|------|------|------|------|
| `MatplotlibBackend` 是否委托 `RobotViewer` | 否，自持 fig/ax | 委托 | `RobotViewer` 是批量 API，无法适配帧级生命周期；`shape_artists` 已有所有绘制逻辑 |
| convex_hull faces 在哪里提取 | `scene_builder`（构建 RenderScene 时） | backend 内重算 | backend-agnostic 原则：scene 是完整快照，backend 不应依赖 physics 对象 |
| `build_render_scene_from_gpu` 是否传入 `merged` | 否，用 `engine.merged` | 调用方传入 | `PhysicsEngine` 已有 `self.merged`，传入是多余参数；若未来需要 override 再加 |
| GPU 桥接是否零拷贝 | 否（CPU numpy copy） | DLPack Warp→Torch | 可视化不在热路径；零拷贝是 Q31 gap item 3，独立优化 |
| `MeshShape` 支持 | 跳过 + warning | 报错 / 实现 | mesh 渲染需要 loader，超出本轮 scope；silent skip 比崩溃更友好 |
| `SensorData` / Step 4 | 延后至 Q51 | 本轮实现 | GPU torque 无干净数据源（Q51 Finding 1）；不阻塞几何渲染主线 |
| `timestamp` 是否必填 | 是 | 可选 | Rerun 需要；统一签名比可选参数更清晰 |

---

## References

- **Drake `SceneGraph`**：role-based geometry separation + pluggable renderer。
- **Isaac Lab 3.0 pluggable renderer**：`VisualizationMarkers` + `InteractiveScene` 分离。
- **Rerun 官方文档**：`rr.Mesh3D(vertex_positions, triangle_indices)`，`rr.save()` headless。
- **Q50 OPEN_QUESTIONS.md**：完整背景、缺陷分析、后端路线图。
- **Q51 OPEN_QUESTIONS.md**：force sensor / torque telemetry contract，Step 4 前置。
- **Codex challenge b60452d**：三个 findings，v2 全部收敛。
