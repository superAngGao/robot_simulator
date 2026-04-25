Initiative: render-physics-pipeline
Stage: proposal
Author: codex
Version: v1
Date: 2026-04-24
Status: draft
Related Files: physics/engine.py, physics/cpu_engine.py, physics/gpu_engine.py, physics/dynamics_cache.py, simulator.py, rendering/render_scene.py, rendering/scene_builder.py, rendering/backends/base.py, OPEN_QUESTIONS.md#Q50, OPEN_QUESTIONS.md#Q51, REFERENCES.md
Owner Summary: 记录 2026-04-24 关于“物理层与渲染/传感器关系、CPU/GPU 两条计算路径、publish/export 阶段、optional views、interface vs contact 兼容关系”的设计收敛结果，并给出可审查的分层接口草案。当前结论：共享物理语义、分离 CPU/GPU 执行路径；GPU path 先设计；publish 是正式 phase；views 是可选消费者，不是 physics step 的必选产物；需要把 `PhysicsModel / PhysicsState / DerivedCache / PublishedFrame / View / Policy` 的边界显式化。

---

## Context

本轮讨论从几个连续问题展开：

1. 仿真和渲染在当前实现中的关系是什么
2. 渲染物理能否与仿真物理区分
3. 传感器与物理层、渲染层的关系是什么
4. 仿真结果当前存在哪里，渲染入口当前需要什么
5. 未来渲染/传感器功能很多，计算管线应该如何组织
6. CPU 和 GPU 两条路径是否应该分别设计
7. 多物理场中 `interface` 的丰富物理意涵如何与当前 rigid-body `contact` 兼容

本文件不是最终设计定案，而是把目前已经形成共识的内容完整落盘，供后续和 Claude 做 review / challenge 时使用。

---

## Current State

### 1. 当前没有统一的“仿真结果存储对象”

现状里最接近统一出口的是 `PhysicsEngine.step()` 返回的 `StepOutput`：

- `q_new`
- `qdot_new`
- `X_world`
- `v_bodies`
- `contact_active`
- `force_state`

定义见 `physics/engine.py`。

但它还不是完整的世界快照，尤其有两个限制：

- 不包含完整 contact 列表；contacts 仍由 `query_contacts()` 单独获取
- GPU 路径里 `StepOutput` 只是从 device buffer 抽出来的一份外显快照，不是 authoritative state 本体

### 2. 当前是“逐 step 保留最近一步缓存”，不是“逐 step 保存完整历史”

当前系统会保留：

- 单步内部缓存：`DynamicsCache`
- 最近一步力分解：`last_force_state`
- 最近一步 contacts：`CpuEngine._last_contacts` / GPU contact buffers
- 当前状态：`q/qdot` 及 GPU resident buffers

不会默认保留：

- 跨步轨迹历史
- 多帧 `RenderScene`
- 可供多个消费者稳定复用的已发布帧对象

### 3. 当前渲染入口是两段式

当前渲染不是直接吃 `StepOutput`，而是：

1. `build_render_scene(...)`
2. `RenderBackend.render_frame(scene, timestamp, env_index)`

`RenderScene` 当前主要承载：

- `shapes`
- `contacts`
- `terrain`
- `skeleton_links`
- `body_positions`
- `body_names`

这说明当前 `RenderScene` 更像 debug/inspection scene snapshot，而不是未来所有渲染需求的统一世界表示。

### 4. GPU 到渲染的桥目前是 debug 风格

`build_render_scene_from_gpu()` 当前路径是：

- `engine.q_wp.numpy()`
- CPU FK
- `build_render_scene(...)`

这条路径的优点是简单、薄、复用现有逻辑；缺点是它明确不是 realtime / high-throughput 级别的最终架构。

---

## Agreed Design Philosophy

本轮讨论已形成的核心共识如下。

### 1. 共享语义，分离执行

CPU 和 GPU 两条计算路径应该分别设计。

但“分别设计”的是：

- 执行模型
- 数据布局
- 缓存策略
- 同步与通信方式

不应该分叉的是：

- 物理世界语义
- 对外数据契约
- consumer 看到的字段含义

一句话：

`shared semantics, separate execution`

### 2. 物理层拥有 authoritative truth

rendering、sensor、RL obs、debug/logging 都是消费者，不拥有世界真值，也不应各自维护一份“真实状态”。

### 3. 不同需求共用真值，不共用末端表示

`RenderScene` 不应该成为未来所有需求的大杂烩。

未来至少要区分：

- debug scene / export scene
- realtime render-facing view
- sensor-facing view

它们共享同一个 physics truth，但不共享一个末端结构体。

### 4. 数据导出必须是仿真 pipeline 的正式阶段

如果不把 export/publish 作为正式 phase 设计进去，系统会退化成：

- renderer 偷读 physics 私有 scratch
- sensor 各自重算 FK / contacts / surfaces
- CPU/GPU 两条路径导出语义不一致

因此，`publish/export` 必须进入 step pipeline 的设计。

但“导出进入 pipeline”不等于“所有导出同步阻塞 physics hot path”。

### 5. GPU path 优先约束架构，CPU path 再做 reference-style 对应实现

本轮讨论已达成明确倾向：

- GPU path 更难，也更决定系统边界
- 应优先设计 GPU path
- CPU path 按共享语义实现“简化、易调试、reference/backbone”版本

理由：

- GPU path 涉及 host/device 通信
- GPU path 涉及 physics kernel 与 CPU orchestration 的并行
- GPU path 直接决定哪些数据可以常驻 device，哪些必须 publish/mirror

---

## Interface vs Contact

这是本轮讨论里最重要的概念清洗之一。

### 1. `contact` 不是总概念，`interface` 才是总概念

在 rigid-body 语境下，`contact` 是自然词汇。

但在 multi-physics 语境下，更一般的词汇应该是 `interface`：

- rigid-rigid contact
- rigid-soft collision
- soft-soft cohesion
- fluid-solid boundary
- thermal interface

因此：

- `contact` 是 rigid-body vocabulary
- `interface` 是 multi-physics vocabulary

### 2. 当前 rigid-only 阶段如何兼容

当前阶段完全可以继续使用 `ContactConstraint`、narrowphase、contact solver 这些 rigid-body 术语。

兼容方式是：

- `InterfaceMaterial` 提供界面律
- `contact` 提供 rigid-body 接触事件与约束形式

也就是说：

`contact` 是 `interface` 在 rigid-body 接触模型里的一个具体实现 / 特例

### 3. 未来 multi-physics 阶段的方向

未来更一般的对象将会是：

- `InterfaceRegion`
- `InterfaceMaterial`
- `CouplingImpulse` / interaction law

此时：

- rigid-rigid 依然可以表现为 contact
- rigid-soft / fluid-solid / hydroelastic 不必伪装成 rigid contact

这与已记录的 SOFA / Drake hydroelastic / Genesis 研究结论一致。

---

## Layered Interface Draft

下面不再只讲哲学边界，而是给出当前建议的**分层接口草案**。这些接口不是最终代码签名，但粒度应该足够让 review 围绕“字段、所有权、phase、依赖方向”展开。

### 1. Layer 0 — `PhysicsModel`

职责：

- 表达静态拓扑与静态资源索引
- 不随 step 改变
- 可构建 GPU resident mirrors
- 为 physics、render、sensor 提供统一 id 空间

建议接口：

```python
@dataclass(frozen=True)
class PhysicsModel:
    # world / env layout
    num_envs: int
    nq: int
    nv: int
    nb: int
    nshape: int

    # kinematic / dynamic topology
    body_parent: ArrayLike[int]          # [nb]
    joint_type: ArrayLike[int]           # [nb]
    joint_q_adr: ArrayLike[int]          # [nb]
    joint_q_dim: ArrayLike[int]          # [nb]
    joint_v_adr: ArrayLike[int]          # [nb]
    joint_v_dim: ArrayLike[int]          # [nb]
    X_tree: ArrayLike                    # [nb] body-local parent transform
    inertia: ArrayLike                   # [nb]

    # shape / interface topology
    shape_body: ArrayLike[int]           # [nshape] -> body id
    shape_type: ArrayLike[int]           # [nshape]
    shape_pose_local: ArrayLike          # [nshape]
    shape_params: ArrayLike              # [nshape, ...]
    shape_interface_adr: ArrayLike[int]  # [nshape] -> interface material id
    collision_filter: ArrayLike          # shape-pair or body-pair filter data

    # terrain / static geometry
    terrain_desc: object | None
    static_geom_desc: object | None

    # render / sensor attachment metadata
    visual_geom_adr: ArrayLike[int]      # [nshape] or [body]
    sensor_mount_body: ArrayLike[int]    # [nsensor]
    sensor_mount_pose: ArrayLike         # [nsensor]
    semantic_id: ArrayLike[int]          # [body] or [shape]
    instance_id: ArrayLike[int]          # [body] or [shape]
```

约束：

- `PhysicsModel` 不应该包含 step-dependent 数据
- render/sensor 只能通过 handle/id 引用静态资产，不在这里存每帧 transform
- `shape_*` 与 `body_*` 应是共享 id 空间，不能为不同消费者重新编号

### 2. Layer 1 — `PhysicsState`

职责：

- 表达某一时刻 authoritative truth
- 由 physics step 更新
- 不直接承载 debug scene / render scene / sensor packet

建议最小接口：

```python
@dataclass
class PhysicsState:
    time: float
    step_index: int

    q: ArrayLike                    # [N, nq] or [nq]
    qdot: ArrayLike                 # [N, nv] or [nv]

    # future subsystem state blocks
    rigid_state: object | None = None
    deformable_state: object | None = None
    fluid_state: object | None = None

    # ownership / validity
    env_mask: ArrayLike | None = None
```

说明：

- rigid-only 当前阶段，`q/qdot` 就是主状态
- 多物理场阶段，`PhysicsState` 不应被 rigid-only 假设绑死；rigid state 只是其中一块
- 如果未来做 asynchronous sensor exposure，`PhysicsState` 可能还需要支持“substep state”或“interpolation anchor”

### 3. Layer 2 — `DerivedPhysicsCache`

职责：

- 从 `PhysicsState` 派生
- 服务 physics hot path 与高频消费者
- 允许 CPU/GPU 各自采用不同缓存策略
- 字段语义一致，但实现不要求完全同构

建议拆成 5 个子块，而不是一个大杂烩：

```python
@dataclass
class KinematicsCache:
    X_world_R: ArrayLike            # [N, nb, 3, 3]
    X_world_r: ArrayLike            # [N, nb, 3]
    v_bodies: ArrayLike             # [N, nb, 6]
    a_bodies: ArrayLike | None = None

@dataclass
class ContactCache:
    contact_count: ArrayLike        # [N]
    body_i: ArrayLike               # [N, max_contacts]
    body_j: ArrayLike               # [N, max_contacts]
    point: ArrayLike                # [N, max_contacts, 3]
    normal: ArrayLike               # [N, max_contacts, 3]
    depth: ArrayLike                # [N, max_contacts]
    active: ArrayLike               # [N, max_contacts]
    lambda_rows: ArrayLike | None = None
    contact_force_world: ArrayLike | None = None

@dataclass
class TelemetryCache:
    qfrc_passive: ArrayLike | None = None
    qfrc_actuator: ArrayLike | None = None
    qfrc_applied: ArrayLike | None = None
    tau_smooth: ArrayLike | None = None
    qacc_smooth: ArrayLike | None = None
    qacc_total: ArrayLike | None = None
    force_sensor: ArrayLike | None = None

@dataclass
class SpatialQueryCache:
    body_aabb_min: ArrayLike | None = None
    body_aabb_max: ArrayLike | None = None
    shape_aabb_min: ArrayLike | None = None
    shape_aabb_max: ArrayLike | None = None

@dataclass
class SurfaceCache:
    render_instance_transform: ArrayLike | None = None
    dynamic_vertices: ArrayLike | None = None
    dynamic_normals: ArrayLike | None = None
    raycast_surface_handle: object | None = None
```

聚合接口：

```python
@dataclass
class DerivedPhysicsCache:
    kinematics: KinematicsCache
    contacts: ContactCache
    telemetry: TelemetryCache
    spatial: SpatialQueryCache
    surface: SurfaceCache
```

关键约束：

- `DerivedPhysicsCache` 是 physics 的派生真值层，不是 consumer-specific scene
- `RenderScene` / `SensorPacket` 不应直接塞进 cache
- `contacts` 必须进入统一 cache，而不是继续仅靠 `query_contacts()` 做旁路

### 4. Layer 3 — `PublishedFrameCore`

职责：

- 建立“本步完成”的正式 frame boundary
- 隔离 mutable state 与 consumer 读取
- 提供 render/sensor/debug 共同锚点

建议接口：

```python
@dataclass
class PublishedFrameCore:
    frame_id: int
    sim_time: float
    step_index: int
    env_mask: ArrayLike | None

    # references or copied slices into published buffers
    state_ref: object
    kinematics_ref: object
    contact_count_ref: object | None
    contacts_ref: object | None
    telemetry_ref: object | None

    # readiness / synchronization
    ready_flag: object | None = None
    completion_event: object | None = None
```

这里最重要的设计点：

- `PublishedFrameCore` 不是等于完整 `PhysicsState`
- 它是“已发布帧”的统一描述符
- 它可以持有引用、buffer slice、slot id，不一定总是深拷贝
- `contact_count` 更接近 frame boundary 所需的轻量真值；完整 dense contact arrays 可视为可选 block

### 5. Layer 4 — Consumer Views

三类 view 都建议围绕 `PublishedFrameCore` 构建，而不是直接从 `GpuMutableState` 偷读。

#### `RealtimeRenderView`

```python
@dataclass
class RealtimeRenderView:
    frame_id: int
    sim_time: float
    env_selector: object

    instance_transform_ref: object
    instance_visibility_ref: object | None
    dynamic_surface_ref: object | None
    material_handle_ref: object
    camera_mount_ref: object | None
```

#### `SensorRenderView`

```python
@dataclass
class SensorRenderView:
    frame_id: int
    sim_time: float
    env_selector: object

    sensor_pose_ref: object
    surface_or_raycast_ref: object
    semantic_id_ref: object | None
    instance_id_ref: object | None
    temporal_window_ref: object | None = None
```

#### `DebugExportView`

```python
@dataclass
class DebugExportView:
    frame_id: int
    sim_time: float
    env_selector: object

    x_world_ref: object
    contacts_ref: object | None
    terrain_desc_ref: object | None
    skeleton_ref: object | None
    telemetry_summary_ref: object | None = None
```

关键约束：

- view 是 consumer-facing，不是 physics cache owner
- view 可以是“引用 + selector”，不一定每次物化成 Python dataclass
- CPU `RenderScene` 应被视为 `DebugExportView` 的一种 host materialization，不应提升为所有 view 的共同基类

### 6. Layer 5 — `PublishPolicy`

本轮已经达成：view 必须可选 skip。  
如果要让这件事工程上可执行，就需要一个正式 policy 对象。

建议接口：

```python
@dataclass(frozen=True)
class ViewPolicy:
    enabled: bool = False
    period_steps: int = 1
    env_selector: object | None = None
    detail_level: str = "default"   # low/default/high
    max_items: int | None = None

@dataclass(frozen=True)
class PublishPolicy:
    publish_core_every_step: bool = True
    realtime_render: ViewPolicy = ViewPolicy()
    sensor_render: ViewPolicy = ViewPolicy()
    debug_export: ViewPolicy = ViewPolicy()
    publish_rigid_block: bool = False
    publish_telemetry_block: bool = True
```

语义：

- `publish_core_every_step=True` 是系统统一帧语义的基础
- 三类 view 的构建由各自 policy 决定
- view 不再是“physics step 的必选副产品”
- 重量级 published block 的写入也应受 policy 控制，不能默认每步无条件开启

### 7. Layer 5.5 — `PublishPlan`

`PublishPolicy` 是低频配置对象，但不能直接成为 GPU kernel 的输入语义来源。

原因：

- kernel 内逐线程判断 `enabled / period / env selection / detail`
  会导致 warp divergence
- per-thread policy branching 会把 optional view 的控制逻辑污染 physics hot path
- 多消费者系统里，真正需要的是“这一帧该发哪些 kernel、维度多大、用哪个 variant”，而不是让 kernel 自己解释 policy

因此建议增加 `PublishPlan` 作为**每帧解析后的执行计划**。

建议接口：

```python
@dataclass
class PublishPlan:
    do_publish_core: bool

    do_realtime_render: bool
    realtime_env_ids: object | None
    realtime_variant: str | None

    do_sensor_render: bool
    sensor_env_ids: object | None
    sensor_variant: str | None

    do_debug_export: bool
    debug_env_ids: object | None
    debug_host_copy: bool

    # optional heavy blocks
    do_rigid_block_write: bool = False
    do_telemetry_block_write: bool = True
```

关系应当明确为：

`PublishPolicy -> PublishPlan -> kernel launches`

而不是：

`PublishPolicy -> kernel internal branching`

### 8. Policy / Plan 的职责分离

建议明确以下责任边界。

#### `PublishPolicy`

低频、用户或系统配置：

- 哪类 view 开启
- 采样周期
- 目标 env 选择模式
- detail level
- 是否需要 host export

#### `PublishPlan`

逐帧、调度器计算结果：

- 这一帧是否发某类 kernel
- 用哪种 kernel variant
- launch 维度是多少
- env 子集是哪些 compacted ids
- 是否排队 host copy
- 是否写入重量级 published blocks（例如 dense `RigidBlock`）

#### kernel

只做已经被 plan 选好的工作：

- 不逐线程判断某个 view 是否启用
- 不逐线程解释 env selection policy
- 不在同一 kernel 内用大分支切换 low/default/high detail

### 9. GPU Kernel Design Rules

基于上面的 policy/plan 分离，本轮新增以下硬约束。

#### A. Launch-level gating 优先

如果某类 view 本帧不需要，就**不发射对应 kernel**。

#### B. Variant selection 优先于 kernel 内 detail 分支

优先：

- `build_sensor_view_low`
- `build_sensor_view_default`
- `build_sensor_view_high`

而不是：

```python
if detail == "low":
    ...
elif detail == "high":
    ...
```

#### C. Env 子集通过 compacted ids / specialized launch 处理

优先：

- 预先生成 `env_ids_subset`
- 以 `dim=len(env_ids_subset)` 发射 kernel

而不是：

```python
if not env_selected[env_id]:
    return
```

#### D. 可接受的 kernel 条件分支

允许：

- 物理求解本身不可避免的数学分支
- shape-type / contact-type dispatch
- launch 级统一常量控制

应尽量避免：

- 每 env 的 consumer policy 分支
- 每 thread 的 export/debug/detail 分支

### 10. `publish_core` 也应拆分 hot / optional 两段

为了避免 optional views 反向污染 physics 主链，建议把 step 末尾分成：

1. `publish_core_always_on`
   - 建立 frame boundary
   - 更新最小 published truth
   - 写 ready/event

2. `publish_optional_views`
   - 按 `PublishPlan` 发射 realtime/sensor/debug 相关 kernels
   - 可完全跳过

这样可以保证：

- physics 主路径始终有稳定 frame 语义
- optional consumers 的控制逻辑不会侵入 core publish kernels

### 11. Layer 6 — `PhysicsPublisher`

如果 `publish` 是正式 phase，那么建议有一个明确的发布器接口，而不是把逻辑散在 engine 内部。

建议接口：

```python
class PhysicsPublisher(ABC):
    @abstractmethod
    def publish_core(
        self,
        model: PhysicsModel,
        state: PhysicsState,
        cache: DerivedPhysicsCache,
    ) -> PublishedFrameCore: ...

    @abstractmethod
    def build_realtime_render_view(
        self,
        frame: PublishedFrameCore,
        policy: ViewPolicy,
    ) -> RealtimeRenderView | None: ...

    @abstractmethod
    def build_sensor_render_view(
        self,
        frame: PublishedFrameCore,
        policy: ViewPolicy,
    ) -> SensorRenderView | None: ...

    @abstractmethod
    def build_debug_export_view(
        self,
        frame: PublishedFrameCore,
        policy: ViewPolicy,
    ) -> DebugExportView | None: ...
```

这层的目的不是做花哨抽象，而是把：

- frame boundary
- optional views
- consumer-specific materialization

从 solver/integrator 热路径里剥离出来。

---

## Publish / Export Stage

### 1. Publish 是正式 phase

建议把每个 step 理解为：

1. physics update
2. derived cache update
3. publish
4. consumer use

其中 `publish` 的职责是：

- 建立“本步已完成”的帧边界
- 产出稳定、可消费的数据视图
- 保证 physics / render / sensor / debug 都围绕同一个 frame 语义工作

### 2. Publish 不等于全量 host 导出

本轮已达成明确原则：

- `publish core`：必须存在
- `heavy export work`：不一定同步执行

也就是说：

- publish 是 pipeline 的语义边界
- copy / encode / serialize / cpu snapshot 可以异步或降频

---

## GPU Path Interface Draft

下面把上面的 shared semantics 压到 GPU path 的执行接口上。

### 1. `GpuModel`

建议把当前 `GpuEngine` 已经拥有的静态数组正式归类为 `GpuModel`：

- `_gpu_parent_idx`
- `_gpu_q_idx_start/_len`
- `_gpu_v_idx_start/_len`
- `_gpu_X_tree_R/_r`
- `_gpu_inertia_mat`
- `_gpu_flat_shape_type/_params/_offset/_rotation`
- `_gpu_body_shape_adr/_num`
- `_gpu_collision_excluded`
- `_gpu_hull_*`
- `_gpu_contact_body_idx`

建议接口：

```python
@dataclass
class GpuModel:
    static: StaticRobotData

    # device-resident mirrors
    joint_type_wp: object
    parent_idx_wp: object
    q_adr_wp: object
    v_adr_wp: object
    X_tree_wp: object
    inertia_wp: object

    shape_type_wp: object
    shape_params_wp: object
    shape_offset_wp: object
    shape_rotation_wp: object
    shape_body_adr_wp: object
    shape_body_num_wp: object
    collision_excluded_wp: object

    interface_material_wp: object | None = None
    semantic_id_wp: object | None = None
    visual_handle_wp: object | None = None
```

### 2. `GpuMutableState`

建议把当前 `_scratch` 中真正属于 mutable truth 的部分单独命名出来：

```python
@dataclass
class GpuMutableState:
    q_wp: object                      # [N, nq]
    qdot_wp: object                   # [N, nv]

    # solver state / warmstart / substep mutable data
    warmstart_wp: object | None = None
    action_wp: object | None = None
    reset_mask_wp: object | None = None
```

注意：

- `x_world_*`、`v_bodies` 不应归到 mutable truth，而应归 `GpuDerivedCache`
- 这样 physics step 的“真值层”和“派生层”才有稳定边界

### 3. `GpuDerivedCache`

建议把当前 `GpuEngine` 中已存在的对外 accessor 和 contact buffers 整理成统一 cache。

可直接映射的现有字段包括：

- `x_world_R_wp`
- `x_world_r_wp`
- `v_bodies_wp`
- `qacc_smooth_wp`
- `qacc_total_wp`
- `contact_active_wp`
- `contact_count_wp`
- `_contact_bi/_bj/_point/_normal/_depth`
- `contact_force_sensor_wp`

建议接口：

```python
@dataclass
class GpuDerivedCache:
    x_world_R_wp: object
    x_world_r_wp: object
    v_bodies_wp: object

    qacc_smooth_wp: object | None = None
    qacc_total_wp: object | None = None
    tau_passive_wp: object | None = None
    tau_total_wp: object | None = None

    contact_count_wp: object | None = None
    contact_active_wp: object | None = None
    contact_bi_wp: object | None = None
    contact_bj_wp: object | None = None
    contact_point_wp: object | None = None
    contact_normal_wp: object | None = None
    contact_depth_wp: object | None = None

    force_sensor_wp: object | None = None

    render_surface_wp: object | None = None
    sensor_surface_wp: object | None = None
```

### 4. `GpuPublishedFrame`

这是 GPU path 最关键的正式对象。

建议接口：

```python
@dataclass
class GpuPublishedFrame:
    slot_id: int
    frame_id: int
    sim_time: float
    step_index: int
    env_mask_wp: object | None

    # immutable-after-publish references into published buffers
    q_wp: object
    qdot_wp: object
    x_world_R_wp: object
    x_world_r_wp: object
    v_bodies_wp: object

    contact_cache_ref: object | None
    telemetry_ref: object | None
    surface_ref: object | None

    ready_event: object
```

建议语义：

- 采用 double 或 triple buffering
- physics 永远写 `next_slot`
- consumers 只读 `committed_slot`
- `slot_id` 是 frame 引用的一部分，不允许消费者绕过它直接读 mutable buffers

### 5. `GpuEngine` 公开控制接口

当前 `GpuEngine.step()` 还是 monolithic。  
如果后续要支持 publish phase 和异步消费者，建议显式拆分控制接口，即使底层实现初期仍可先串行。

建议外部接口：

```python
class GpuEngine(PhysicsEngine):
    def submit_actions(self, tau_wp_or_np, env_ids=None) -> None: ...
    def submit_resets(self, env_ids, q0=None, qdot0=None) -> None: ...

    def step_async(self, dt: float | None = None) -> int:
        """Launch one physics step on stream_physics and return target frame_id."""

    def step_wait(self, frame_id: int | None = None) -> GpuPublishedFrame:
        """Wait until publish_core of target frame completes."""

    def latest_published_frame(self) -> GpuPublishedFrame | None: ...

    def export_debug_snapshot(self, frame: GpuPublishedFrame, env_idx: int = 0) -> DebugExportView: ...
```

这里的重点不是强行模仿某个框架，而是：

- physics launch
- publish completion
- consumer reads

要有明确的时序 API，而不是全靠调用方猜“现在 buffer 里是什么”。

### 6. `GpuPublisher` 的 phase 接口

建议 GPU path 把 publish phase 做成显式私有方法：

```python
class GpuPublisher(PhysicsPublisher):
    def publish_core_gpu(
        self,
        model: GpuModel,
        state: GpuMutableState,
        cache: GpuDerivedCache,
        slot_id: int,
        frame_id: int,
        sim_time: float,
    ) -> GpuPublishedFrame: ...

    def maybe_build_realtime_view_gpu(
        self,
        frame: GpuPublishedFrame,
        policy: ViewPolicy,
    ) -> object | None: ...

    def maybe_build_sensor_view_gpu(
        self,
        frame: GpuPublishedFrame,
        policy: ViewPolicy,
    ) -> object | None: ...

    def maybe_queue_host_export(
        self,
        frame: GpuPublishedFrame,
        policy: ViewPolicy,
    ) -> None: ...
```

### 7. GPU stream / event contract

本轮讨论已明确：GPU path 不能只靠单 stream 串行推进所有 consumer。

建议先把 contract 写清楚，即使实现第一版仍然退化成同步串行：

```python
@dataclass
class GpuExecutionContext:
    stream_physics: object
    stream_publish: object
    stream_realtime: object | None = None
    stream_sensor: object | None = None
    stream_hostcopy: object | None = None
```

建议时序：

- `stream_physics`: step physics kernels
- `stream_publish`: derive + publish_core
- `stream_realtime`: wait(frame.ready_event) -> build/use realtime render view
- `stream_sensor`: wait(frame.ready_event) -> build/use sensor view
- `stream_hostcopy`: wait(frame.ready_event) -> sampled async copies

### 8. GPU path 的 skip / sampling 不是附加优化，而是接口的一部分

建议 `GpuPublisher` 必须支持：

- view disabled
- lower publish rate
- env subset selection
- detail-level downgrade

也就是说，`PublishPolicy` 必须在 GPU path 中是一等对象，而不是 optional helper。

---

## CPU Path Interface Draft

虽然本轮优先级是 GPU path，但为了保证共享语义，CPU path 也应有对应接口草案。

建议 CPU path 对应对象如下：

```python
@dataclass
class CpuModel:
    merged: MergedModel

@dataclass
class CpuMutableState:
    q: NDArray
    qdot: NDArray
    time: float
    step_index: int

@dataclass
class CpuDerivedCache:
    dynamics: DynamicsCache
    contacts: list[ContactInfo]
    force_state: ForceState | None

@dataclass
class CpuPublishedFrame:
    frame_id: int
    sim_time: float
    step_index: int
    q: NDArray
    qdot: NDArray
    X_world: list
    v_bodies: list
    contacts: list[ContactInfo]
    force_state: ForceState | None
```

CPU path 和 GPU path 的差异可以很大，但语义上最好保持：

- 都有 mutable truth
- 都有 derived cache
- 都有 published frame
- 都能按 `PublishPolicy` 构建三类 view

### CPU path 建议公开接口

```python
class CpuEngine(PhysicsEngine):
    def step(self, q, qdot, tau, dt=None) -> StepOutput: ...
    def step_and_publish(self, q, qdot, tau, dt=None) -> CpuPublishedFrame: ...
    def latest_published_frame(self) -> CpuPublishedFrame | None: ...
```

这里 `step()` 可以继续保留向后兼容，  
但后续 render/sensor/debug 更推荐消费 `CpuPublishedFrame`。

---

## Mapping from Current Code to Proposed Layers

为了方便 review，下面把当前已有对象映射到上面草案中的层。

### 已存在并可直接复用的部分

- `MergedModel` -> `PhysicsModel` 的 rigid-only host 实现雏形
- `StepOutput` -> `PublishedFrame` 的不完整外显雏形
- `DynamicsCache` -> `DerivedPhysicsCache` 的 CPU 单步子集
- `ForceState` -> `TelemetryCache` 的 CPU 子集
- `GpuEngine` 的 `q_wp/qdot_wp/x_world_*/v_bodies_wp/contact_*_wp/qacc_*_wp` -> `GpuMutableState + GpuDerivedCache` 的现成字段基础

### 当前缺失的关键层

- 正式的 `PublishedFrameCore`
- 正式的 `PublishPolicy`
- 正式的 `PhysicsPublisher`
- 将 contact 列表纳入统一 `DerivedPhysicsCache` / published frame，而不是靠 `query_contacts()` 旁路
- 区分 `GpuMutableState` 和 `GpuDerivedCache` 的显式对象边界

### 当前最可能踩坑的点

- 继续让 `RenderScene` 兼任未来所有 consumer 的统一末端表示
- 继续把 GPU export 当作 `q_wp.numpy() + CPU build scene`
- 继续让 telemetry / sensor / render 各自偷读不同 buffer，而没有统一 frame boundary

---

## Optional Views

这一点本轮已经收敛为明确原则。

### 1. 三个 view 都应该可选 skip

可以 skip 的是：

- `RealtimeRenderView`
- `SensorRenderView`
- `DebugExportView`

不能 skip 的是：

- authoritative `PhysicsState`
- 最小 `PublishedFrame core`

### 2. 更好的表达不是“是否生成”，而是“publish core + optional views”

即：

- `publish core`: always on
- `build realtime render view`: optional
- `build sensor render view`: optional
- `build debug export view`: optional

### 3. Optional 不只意味着 on/off，还应包括 policy

每个 view 最好都有独立 policy：

- `enabled`
- `rate`
- `env_selector`
- `detail_level`

这对大规模 RL / sampled debug / multi-env render 都很关键。

### 4. Optional view 的实现必须服从 `PublishPlan`

也就是说：

- view 是否生成，不应在 kernel 内部逐线程判断
- view 对哪些 env 生效，不应由 kernel 读取一个任意 mask 自己解释
- view 的 detail level，不应在一个通用 kernel 里靠大分支切换

optional view 的工程实现应当是：

- 先由 policy 解析出本帧 plan
- 再根据 plan 发射相应 kernel variant
- kernel 本身只处理已选定的目标数据

---

## GPU Path First

本轮已经明确决定：优先从 GPU path 设计系统边界。

原因：

- CPU path 更容易做成“舒服但不适合设备端发布”的结构
- GPU path 先定，CPU path 可以自然变成共享语义下的简化实现

### 1. GPU path 的核心判断

GPU path 里最难的不是 dynamics kernel 本身，而是：

**step 完成后，世界状态以什么形态被发布给多个消费者**

### 2. GPU path 的三条硬约束

#### A. Authoritative truth 在 device

对于大规模 RL / realtime 主路径：

- authoritative state 在 GPU
- CPU 只持有 mirror / sampled snapshot / metadata / control inputs

#### B. Physics 不直接产出最终 render object

physics 产出的是：

- state
- derived caches
- published frame

不是直接产出：

- CPU `RenderScene`
- images
- sensor packets

#### C. 发布必须是 frame boundary

如果没有正式的 published-frame 边界，physics 下一步写入和 render/sensor 上一帧读取会互相踩。

因此 GPU path 需要显式的帧发布对象。

---

## Proposed GPU Object Split

下面是本轮已经形成的 GPU 路径对象分层。

### 1. `GpuModel`

静态，不变，常驻 GPU / Host metadata。

### 2. `GpuMutableState`

仅 physics 写入的当前可变真值。

至少包括：

- `q`
- `qdot`
- reset/action buffers
- warmstart / solver scratch ownership

### 3. `GpuDerivedCache`

服务 physics 与高频消费者的派生量。

建议至少分成四类 cache：

#### `KinematicsCache`

- `X_world`
- `v_bodies`
- optional `a_bodies`

#### `ContactCache`

- `contact_count`
- body/shape pairs
- points
- normals
- depth
- lambda / contact force

#### `TelemetryCache`

- `qacc_smooth`
- `qacc_total`
- `tau_passive`
- `tau_total`
- force sensor buffers

#### `SurfaceOrVisualCache`

- render-facing transform buffers
- dynamic surface vertices
- AABB / visibility seeds

### 4. `GpuPublishedFrame`

这是本轮讨论中最关键的新对象。

其职责不是当 scratch，而是充当：

- 已完成 step 的已发布帧
- 可被 render / sensor / host-export 消费的冻结视图

建议做成：

- double buffer 或 triple buffer
- physics 写下一槽位
- consumers 读已提交槽位

这样可以避免 physics 和 consumer 直接共享同一份可写数据。

---

## Proposed GPU Step Phases

目前讨论已基本收敛到如下 phase 切分。

### Phase 1. `ingest`

- 处理 reset / action / config update requests
- CPU 主要在这个阶段写输入，不读结果

### Phase 2. `physics`

- FK / dynamics / collision / solve / integrate
- 只碰 `GpuMutableState + internal scratch`

### Phase 3. `derive`

- 更新本步需要发布的派生量
- 例如 `X_world`、contacts、telemetry、surface reps

### Phase 4. `publish`

- 将当前 step 的关键信息写入 `GpuPublishedFrame[slot]`
- 写入 `frame_id / sim_time / ready flag / event`

### Phase 5. `consume`

分成多路：

- realtime render stream
- sensor stream
- host export / debug stream

关键原则是：

- physics 不等 host copy
- physics 不等 debug render
- host 不应通过全局 `synchronize()` 驱动主循环

---

## Host / Device Boundary Principles

本轮讨论已明确若干硬规则。

### 1. `GPU -> CPU -> GPU` 禁止进入 realtime 主路径

如果一条路径包含：

- device 结果拷回 host
- host 构建中间 scene
- 再喂回 device / realtime consumer

则它应被视为：

- debug path
- offline path

而不是最终 realtime 主路径。

### 2. Host 侧主要持有这些东西

- 静态 metadata
- frame descriptors / publish state
- sampled snapshots
- export queues
- control inputs

### 3. Device 侧主要持有这些东西

- authoritative mutable state
- derived caches
- published frames
- realtime render / sensor input buffers

---

## Published Ring Sizing And Consumer Modes

本节记录 2026-04-24 关于 `PublishedSlot`、默认 buffer 数量、显存量级、以及阻塞/不阻塞消费模式的进一步收敛。

### 1. 默认 `PublishedRing` 大小

当前默认建议：

- `published_ring_size = 3`

即默认采用 **triple buffering**。

原因：

- `2` 个 slot 容易让 publish 与 consumer 形成紧耦合乒乓，consumer 一慢就逼近阻塞
- `3` 个 slot 能给 physics、当前 consumer、下一帧 publish 留出更安全的重叠空间
- 继续增大 ring size 是合理扩展项，但不应作为默认前提

### 2. `PublishedSlot` 的建议组织

当前建议 `PublishedSlot` 逻辑上拆成：

- `CorePublishedBlock`
- `RigidBlock`
- `TelemetryBlock`
- future: `DeformableBlock`
- future: `FluidBlock`
- future: `InterfaceBlock`

第一版实现上可接受：

- `core / rigid / telemetry` 都做成每个 slot 固定分配
- 逻辑上仍然保留 block 边界

这样可以避免在 rigid-only 阶段过早引入复杂动态显存管理。

进一步细化建议：

- `contact_count` 可作为轻量 contact 边界信息进入 core / core-ref
- dense `RigidBlock` 即使预分配，也不应默认每步无条件写入
- `RigidBlock` 的实际写入应由 `PublishPlan.do_rigid_block_write` 控制

### 3. `PublishedSlot` 中 core buffers 的当前建议

每个 slot 当前建议固定持有：

- `q`
- `qdot`
- `X_world_R`
- `X_world_r`
- `v_bodies`
- `contact_count`
- `frame metadata`

其中 `frame metadata` 至少包括：

- `frame_id`
- `step_index`
- `sim_time`
- `env coverage`
- `ready flag / completion event`

### 4. `v_bodies` 的成本判断

当前代码里，GPU 路径在 FK 阶段已经计算：

- `X_world_R`
- `X_world_r`
- `v_bodies`

因此把 `v_bodies` 带入 published slot 的新增成本主要是：

- published storage
- device-to-device copy / slot write

而不是新增一个昂贵 kernel。

基于当前 Warp buffer 形状：

- `v_bodies: (N, nb, 6)` `float32`

其每 env 每 slot 成本约为：

`24 * nb bytes`

对比：

- `X_world_R + X_world_r` 约为 `72 * nb bytes`

因此：

- `v_bodies` 大约是 `X_world` 存储成本的 `1/3`
- 单独看姿态块内部，`v_bodies` 大约是 `X_world_R + X_world_r` 的 `1/2`

结论：

- 计算成本：基本已支付
- 存储成本：中等、可接受
- 设计上：值得作为 recommended core field 保留

### 5. Published buffers 的增量显存估算公式

下面估算的是 **新增的 published ring 增量**，不是整个 physics engine 当前总显存。

#### CorePublishedBlock

若采用当前推荐字段：

- `q:        4 * nq`
- `qdot:     4 * nv`
- `X_world:  48 * nb`
- `v_bodies: 24 * nb`
- `contact_count: 4`

则每 env 每 slot 约为：

`core_bytes_per_env_per_slot = 4 * (nq + nv) + 72 * nb + 4`

总量：

`core_total = ring_size * N * (4 * (nq + nv) + 72 * nb + 4)`

#### RigidBlock

若当前 rigid contact block 包含：

- `contact_bi: 4 * max_contacts`
- `contact_bj: 4 * max_contacts`
- `contact_active: 4 * max_contacts`
- `contact_depth: 4 * max_contacts`
- `contact_normal: 12 * max_contacts`
- `contact_point: 12 * max_contacts`

则每 env 每 slot 约为：

`rigid_bytes_per_env_per_slot = 40 * max_contacts`

总量：

`rigid_total = ring_size * N * (40 * max_contacts)`

注：

- 若未来把 `lambda/contact_force` 也纳入 block，需要再加对应条目
- 当前阶段 `RigidBlock` 往往是 published ring 中最大的增量项之一
- 上式默认 `max_contacts` 是 **per-env 上限**
- 物理求解层仍可保留 dense preallocated contact buffers；这里讨论的是 published ring 的增量
- 如果 `PublishPlan.do_rigid_block_write=False`，则该 block 可以预分配但本帧不写入

#### TelemetryBlock

若当前 telemetry block 含：

- `qacc_smooth: 4 * nv`
- `qacc_total: 4 * nv`
- `tau_passive: 4 * nv`
- `tau_total: 4 * nv`
- `force_sensor: 12 * nc_sensor`

则每 env 每 slot 约为：

`telemetry_bytes_per_env_per_slot = 16 * nv + 12 * nc_sensor`

总量：

`telemetry_total = ring_size * N * (16 * nv + 12 * nc_sensor)`

#### Example A. 四足机器人量级

若取：

- `N = 4096`
- `nb = 13`
- `nq = 19`
- `nv = 18`
- `max_contacts = 64`
- `nc_sensor = 4`
- `ring_size = 3`

则量级约为：

- `CorePublishedBlock`: 约 `13.3 MB`
- `RigidBlock`: 约 `31.5 MB`
- `TelemetryBlock`: 约 `4.1 MB`

合计：

- 约 `48.9 MB`

#### Example B. 更大刚体系统量级

若取：

- `N = 4096`
- `nb = 30`
- `nq = 37`
- `nv = 36`
- `max_contacts = 256`
- `nc_sensor = 8`
- `ring_size = 3`

则量级约为：

- `CorePublishedBlock`: 约 `30.1 MB`
- `RigidBlock`: 约 `120.0 MB`
- `TelemetryBlock`: 约 `8.3 MB`

合计：

- 约 `158.4 MB`

结论：

- 对当前设计，published ring 的主要显存压力通常来自 `RigidBlock`
- `max_contacts` 是最需要重点审查和约束的参数之一
- 对多数 RL 训练场景，平均 active contacts 远小于 `max_contacts`，因此 dense published `RigidBlock` 有显著 padding 浪费
- 第一版接受 dense preallocation，但应把 `max_contacts` 暴露为用户可调参数，并在文档中持续强调其显存影响

### 5.5 `max_contacts` 与 ring sizing 的选取指引

建议显式记录以下工程判断：

- `max_contacts` 在当前 GPU 实现里是 **per-env 上限**
- 它既影响 physics dense contact buffers，也直接影响 dense published `RigidBlock` 的上界显存
- 对多数 RL 训练场景，平均 active contacts 往往远小于 `max_contacts`，因此 published `RigidBlock` 的 padding 浪费很常见

对 `lossless` consumer，还应给出一个 ring sizing 的经验下界：

`published_ring_size ~= 1 + ceil(max_lossless_latency / effective_publish_period)`

其中：

- `max_lossless_latency` 是最慢 `lossless` consumer 从 frame ready 到 ack 的最长时延
- `effective_publish_period` 是该 consumer 实际需要的 frame 发布周期，不一定等于 physics `dt`

这不是严格证明后的充分条件，但作为工程预警非常有价值：如果最慢 `lossless` 路径的处理时延远大于 ring 可容纳的 frame 数，physics 被反压就是预期行为，而不是 bug。

### 6. 同步语义：`PublishedFrame` 不引用 mutable scratch

当前已收敛为明确结论：

- `PublishedFrame` 不能直接引用 mutable scratch
- 它必须引用 dedicated published slot buffers

原因：

- 下一步 physics 会覆盖 mutable scratch
- 如果 frame 直接引用 scratch，则“这一帧”的语义不稳定
- render/sensor/debug 的异步消费将无法安全成立

因此系统语义应为：

1. physics 写 `MutableState + DerivedCache`
2. `publish_core` 写入 `PublishedSlot[k]`
3. 写 `ready/event`
4. consumers 只读 `PublishedSlot[k]`
5. physics 继续推进下一步

### 7. 两种正式 consumer mode

当前建议不要只有一个统一同步模式，而是把 consumer 分成两类服务质量（QoS）模式。

#### A. `best_effort`

典型目标：

- realtime rendering
- 在线监看
- 非关键调试视图

语义：

- physics 不被该 consumer 阻塞
- consumer 允许跳过旧帧，直接读最近已提交帧
- ring slot 不因该 consumer 长时间保留

这意味着：

- “看最新状态” 优先于 “一帧不漏”
- 允许 drop frame
- 不允许反压 physics 主循环

#### B. `lossless`

典型目标：

- high-fidelity simulation rendering
- 离线高精渲染
- 不能丢帧的精确录制/数据集生成

语义：

- consumer 必须看到每一个 published frame
- physics 允许因该 consumer 而阻塞
- ring slot 在该 consumer 确认消费前不可回收

这意味着：

- “一帧不漏” 优先于 “physics 永远不停”
- 不允许 drop frame
- 必须有正式 backpressure 机制

### 8. 默认同步策略建议

基于当前讨论，建议默认规则如下：

- `realtime rendering` -> `best_effort`
- `high-fidelity simulation rendering` -> `lossless`

对应到用户侧语言：

- 开启 realtime rendering 时，不应阻塞 physics
- 开启高精仿真渲染时，应阻塞并确保不丢帧

### 9. Slot 回收规则

建议把 slot 生命周期与 consumer mode 显式绑定。

#### 对 `best_effort` consumer

- slot 不因该 consumer 而被 pin
- consumer 如果落后，可直接跳到最新 ready frame
- 若 ring 即将覆盖其未读旧帧，该旧帧可被丢弃

#### 对 `lossless` consumer

- slot 会被 pin 直到该 consumer 记录“已消费”
- physics 想覆盖最老未释放 slot 时，必须阻塞等待
- 若存在多个 `lossless` consumer，则 slot 回收取决于它们的最慢者

### 10. Host export 必须异步，但其 backpressure 取决于 consumer mode

当前已形成更细的结论：

- host export 本身应始终异步
- 但是否允许丢帧、是否反压 physics，取决于对应 consumer 的 QoS

因此建议区分两层缓冲：

- `PublishedRing`：device-side published slots
- `HostExportQueue`：host-side staging / serialization queue

#### 对 `best_effort` host export

例如：

- 低频 debug snapshot
- 非关键日志

策略建议：

- async copy 到 host staging
- 若 queue 满，可丢帧或合并到“最新快照”
- 不反压 physics

#### 对 `lossless` host export

例如：

- 高精录制
- 数据集生成

策略建议：

- async copy 仍然保留
- 但若 `HostExportQueue` 满，必须在“入队 / staging 分配”处形成 backpressure
- 一旦本帧数据已安全进入 host staging，device slot 可释放
- 后续编码/写盘继续异步进行，不必一直 pin 住 device slot

关键原则：

- host 文件写出速度不应直接决定 device slot 生命周期
- 真正需要阻塞时，应阻塞在“是否能保证本帧已被可靠转移到下一层队列”这一步

### 11. 当前推荐结论

截至 2026-04-24，本轮讨论在这一层的推荐结论为：

1. `PublishedRing` 默认大小取 `3`
2. `PublishedFrame` 引用 dedicated published slots，不引用 mutable scratch
3. `realtime rendering` 默认采用 `best_effort`
4. `high-fidelity simulation rendering` 默认采用 `lossless`
5. host export 始终异步，但是否反压 physics 由对应 consumer mode 决定
6. 在当前 rigid-only 阶段，优先控制 `max_contacts`，因为它最容易主导 published ring 显存增长

### 12. Consumer 读取语义必须区分 `borrow` 与 `snapshot`

本轮进一步确认：仅仅说“consumer 读一帧”是不够的，必须区分两种完全不同的读取契约。

#### A. `borrow`

含义：

- consumer 借用一个当前 ready 的 published slot view
- 该 view 只在 slot 生命周期内有效
- consumer 不获得该 slot 的保留权
- slot 可在后续被 ring 覆盖

适用场景：

- realtime rendering
- 立即执行的 device-side sensor render
- 只做短时读取、不跨帧滞留的 consumer

约束：

- `borrow` 不允许被理解为“长期持有这一帧”
- 如果 consumer 处理时间可能跨越 slot 生命周期，就不能只靠 `borrow`

#### B. `snapshot`

含义：

- consumer 从 published slot 中把所需数据复制/转存到自己的私有 staging
- 后续处理不再依赖原 slot 是否仍存在
- slot 在 snapshot 完成后即可按正常回收规则释放

适用场景：

- host export
- 离线编码/录制
- 长耗时 sensor assembly
- 任何需要异步慢处理的 consumer

### 13. `best_effort` 与 `lossless` 不是一回事，正交于 `borrow/snapshot`

当前建议正式区分两组概念：

- QoS 模式：
  - `best_effort`
  - `lossless`
- 读取方式：
  - `borrow`
  - `snapshot`

它们不是同一维度。

#### 常见组合

- `best_effort + borrow`
  - 最典型 realtime rendering
  - 只看最新帧，不保留旧帧

- `best_effort + snapshot`
  - 低频 debug 导出
  - 如果来不及 snapshot，可以跳过

- `lossless + snapshot`
  - 高精录制/数据集导出
  - 每帧必须安全转移到下一层队列

原则上也可存在：

- `lossless + borrow`

但这只适合极少数“consumer 与 physics 严格锁步、并且在 slot 生命周期内完成处理”的场景。默认不建议把它当主路径。

### 14. 推荐 API 语义

为了避免调用方误以为“拿到 frame 就能一直用”，建议在 API 上显式区分借用和快照。

建议接口层表达为：

```python
@dataclass(frozen=True)
class HostSnapshotSpec:
    fields: frozenset[str]
    env_ids: object | None = None

@dataclass(frozen=True)
class DeviceSnapshotSpec:
    fields: frozenset[str]
    env_ids: object | None = None

class PublishedFrameConsumer(Protocol):
    def borrow_latest_frame(self) -> object: ...
    def borrow_frame(self, frame_id: int) -> object: ...

    def snapshot_frame_to_host(self, frame_id: int, spec: HostSnapshotSpec) -> object: ...
    def snapshot_frame_to_device(self, frame_id: int, spec: DeviceSnapshotSpec, target_buffer: object) -> object: ...
```

语义要求：

- `borrow_*` 返回的应是带 context-manager 语义的 **ephemeral lease**
- `snapshot_*` 返回的是 **owned copy / owned staging handle**

推荐形态：

```python
with consumer.borrow_latest_frame() as frame:
    use(frame)
```

建议 `borrow` 对象在离开 `with` 作用域后主动失效，以尽早暴露“把 borrowed frame 存起来跨步使用”的误用。

### 15. 对 `best_effort` consumer 的正式约束

如果一个 consumer 选择 `best_effort + borrow`，则它必须接受：

- 旧 slot 可能在其下一次处理前已经被覆盖
- 系统不会因为它而 pin slot
- 它若想跨帧持有数据，必须主动执行 `snapshot`

因此对这类 consumer，正确理解不是：

- “我拥有这一帧”

而是：

- “我被允许短暂查看当前这帧”

### 16. 对 `lossless` consumer 的正式约束

如果一个 consumer 声明 `lossless`，则它必须明确自己的 ack 点。

推荐 ack 语义：

- `lossless + snapshot`
  - 当 snapshot 已经完成 staging，且其生命周期不再依赖原 device slot 时 ack

- `lossless + borrow`
  - 当借用处理已在 slot 生命周期内完成时 ack

这样 slot 回收逻辑才有可判定的稳定边界。

### 17. Ack / Reclaim 控制平面草案

为了把上面的语义变成可实现的运行时协议，建议显式引入三类控制对象：

- `ConsumerState`
- `AckPolicy`
- `SlotReclaimer`

#### `ConsumerState`

```python
@dataclass
class ConsumerState:
    consumer_id: str
    consumer_kind: str          # realtime_render / sensor / host_export / recorder
    qos_mode: str               # best_effort / lossless
    access_mode: str            # borrow / snapshot

    latest_seen_frame_id: int = -1
    acked_frame_id: int = -1

    enabled: bool = True
```

语义：

- `latest_seen_frame_id` 主要服务监控/debug，不参与 slot 回收判定
- `acked_frame_id` 只对 `lossless` consumer 有约束意义
- `best_effort` consumer 可以不维护严格 ack

#### `AckPolicy`

```python
@dataclass(frozen=True)
class AckPolicy:
    consumer_id: str
    qos_mode: str               # best_effort / lossless
    access_mode: str            # borrow / snapshot
    ack_point: str              # none / on_borrow_complete / on_snapshot_staged
```

推荐默认映射：

- `best_effort + borrow` -> `ack_point = none`
- `best_effort + snapshot` -> `ack_point = none`
- `lossless + borrow` -> `ack_point = on_borrow_complete`
- `lossless + snapshot` -> `ack_point = on_snapshot_staged`

这里的 `on_snapshot_staged` 指：

- host staging buffer 中已经拥有该帧的一份完整、自持（self-contained）副本
- 该副本的生命周期不再依赖原 device slot
- 若经过 device-to-host async copy，则 copy completion event 已经 signal
- 不要求磁盘写出、编码完成或最终外部消费完成

#### `PublishedSlotMeta`

```python
@dataclass
class PublishedSlotMeta:
    slot_id: int
    frame_id: int = -1
    step_index: int = -1
    sim_time: float = 0.0

    state: str = "free"         # free / writing / ready

    publish_event: object | None = None
    host_export_queued: bool = False
```

#### `SlotReclaimer`

```python
class SlotReclaimer(ABC):
    @abstractmethod
    def min_lossless_acked_frame_id(self) -> int: ...

    @abstractmethod
    def reclaimable(self, slot: PublishedSlotMeta) -> bool: ...

    @abstractmethod
    def wait_until_reclaimable(self, slot: PublishedSlotMeta) -> None: ...
```

建议默认规则：

```python
def reclaimable(slot_frame_id: int, min_lossless_ack: int) -> bool:
    return slot_frame_id <= min_lossless_ack
```

如果当前没有任何启用中的 `lossless` consumer，则：

- `min_lossless_acked_frame_id` 可视为 `+inf`
- slot 不因 consumer 而被 pin

### 18. 推荐 ack 时序

#### A. `best_effort + borrow`

1. consumer 借用最新 ready frame
2. 立即读取并使用
3. 更新 `latest_seen_frame_id`
4. 不推进正式 `acked_frame_id`

结论：

- 不参与 reclaim
- 被覆盖是允许语义

#### B. `best_effort + snapshot`

1. consumer 选择某个 frame 做 snapshot
2. 若 snapshot 成功，更新 `latest_seen_frame_id`
3. 不要求正式 `ack`

结论：

- queue 满时可丢帧
- 不反压 physics

#### C. `lossless + snapshot`

1. consumer 请求 frame snapshot
2. host/device staging 中获得一份完整、自持副本；若涉及 async copy，则 copy completion event 已 signal
3. consumer 更新 `acked_frame_id = frame_id`
4. slot 可参与 reclaim

结论：

- `ack` 点在“副本已经真正 staged 完成”
- 不在“最终慢处理完成”

#### D. `lossless + borrow`

1. consumer 借用指定 frame
2. 在 slot 生命周期内完成处理
3. 更新 `acked_frame_id = frame_id`

结论：

- 如果处理超出 slot 生命周期，该模式不成立
- 默认不建议作为通用主路径

### 19. 多个 `lossless` consumer 并存时的规则

建议明确采用最保守但最清晰的回收判定：

- slot 回收取决于所有启用中的 `lossless` consumer 的最小 `acked_frame_id`

即：

`min_lossless_ack = min(c.acked_frame_id for c in lossless_consumers if c.enabled)`

这意味着：

- 最慢的 `lossless` consumer` 决定 ring 回收速度
- 这是刻意设计，而不是副作用
- 如果用户同时开启多个不能丢帧的下游，就应接受它们共同形成 backpressure

建议配套两类可观测性机制：

- `ring_pressure_stats`
  - 当前最小 `lossless ack`
  - ring occupancy
  - publish wait time
  - 按 consumer 拆分的 ack lag / wait attribution

- `max_lag_frames`
  - 可作为 `lossless` consumer 的可选预警阈值
  - 超阈值时至少报警；是否自动降级为 `best_effort` 应保持为显式策略选择，而不应偷偷发生

关于 `lossless` 的 QoS 可信度，进一步明确：

- `lossless` 默认行为应是：监控 + 报警 + 阻塞等待
- 系统不应在用户不知情的情况下把 `lossless` 隐式降级为 `best_effort`
- 若支持降级，必须由用户显式 opt-in，例如 `fallback_qos="best_effort"`
- 一旦发生降级，必须产生清晰日志/事件记录，不能静默发生

### 20. 与 HostExportQueue 的关系

对 `lossless + snapshot` 的 host export，推荐把 ack 分两层理解：

- `device-side ack`
  - frame 已完成 staging，host/export 侧已有一份自持副本
  - device slot 可以释放

- `pipeline completion`
  - 编码/写盘/外部提交最终完成
  - 不再影响 device ring 回收

这样可以明确避免一个坏结果：

- 磁盘慢导致 device slots 长时间全部被 pin 住

真正需要 backpressure 的位置应是：

- host staging / export queue 是否还能可靠接住下一帧

而不是：

- 最终磁盘写出是否已经结束

### 21. Ring Pressure Behavior Matrix

本节把前面已经口头收敛的规则压成显式行为表。

这里的 “ring 满” 指：

- 下一次 publish 想写入的 slot 仍不可回收

这里的 “host queue 满” 指：

- 对应 snapshot 路径的 staging / export queue 无法再可靠接收新帧

#### A. `best_effort + borrow`

- 典型场景：
  - realtime rendering
  - 在线 viewport

- ring 满时行为：
  - 允许覆盖最旧未被 `lossless` pin 住的旧帧
  - consumer 若还引用旧 slot，需自行承担该引用失效的语义

- host queue 满时行为：
  - 不适用

- 是否允许丢帧：
  - 是

- 是否阻塞 physics：
  - 否

#### B. `best_effort + snapshot`

- 典型场景：
  - 低频 debug 导出
  - 非关键日志采样

- ring 满时行为：
  - 与 `best_effort + borrow` 一样，不因该 consumer 阻塞

- host queue 满时行为：
  - 可丢弃本次 snapshot
  - 可合并为“只保留最新快照”
  - 不要求保留所有中间帧

- 是否允许丢帧：
  - 是

- 是否阻塞 physics：
  - 否

#### C. `lossless + snapshot`

- 典型场景：
  - 高精仿真渲染
  - 数据集录制
  - 逐帧可靠导出

- ring 满时行为：
  - 不能覆盖尚未被该 consumer ack 的 frame
  - 必须等待可回收 slot

- host queue 满时行为：
  - 必须阻塞在 snapshot 入队 / staging 分配处
  - 直到该帧能够被可靠转移到下一层队列

- 是否允许丢帧：
  - 否

- 是否阻塞 physics：
  - 是

#### D. `lossless + borrow`

- 典型场景：
  - 极少数严格锁步、处理极短的专用 consumer

- ring 满时行为：
  - 不能覆盖其仍未完成处理的 slot
  - 必须等待该 consumer 完成 borrow 处理并 ack

- host queue 满时行为：
  - 不适用

- 是否允许丢帧：
  - 否

- 是否阻塞 physics：
  - 是

### 22. 行为矩阵的归纳结论

可以把上面的矩阵压缩成两条总规则：

#### 规则 1. `best_effort` 不形成硬 backpressure

无论是 `borrow` 还是 `snapshot`：

- 允许跳帧
- 允许丢弃旧帧或放弃本次 snapshot
- 不应阻塞 physics 主循环

#### 规则 2. `lossless` 必形成硬 backpressure

无论是 `borrow` 还是 `snapshot`：

- 不允许覆盖尚未可靠消费的 frame
- ring 无可回收 slot 时必须等待
- 如果下一层队列接不住，也必须等待

这正是本轮讨论里“ring 满行为要和读出阻塞语义对应”的核心落点。

### 23. 多 consumer 并存时的组合规则

建议把系统整体行为定义为：

- 所有 `best_effort` consumer` 只影响“谁能看到什么帧”，不影响 slot 回收
- 任意一个启用中的 `lossless` consumer` 都会影响 slot 回收
- 多个 `lossless` consumer` 并存时，系统按最慢者形成 backpressure

因此：

- 可以同时存在 realtime viewport 和 lossless dataset export
- 此时 viewport 仍是 `best_effort`
- 但整个系统是否阻塞，由 lossless export 决定

### 24. 调度器级建议

为了让上述矩阵真正落地，建议调度器在每次 publish 前显式执行：

1. 计算目标 slot 是否可回收
2. 若不可回收，判断原因是否来自 `lossless` consumer
3. 若仅有 `best_effort` consumer` 落后，不等待，继续覆盖旧帧
4. 若存在 `lossless` backpressure`，等待相应 ack / staging capacity

这样 “ring pressure behavior” 会体现在调度器逻辑中，而不是散落在各 consumer 实现里。

---

## CPU Path Positioning

本轮没有开始具体设计 CPU path，但已形成清晰定位：

- CPU path 不应成为 GPU path 的架构来源
- CPU path 应是共享语义下的 reference / baseline / debug-friendly 实现

适合承担的职责：

- correctness baseline
- offline export
- small-scale high-precision inspection
- debug / architecture validation

不应成为：

- future GPU publish/export constraints 的反向绑架源

---

## Relation to Existing Project Research

本轮讨论与仓库中已有参考项目结论高度一致。

### MuJoCo / MJWarp

支持：

- `Model + Data` / `Model + State`
- authoritative state 统一
- GPU 只是更换执行实现，不改变世界语义

### Newton

支持：

- flat SoA model/state
- preallocated contacts
- multi-world / GPU-friendly buffers

### Drake

支持：

- physics truth 与 geometry registration 分离
- 同一几何可有多种 role
- shared world semantics，多消费者派生

### Isaac Lab

支持：

- physics 资产/view 与 observation/reward/termination manager 分离
- consumer 不反向侵入 physics 热路径

### SOFA

支持：

- 机械状态 / 碰撞表示 / 视觉表示 / mapping 分层

但本项目已明确拒绝其：

- scene graph visitor execution model
- component-level hot-loop dispatch

---

## Open Questions for Review

以下问题尚未定案，适合在与 Claude 的审查中重点挑战。

### 1. `GpuPublishedFrame` 的最小字段集

是否应拆成：

- `core truth`
- `telemetry`
- `render-facing`
- `sensor-facing`
- `host-exportable`

如果拆，边界在哪里。

### 2. `PublishPolicy` 的粒度

是否为每种 view 独立配置：

- enabled
- rate
- env selector
- detail level

还是更集中统一。

补充要审查的问题：

- `PublishPlan` 是否应成为正式类型，而不是 engine 内部临时局部变量
- env 子集选择是否统一抽象成 compacted `env_ids`，还是保留 `all/subset` 双路径
- detail level 是否应该只允许有限个 compile-time / launch-time variant，而不是任意配置字符串

### 3. `DerivedPhysicsCache` 与 `PublishedFrame` 的关系

哪些数据：

- 只存在于 cache
- 哪些数据必须复制或冻结到 published frame
- 哪些只保留 handle / pointer / descriptor

### 4. GPU realtime renderer 和 sensor pipeline 是否共享一套 surface cache

例如：

- camera / lidar / realtime renderer 是否可以共享 dynamic surface rep
- 还是必须按 consumer 分开物化

### 5. CPU path 是否也应显式建 `PublishedFrame`

从语义一致性角度看，答案倾向于 yes；  
但需要评估是否值得在 CPU reference path 中也保留同样的 frame-boundary object。

### 6. `contact` 到未来 `InterfaceInteraction` 的类型关系

是否应该正式确立：

“所有 solver-facing `contact` objects 都是某类 interface interaction 的具体实现”

这会影响：

- 类型命名
- narrowphase 输出
- solver 输入结构
- multi-physics 扩展路径

---

## Working Summary

当前已经收敛出的中间结论可以压缩成下面几句：

1. 当前系统还没有统一的仿真结果存储层，只有 `StepOutput + query_contacts + recent-step caches`
2. `RenderScene` 是 debug/inspection scene，不应被扩张为所有未来需求的统一 world representation
3. CPU / GPU 两条路径应该分别设计，但必须共享物理语义和对外数据契约
4. 数据导出必须进入仿真 pipeline，作为正式的 `publish/export` 阶段
5. GPU path 应优先设计，因为它决定 host/device 边界和多消费者同步方式
6. GPU path 需要 `GpuPublishedFrame` 作为 frame boundary，而不是让消费者直接读 mutable state
7. 三类 view 都应该是可选消费者，最好通过 `PublishPolicy` 控制 on/off、频率、env 子集和 detail
8. `PublishPolicy` 不能直接进入 kernel 热路径；必须先解析成逐帧 `PublishPlan`
9. kernel 只执行 plan 选定的 phase / variant / env subset，不承担逐线程策略解释
10. `interface` 是更一般的 multi-physics 概念，`contact` 是其在 rigid-body 接触模型中的特例

---

## Recommended Next Step

后续设计讨论建议按如下顺序推进：

1. 先定 `GpuPublishedFrame` 字段边界
2. 再定 `PublishPolicy`
3. 再定 GPU path 各 stream / event / synchronization
4. 最后让 CPU path 实现相同语义的简化版本

这会让后续 CPU/GPU 两条路径的讨论更具体，也更不容易在原则层反复横跳。

---

## Review Resolution Snapshot

截至 2026-04-24，本轮与 Claude 的 review 形成了以下补充共识：

1. `max_contacts` 在当前公式中是 **per-env 上限**，并且是 dense published `RigidBlock` 的主要显存放大器。
2. dense `RigidBlock` 第一版可以继续预分配，但其写入应由 `PublishPlan.do_rigid_block_write` 控制，而不是默认每步无条件写入。
3. `lossless + snapshot` 的 ack 点应理解为 `on_snapshot_staged`：
   staging 中已经拥有完整、自持副本；若涉及 async copy，则 copy completion event 已 signal。
4. `borrow` 与 `snapshot` 的 API 必须保持强区分；`borrow` 应采用 context-manager / ephemeral lease 语义，以降低误用风险。
5. 多个 `lossless` consumer 并存时，最慢者决定 backpressure 是接受的，但需要 `ring_pressure_stats`、`max_lag_frames` 等监控配套。
6. `lossless` 的 QoS 承诺不能被系统静默打破：
   默认行为应是监控 + 报警 + 阻塞等待；任何降级都必须是用户显式 opt-in，并伴随明确日志/事件。
