# Robot Simulator — Project Manifest

> 面向具身智能研究的多物理仿真平台——多物理统一耦合、GPU 原生、渲染与合成数据生成、从第一性原理出发的 API 设计。
> Last updated: 2026-06-03 (Q54 P9 source/clock metadata)

## 一句话

从第一性原理出发构建的物理仿真平台，支持刚体、可变形体、流体的统一耦合，GPU 原生并行，内置渲染管线，面向具身智能研究的高质量物理标注合成数据生成。

## 架构

```
Layer 4  VecEnv              GPU 并行 RL 训练 (N=4096, torch.Tensor on CUDA)
Layer 3  Env                 Gymnasium 接口 + ObsManager + 域随机化(stub)
Layer 2  Simulator           多机器人编排: Scene → CollisionPipeline → PhysicsEngine
Layer 1  Physics Core        刚体动力学管线 (StepPipeline → ForceSource → ConstraintSolver → integrate)
Layer 0  Math                空间代数 (6D Plücker, [linear; angular] 约定)

rendering/                   Backend-agnostic RenderScene + matplotlib viewer
sensing/                     Sensor specs/readings/views + optical camera postprocess
optics/                      Optical registry/scene/executor/result pipeline
tools/optical_pipeline_lab/   Q54 optical pipeline scenario/timing/preset lab tooling
```

依赖方向：`rl_env/ → simulator.py → robot/ → physics/`（不可逆）。
`sensing/` 与 `rendering/` 都消费 physics/public published state；`optics/`
作为独立 integration layer 消费 `PublishedFrame + OpticalWorldRegistry`，不把
Rerun/debug sink 放进计算路径。

## 物理管线（StepPipeline）

```
DynamicsCache ← FK + body_v（算一次，全链路复用）

Stage 1 (smooth):     PassiveForceSource → qfrc_passive
                      tau_smooth = tau + qfrc_passive
                      qacc_smooth = ABA(无接触) 或 CRBA(有接触)

Stage 2 (constraint): ConstraintSolver.solve() → qacc
                      PGS-SI (CPU RL) 或 ADMMQPSolver (CPU precision)

Integration:          qdot += dt × qacc;  q = tree.integrate_q(q, qdot, dt)

Output:               ForceState (可观测力分解)
```

## 求解器

| 求解器 | 级别 | 平台 | 定位 |
|--------|------|------|------|
| **PGSSplitImpulseSolver** | velocity | CPU | RL 训练快速路径 |
| **ADMMQPSolver** | acceleration | CPU | 高精度 MuJoCo 对标 |
| **Jacobi-PGS-SI** | velocity | GPU (Warp) | 大规模 RL (N=4096) |
| **Jacobi-PGS-MS** | velocity | GPU (Warp) | Mass splitting, 多点接触稳定 |
| **Colored-PGS** | velocity | GPU (Warp) | Graph-colored GS, GS 收敛性 |
| **GPU ADMM** | velocity | GPU (Warp) | 高精度 GPU 仿真 |

## GPU 碰撞检测

GpuEngine 使用解析碰撞函数（shape type dispatch）+ S-H 面裁剪：

| 碰撞对 | GPU 状态 |
|--------|---------|
| Sphere/Capsule/Box/Cylinder vs Ground | ✅ 解析 @wp.func |
| Box vs Ground (多点) | ✅ 4 点顶点枚举 |
| Sphere-Sphere, Sphere-Capsule, Capsule-Capsule, Sphere-Box | ✅ 解析 @wp.func |
| Box-Box (多点 manifold) | ✅ SAT + S-H 面裁剪 (1-4 点) |
| Capsule-Box | ⬜ fallback 球近似 |
| ConvexHull (凸分解) | ⬜ 待 GPU GJK kernel (Q41) |
| Mesh (BVH) | ⬜ 待 BVH 集成 (Q17) |

## 渲染

```
RenderBackend (ABC)
  ├── MatplotlibBackend   调试可视化 + GIF 导出
  └── RerunBackend        实时流式可视化 (rerun-sdk >= 0.16)

RenderScene (backend-agnostic dataclass)
  ├── PositionedShape: 碰撞形状 + world pose
  ├── ContactPoint: 接触点可视化
  ├── TerrainInfo: 地形描述
  └── skeleton_links: 骨骼连线

scene_builder.build_render_scene() → RenderScene
  ├── 从 RobotTree FK 提取形状位置
  └── 从 ContactInfo 提取接触点

build_render_scene_from_gpu() → RenderScene (GPU engine 路径)

RerunBackend 也可显式消费已经算好的 `OpticalCameraReading`，记录
depth/range/segmentation/RGB preview；这只是 visualization sink，不参与光学计算。
```

## 感知与光学

```
sensing/
  ├── StateSampleView + IMU/Joint/Force/Contact/Range readings
  ├── SurfaceQuerySpec + CpuPlaneSurfaceQueryExecutor
  └── OpticalPinholeCameraSpec + OpticalCameraImageResult/Reading

optics/
  ├── OpticalWorldRegistry: geometry/material/light/instance identity
  ├── OpticalSceneCache: PublishedFrame + registry → OpticalSceneSnapshot
  ├── CpuReferenceOpticalExecutor: first-hit range/material/instance ids
  ├── CpuBvhOpticalExecutor: scene-owned CPU BVH acceleration
  ├── CpuDirectLightOpticalExecutor: deterministic two-sided Lambertian RGB
  ├── GpuBruteForceOpticalExecutor: L5A Warp first-hit device result path
  ├── execute_optical_on_gpu_published_frame: L5B.1 Q52 GPU optical runtime helper
  ├── DeviceOpticalSceneCache: L5C.0 device-resident optical scene update
  ├── GpuDeviceSceneOpticalExecutor: L5C.1 first-hit executor over device scene buffers
  ├── GpuDeviceBvhOpticalExecutor: device-scene BVH first-hit traversal
  ├── GpuDeviceBvhDirectLightOpticalExecutor: GPU direct-light + hard shadows
  └── build_cuda_lbvh_from_snapshot: CUDA LBVH builder for warmed-session BVH rebuild

Q54 current flow:
  OpticalWorldRegistry + OpticalFrameInputs
    → OpticalSceneSnapshot
    → OpticalExecutor.execute(snapshot, sensor_spec)
    → OpticalComputeResult(host/device/external)
    → sensing readings / Rerun sink / examples / future RL consumers
```

当前 L5 GPU optical 已从 first-hit bridge 推进到 device-scene BVH、
direct-light/shadow、CUDA LBVH build spike 和 GPU pinhole camera raygen。
L5B.1 已接入 `GpuPublishedFrame` Q52 borrow/complete lifecycle，并支持
world-static 与 rigid body-bound optical instances。L5C.0/L5C.1b 让 registry
geometry/metadata 长驻 device，并以 derived triangle payload
（`v0/e1/e2/normal`）生成 world primitive buffers。L5C.2 现有 flat BVH
first-hit traversal 与 refit instrumentation；L5C.3 加入 deterministic RGB、
ambient/directional/point light 和 inline shadow any-hit；L5C.4 CUDA LBVH spike
已把 warmed tree build 压到 ms 级。Video benchmark 使用 `--video-readback`
区分 blocking host readback，不把它误称为 pipeline stage；`--video-raygen gpu`
可在 GPU 上按 pixel id 生成 pinhole rays，保留 `OpticalRaySensorSpec` 作为
LiDAR/arbitrary ray query 和 CPU/GPU parity reference。GPU optical/rendering
管线的长期设计基线已固化到 `GPU_OPTICAL_PIPELINE_DESIGN.md`；`collab/`
继续作为 Codex/Claude review 与讨论工作区。`tools/optical_pipeline_lab/`
现已承接 source-driven render foundation、dynamic frame prep、frame-context
providers、generic video loop、video/export tuning 的 scenario config、timing schema、preset metadata
和 thin runner。动态渲染入口应来自 physics-published `GpuPublishedFrame`；
Go2/Menagerie 等 concrete static asset source 只用于静止/benchmark 场景的
Menagerie model defaults、camera preset 和 preview/reporting glue，不是物理帧提供者、
渲染 pipeline，也不是 renderer/backend adapter；通用静态 asset source 构建已收敛到
`static_asset_source.py` 的 `build_static_asset_render_source(...)`，不再使用
Go2 命名承载通用静态组件；
`OpticalLabRenderSource` / `OpticalLabRenderOptions` / `OpticalLabRenderWorkspace`
/ `OpticalLabRenderSession` / `OpticalLabRenderFrameContext` /
`OpticalLabRenderPipeline` 是通用 lab render 入口，旧 `Go2Render*`
兼容 alias 与 `go2_session.py` shim 已清理完毕。Stage J 已开始把真实
physics-published `GpuPublishedFrame` 接入同一 source vocabulary：
scenario config 现在用 `frame_source` / `clock_owner` 区分 static asset
builder、synthetic frame sequence 与 physics-published frame source，
`physics_source.py` 提供 physics frame → lab render source 的最小桥接、
physics pipeline factory 与 borrow/begin/complete frame lease helper，
并封装 render-backed device consumer 注册，
`PhysicsLabRenderRuntime` 将 engine/pipeline/consumer 组合成物理渲染循环入口，
`frame_contexts.py` 提供 static、synthetic frame sequence 与 physics-published
三类 context provider，让 video 层消费已取得的 `OpticalLabRenderFrameContext`，
`video_loop.py` 也提供 provider-backed benchmark 入口和 provider-backed
torch async warmup helper，
`frame_runtime.py` 提供窄版 lab-internal `FrameWorkflowRunner`，
负责 provider lifecycle、video consumer 与 delivery 的一帧工作流编排，
`physics_runtime.py` 提供 lab-internal `PhysicsLabScenarioRuntime` lifecycle
owner 与 `create_physics_body_triangle_lab_runtime(...)`，用于显式管理
engine/registry/base_frame/bounds/step_frame/close 边界，
`runner.py` 提供显式 `run_physics_video_scenario(...)`，要求调用方提供
engine、registry、base frame 与逐帧 `published_frame_for_index` callback；
并提供 `run_physics_stepped_video_scenario(...)`，用
`PhysicsPublishedFrameStepper` 表达调用方负责推进或选择 physics 时间后返回
published frame。因此 physics 时间推进仍由物理侧所有，
GPU smoke 覆盖 `GpuEngine.step()` → `OpticalLabRenderPipeline.begin_frame(...)`
→ dynamic snapshot/BVH → direct-light ray render、GPU pinhole camera raygen render，
并已有 physics video runner smoke 验证 provider/runtime/delivery/timing
串接与 `frame_source=physics_published_frame`,
`clock_owner=external_physics_runtime` CSV metadata；stepped runner smoke 验证
`PhysicsPublishedFrameStepper` 真实推进 `GpuEngine.step(...)` 后，rendered
`range_m` 随当前 physics body height 改变而不是读取 stale base frame；
runtime-owner GPU smoke 验证 `PhysicsLabScenarioRuntime` context manager 可驱动
同一 stepped runner 并在 workflow 后关闭。
普通 CLI `run_scenario(...)`
仍不会构造 physics engine。

## 关键文件

| 文件 | 职责 |
|------|------|
| `physics/step_pipeline.py` | 刚体动力学管线（两阶段 + 内联积分） |
| `physics/robot_tree.py` | 运动学树: FK, ABA, CRBA, RNEA, integrate_q |
| `physics/solvers/admm_qp.py` | ADMMQPSolver (acceleration-level QP) |
| `physics/solvers/pgs_split_impulse.py` | PGS + split impulse |
| `physics/gjk_epa.py` | GJK/EPA + convex margin (gjk_distance) + CPU face clipping manifold |
| `physics/geometry.py` | CollisionShape + FaceTopology + ConvexHullShape |
| `physics/backends/warp/analytical_collision.py` | GPU 解析碰撞 + S-H 面裁剪 manifold |
| `physics/backends/warp/admm_kernels.py` | GPU ADMM 求解器 (Cholesky + 锥投影) |
| `physics/backends/warp/mass_splitting_kernels.py` | Jacobi PGS mass splitting (Tonge 2012) |
| `physics/backends/warp/colored_pgs_kernels.py` | Graph-colored GS (PhysX 方案) |
| `physics/gpu_engine.py` | GPU 物理引擎 (Warp kernel 管线, 4 solver dispatch) |
| `physics/cpu_engine.py` | CPU 物理引擎 (GJK/EPA + PGS/ADMM) |
| `sensing/state_sample.py` | PublishedFrame → sensor-facing state sample |
| `sensing/surface_query.py` | Surface/range query spec + CPU plane executor |
| `sensing/optical.py` | Optical ray/camera specs + camera image postprocess |
| `sensing/readings.py` | Sensor-facing readings, including `OpticalCameraReading` |
| `sensing/builders.py` | State/query/result → host-owned sensor readings |
| `optics/registry.py` | Optical materials/lights/geometry/instances and stable numeric ids |
| `optics/builder.py` | RobotModel collision geometry → OpticalWorldRegistry builder |
| `optics/scene.py` | OpticalFrameInputs + OpticalSceneSnapshot + CPU BVH payload |
| `optics/execution.py` | CPU reference/BVH/direct-light optical executors |
| `optics/device.py` | Device workload packing + device→host result staging/readback helpers |
| `optics/device_bvh.py` | Device BVH layout, CPU-build upload, refit, and traversal support buffers |
| `optics/device_scene.py` | L5C.0/L5C.1b device-resident registry geometry + derived world primitive update |
| `optics/cuda_lbvh.py` | CUDA LBVH extension spike and device BVH builder backend |
| `optics/warp_execution.py` | GPU optical Warp executors: brute-force, device scene, BVH, direct-light/shadow, GPU camera raygen |
| `optics/gpu_runtime.py` | L5B.1 Q52 `GpuPublishedFrame` optical runtime helper |
| `GPU_OPTICAL_PIPELINE_DESIGN.md` | Q54 GPU optical/rendering pipeline repo-level design baseline: scenarios, delivery policies, Optical Pipeline Lab, roadmap |
| `tools/optical_pipeline_lab/` | Optical Pipeline Lab: scenario configs, presets, timing CSV schema, report helpers, static asset source builders, thin Go2 runner |
| `tools/optical_pipeline_lab/async_readback.py` | Optical Pipeline Lab async D2H readback ring helper for pinned Torch copies |
| `tools/optical_pipeline_lab/dynamic_frames.py` | Lab-only synthetic GPU published-frame clone/perturb helpers for dynamic optical smokes |
| `tools/optical_pipeline_lab/frame_contexts.py` | Static/synthetic/physics OpticalLabRenderFrameContext provider helpers |
| `tools/optical_pipeline_lab/frame_providers.py` | Product-facing tick frame-context provider adapters for video products |
| `tools/optical_pipeline_lab/frame_runtime.py` | Lab-internal FrameWorkflowRunner for provider/video/delivery frame workflow |
| `tools/optical_pipeline_lab/static_asset_source.py` | Static asset OpticalLabRenderSource builders for Go2/Menagerie and synthetic lab scenes |
| `tools/optical_pipeline_lab/menagerie_static_runner.py` | Legacy Go2/Menagerie static preview and benchmark runner for lab runner and example CLI |
| `tools/optical_pipeline_lab/go2_backend.py` | Deprecated compatibility shim for legacy Go2 backend imports |
| `tools/optical_pipeline_lab/physics_source.py` | Physics-published `GpuPublishedFrame` → Optical Pipeline Lab render source bridge |
| `tools/optical_pipeline_lab/render_session.py` | OpticalLabRenderSource / Options / Workspace / PreparedFrame / Session / FrameContext / Pipeline lab render foundation |
| `tools/optical_pipeline_lab/video_loop.py` | Generic Optical Pipeline Lab video plan/render-context/delivery helpers parameterized by camera builder |
| `tools/optical_pipeline_lab/camera_presets.py` | Generic Optical Pipeline Lab fixed/orbit video camera preset builders |
| `tools/optical_pipeline_lab/rgb_pack.py` | Optical Pipeline Lab GPU RGB8 preview packing helper |
| `tools/optical_pipeline_lab/frame_tick.py` | Shared `SimulationFrameTick` contract for multi-product lab workflows |
| `tools/optical_pipeline_lab/frame_products.py` | `FrameProductResult` + product protocol contracts for tick-based workflows |
| `tools/optical_pipeline_lab/product_workflow.py` | Generic/physics product workflow runners, named product run results, and P10/P11 Optical Lab workflow convenience entry points |
| `tools/optical_pipeline_lab/product_specs.py` | `DebugProductSpec`, `ObservationProductSpec`, and runtime-specific `VideoProductSpec` declarative builders for P10/P11 workflows |
| `tools/optical_pipeline_lab/preset_runtime.py` | P11 reviewed preset -> physics/static workflow runtime factory registry for Optical Lab workflows |
| `tools/optical_pipeline_lab/preset_products.py` | P11 user-facing preset product string resolver, including reviewed physics and static video product specs |
| `tools/optical_pipeline_lab/preset_workflows.py` | P11 public preset workflow entry point that delegates to generic P10 product workflows |
| `tools/optical_pipeline_lab/video_products.py` | P11 reviewed physics/static video product spec factories for Optical Lab presets |
| `tools/optical_pipeline_lab/observation_products.py` | `PublishedStateObservationProduct` — published frame → RL observation vector (P9.4) |
| `tools/optical_pipeline_lab/physics_runtime.py` | Explicit physics runtime owners for Optical Pipeline Lab scenarios |
| `tools/optical_pipeline_lab/delivery.py` | Video delivery helpers for Optical Pipeline Lab experiments |
| `tools/optical_pipeline_lab/matrix.py` | Matrix suite runner for Optical Pipeline Lab baselines |
| `tools/optical_pipeline_lab/presets.py` | Preset registry for Optical Pipeline Lab scenarios |
| `tools/optical_pipeline_lab/reports.py` | Report formatting helpers for Optical Pipeline Lab output |
| `tools/optical_pipeline_lab/runner.py` | Runner entry points and generic/physics video product builders for the Optical Pipeline Lab |
| `tools/optical_pipeline_lab/scenarios.py` | Scenario configuration vocabulary for the Optical Pipeline Lab |
| `tools/optical_pipeline_lab/timing.py` | Timing and CSV helpers for optical pipeline lab runs |
| `benchmarks/bench_optical_device_scene.py` | L5C.1c AABB/BVH decision benchmark harness |
| `benchmarks/robot_optical_scene.py` | Shared robot-like optical scene generator for benchmarks/examples |
| `rendering/render_scene.py` | Backend-agnostic 场景描述 |
| `rendering/scene_builder.py` | Physics state → RenderScene |
| `rendering/shape_artists.py` | matplotlib 形状绘制 |
| `examples/optical_direct_light_preview.py` | In-repo CPU optical RGB/depth/segmentation preview |
| `examples/optical_robot_scene_preview.py` | Robot-like optical preview writer using benchmark scene generator |
| `examples/mujoco_menagerie_robot_preview.py` | Open-source MuJoCo Menagerie visual mesh preview importer/writer |
| `examples/mujoco_menagerie_gpu_preview.py` | Legacy CLI wrapper for the shared Menagerie static GPU runner |
| `examples/optical_readback_microbench.py` | Device→host readback/materialization microbenchmark |
| `examples/optical_lab/README.md` | P11 Optical Lab preset workflow examples overview |
| `examples/optical_lab/physics_body_triangle_video_debug.py` | P11 example: physics body-triangle preset with video/debug products |
| `examples/optical_lab/physics_body_triangle_observation.py` | P11 example: physics body-triangle preset with explicit observation product |
| `simulator.py` | 多机器人编排 |
| `robot/urdf_loader.py` | URDF → RobotModel |
| `robot/mesh_loader.py` | trimesh 网格加载 → ConvexHullShape |

## 规模

- Q54 sensing/optics 子系统当前收集 **267 个测试**：
  `tests/unit/optics` + `tests/unit/sensing` + `tests/gpu/test_optical_warp_executor.py`
  + `tests/gpu/test_optical_gpu_runtime.py`
  （185 unit optics/lab + 40 unit sensing + 42 GPU optical）
- 当前 push gate：2026-05-22 在 `env_tilelang_20260119` 中完成全量
  `python -m pytest -q`，结果 **1555 passed, 1 skipped**，耗时 31:07；
  `ruff check .` 与 `ruff format --check .` clean。
- physics/ ~16,000 行，rendering/ ~960 行；新增 sensing/、optics/ 与
  tools/optical_pipeline_lab/ 作为独立感知/光学与 pipeline tuning 子系统
- 支持多机器人场景 + 静态几何 + 碰撞过滤 + 多点接触 manifold

## 进度

| Phase | 状态 |
|-------|------|
| 1 — CPU 物理核心 | ✅ |
| 2a-2e — 重构 + URDF + Simulator + RL env + GPU 后端 | ✅ |
| 2f — 高精度接触 (GJK/EPA + 6 求解器) | ✅ |
| 2g — CRBA + tensor core | ✅ |
| 2h — 力系统重构 (StepPipeline) | ✅ |
| 2i — PhysicsEngine 统一 (CpuEngine + GpuEngine) | ✅ |
| 2j — GPU 解析碰撞 + MuJoCo 亚毫米对标 | ✅ |
| 2k — GPU ADMM 求解器 + solver dispatch | ✅ |
| 2l — GPU 多点接触 manifold + solver stability (session 27-29) | ✅ |
| 2m — EPA 鲁棒性 + convex margin 两道防线 (session 30) | ✅ |
| 3 — 渲染 (RenderBackend ABC + MatplotlibBackend + RerunBackend) | ✅ |
| Q53 — sensing/rendering 边界 + sensor-facing readings | ✅ |
| Q54 L0-L3 — optical registry/scene/CPU BVH/direct-light | ✅ |
| Q54 L5A — Warp brute-force GPU optical first-hit executor | ✅ |
| Q54 L5B.0 — GpuPublishedFrame/Q52 world-static optical device lifecycle | ✅ |
| Q54 L5B.1 — host-staged body-bound GPU optical runtime | ✅ |
| Q54 L5C.0 — device-resident optical scene cache/update | ✅ |
| Q54 L5C.1a — GPU executor over device scene buffers | ✅ |
| Q54 L5C.1b — derived triangle buffers | ✅ |
| Q54 L5C.1c — AABB traversal variant + benchmark harness | 🟡 |
| Q54 L5C.2 — GPU BVH traversal/refit correctness bridge | ✅ |
| Q54 L5C.3 — GPU direct-light + shadow any-hit | ✅ |
| Q54 L5C.4 — CUDA LBVH build + GPU raygen/readback optimization | 🟡 |
| Q54 Stage B/C1-C5 + cleanup — Optical Pipeline Lab source-driven foundation + video loop split | ✅ |
| Q54 Stage J.0 — physics-published frame → Optical Pipeline Lab ray/camera render smoke | ✅ |
| 4 — 域随机化 | ⬜ |
| 5 — Sim-to-Real | ⬜ |

## PhysicsEngine 统一

CPU/GPU 已统一到 PhysicsEngine 接口：
```
Scene.build_merged() → MergedModel（多 robot 合并为单一多根树）
  → PhysicsEngine.step(q, qdot, tau) → StepOutput
    ├─ CpuEngine: GJK/EPA ground + body-body + PGS-SI/ADMMQPSolver
    └─ GpuEngine: 解析碰撞 + S-H manifold + 4 solver backends
                  (jacobi_pgs_si / jacobi_pgs_ms / colored_pgs / admm)
```

## 设计原则

- **physics/ 是未来独立库** — 不依赖上层，可单独发布
- **积分器属于物理子系统** — 不同物理（刚体/柔体/流体）各自积分
- **MuJoCo 命名 + Isaac flat tensor 布局** — qfrc_*, ForceState 可观测
- **每个求解器输出 qacc (nv,)** — 统一接口，不管内部是 acceleration 还是 velocity level
- **ContactBuffer 解耦碰撞与求解** — 碰撞检测和约束求解通过统一 buffer 格式正交组合
