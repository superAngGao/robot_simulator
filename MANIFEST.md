# Robot Simulator — Project Manifest

> 从零构建的刚体物理仿真器，面向 sim-to-real 腿足机器人 RL 训练。
> Last updated: 2026-04-13 (session 29)

## 一句话

基于 Featherstone 算法的刚体动力学引擎 + 约束求解器 + GPU 并行环境，目标是在自有硬件上完成四足/双足机器人的 sim-to-real 训练。

## 架构

```
Layer 4  VecEnv              GPU 并行 RL 训练 (N=4096, torch.Tensor on CUDA)
Layer 3  Env                 Gymnasium 接口 + ObsManager + 域随机化(stub)
Layer 2  Simulator           多机器人编排: Scene → CollisionPipeline → PhysicsEngine
Layer 1  Physics Core        刚体动力学管线 (StepPipeline → ForceSource → ConstraintSolver → integrate)
Layer 0  Math                空间代数 (6D Plücker, [linear; angular] 约定)

rendering/                   Backend-agnostic RenderScene + matplotlib viewer
```

依赖方向：`rl_env/ → simulator.py → robot/ → physics/`（不可逆）

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
RenderScene (backend-agnostic dataclass)
  ├── PositionedShape: 碰撞形状 + world pose
  ├── ContactPoint: 接触点可视化
  ├── TerrainInfo: 地形描述
  └── skeleton_links: 骨骼连线

scene_builder.build_render_scene() → RenderScene
  ├── 从 RobotTree FK 提取形状位置
  └── 从 ContactInfo 提取接触点

shape_artists.py → matplotlib 3D artists
  ├── Box/Sphere/Cylinder/Capsule/ConvexHull 绘制
  └── 接触法线箭头 + 力大小标注

viewer.py → matplotlib 交互/导出
```

## 关键文件

| 文件 | 职责 |
|------|------|
| `physics/step_pipeline.py` | 刚体动力学管线（两阶段 + 内联积分） |
| `physics/robot_tree.py` | 运动学树: FK, ABA, CRBA, RNEA, integrate_q |
| `physics/solvers/admm_qp.py` | ADMMQPSolver (acceleration-level QP) |
| `physics/solvers/pgs_split_impulse.py` | PGS + split impulse |
| `physics/gjk_epa.py` | GJK/EPA + CPU face clipping manifold |
| `physics/geometry.py` | CollisionShape + FaceTopology + ConvexHullShape |
| `physics/backends/warp/analytical_collision.py` | GPU 解析碰撞 + S-H 面裁剪 manifold |
| `physics/backends/warp/admm_kernels.py` | GPU ADMM 求解器 (Cholesky + 锥投影) |
| `physics/backends/warp/mass_splitting_kernels.py` | Jacobi PGS mass splitting (Tonge 2012) |
| `physics/backends/warp/colored_pgs_kernels.py` | Graph-colored GS (PhysX 方案) |
| `physics/gpu_engine.py` | GPU 物理引擎 (Warp kernel 管线, 4 solver dispatch) |
| `physics/cpu_engine.py` | CPU 物理引擎 (GJK/EPA + PGS/ADMM) |
| `rendering/render_scene.py` | Backend-agnostic 场景描述 |
| `rendering/scene_builder.py` | Physics state → RenderScene |
| `rendering/shape_artists.py` | matplotlib 形状绘制 |
| `simulator.py` | 多机器人编排 |
| `robot/urdf_loader.py` | URDF → RobotModel |
| `robot/mesh_loader.py` | trimesh 网格加载 → ConvexHullShape |

## 规模

- **866 个非慢速测试**（+112 slow），全部通过
- physics/ ~16,000 行，rendering/ ~960 行，总计 ~44,000 行
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
| 3 — 渲染 (RenderScene 抽象 + matplotlib) | 🔄 开始 |
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
