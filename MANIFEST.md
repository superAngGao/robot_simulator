# Robot Simulator — Project Manifest

> 从零构建的刚体物理仿真器，面向 sim-to-real 腿足机器人 RL 训练。
> Last updated: 2026-03-30 (session 12)

## 一句话

基于 Featherstone 算法的刚体动力学引擎 + 约束求解器 + GPU 并行环境，目标是在自有硬件上完成四足/双足机器人的 sim-to-real 训练。

## 架构

```
Layer 4  VecEnv              GPU 并行 RL 训练 (N=4096, torch.Tensor on CUDA)
Layer 3  Env                 Gymnasium 接口 + ObsManager + 域随机化(stub)
Layer 2  Simulator           多机器人编排: Scene → CollisionPipeline → PhysicsEngine
Layer 1  Physics Core        刚体动力学管线 (StepPipeline → ForceSource → ConstraintSolver → integrate)
Layer 0  Math                空间代数 (6D Plücker, [linear; angular] 约定)
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

| 求解器 | 级别 | 平台 | 定位 | vs MuJoCo |
|--------|------|------|------|-----------|
| **PGSSplitImpulseSolver** | velocity | CPU | RL 训练快速路径 | x=18.5mm z=3.8mm |
| **ADMMQPSolver** | acceleration | CPU | 高精度 MuJoCo 对标 | **x=0.8mm z=0.3mm** |
| **Jacobi-PGS-SI** | velocity | GPU (Warp) | 大规模 RL (N=4096) | ~18mm |
| **GPU ADMM** | velocity | GPU (Warp) | 高精度 GPU 仿真 | **z-L2=0.3µm** |

## GPU 碰撞检测

GpuEngine 使用解析碰撞函数（shape type dispatch），不依赖 GJK/EPA：

| 碰撞对 | GPU 状态 |
|--------|---------|
| Sphere/Capsule/Box/Cylinder vs Ground | ✅ 解析 @wp.func |
| Sphere-Sphere, Sphere-Capsule, Capsule-Capsule, Sphere-Box | ✅ 解析 @wp.func |
| Capsule-Box, Box-Box | ⬜ fallback 球近似 |
| ConvexHull (凸分解) | ⬜ 待几何系统升级 |
| Mesh (BVH) | ⬜ 待 wp.Mesh 集成 |

## GPU 后端

4 个后端实现，通过 `get_backend()` 工厂选择：

| 后端 | N=1000 steps/s | vs NumPy |
|------|----------------|----------|
| NumPy (CPU) | 533 | 1× |
| TileLang | 438,700 | 823× |
| Warp | 750,363 | 1,408× |
| **CUDA C++** | **2,204,524** | **4,136×** |

## 关键文件

| 文件 | 职责 |
|------|------|
| `physics/step_pipeline.py` | 刚体动力学管线（两阶段 + 内联积分） |
| `physics/robot_tree.py` | 运动学树: FK, ABA, CRBA, RNEA, integrate_q |
| `physics/solvers/mujoco_qp.py` | ADMMQPSolver (acceleration-level QP) |
| `physics/solvers/pgs_split_impulse.py` | PGS + split impulse |
| `physics/backends/warp/analytical_collision.py` | GPU 解析碰撞函数 (14 个 @wp.func) |
| `physics/backends/warp/admm_kernels.py` | GPU ADMM 求解器 (Cholesky + 锥投影) |
| `physics/gpu_engine.py` | GPU 物理引擎 (Warp kernel 管线, solver dispatch) |
| `physics/cpu_engine.py` | CPU 物理引擎 (GJK/EPA 地面 + 球近似 body-body) |
| `collision_pipeline.py` | GJK/EPA 统一碰撞检测 (Scene legacy path) |
| `simulator.py` | 多机器人编排 |
| `robot/urdf_loader.py` | URDF → RobotModel |
| `rl_env/vec_env.py` | GPU 并行 VecEnv |

## 规模

- **616 个测试**，全部通过
- physics/ ~25 个模块，~6000 行核心代码
- 支持多机器人场景 + 静态几何 + 碰撞过滤

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
| 3-5 — 渲染 / 域随机化 / sim-to-real | ⬜ |

## PhysicsEngine 统一

CPU/GPU 已统一到 PhysicsEngine 接口：
```
Scene.build_merged() → MergedModel（多 robot 合并为单一多根树）
  → PhysicsEngine.step(q, qdot, tau) → StepOutput
    ├─ CpuEngine: GJK/EPA ground + sphere body-body + PGS-SI/ADMMQPSolver
    └─ GpuEngine: 解析碰撞 kernel + Jacobi-PGS-SI (Warp)
```

## 设计原则

- **physics/ 是未来独立库** — 不依赖上层，可单独发布
- **积分器属于物理子系统** — 不同物理（刚体/柔体/流体）各自积分
- **MuJoCo 命名 + Isaac flat tensor 布局** — qfrc_*, ForceState 可观测
- **每个求解器输出 qacc (nv,)** — 统一接口，不管内部是 acceleration 还是 velocity level
- **ContactBuffer 解耦碰撞与求解** — 碰撞检测和约束求解通过统一 buffer 格式正交组合
