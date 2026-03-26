# Robot Simulator — Project Manifest

> 从零构建的刚体物理仿真器，面向 sim-to-real 腿足机器人 RL 训练。
> Last updated: 2026-03-26 (session 10)

## 一句话

基于 Featherstone 算法的刚体动力学引擎 + 约束求解器 + GPU 并行环境，目标是在自有硬件上完成四足/双足机器人的 sim-to-real 训练。

## 架构

```
Layer 4  VecEnv              GPU 并行 RL 训练 (N=4096, torch.Tensor on CUDA)
Layer 3  Env                 Gymnasium 接口 + ObsManager + 域随机化(stub)
Layer 2  Simulator           多机器人编排: Scene → CollisionPipeline → per-robot StepPipeline
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

| 求解器 | 级别 | 平台 | 定位 |
|--------|------|------|------|
| **PGSSplitImpulseSolver** | velocity | CPU | RL 训练快速路径 |
| **ADMMQPSolver** | acceleration | CPU | 高精度，与 MuJoCo 匹配 |
| Jacobi-PGS-SI (待做) | velocity | GPU | 大规模 RL |
| ADMM-TC (待做) | acceleration | GPU | tensor core 高精度 |

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
| `collision_pipeline.py` | GJK/EPA 统一碰撞检测 |
| `simulator.py` | 多机器人编排 |
| `robot/urdf_loader.py` | URDF → RobotModel |
| `rl_env/vec_env.py` | GPU 并行 VecEnv |

## 规模

- **542 个测试**，全部通过
- physics/ ~20 个模块，~5000 行核心代码
- 支持多机器人场景 + 静态几何 + 碰撞过滤

## 进度

| Phase | 状态 |
|-------|------|
| 1 — CPU 物理核心 | ✅ |
| 2a-2e — 重构 + URDF + Simulator + RL env + GPU 后端 | ✅ |
| 2f — 高精度接触 (GJK/EPA + 5 求解器) | ✅ |
| 2g — CRBA + tensor core | ✅ |
| 2h — 力系统重构 (StepPipeline) | ✅ |
| 2i — GPU 求解器 (Jacobi-PGS-SI, ADMM-TC) | ⬜ 下一步 |
| 3-5 — 渲染 / 域随机化 / sim-to-real | ⬜ |

## 设计原则

- **physics/ 是未来独立库** — 不依赖上层，可单独发布
- **积分器属于物理子系统** — 不同物理（刚体/柔体/流体）各自积分
- **MuJoCo 命名 + Isaac flat tensor 布局** — qfrc_*, ForceState 可观测
- **每个求解器输出 qacc (nv,)** — 统一接口，不管内部是 acceleration 还是 velocity level
