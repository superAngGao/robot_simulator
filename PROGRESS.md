# Robot Simulator — Progress Tracker

> Last updated: 2026-03-16
> Reference plan: [PLAN.md](./PLAN.md)

---

## Overall Status

| Phase | 状态 | 完成度 |
|-------|------|--------|
| Phase 1 — Basic Physics + Simple Rendering | ✅ 完成（含修复） | 100% |
| Phase 2 — GPU Acceleration + Parallel Envs | ⬜ 未开始 | 0% |
| Phase 3 — High-Fidelity Rendering          | ⬜ 未开始 | 0% |
| Phase 4 — Domain Randomization             | ⬜ 未开始 | 0% |
| Phase 5 — Sim-to-Real Validation           | ⬜ 未开始 | 0% |

---

## Phase 1 — Basic Physics + Simple Rendering ✅

### 核心交付物

| 文件 | 状态 | 说明 |
|------|------|------|
| `physics/spatial.py` | ✅ | 空间代数（6D 向量、Plücker 变换、SpatialInertia） |
| `physics/joint.py` | ✅ | Revolute / Prismatic / Fixed / Free 关节模型；含关节限位 |
| `physics/robot_tree.py` | ✅ | 运动学树 + FK + RNEA + ABA；含 `joint_limit_torques()` |
| `physics/contact.py` | ✅ | Penalty spring-damper 地面接触模型 |
| `physics/integrator.py` | ✅ | 半隐式欧拉 + RK4 + `simulate()` |
| `physics/self_collision.py` | ✅ | AABB 自碰撞检测（**新增**，2026-03-16） |
| `rendering/viewer.py` | ✅ | matplotlib 3D 可视化 + GIF/MP4 导出 |
| `examples/simple_quadruped.py` | ✅ | 四足机器人 drop-test 验证（含完整修复） |

### 实现的算法

- **Featherstone ABA**：O(n) 前向动力学，已验证与解析解吻合（自由落体误差 < 5 mm @ t=1s）
- **RNEA**：逆动力学（关节力矩计算）
- **Penalty contact**：地面 spring-damper + 正则化 Coulomb 摩擦
- **Semi-implicit Euler**：接触丰富场景下的稳定积分器（dt = 2e-4 s）
- **RevoluteJoint 限位**：penalty spring-damper，单向阻尼，参数化（q_min/q_max/k/b）
- **AABB 自碰撞**：OBB→世界AABB 投影 + MTV penalty force，自动排除相邻体对

### Phase 1 期间修复的 Bug（按时间顺序）

| 日期 | Bug | 修复方式 |
|------|-----|---------|
| 2026-03-16 | ABA 重力方向反转 | root 加速度初始化改为 `−a_gravity` |
| 2026-03-16 | 接触点世界位置错误（转置 R） | 改为 `R @ pos + r` |
| 2026-03-16 | 接触力坐标系错误 | 改为 `X.inverse().apply_force()` |
| 2026-03-16 | 接触在首次触地时发散 | 接触点临时移至 calf 原点（zero moment arm） |
| 2026-03-16 | `standing_state` 站立高度错误 | 改用 FK 测量最低足端 z |
| 2026-03-16 | 初始关节角度设置用了 `v_idx` | 改为 `q_idx` |
| 2026-03-16 | **接触点偏离足尖 0.2 m** | 新增独立 foot body + FixedJoint（几何精确） |
| 2026-03-16 | **关节角度无限制** | RevoluteJoint 添加 q_min/q_max + penalty 限位力矩 |
| 2026-03-16 | **腿部可穿透躯干** | 新建 AABB 自碰撞模块 |

### 验证结果

| 测试 | 结果 |
|------|------|
| 自由落体精度（解析 vs 仿真，t=1s） | 误差 < 5 mm ✅ |
| 四足 drop-test：四脚着地时间 | ~0.08 s ✅ |
| 站立稳定性（2 s，10 000 步） | 无发散 ✅ |
| 足端几何精度（foot_z vs calf_z） | 差 0.2000 m ✅ |
| 限位有效性（超限时产生恢复力矩） | 验证通过 ✅ |
| 0.5 s 仿真后 max\|q\| | 0.059 rad（远低于所有限位） ✅ |

### 已知剩余限制（Phase 2 解决）

- `_compute_body_velocities()` 每步重复前向递推，与 ABA 内部冗余，应缓存
- matplotlib 渲染速度慢，无法实时；Phase 3 替换为 Vulkan
- AABB 以 body origin 为中心，CoM 偏心时精度下降
- 无传感器模型（IMU / 力传感器）

---

## Phase 2 — GPU Acceleration + Parallel Environments ⬜

> 目标：将 Phase 1 的 NumPy 后端移植到 NVIDIA Warp（GPU 原生 Python），实现
> 1000+ 并行环境供 RL 训练。

**待完成：**
- [ ] `physics/warp_kernels/` — Warp 版 ABA、FK、contact
- [ ] `rl_env/base_env.py` — Gymnasium 兼容接口
- [ ] `rl_env/vec_env.py` — 并行向量化环境
- [ ] Phase 1 vs Phase 2 性能 benchmark（吞吐量 steps/s）
- [ ] 单环境物理精度与 Phase 1 NumPy 结果对比验证
- [ ] 修复：缓存 ABA 内部 body velocity，供 contact / self-collision 直接复用

---

## Phase 3 — High-Fidelity Rendering + Sensor Simulation ⬜

**待完成：**
- [ ] `rendering/vulkan_renderer/` — Vulkan + ray tracing 渲染器
- [ ] `rendering/camera_sim.py` — 相机模型（噪声、畸变、运动模糊）
- [ ] `rendering/lidar_sim.py` — LiDAR 点云仿真
- [ ] IMU 噪声模型

---

## Phase 4 — Domain Randomization ⬜

**待完成：**
- [ ] `domain_rand/physics_rand.py` — 质量、摩擦、阻尼、关节刚度随机化
- [ ] `domain_rand/visual_rand.py` — 纹理、光照、颜色随机化
- [ ] `domain_rand/noise_models.py` — 传感器噪声模型
- [ ] 结构化随机化课程（curriculum）

---

## Phase 5 — Sim-to-Real Transfer Validation ⬜

**待完成：**
- [ ] `deploy/policy_export.py` — ONNX / TorchScript 导出
- [ ] `deploy/hardware_bridge.py` — ROS2 / 硬件 SDK 接口
- [ ] 真实机器人部署 + sim-to-real gap 测量
- [ ] 系统辨识（system identification）迭代优化仿真参数
