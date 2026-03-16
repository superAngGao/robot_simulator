# Robot Simulator — 文件清单

> Last updated: 2026-03-16

---

## `physics/` — 物理内核

### `spatial.py`
空间代数基础（Featherstone 6D 向量体系）。

| 类 / 函数 | 说明 |
|-----------|------|
| `SpatialTransform(R, r)` | 刚体坐标变换；`apply_velocity`, `apply_force`, `inverse`, `@` 复合；工厂：`identity`, `from_rotation`, `from_translation`, `from_rpy` |
| `SpatialInertia(mass, inertia, com)` | 6×6 空间惯量；工厂：`from_box`, `from_cylinder`, `point_mass`；`matrix()` 返回矩阵；支持 `+` 合并 |
| `rot_x/y/z(angle)` | 3×3 基本旋转矩阵 |
| `skew(v)` | 3×3 反对称矩阵 |
| `quat_to_rot(q)` / `rot_to_quat(R)` | 四元数 ↔ 旋转矩阵 |
| `spatial_cross_velocity(v)` | 速度叉积算子（6×6） |
| `spatial_cross_force(v)` | 力叉积算子（6×6，对偶） |
| `gravity_spatial(g)` | 空间重力向量 |

---

### `joint.py`
关节模型，每种关节提供运动子空间 S、变换 X_J(q)、偏置加速度。

| 类 | DOF | 说明 |
|----|-----|------|
| `RevoluteJoint(name, axis, q_min, q_max, k_limit, b_limit)` | 1 | 转动关节（X/Y/Z 轴）；含关节限位，`compute_limit_torque(q, qdot)` 返回 penalty 恢复力矩 |
| `PrismaticJoint(name, axis)` | 1 | 移动关节 |
| `FixedJoint(name)` | 0 | 刚性连接（用于 foot body） |
| `FreeJoint(name)` | 6 | 浮动基（用于躯干）；`integrate_q()` 四元数积分 |

---

### `robot_tree.py`
运动学树，核心调度层。

| 类 / 方法 | 说明 |
|-----------|------|
| `Body` (dataclass) | 单个刚体节点：`name, index, joint, inertia, X_tree, parent, children, q_idx, v_idx` |
| `RobotTree(gravity)` | 树结构容器 |
| `add_body(body)` | 添加 body，返回 index |
| `finalize()` | 锁定树，分配 q/v 切片索引 |
| `forward_kinematics(q)` | → 每个 body 的世界位姿列表 |
| `rnea(q, qdot, qddot)` | 逆动力学：给定加速度 → 所需力矩 |
| `aba(q, qdot, tau, ext_forces)` | **前向动力学**（O(n) Featherstone ABA）：给定力矩 → 加速度 |
| `joint_limit_torques(q, qdot)` | 全树关节限位恢复力矩，叠加到 tau |
| `body_by_name(name)` | 按名查找 body |
| `default_state()` | 返回零位 (q, qdot) |

---

### `contact.py`
地面接触模型（penalty spring-damper）。

| 类 / 方法 | 说明 |
|-----------|------|
| `ContactParams` (dataclass) | `k_normal, b_normal, mu, slip_eps, ground_z` |
| `ContactPoint(body_index, position, name)` | 单个接触点（body 局部坐标）；`world_position`, `world_velocity` |
| `ContactModel(params)` | 多点接触管理器 |
| `add_contact_point(cp)` | 注册接触点 |
| `compute_forces(X_world_list, v_body_list, num_bodies)` | → 每个 body 的空间接触力列表（body 坐标系） |
| `active_contacts(X_world_list)` | → 当前入地的接触点 `[(name, pos)]` |

法向力：`F_n = max(0, k·δ − b·δ̇)`；摩擦：正则化 Coulomb 锥，无零速不连续。

---

### `self_collision.py`
AABB 自碰撞检测（防止腿穿透躯干）。

| 类 / 方法 | 说明 |
|-----------|------|
| `BodyAABB(body_index, half_extents)` | 单个 body 的包围盒描述（body 局部坐标系半尺寸） |
| `AABBSelfCollision(k_contact, b_contact)` | 自碰撞模型 |
| `add_body(babb)` | 注册一个 body 的包围盒 |
| `build_pairs(parent_list)` | 构建碰撞候选对，**自动排除直接父子体对** |
| `compute_forces(X_world_list, v_body_list, num_bodies)` | → penalty 碰撞力列表（body 坐标系） |

OBB → 世界 AABB：`world_half[i] = Σ_j |R[i,j]| · local_half[j]`；碰撞响应沿最小穿透轴（MTV）施加等大反向弹簧力。

---

### `integrator.py`
数值积分器，驱动仿真时间推进。

| 类 / 函数 | 说明 |
|-----------|------|
| `SemiImplicitEuler(dt)` | **推荐主力积分器**；symplectic Euler，离散能量守恒，接触场景稳定；NaN/Inf 检测 |
| `RK4(dt)` | 4 阶 Runge-Kutta，每步 4 次 ABA，精度更高，适合验证 |
| `simulate(tree, q0, qdot0, controller_fn, contact_fn, dt, duration)` | 完整仿真循环 → `(times, qs, qdots)` 轨迹数组 |

两者均正确处理 `FreeJoint` 的四元数积分（调用 `FreeJoint.integrate_q`）。

---

## `rendering/`

### `viewer.py`
matplotlib 3D 可视化器（调试用，非实时）。

| 类 / 方法 | 说明 |
|-----------|------|
| `RobotViewer(tree, floor_size, contact_names)` | 可视化器 |
| `render_pose(q, show, save_path)` | 渲染单帧姿态，可保存图片 |
| `animate(times, qs, interval, show, save_path)` | 回放仿真轨迹，可导出 `.gif` / `.mp4` |

可视化内容：body 节点（蓝色球）、连杆（橙色线）、地面（灰色网格）；`contact_names` 中的 body 高亮红色。

---

## `examples/`

### `simple_quadruped.py`
17 体四足机器人完整 demo，兼作 Phase 1 集成测试。

| 函数 | 说明 |
|------|------|
| `build_quadruped()` | 构建 `(RobotTree, ContactModel, AABBSelfCollision)` 三元组；12 转动关节 + 4 FixedJoint foot body |
| `standing_state(tree)` | 用 FK 精确计算站立高度（足端恰好接地 + 1 mm 间隙） |
| `_compute_body_velocities(tree, q, qdot)` | 前向递推各 body 空间速度（接触阻尼 / 自碰撞阻尼用） |
| `main(save_path)` | 完整仿真：PD 稳态控制 + 关节限位 + 地面接触 + 自碰撞 → matplotlib 动画 |

运行：`python -m robot_simulator.examples.simple_quadruped [--save out.gif]`

---

## 空壳目录（仅 `__init__.py`，功能待实现）

| 目录 | 计划用途 | 对应 Phase |
|------|---------|-----------|
| `robot/` | URDF loader、FK/IK 工具、质量矩阵 | Phase 2 |
| `rl_env/` | Gymnasium 接口、VecEnv 并行环境 | Phase 2 |
| `domain_rand/` | 物理 / 视觉随机化、传感器噪声 | Phase 4 |
| `deploy/` | ONNX/TorchScript 导出、ROS2 接口 | Phase 5 |
| `tests/` | 单元测试（自由落体、单摆、能量守恒） | 待补充 |

---

## 当前整体能力

```
机器人描述（Body + Joint + Inertia）
    → RobotTree.forward_kinematics()   # 各 body 世界位姿
    → RobotTree.aba()                  # O(n) 前向动力学
    → ContactModel.compute_forces()    # 地面接触力
    → AABBSelfCollision.compute_forces() # 自碰撞力
    → RobotTree.joint_limit_torques()  # 关节限位力矩
    → SemiImplicitEuler.step()         # 状态积分
    → RobotViewer.animate()            # 轨迹可视化
```

**当前是功能完整的 CPU 单环境仿真器。** 下一步：GPU 并行（Phase 2）+ RL 接口。
