# Robot Simulator — Project Plan

> Created: 2026-03-16
> Goal: Build a custom robot simulator with sim-to-real capability for legged robots.

---

## Background & Motivation

- Target robot type: **legged robots** (quadruped / biped)
- Primary goal: **Sim-to-Real transfer** (train in simulation, deploy on real hardware)
- Hardware: High-end NVIDIA GPU available
- Approach: Build from scratch for full customizability and deep understanding

### Why not NVIDIA Isaac Sim?
Isaac Sim is a powerful industrial tool, but we build our own because:
- Full control over physics model and contact dynamics
- Lightweight and embeddable in products
- Deep understanding of underlying algorithms
- Custom domain randomization strategies
- Research platform, not a production dependency

---

## Architecture Overview

### Layered Architecture (5 layers)

```
┌──────────────────────────────────────────────────────────┐
│  Layer 4: Application                                    │
│  rl_env/vec_env.py  —  parallel VecEnv for RL training   │
├──────────────────────────────────────────────────────────┤
│  Layer 3: Task / Environment                             │
│  rl_env/base_env.py  —  Gymnasium interface              │
│  domain_rand/        —  physics & visual randomization   │
├──────────────────────────────────────────────────────────┤
│  Layer 2: Simulator  (single-env step, auto-manages      │
│  passive forces; wraps Layer 1 + contact + self-collision)│
├──────────────────────────────────────────────────────────┤
│  Layer 1: Physics Core  (backend-agnostic algorithms)    │
│  physics/{joint, robot_tree, contact, self_collision,    │
│           integrator}  —  NumPy now, Warp in Phase 2     │
├──────────────────────────────────────────────────────────┤
│  Layer 0: Math Primitives                                │
│  physics/spatial.py  —  pure spatial algebra, no physics │
└──────────────────────────────────────────────────────────┘

Robot Description (orthogonal config axis):
  URDF / programmatic builder
       → robot/urdf_loader.py → RobotModel
             (bundles RobotTree + ContactModel + AABBSelfCollision)
       → feeds into Layer 2 Simulator
```

### External-Facing APIs (two primary entry points)

```
1. load_urdf("robot.urdf", ...)  →  RobotModel     # bring-your-own robot
2. Env(model, ...)               →  Gymnasium env  # RL training interface
```

Everything below these two interfaces is implementation detail.

### Module Map

```
robot_simulator/
├── physics/               # Layer 0 + Layer 1
│   ├── spatial.py         # Layer 0: 6D spatial algebra (Plücker)
│   ├── joint.py           # Layer 1: joint kinematics + passive torques
│   ├── robot_tree.py      # Layer 1: kinematic tree, FK, RNEA, ABA
│   ├── contact.py         # Layer 1: penalty ground contact
│   ├── self_collision.py  # Layer 1: AABB self-collision
│   ├── integrator.py      # Layer 1: Semi-implicit Euler / RK4
│   └── warp_kernels/      # Layer 1 (Phase 2): GPU backend
│
├── robot/                 # Robot Description axis
│   ├── model.py           # RobotModel dataclass
│   └── urdf_loader.py     # URDF → RobotModel
│
├── rendering/
│   ├── viewer.py          # matplotlib debug viewer (Phase 1)
│   ├── camera_sim.py      # camera noise model (Phase 3)
│   └── lidar_sim.py       # LiDAR point cloud (Phase 3)
│
├── domain_rand/           # Layer 3: physics & visual randomization
│   ├── physics_rand.py
│   ├── visual_rand.py
│   └── noise_models.py
│
├── rl_env/                # Layer 3 + Layer 4
│   ├── base_env.py        # Layer 3: Gymnasium-compatible single env
│   └── vec_env.py         # Layer 4: parallel VecEnv
│
├── deploy/                # Phase 5 (deferred)
│   ├── policy_export.py
│   └── hardware_bridge.py
│
├── examples/
└── tests/
```

---

## Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Phase 1 physics | Python + NumPy | Validate correctness first |
| Phase 2 physics | NVIDIA Warp / CUDA | GPU parallelism for RL training |
| Rendering (early) | matplotlib 3D | Quick visualization |
| Rendering (later) | Vulkan + ray tracing | Sim-to-Real visual fidelity |
| RL training | PyTorch + RL Games / SB3 | |
| Real robot interface | ROS2 / vendor SDK | |

---

## Development Phases

### Phase 1 — Basic Physics + Simple Rendering ✅ DONE
Validate dynamics correctness for a single legged robot.

Key algorithms:
- **Spatial algebra** — 6D force/velocity vectors, Plücker coordinate transforms
- **Featherstone ABA** — Articulated Body Algorithm for forward dynamics  O(n)
- **Penalty method contact** — Spring-damper foot-ground contact
- **Semi-implicit Euler** — Stable integrator for contact-rich simulation
- **matplotlib 3D viewer** — Visualize robot skeleton and motion

Deliverables:
- [x] `physics/spatial.py` — Spatial algebra utilities
- [x] `physics/joint.py` — Revolute, Prismatic, Fixed, Free joint models
- [x] `physics/robot_tree.py` — Kinematic tree, FK, RNEA, ABA
- [x] `physics/contact.py` — Spring-damper contact model
- [x] `physics/integrator.py` — Semi-implicit Euler + RK4 + simulate()
- [x] `rendering/viewer.py` — Simple 3D visualization + animation export
- [x] `examples/simple_quadruped.py` — Quadruped drop-test validation

### Phase 2 — GPU Acceleration + Parallel Environments
- Port physics core to NVIDIA Warp (GPU-native Python)
- Implement parallel VecEnv for RL training (1000+ envs simultaneously)
- Benchmark speedup vs Phase 1 NumPy baseline

### Phase 3 — High-Fidelity Rendering + Sensor Simulation
- Vulkan renderer with ray tracing
- Realistic camera simulation (noise, distortion, motion blur)
- LiDAR point cloud simulation
- IMU noise models

### Phase 4 — Domain Randomization
- Physics parameter randomization (mass, friction, damping, joint stiffness)
- Visual randomization (textures, lighting, object placement)
- Structured randomization schedules (curriculum)

### Phase 5 — Sim-to-Real Transfer Validation
- Deploy trained policy to real hardware
- Measure sim-to-real gap
- Iterative refinement of simulation parameters (system identification)

---

## Key References

- Featherstone, R. — *Rigid Body Dynamics Algorithms* (2008)
- Spatial algebra: http://royfeatherstone.org/spatial/index.html
- Penalty-based contact: Mirtich & Canny (1995)
- Sim-to-Real: OpenAI "Learning Dexterous In-Hand Manipulation" (2019)
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
