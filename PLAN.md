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

```
robot_simulator/
├── physics/               # GPU-accelerated physics core
│   ├── warp_kernels/      # NVIDIA Warp / CUDA kernels (Phase 2+)
│   ├── spatial.py         # Spatial algebra (6D vectors, Plücker transforms)
│   ├── joint.py           # Joint models (revolute, fixed)
│   ├── robot_tree.py      # Kinematic tree + FK + ABA forward dynamics
│   ├── contact.py         # Spring-damper contact model (penalty method)
│   └── integrator.py      # Semi-implicit Euler / RK4
│
├── robot/
│   ├── urdf_loader.py     # URDF parser
│   ├── kinematics.py      # FK / IK
│   └── dynamics.py        # Mass matrix, Coriolis
│
├── rendering/
│   ├── vulkan_renderer/   # High-fidelity rendering (Phase 3+)
│   ├── viewer.py          # Simple 3D visualization (matplotlib / PyOpenGL)
│   ├── camera_sim.py      # Camera model + noise
│   └── lidar_sim.py       # Point cloud simulation
│
├── domain_rand/           # Sim-to-Real key module
│   ├── physics_rand.py    # Randomize mass / friction / damping
│   ├── visual_rand.py     # Randomize textures / lighting / colors
│   └── noise_models.py    # Sensor noise models
│
├── rl_env/                # Reinforcement learning interface
│   ├── base_env.py        # Gymnasium-compatible interface
│   └── vec_env.py         # Parallel environments
│
├── deploy/                # Real robot deployment
│   ├── policy_export.py   # ONNX / TorchScript export
│   └── hardware_bridge.py # ROS2 / vendor SDK interface
│
├── examples/
│   └── simple_quadruped.py
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
