# Robot Simulator

**Robot Simulator** *(working title)* is a physics simulation platform for embodied intelligence research, built on four principles:

- **Multi-physics from the ground up** — rigid bodies, deformables, and fluids under a unified coupling architecture, designed to scale across physics domains
- **GPU-native** — algorithms and data structures designed for massively parallel execution, targeting large-scale simulation for reinforcement learning and beyond
- **Rendering and synthetic data** — a rendering pipeline that produces high-quality visual output with full physical annotation, not just debug visualization
- **First-principles API design** — every interface boundary and material parameter traces back to a physical concept; engineering convenience is never a reason to obscure the underlying physics

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | CPU physics core — ABA, penalty contact, joint limits, AABB self-collision | ✅ Complete |
| 2 | GPU acceleration — NVIDIA Warp, parallel VecEnv | 🔄 In progress |
| 3–5 | Rendering pipeline, deformable/fluid subsystems, synthetic data generation | ⬜ Planned |

## Optical rendering preview

The optical pipeline can render RGB, metric depth, and numeric instance segmentation from the same scene query. The preview below uses the Unitree Go2 visual mesh from Google DeepMind's MuJoCo Menagerie, imported into the in-repo optical registry and rendered with the CPU BVH/direct-light reference executor.

![Unitree Go2 optical preview, front view](docs/assets/optical/menagerie_go2_front/panel.png)

![Unitree Go2 optical preview, side view](docs/assets/optical/menagerie_go2_side/panel.png)

The Go2 assets are not vendored in this repository. They were loaded from a local checkout of `google-deepmind/mujoco_menagerie`; consult the model directory's BSD-3-Clause `LICENSE` before redistributing those assets.

## Running tests

| Scope | Command |
|-------|---------|
| Commit gate (~2 min) | `python -m pytest tests/ -m "not (slow or gpu)"` |
| Including GPU | `python -m pytest tests/ -m "not slow"` |
| Full suite (~21 min) | `python -m pytest tests/ -v` |

## Extras

| Extra | Installs | When needed |
|-------|----------|-------------|
| `dev` | pytest, scipy, pillow, trimesh, gymnasium, torch | development and testing |
| `mesh` | trimesh | mesh loading (runtime) |
| `rl` | gymnasium, torch | RL environment (runtime) |
| `rerun` | rerun-sdk>=0.16 | Rerun visualisation backend |

```bash
pip install -e ".[dev,rerun]"
```

## GPU environment

Warp (NVIDIA) must be installed manually — the correct version depends on your local CUDA driver:

```bash
pip install warp-lang  # match to your CUDA version
python -m pytest tests/ -m "not slow" -v
```

See the [Warp repository](https://github.com/NVIDIA/warp) for the CUDA compatibility matrix.
