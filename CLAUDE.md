# Robot Simulator — Claude Project Context

> Phase 1 complete. Next: Phase 2 — GPU acceleration with NVIDIA Warp.

## Quick Reference

| Task | Command |
|------|---------|
| Install (editable) | `pip install -e .` |
| Run example | `MPLBACKEND=Agg python -m robot_simulator.examples.simple_quadruped [--save out.gif]` |
| Run tests | `python -m pytest tests/ -v` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Pre-commit (manual) | `pre-commit run --files <file1> <file2>` |

## Project Status

- **Phase 1** ✅ CPU physics core (ABA, penalty contact, joint limits, AABB self-collision)
- **Phase 2** ⬜ GPU acceleration — NVIDIA Warp + parallel VecEnv for RL training
- **Phase 3–5** ⬜ See PLAN.md

## Architecture

```
physics/         # Core physics (do not modify ABA without unit test)
  spatial.py     # Spatial algebra (6D vectors, Plücker transforms)
  joint.py       # Joint models; RevoluteJoint has penalty joint limits
  robot_tree.py  # Kinematic tree + FK + RNEA + ABA
  contact.py     # Penalty spring-damper contact
  self_collision.py  # AABB self-collision
  integrator.py  # Semi-implicit Euler (recommended) + RK4

rendering/viewer.py        # matplotlib 3D (debug only, not real-time)
examples/simple_quadruped.py  # Integration test / demo
tests/           # Unit tests (expand in Phase 2)
```

## Dependency Direction — HARD RULE

`physics/` is the future independent library. It must never import from any
other layer of this repo. The allowed dependency graph is strictly:

```
rl_env/  →  simulator.py  →  robot/  →  physics/  (no reverse edges)
```

**Violation = blocking review comment.** If you find yourself adding an import
from `physics/` to `simulator.py`, `robot/`, or `rl_env/`, the design is wrong —
restructure so the dependency flows downward only.

## Key Algorithms & Invariants

- **ABA (Featherstone)**: root body initialized with `a_p = -gravity` (not +gravity).
- **Contact forces**: applied via `X.inverse().apply_force()` (body frame), NOT `X.apply_force()`.
- **Contact point** position: foot body origin = true foot tip (NOT calf origin).
- **Semi-implicit Euler**: `dt = 2e-4 s` with `k_normal = 3000`. Larger dt diverges.
- **FreeJoint q layout**: `[qx, qy, qz, qw, px, py, pz]` (quaternion first).

## Code Conventions

- **Physics variables**: single-letter names are acceptable (`I`, `R`, `q`, `v`, `f`).
- **Spatial vectors**: `[linear; angular]` ordering (Pinocchio / Isaac Lab convention).
- **New physics modules**: must include a docstring citing the reference equation/section.
- **No type annotations required**, but add them to new public APIs.

## At the Start of Every Session

Read **OPEN_QUESTIONS.md**. It lists all unresolved design questions and
deferred work. Check whether the current task touches any of them, and
update the file if a question gets resolved or a new one comes up.

## Before Making Any Design Decision

Read **REFERENCES.md** first. For every non-trivial design choice (API shape,
abstraction boundary, algorithm, data structure), check the quick-reference
matrix and find at least one established project that handles the same problem.
Record the finding in REFLECTIONS.md under the decision.

Concretely: if you are about to design a joint model, a collision interface, an
env reset API, or a geometry representation — open REFERENCES.md, find the
relevant row, read the project detail, then proceed.

## After Every Change

1. Run `python -m pytest tests/ -v` — all tests must pass.
2. If behaviour changed: update **PROGRESS.md** and **REFLECTIONS.md**.
3. Use `/commit` skill for git commits.
4. Use `/review` skill to check test coverage before committing new modules.

## Reference Files

- **PLAN.md** — full roadmap and architecture
- **PROGRESS.md** — current status per phase
- **REFLECTIONS.md** — decisions and lessons learned
- **repo_list.md** — API reference for all modules
