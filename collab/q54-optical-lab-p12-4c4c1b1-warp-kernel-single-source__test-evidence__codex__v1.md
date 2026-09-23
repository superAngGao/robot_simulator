# C4c1b1 Warp kernel single-source extraction — test evidence

Author: Codex
Status: review evidence
Implementation commit: `00f1790`

## Target

- Host: `183.166.183.33` / `hd03-gpu2-0031`
- Repository: `/data2/shared/ang.gao/robot_simulator`
- Base revision: `2b41068`
- GPU: NVIDIA H200; tests exposed devices 0 and 1
- Python: 3.12.13
- PyTorch: 2.9.1+cu128; runtime CUDA 12.8
- Warp: 1.12.0
- NVCC: 12.8.93

The environment was created from the tracked Conda explicit manifest. All 62
tracked wheel SHA-256 entries verified before offline installation, and the
resulting `pip freeze --exclude-editable` exactly matched
`deploy/robot_sim_cuda128/python-packages.txt`.

## Results

- Ruff check: passed.
- Ruff format check: passed.
- Source-ownership unit test: `2 passed in 4.37s`.
- Warp executor/runtime/C3 command GPU regressions: `60 passed in 212.08s`.
- Closed-worker clean-tree regression: `8 passed in 4.55s`.
- Full non-slow suite from a clean `00f1790` worktree:
  `1826 passed, 39 skipped, 117 deselected, 1 warning in 330.48s`.
- `git show --check 00f1790`: passed.

The one warning is the existing pytest 10 class-scoped instance-fixture
deprecation in `tests/unit/solvers/test_pgs_split_impulse.py`.

## Clean-tree observation

An initial full-suite run before the implementation commit produced seven
expected C4c0 closed-worker rejections because that protocol intentionally
requires a clean Git worktree. After committing the reviewed implementation,
the closed-worker file and the entire non-slow suite passed without weakening
that gate. No test bypass or manifest relaxation was introduced.

This evidence does not expand scope beyond C4c1b1. Caller-owned buffers, fixed
shade ABIs, recorded render bundles, C4c1c execution, product GPU preparation,
a3b, results, baselines, host payloads, and shipping paths remain closed.
