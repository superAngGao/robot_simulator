# Shared toolchain scanner: first real filesystem inventory

Initiative: Q54 / P12.4 / C4c2 / remaining parent C2d
Stage: CPU/source-only evidence, not compiler or native execution approval
Author: codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: compilation-admission design-request/design-review v3; implementation-note v4; toolchain-layout test-evidence v1
Owner Summary: The shared fixed-profile scanner completed against the existing primary-host deployment, and its CPU counterexamples passed on Linux. This verifies scanner execution and produces an exact inventory for the future review packet. It does not establish actual compilation, runtime library selection, native ABI/process behavior or full dependency coverage.

## Exact candidate and transfer

Code/test candidate52fb36a8960596ddb2ac3decd8d129188e4b1eb4 was committed with a
clean local tree. Bundle `cuda-toolchain-inventory-wip-v1.bundle` requires remote
predecessorb42475e976c568b6a56a10a016a96e78bdf5311c and has SHA256
5ff2c6f3f5f3e8aa8503cc4cfcc20e325a7e07e05d410a02a77aea624e3fe293.
The primary-host isolated tree
`/data2/shared/ang.gao/robot_sim_cuda_compilation_wip` was verified clean at that
predecessor, then received the hash-checked/bundle-verified candidate. It remained
clean at52fb36a after both executions. No original runtime/other worktree changed.

## Actual scanner execution

Interpreter `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`, flags-u -I -S -B.
A stdlib runpy wrapper printed its PID, set the sole CLI arguments and timed
`tools/optical_cuda_toolchain_inventory.py --output
/data2/shared/ang.gao/cuda-toolchain-52fb36a-inventory-v1.json`.

Observed PID4048193, exit0, 15.100253608077765seconds. The scanner read existing
files only: 17023 canonical file rows,60 root descriptors,1169 symlink rows,
12502301277 summed file bytes. Sixteen root descriptors record absence, including
Conda lib64 and the listed hardware-capability/fallback roots. Fourteen tool rows
resolve to the actual Conda CUDA12/Ninja and fixed system GCC/binutils/shell paths.
Duplicate aliases are hashed as one canonical file per scan. This is a filesystem
inventory duration, not compile or optical performance evidence.

Output size3420577bytes, SHA256
ec4b2cfc004ee3e387c1c6bce1251e718825bf9b96889221b8f2beba8e74e65d.
The original remote JSON is retained at the path above. Its local copy
`C:/Users/AG/新建文件夹/c4_assets/cuda-toolchain-52fb36a-inventory-v1.json`
has the identical digest. The CLI writes with exclusive creation, never replaces
an old inventory, and asserts that neither Torch nor Warp was imported. No tool
binary was executed, no library was loaded by the scanner, and no PCH was changed.
Python's ordinary interpreter/stdlib execution is not claimed to be native-free;
no new backend/observer/compiler/process-policy execution occurred.

## Linux CPU counterexamples and cleanup

In the same exact clean tree, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1,
python-u -B ran pytest-q -p no:cacheprovider on:

- tests/unit/optics/test_cuda_toolchain_inventory.py
- tests/unit/optics/test_cuda_toolchain_environment.py
- tests/unit/optics/test_cuda_compilation_unwind.py

Observed PID4048188, **34 passed in0.20s**, exit0. The wrapper asserted no Torch/Warp
imports and only the main Python thread remained. The scanner and tests ran as
separate CPU processes; environment/fixture monkeypatches did not affect scanning.
Subsequent read-only `/proc` checks found both recorded PIDs absent. SSH sessions
completed and the remote tree stayed clean. No compiler/GPU worker or dedicated
native supervisor was started.

## What remains unproven

The profile is now concrete and executable, not independently approved. Its
coverage of every actual compiler/header/linker/native dependency still requires
the exact source/manifest/runner/ABI/observer preflight and then real build/process
evidence. A source-only scanner cannot prove what a future compiler will load or
that the build fits the unchanged setup deadline. Source policy/catalog updates,
the external build-helper closure, complete live collector/source/parser proof and
fixed failure matrix also remain outstanding. No compile or readiness gate opens.
Continue the full compilation checkpoint and remaining C4 work, including the
owner-required systematic refactor, rather than treating this inventory as final.
