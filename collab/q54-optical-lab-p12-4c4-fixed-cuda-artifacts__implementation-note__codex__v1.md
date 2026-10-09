# Fixed artifact replacement: implementation in progress

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: implementation note, unfinished integrated candidate
Author: Codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: fixed-cuda-artifacts design-request/design-review v2;
  fixed-cuda-artifacts design-review v1; scope-convergence proposal v1
Owner Summary: The simpler replacement design is independently approved. Work has
  started on explicit extension bindings and the standard deployment build leaf.
  The deployment publisher, ordinary worker loading, factory integration and old
  machinery removal are still unfinished; this is not an accepted code checkpoint.

## Design authority

Fresh reviewer `/root/fixed_artifacts_design_v2` approved exact design v2 at
`3e31778766d7ea16ebe6b936dd00b9de0a0af5f0`, SHA256
`4f7cca3d448f5fd243f2786571193790dbaf2fc7604e755b86a2328e1c4dcc7d`.
The complete unchanged report is archived in commit `dcfa7a7`, report SHA256
`28315968c2b05479a10b1e08a67f733edcb3bc716e771d5ad8e537acf7b1b5d5`.
It closes v1 R1 (clean-signal false success), R2 (native availability probe) and
R3 (named entry/set adoption joins) at design level only. Original report v1 is
preserved in `0b8f4f5`. The interrupted older pidfd campaign remains stopped.

## Implemented so far, based on dcfa7a7

- Explicit CUDA headers and the existing six wrapped direct-light pybind entries
  now live in the sole source modules. The recipe owns C++/CUDA flag tuples.
  The old load_inline compatibility helper still exists temporarily; no replacement
  or deletion is claimed until the whole new route is connected and verified.
- `tools/optical_cuda_build_recipe.py` contains the fixed isolated Linux build
  leaf using standard setuptools CUDAExtension/BuildExtension, private staging,
  explicit architecture/visibility and no result import or publication. It has
  not been executed. Its launcher, inventory binding and artifact publisher do
  not exist yet; the leaf alone is not a complete deployment command.
- `tools/optical_cuda_build_status.py` records manager termination variables from
  ExecStopPost and provides the shared normal-exit predicate. It does not manage
  processes or claim cgroup closure; the future launcher must observe closure
  after the actual waiter exits. No service or recorder was launched.

## Verification, with actual limits

Local `C:/Python314/python.exe -B -m pytest -q` selected
`tests/unit/optics/test_cuda_artifact_build_status.py` and the original-kernel-byte
and cold-recipe-import tests in `test_cuda_extension_recipes.py`: **30 passed in
0.30s**, terminal exit0. These are CPU contract/source checks only. They cover the
clean-signal counterexample, incomplete/failed/wrong-unit facts, required build
environment and forbidden overrides, and preserve original kernel/declaration
hashes after removing only the migrated headers/binding block. The cold recipe
import still imports neither backend. They do not prove compilation, numerical
behavior, manager cleanup, absence of all native build effects or worker custody.

The first environment-test attempt had8 failures because a Windows Path converted
the fixed Linux prefix into backslashes. The production constant now uses
PurePosixPath; the same independent Linux-string expectations then passed. This
was a local implementation failure, not a target/compiler problem.

No target modification, build, systemd unit, GPU/backend job, process fault matrix,
product preparation or benchmark ran. Inspection commands and pytest completed.
Design reviewer v1's owned SSH81726 was explicitly cancelled and observed exit1;
v2 used no SSH. No reviewer remains running. The local source catalog has not yet
been regenerated for this unfinished candidate and cannot authorize native runs.

## Next integrated work and remaining goal

Finish the one deployment launcher/manifest/publication path, reuse/move the
existing toolchain inventory, implement ordinary guard3 extension admission and
v3 custody joins, pass the exact issued set into all factories and LBVH workspace
sizing, remove runtime JIT/setuptools/compiler branches and obsolete tests, and
refresh the actual source/profile closure. Then freeze the complete implementation
and bounded commands for fresh independent code preflight before any real build,
service or GPU acceptance run. Do not review this build leaf as a substitute for
that integrated packet or resume the old pidfd campaign.

Continue C2e/f, concrete Warp compilation custody, warmup/final graph, C4c3/c4,
C4d/B3, whole-C4 refactor and independent acceptance. C4 remains active and
incomplete; measurement, product activation and backend selection remain closed.
