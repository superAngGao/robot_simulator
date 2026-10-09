# Fixed CUDA artifacts: whole replacement and execution preflight

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: whole replacement code preflight, before real execution
Author: Codex
Version: v1
Date: 2026-10-09
Status: in_review
Related Files: fixed-cuda-artifacts design-request/design-review v2;
  implementation-note v3/v4; runtime-integration design-request/review v3;
  remaining-setup gap-scan v1; systematic-refactor implementation-note v1
Owner Summary: Review the integrated replacement once, then execute its fixed
  manager/build/Go2 commands. Worker JIT and its compiler supervision are removed.
  This requests execution preflight, not acceptance without real GPU evidence.

## Candidate and authority

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
Branch: `implementation/c4-supervised-waits`.
Whole implementation base: `dcfa7a7120690a4c79bfbc5baa36a116c641ab28`.
Code/execution HEAD: `a43f0c1e292e586b2740bfb2142de142e5284639`.
The review packet is a documentation-only descendant of that code. The reviewer
handoff records its exact HEAD; execute the code revision above, so requests,
source selection and the clean-tree assertion agree. Candidate stays frozen.

Read AGENTS.md, collab/README.md and REVIEW_PRINCIPLES.md. The accepted replacement
is `q54-optical-lab-p12-4c4-fixed-cuda-artifacts__design-request__codex__v2.md`,
design commit `3e31778766d7ea16ebe6b936dd00b9de0a0af5f0`, SHA256
`4f7cca3d448f5fd243f2786571193790dbaf2fc7604e755b86a2328e1c4dcc7d`.
Its unchanged independent review is design-review v2, archived at the base above.
Also inspect the accepted parent runtime-integration v3 and remaining-setup
gap-scan v1 for the bounded parent factory preparation after input validation.

This scope includes standard deployment builds, cold selection, ordinary guarded
native loading, session-issued extension identities, actual factory preparation,
v3 replay, removal of superseded worker machinery and the commands below.
Do not limit review to a helper or unit-test subset. Check maintainability and
unnecessary duplication as well as correctness; the owner explicitly objects to
continued infrastructure growth. No new threat model or hostile-Python sandbox
is proposed. Review independently; author summaries and test counts are claims.

## Implementation since note v4

`0a26ad03d2527476f1751b47a512311f96f670d6` connects actual CUDA factories after
the six input phases. `SourceBootstrapSession.prepare_cuda_factories()` uses
the original guarded controller, exact uploaded/prepared storage, runtime binding
and issued extensions. `native_setup._prepare_cuda_factory_owners` constructs
the existing context, scene, LBVH and renderer factories within explicit target/
stream contexts, retaining each return before later fallible work. It does not
allocate a final output slot, launch rendering, warm up or declare readiness.
Factory sources are admitted only in the CUDA parent preparation phases.

`tools/optical_cuda_environment.py` verifies the selected base digest, running
interpreter ABI, expected platform and deployment, and writes the existing v5
environment. `183_platform_v1.json` is an independent selected expectation;
in particular CXX11 ABI=true is not a claimed native observation. The real builder
must agree; a mismatch is a failed selection, not permission to rewrite this file.

The builder records its own driver-library mappings, GPU device FDs and Torch
initialization state before dependencies, after dependencies and after build.
These are OS checkpoints, not a syscall history or compiler-child observation.
They supplement the installed-source/environment argument below. The reviewer
must assess whether that combined proof meets the accepted no-GPU-init build
requirement; do not silently promote checkpoint absence into complete history.

`tests/support/cuda_artifact_runtime_acceptance.py` uses the existing supervisor
and actual Go2 source for ten cases: fresh CUDA, reused deployment in a fresh
worker, CUDA full resolution, Warp at both resolutions, create-return failure,
exec-return failure, entry-adoption failure, factory-return failure and factory
hang. Target1/ambient0 observations and actual factory/module/workspace bindings
are asserted by `native_runtime_process.py`. The new code revision adds missing,
changed and partial artifact checks on private copies before any worker launch.
The selected real deployment is reverified and never altered by these tests.

Check the whole diff, particularly tools/optical_cuda_*.py,
optics/cuda_extension_*.py and three factories/source modules,
optical_worker/native_bootstrap.py, native_setup.py, runtime_custody*.py,
native_scope*.py, historical/, source catalog/policy, and changed support/tests.
Live cuda_compilation/setuptools/compiler-supervision machinery is deleted;
historical readers are outside the captured worker source admission.

## Observed verification and its limits

Windows CPU interpreter: `C:/Python314/python.exe -B`. Prior evidence is in notes
v3/v4. Additional completed commands (pytest -q, paths under tests/unit/optics/):

- test_cuda_artifact_build_status.py, test_cuda_extension_loading.py,
  test_guarded_runtime_binding.py, test_cuda_artifact_manifest.py,
  test_cuda_artifact_runtime.py, test_cuda_artifact_replay.py,
  test_native_scope_observer.py: 162 passed, 9 skipped, 18.42s.
- test_cuda_artifact_build_status.py, test_cuda_artifact_manifest.py,
  test_cuda_extension_recipes.py: 59 passed, 4.98s after builder checks.
- test_cuda_artifact_manifest.py: 21 passed, 5.57s with private-copy negatives.
- test_native_source_policy.py, test_native_sources.py, test_native_bootstrap.py,
  test_runtime_import_cold.py: 295 passed, 13 skipped, 348.98s, exit0. This ran at
  committed 0a26ad0; a43f0c1 changes only the two acceptance/test files. Session
  51816 was polled to terminal exit0; no owned pytest is still running.

Initial runtime-binding tests failed because the old mock fixture lacked the now
mandatory artifact inventory/guard3. Its unit isolation now explicitly stubs the
artifact boundary while checking runtime binding/armed guard; production rules
were not weakened. It is not proof of native loading. Full v3 CPU replay models
native facts and exercises the real lifecycle/parser. None of these CPU results
establish actual compilation, GPU loading, numerical rendering or performance.
The source catalog has 83 files, policyv15/catalogv14, SHA256
`3a21fcc175518b0e8bdbdfe3e484278f4d97056c03d28ae91e7525fbdc028b79`.

Read-only SSH on this turn confirms the isolated target remains clean, detached
at `92a8ac646ac182f2dac4dc4eef8a3dd0b331f1af`, user-manager cgroup
`/user.slice/user-1006.slice/user@1006.service`. No service/build/GPU/fault run
has occurred for this candidate. Use only this isolated checkout:
`/data2/shared/ang.gao/robot_sim_cuda_compilation_wip` on `tileops-dev`.

Installed source inspection from the preceding implementation turn:
Torch cpp_extension.py SHA256
`52ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6`;
torch/cuda/__init__.py SHA256
`9ba786aee9b5aac710f49c3beade1722610a0d6d8ec406fad9a9793df1316e99`.
Under the frozen build environment, _find_cuda_home calls is_available via the
NVML-based branch; empty visible devices yields an empty parsed list and zero
without NVML discovery or fallback _cuda_getDeviceCount. Explicit arch9.0 bypasses
architecture auto-detection. PCH create/remove calls are in load_inline, which the
standard build path does not call. Inspect the real installed closure yourself;
these source observations are not a previously executed build or dynamic proof.

## Exact proposed execution sequence

Only after fresh independent code preflight approves this scope: transfer the
reviewed commits with Git into the isolated target, verify it is clean and at the
code revision above, preserving existing history. Do not reset, alter the original
repository, install dependencies or change GPU visibility for runtime workers.
All outputs below must be absent; existing output is a stop, never an implicit
overwrite/retry. Execute sequentially, inspect each terminal result before the
next command. The shell variables here are literal paths/identities, not arbitrary
commands. No command below has been executed.

```sh
set -eu
repo=/data2/shared/ang.gao/robot_sim_cuda_compilation_wip
py=/home/ang.gao/miniforge3/envs/robot_sim/bin/python
revision=a43f0c1e292e586b2740bfb2142de142e5284639
test "$(git -C "$repo" rev-parse HEAD)" = "$revision"
test -z "$(git -C "$repo" status --porcelain)"
```

For every Python invocation below use this exact environment prefix, followed by
`"$py" -I -S -B ...`. This admits an empty build LD_LIBRARY_PATH and prevents
inherited native configuration in the acceptance parent. The build service uses
its own fixed env -i selection (including empty GPU visibility); workers retain
normal visibility and the existing explicit target selection.

```sh
/usr/bin/env -i HOME=/home/ang.gao USER=ang.gao LOGNAME=ang.gao PATH=/usr/bin:/bin LC_ALL=C XDG_RUNTIME_DIR=/run/user/1006 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1006/bus
```

1. Manager fixture (6s runtime/start, 1s stop, six cases; isolated sentinel):

```sh
"$py" -I -S -B "$repo/tests/support/cuda_artifact_service_acceptance.py" --output /data2/shared/ang.gao/c4-fixed-artifact-manager-v1
```

Require exit0, all six results and actual closure/sentinel evidence. The fixture
does not publish artifacts. Do not infer closure from intended stop commands.

2. Three standard real builds, each 120s runtime/start and 10s stop:

```sh
"$py" -I -S -B "$repo/tools/optical_cuda_deploy.py" --output /data2/shared/ang.gao/c4-fixed-cuda-artifacts-v1 --dependency-profile "$repo/tests/data/runtime_environments/183_cuda_compilation_author_v1.json" --profile-sha256 a9f42cc833088f308545749e4f239c246f787d9ec39096825815ae9557bb4829
```

Require exit0 and the complete published set. Inspect raw job stdout/stderr,
manager footers/cgroup observations and all builder checkpoints. The deployment
prints progress as well as the final digest: never treat all stdout as a hash.
After successful publication only, derive the actual selected manifest digest:

```sh
manifest_sha=$(sha256sum /data2/shared/ang.gao/c4-fixed-cuda-artifacts-v1/cuda-artifacts.json | cut -d ' ' -f 1)
```

3. Cold environment composition (no worker/native imports):

```sh
"$py" -I -S -B "$repo/tools/optical_cuda_environment.py" --base-profile "$repo/tests/data/runtime_environments/183_cuda_go2_author_v1.json" --base-sha256 79c3580baff1ef3a019c97a07177558808918b7e133caa430239e25b82853919 --deployment /data2/shared/ang.gao/c4-fixed-cuda-artifacts-v1 --manifest-sha256 "$manifest_sha" --expected-platform "$repo/tests/data/cuda_artifacts/183_platform_v1.json" --output /data2/shared/ang.gao/c4-fixed-cuda-environment-v1.json
```

Require exit0. This file is canonical, so its byte SHA is its environment digest:

```sh
environment_sha=$(sha256sum /data2/shared/ang.gao/c4-fixed-cuda-environment-v1.json | cut -d ' ' -f 1)
```

4. Native acceptance, using the existing worker supervisor:

```sh
"$py" -I -S -B "$repo/tests/support/cuda_artifact_runtime_acceptance.py" --revision "$revision" --cuda-environment /data2/shared/ang.gao/c4-fixed-cuda-environment-v1.json --cuda-environment-sha256 "$environment_sha" --output /data2/shared/ang.gao/c4-fixed-artifact-runtime-v1
```

Each setup operation retains the existing 120000ms deadline/1000ms ACK; the outer
fixture wait is 1235s with parent-liveness cleanup, not a new worker budget. Cases
run sequentially and stop at first unexpected result. The Warp base profile's
canonical digest is `88d295b90bc4047b2f2cd4c3d0e48c33ead1a1050919b47d526b4576159146e8`.
Require real Go2 both-lane validation/readback, exact source/module/factory binding,
target1/ambient0 restoration, fault reached at the intended original guard, actual
terminal worker/supervisor facts and public custody replay. Preserve every failed
run; do not relabel an unentered fault or partial run as passing. Inspect all owned
PIDs/jobs to terminal state without signaling unrelated processes.

## Requested review and remaining gates

Use a fresh independent session, no implementation history. CPU tests and read-only
SSH/source inspection are allowed now; the reviewer must not launch real services,
builds, GPU imports/getters or faults before preflight. Write the complete report
outside the candidate, with exact base/HEAD/status, findings/repros, commands and
limits, and return its location for verbatim archival. An approval here authorizes
only the reviewed bounded execution; real evidence then needs independent review.

Fixed-artifact acceptance is not C2d3/C4 acceptance. Warp NVRTC/cache/ModuleExec
custody, parent timeline/arena, one-slot graph, ten warmups, final readiness/graph/
seal, C4c3/c4, C4d/B3 and the owner-required whole-C4 structural refactor/review
remain. Product PREPARE, benchmarks/results/baselines and backend selection remain
closed. Do not add another helper-level review campaign or verification framework
unless a concrete accepted requirement cannot be met with the existing machinery.
