# Fixed artifacts: complete replay and bounded service acceptance fixture

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: implementation note, unfinished integrated candidate
Author: Codex
Version: v4
Date: 2026-10-09
Status: draft
Related Files: fixed-cuda-artifacts design-request/design-review v2;
  fixed-cuda-artifacts implementation-note v3
Owner Summary: Full v3 aggregate replay now covers file/native/adoption joins.
  A fixed installed-manager acceptance script reuses deployment service settings.
  Neither actual services/builds nor GPU loading have run; whole-packet preflight
  remains required. No additional worker supervision mechanism was introduced.

Candidate code is `3912a273b99d78d538ad0508937ce28673df62d3`, from clean
`4b5173c` on `implementation/c4-supervised-waits`. The latter is the evidence-only
record of the prior 275-pass/13-skip cold-source run against `1ff696a`.

## Changes and verification

- LBVH factory preparation previously queried CUDA availability/device count
  before validating a supplied captured extension set. Request shape checks are
  now CPU-only; device queries follow module/session/context resolution. A real
  factory regression rejects an unregistered set before the CUDA-query canary.
- `test_cuda_artifact_replay.py` drives the public `RuntimeCustodyRecord` parser
  from source binding through original guard 3 completion. The fixture uses real
  CPU guard/controller/adoption events and explicitly modeled module/native facts.
  It checks five complete file admissions, six create/exec pairs and exact named
  entry/set adoptions, including the component call table and first-use table.
  Eleven mutations hold the lifecycle fixed; each must fail at its specific
  identity/order check, not merely at a stale journal index. This closes the
  previous direct-binding-method-only CPU coverage gap, not native execution.
- `tests/support/cuda_artifact_service_acceptance.py` contains the six fixed
  installed-manager cases. It reuses production argv, footer validation, normal
  main-exit gate and cgroup closure checks. Only the fixture command and budgets
  change (6s runtime/start, 1s stop). Descendants call setsid; the parent-loss case
  kills only its owned systemd-run waiter. A separately owned sentinel must stay
  alive through every case and is waited at the end. The lost-waiter path observes
  service closure before reading ExecStopPost's raw record, avoiding a partial
  footer read. Evidence is retained in a new private directory. No publication
  occurs in this fixture; publisher integration remains separate.
- Production source policy accepted all 83 prospective LF sources. Catalog v14
  was refreshed for the LBVH change; new SHA256:
  `663b26b8c191f3505ec37c98e5de6742fc36f95347f446c538fac61384661412`.

Local Windows, `C:/Python314/python.exe -B`:

```text
python -B -m pytest -q
  tests/unit/optics/test_cuda_artifact_build_status.py
  tests/unit/optics/test_cuda_artifact_manifest.py
  tests/unit/optics/test_cuda_extension_loading.py
  tests/unit/optics/test_cuda_lbvh_native.py
  tests/unit/optics/test_cuda_artifact_runtime.py
85 passed in 9.87s, exit 0

python -B -m pytest -q tests/unit/optics/test_cuda_artifact_replay.py
12 passed in 4.03s, exit 0

python -B -m pytest -q
  tests/unit/optics/test_native_source_policy.py
  tests/unit/optics/test_native_sources.py
206 passed, 4 skipped in 43.85s, exit 0 (committed 3912a27)
```

The stdlib fixture CLI `python -I -S -B .../cuda_artifact_service_acceptance.py
--help` also exits 0; this executes no case. `git diff --check` passed before
the code commit. The earlier focused selection passed 48 in 3.22s. No independent
review was requested for this still-incomplete packet. All local pytest/CLI/SSH
commands are terminal; no owned test process or service remains running.

## Prepared manager command; not execution approval

After the whole replacement has an exact independent preflight and the reviewed
revision is installed in the isolated target checkout, the manager fixture's
bounded command is:

```sh
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/robot_sim_cuda_compilation_wip/tests/support/cuda_artifact_service_acceptance.py --output /data2/shared/ang.gao/c4-fixed-artifact-manager-v1
```

The output must not exist. No overwrite/retry is implicit. Every case retains its
exact invocation, waiter PID/log, main/child facts, production footer and observed
closure. `results.json` exists only after all six pass; the sentinel termination
is separately recorded. On failure preserve the exact unit and inspect its actual
terminal state; do not declare cleanup from the fixture's intended deadline.
This script does not prove compilation, builder driver inactivity or publication.

Read-only SSH this turn confirms the target isolated tree is still clean at
`92a8ac646ac182f2dac4dc4eef8a3dd0b331f1af`, with user-manager cgroup
`/user.slice/user-1006.slice/user@1006.service`. No target files were changed and
no service/compiler/GPU/fault campaign ran.

Remaining in the one replacement packet: exact installed builder dependency
closure and driver-inactivity evidence plan; executable cold v5 launch selection;
actual factory preparation under its approved parent scope; bounded build/load/
failure/reuse/Go2 commands. Then fresh independent code preflight, real execution,
and independent acceptance. Parent Warp custody, timeline/arena, warmups/readiness,
C4c3/c4, C4d/B3 and whole-C4 refactor remain. C4 is active; product PREPARE,
benchmarks/results and backend selection remain closed.
