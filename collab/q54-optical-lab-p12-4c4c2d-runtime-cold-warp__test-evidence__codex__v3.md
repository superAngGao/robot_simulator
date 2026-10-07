# C2d3 optional-adapter WIP and schema-v2 regression

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: test-evidence
Author: codex
Version: v3
Date: 2026-10-08
Status: draft
Related Files: optional-adapter design-request/design-review v2;
runtime-adoption implementation-note v1; runtime-cold-warp test-evidence v2
Owner Summary: The bounded denial implementation has CPU fixture coverage and
existing real Warp/C0/Go2-CPU regressions pass. This is author WIP evidence, not
whole C2d3 code approval or completion of C4. All later gates stay closed.

Candidate: b5f7737fb3788b15d02bad420533ac1fe94db1f9, based on8128e79.
Primary183 alias tileops-dev, hd03-gpu2-0031, H200, robot_sim Python3.12.13.
Repository: /data2/shared/ang.gao/robot_sim_c2d3_binding_wip.
All commands below use /home/ang.gao/miniforge3/envs/robot_sim/bin/python.
The isolated tree was clean before/after tests. No101 use, dependency install,
original-worktree edits, GPU reset, push or unrelated process termination.

## Changed contract and narrow CPU proof

Runtime environment v2 contains the required disabled_optional_modules table.
Only the two design-approved present adapters may be denied before execution.
The existing production finder retains the resolution/error graph and consumes
an identity-registered token; requester completion verifies the exact fallback.
Cached sys.modules entries are separately checked because None can bypass the
finder. Tests exercise schema collisions, wrong requester/phase, file changes,
prerequisite presence, token copying/reuse, fallback and exception identity drift.

Cold CPU fixture tests use the actual captured source session, guardian and
distinct guarded segments. Initial configuration permits unrequested rules;
later fixture imports confirm both. Adoption/evidence-record failures retain
the admission graph and quarantine. These are not actual trimesh/Go2 admission
or GPU fault proof. The full same-session Go2 materialization obligation remains.

Active synthetic fixture: tests/data/runtime_environments/183_warp_synthetic_author_v2.json;
canonical SHA-256 f42485bdf9580d56b86080aad18d2d3e5a8832c23bd0868419852b2da4b734af.
It has no disabled adapters and does not admit trimesh/Torch. Historical v1 is
unchanged. Catalogue v5 raw SHA-256:
d60c614af983399f85e7e0689414f2168c7cb3141194546c30ff62bf2092e206.
AST policy is v6; C0 policy remains v1. Pins use exact Git LF release bytes.

## Remote unit/cold-process regression

```text
timeout --signal=TERM --kill-after=5s 180s PY -m pytest \
 tests/unit/optics/test_runtime_environment.py \
 tests/unit/optics/test_runtime_import_admission.py \
 tests/unit/optics/test_runtime_import_cold.py \
 tests/unit/optics/test_native_source_policy.py \
 tests/unit/optics/test_native_sources.py \
 tests/unit/optics/test_native_runtime_driver_process.py -q
```

332 passed,2 skipped in110.42s; actual wait/SSH exit0. Start00:22:05+08;
wrapper3406729, Python3406731, parent3406727. The two unconfigured Go2 tests
were subsequently run with the real asset path below, not counted as passes here.

## Actual cold Warp regression

```text
ROBOT_SIM_TEST_C2D3_COLD_WARP=1 \
 timeout --signal=TERM --kill-after=5s 240s PY -m pytest \
 tests/gpu/test_native_runtime_cold_warp.py -q -s \
 --basetemp=/data2/shared/ang.gao/c2d3-b5f7737-cold-warp-proof-v1
```

The named artifact directory was checked absent before pytest. Result:6 passed
in160.90s, no skips, actual wait/SSH exit0; outer bound did not fire. Start
00:24:27+08; wrapper3410468, Python3410470, parent3410466. Continued observing
the same tool session57040 until exit; no restart on an observation yield.

| Case | Fault | Supervisor/worker | Exit | Artifact suffix |
| --- | --- | --- | --- | --- |
|asymmetric|success|3410539/3410541|0|war0|
|dynamic|success|3410884/3410889|0|war1|
|asymmetric|terminal sync raises|3411204/3411205|74|war2|
|dynamic|terminal sync raises|3411586/3411587|74|war3|
|asymmetric|terminal sync hangs|3411894/3411895|-9|war4|
|dynamic|terminal sync hangs|3412727/3412730|-9|war5|

Full case IDs are asymmetric_synthetic_7x3_v1 and
synthetic_dynamic_rigid_960x640_v1. Sole canonical builders independently
provide parent expectations for42/27 arrays respectively; actual cold upload,
readback, layout/content/scene/fixture/topology checks run, including replay.
Input validation does not issue final optical readiness or benchmark results.

Each hang is deliberately injected at terminal sync after actual GPU setup;
the real30s supervisor deadline expires with the parent alive, kills and reaps
the worker. A killed worker's prefix is not upgraded to release/completion.
These tests do not reproduce or resolve the previously reported hardware hang.

Each suffix above maps to test_actual_cold_canonical_SUFFIX/custody_artifact.json
under the artifact root. SHA-256 values:

```text
war0 61cf62881d3ab5e8a23399480e78cde3bce7895a57114d26b2d30fad31410709
war1 b779d02143c7f16341ab8aef8c1b43ae1341e0f8a838582898a94e14a0e13089
war2 bd26c7993551eee3f03a5a813bf3e3906245e14270c374bcba5bba01d8d049d9
war3 1c2143c93301e0b7b9f057d4d33f0542af18cb97c82261715515b114a7fda232
war4 539158b7f0a3babd57add7f46c3254485af1ef3c0c0298748f072469234077e3
war5 c33b662067ba635b072d33df16de288365b696a9ab20a6a2410404b3ddad6933
```

## Existing C0 and Go2 CPU paths

```text
timeout --signal=TERM --kill-after=5s 180s PY -m pytest \
 tests/unit/benchmarks/test_optical_closed_worker.py \
 tests/unit/benchmarks/test_closed_worker_protocol.py \
 tests/gpu/test_optical_closed_worker_setup.py -q
```

30 passed in13.52s, no skips, actual wait/SSH exit0. Start00:27:34+08;
wrapper3413954, Python3413956, parent3413952. This includes the existing real
both-lane setup/partial-failure/foreign-device/warmup tests, not the unfinished
C2d3 cold admission fault matrix.

```text
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2 \
 timeout --signal=TERM --kill-after=5s 150s PY -m pytest \
 tests/unit/optics/test_native_sources.py -k real_go2 -q -s
```

2 passed,39 deselected in45.38s, actual wait/SSH exit0. Start00:28:45+08;
wrapper3415228, Python3415230, parent3415226. Both lane labels each materialize
both real Go2 resolutions through the existing CPU import path. This is not
new disabled-optional admission, GPU upload, render or shipping proof.

## Hygiene and remaining work

At00:27:33 all14 sampled Warp wrapper/Python/supervisor/worker PIDs were absent.
At00:29:59 the ang.gao process list contained no Python/test/worker/supervisor,
only user-session/current SSH readers. GPU listing had only existing MPS2238410,
60MiB, untouched. Local Win32_Process query after Go2 wait found no Python/SSH
residual. Short-lived CPU child PIDs were not all sampled, so this is not a
complete per-child process ledger. Tests use bounded waits and observed exits.

Original remote robot_simulator remains clean at
f771e612beaeea126e26926738097512cd2d809c; isolated candidate clean b5f7737.
Ruff and diff checks pass. The author's local earlier full group had one stale
policy-version assertion (fixed and specifically rerun); remote expanded run
above is on the corrected revision, not reuse of that earlier output.

Still pending: actual frozen Go2/CUDA external/native closure; both lanes and
both Go2 cold admission/upload/readback; versioned runtime-custody evidence and
replay; remaining real device-scope/fault matrix; whole independent code review.
No code approval is claimed for this WIP subset. C4 remains incomplete; C2e/f,
C4c3/c4, C4d/B3, product GPU preparation, host/shipping, a3b and result/baseline
remain closed. Shared collab/README and review principles were not modified.
