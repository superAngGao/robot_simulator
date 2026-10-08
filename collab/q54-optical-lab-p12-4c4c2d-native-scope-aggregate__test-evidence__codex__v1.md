# Actual v2 native-scope execution and repaired foundation regression

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-08
Status: implemented (evidence only; whole-stage acceptance outstanding)
Related Files: native-scope-aggregate design-request/design-review v2;
native-scope-cache design-request/design-review v1;
native-scope-aggregate implementation-note/code-preflight-review v3;
runtime-custody implementation-plan v3; runtime-faults test-evidence v2
Owner Summary: The exact approved preflight candidate passes the 16 required
actual native observations and the 637-case affected CPU regression. Raw
records, external live identities, failure history and scoped residual checks
are preserved. The complete 85-case fault rerun and independent whole-C2d3
review remain outstanding; no later gate opens.

## Revision, authorization and environment

Executed code is clean detached e585956eb29e3a33483d8cb0c7444f893b390ac4 in
/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip on tileops-dev
(183.166.183.33:40031). Original binding worktree remains separate and unchanged.
The local implementation/c4-supervised-waits branch at fb2297c differs only by
the unchanged archived independent preflight report. The reviewer session was
/root/native_scope_code_preflight_v3, report SHA256
b874bcd8fd9b313d115fa3d4f2fc378f944732d3a15fc74656b8bdd394f6a5bc.
Its Approve permits these 16 actual cases and closes the exact C0 compatibility
repair; it is not approval of the full C2d3 implementation or native outcomes.

Interpreter: /home/ang.gao/miniforge3/envs/robot_sim/bin/python (CPython3.12.13),
NumPy2.5.1, Warp1.12, Torch2.9.1+cu128. The fixed tests verify the pinned runtime
profiles, sources, GPU mapping and driver595.71.05. Target GPU1 is
GPU-3a3112ff-837c-3b87-38d1-cb203d656589; foreign GPU0 is
GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274. No device remapping, dependency change,
reset, arbitrary deadline change or fallback host was used.

## Actual native cases

External runner run_c2d3_native_scope_v1.py, SHA256
242df56b990f466d2c8a197d3a62b6d60c01c4848293d78ea722942e45af455c,
asserts the clean exact revision and runs the existing fixed suite:

```text
ROBOT_SIM_TEST_NATIVE_SCOPE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -B -m pytest tests/gpu/test_native_scope_runtime.py -x -q -s
  -p no:cacheprovider --basetemp /data2/shared/ang.gao/c2d3-e585956-native-scope-v1
16 passed in 1271.47s (0:21:11)
```

Raw stdout/stderr and author_run_summary.json confirm returncode0, no outer
timeout, 16 live external identity chains and no observer errors. Runner1392319
and pytest1392324 are distinct from each actual supervisor/worker pair.

Eight normal cases cover each lane's asymmetric_synthetic_7x3_v1,
synthetic_dynamic_rigid_960x640_v1, go2_static_960x640_v1 and
go2_static_1920x1080_v1. Real Go2 assets are the frozen unitree_go2 model under
/data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie. Each normal case has
actual input validation, current durable selection, v2 native_scope rows joined
to the actual helper occurrences, full terminal evidence, one supervised
terminal sync/release and worker exit0/reap. This is input preparation and
readback validation, not optical rendering or final execution readiness.

Eight negatives cover both lanes' getter error, failure after actual returned
stream retention, failure after allocation snapshot append, omitted Warp
readback scope, and CUDA stream-only C1 copy. Each reaches its intended marker
and diagnostic, exits74 with UNKNOWN, zero emergency-sync/terminal-ARM attempts,
no ordinary release, no successful input-validation report and current
selection. A safe failure is counted only for its specified negative contract.

The same-run external observer reads procfs while each worker and supervisor is
live and records PID, parent, start ticks, executable and command. The tests
join these to their actual supervisor/pytest identities. All16 chains exist;
no historical ancestry is synthesized from old identity files.

Final residual_check.json checks all34 unique owned PID references, including
runner and pytest: no /proc entry and no owned PID in the successful nvidia-smi
compute query. Only these owned identities were checked; no process was killed
by the author. After remote completion and absence were independently verified,
the stale local SSH session55791 was closed with Ctrl-C. That transport cleanup
did not terminate a test or alter its result.

Archive, retained remotely under /data2/shared/ang.gao and in local sibling
c4_assets with matching downloaded hash:

```text
c2d3-e585956-native-scope-v1.tar.gz
bytes: 7821994
SHA256: c84af059c9e32a8048b1b6d9bb5efa7af09e62a8a7e38d54fa3b5d0b444dc230
```

It contains raw requests, trusted launch inputs, runtime records/sidecars,
artifacts, reports/diagnostics, identity files, stdout/stderr, run summary and
the residual result. No files were rewritten into successful observations.

## CPU replay of these actual v2 observations

The external stdlib-only replay_actual_native_scope_v1.py parses all8 normal
records with their actual environment and independently captured launch inputs,
requires byte-exact roundtrip, validates current selection against each full
terminal trace, attaches its actual artifact and checks native success against
the same-run external identity. There are1988 native rows: Warp1002/CUDA986.
Warp counts per case are273/183/273/273; CUDA269/179/269/269. Warp additionally
has paired upload/readback scope observations; helper counts come from the
actual allocations/copies, not an invented uniform count.

All56 in-memory negative mutations reject: future operation leaf, wrong thread,
wrong module, crossed pair, wrong owner, claiming native rows in v1, and an
ambient contradiction. The last remains representable as a descriptive record
but fails live-success acceptance. Original records remain unchanged. NumPy,
Torch and Warp imports are absent. This replay does not create new native facts.
Log replay_actual_native_scope_v1.log SHA256
a1b1a28fe93106fbe9bbc955da7e2af8e44de0ef4a91c6e40ad7d82bb8e01153,
retained on both hosts with its external script.

## Exact-revision foundation regression and preserved failures

run_c2d3_foundation_cpu_v2.py executes 14 existing suites under the clean exact
e585956 revision: C0 protocol/actual child processes, guarded wait core/process/
lifecycle, native/CUDA arena contracts, cold custody failure doubles, observer,
custody/readback/native codec and source/source-policy tests. The environment
uses real Go2 CPU assets and the retained historical archive where the existing
codec suite requires it. Result **637 passed in125.94s**, no skips, empty stderr,
returncode0. Runner1375393/pytest1375398 and all65 procfs-observed live owned
identities were absent afterward from OS/GPU records. Polling can miss very
short-lived descendants; this is not complete attestation of every spawn.

```text
c2d3-e585956-foundation-cpu-v2.tar.gz
bytes: 16327350
SHA256: 7396ebb1ae39d692a3d58648ce4dae29085fe4b50dac23f23d47073121d7ba9d
```

Both-host raw archives and local hashes are preserved. Independent preflightv3
inspected the actual revised C0 manifests, logs, failure journals and source
hashes; its complete report states which supplied executions it did not rerun.

This follows the genuine4da4ed2 foundation failure:303passed/5failed46.38s.
All five C0 failures named the extracted lane_contract owner missing from the
old exact allowlist. The approved repair adds exactly lane_contract and
native_manifest to the two matching literals, preserving all enforcement.
The original archive remains c2d3-4da4ed2-foundation-cpu-v1.tar.gz,
SHA256 b87d22bb0caf9fd1eefe8172c079645af3bf7e38cb538bb93490938ec8d78ae5.
Earlier4da4ed2 focused110passes and b0f22fb501passes/one platform skip remain
separate historical CPU evidence; they are not added to the new637 count.

## Remaining acceptance work

The full85 existing fault cases are now running on the same frozen e585956,
without subset filtering, through run_c2d3_runtime_fault_matrix_v4.py SHA256
9a84e404e2def345c44dbeaadf92b42706b15e3021bdc0ff5a76dba9899d15cc.
Output directory: /data2/shared/ang.gao/c2d3-e585956-runtime-fault-matrix-v4.
Runner1598724 and pytest1598730 were verified live; no result is claimed here.
The prior85 passes at older test revisions remain historical, not proof of this
candidate's full fault lifecycle. Existing actual-record CPU mutation tests
are also being applied to all8 new positives; their result will be recorded
separately. No implementation change is being made under either execution.

Fresh independent whole-C2d3 review still follows the complete required matrix
and evidence audit. C2d3/C2d/C4 acceptance is not issued. C2e/f, C4c3/c4,
C4d/B3, product GPU preparation, a3b, result/baseline, host,R2 and shipping stay
closed. These evidence files do not authorize rendering, performance claims,
backend selection or losing-lane deletion. The continuous objective remains C4.
