# Actual new-schema runtime custody and input validation

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: test-evidence and continued WIP validation
Author: Codex (implementation session; not independent review)
Version: v1
Date: 2026-10-08
Status: implemented
Related Files: runtime-custody implementation-plan v3; implementation-note v1;
tests/unit/optics/test_runtime_custody_actual_records.py;
tests/gpu/test_native_runtime_fault_matrix.py
Owner Summary: Both lanes now execute all four canonical inputs with actual v1
runtime-custody records. The existing six Warp normal/terminal tests also pass.
This is bounded execution evidence, not whole C2d3 or C4 approval. The newly
added fixed fault selectors are still awaiting their real executions.

## Exact execution candidate and environment

All executions below use clean detached
`9b5702b4016824ea7fc5b5e7c108c5adf02a1bd1` in the new owned worktree
`/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip` on
`tileops-dev` / `hd03-gpu2-0031`. The original binding worktree remains clean at
`a3d750bbaf79d760a6727e56146d277e1cb3b770`; it was not reset or overwritten.
The new checkout came from a verified Git bundle and `git worktree add --detach`.
Python is `/home/ang.gao/miniforge3/envs/robot_sim/bin/python` (CPython3.12.13),
NumPy2.5.1, Warp1.12.0, Torch2.9.1+cu128. No dependency changes.

Primary SSH briefly timed out before the GPU launch. Failed upload/transport
attempts did not execute tests. Primary recovered and exact revision/clean tree
and uploaded driver digest were checked before launch. A read-only fallback
SSH attempt was stopped locally; no host switch or write on101 occurred.

GPU target remains ordinal1, UUID
`GPU-3a3112ff-837c-3b87-38d1-cb203d656589`; ordinal0 is
`GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274`. No visibility remapping, broad kill,
GPU reset, runtime package change or modification of other users' work.

## Executed checks and raw locations

The remote prefix for the paths below is `/data2/shared/ang.gao/`.

* CPU/POSIX: `c2d3-9b5702b-cpu-posix-v1`, log with the same name plus `.log`:
  **57 passed in23.50s**, comprising the actual POSIX runtime driver negative
  cases, runtime-custody parser/persistence and manifest tests. Four owned
  workers/supervisors were checked absent, recorded in
  `c2d3-9b5702b-cpu-posix-v1-residual.json`. These negative workers did not enter
  a dependency native body or GPU operation.
* `run_c2d3_runtime_custody_warp_matrix_v1.py`, SHA256
  `1b3e84b254ab8015c92eb3a85c7db9f7a1ffaed371efefb1b230f0c63b18583f`:
  **6 passed in176.32s** at
  `c2d3-9b5702b-runtime-custody-warp-matrix-v1`.
  Asymmetric and replay each cover normal disposition, terminal sync exception
  and terminal sync hang. ACK1000ms and operation30000ms are unchanged.
* `run_c2d3_runtime_custody_normals_v1.py` executes six pinned original-driver
  invokers sequentially. `runtime_custody_normal_invokers_v1.json` carries their
  exact SHA256 values. Successful directories are
  `c2d3-9b5702b-runtime-custody-cuda-upload-author-v1` through `v4`
  (asymmetric, replay, Go2 960x640, Go2 1920x1080), and
  `c2d3-9b5702b-runtime-custody-warp-go2-author-v1` and `v2`.
  These preserve ACK1000ms / operation120000ms from the prior fixed invokers.
  Each script's `.py.log` and `.py.summary.json` preserve its output, digest,
  exact PIDs, observer result and residual check. Each exited0 with one actual
  worker/supervisor/parent OS identity chain and no observer error.

The scripts add independent cold trusted-launch capture, new record/sidecar
parsing and attachment checks. They call the same actual supervisor, captured
production imports, canonical builders, sole uploader and readback validator.
The CPU observer reads only procfs. It does not enter a GPU API or renew guards.
The twelve inherited native environment settings are asserted absent before
launch. Existing frozen environment profiles and Go2 asset directory are used.

All twelve runs produced v1 records selected as `current` and passed existing
custody validation followed by the new exact lifecycle-prefix attachment.
All report `input_validation_observed=true` and no final execution readiness.
Normal runs have worker0, one terminal synchronization, RELEASED guardian and
supervisor reap. Sync exceptions have worker74/full quarantine; hangs have
worker-9 under the original supervisor deadline. Their last worker/aggregate
prefix still has CONSTRUCTING guardian and grants no release: the supervisor,
not an invented worker terminal snapshot, proves the kill/reap.

| Input | Arrays | Active bytes | Zero-size arrays |
| --- | ---: | ---: | ---: |
| asymmetric | 42 | 5,376 | 9 |
| replay | 27 | 5,936 | 7 |
| Go2 960x640 | 42 | 70,214,100 | 0 |
| Go2 1920x1080 | 42 | 70,214,100 | 0 |

These are the actual observed registered-storage descriptors and shared CPU
readback results, not renderer timing or readiness. Both Go2 cases retain their
distinct manifest identity despite identical arrays/content/topology.

`c2d3-9b5702b-runtime-custody-observed-summary-v1.json` records each run's input,
record digest, lifecycle summaries, source manifest digest, process IDs and the
post-run GPU mapping/process query. All twelve owned worker/supervisor chains
were absent by PID/start-time comparison; no owned worker PID remained in the
GPU process listing. Unrelated GPU processes were only observed.

Raw run files, external identities, fixed invokers and their logs/summaries are
archived in `c2d3-9b5702b-runtime-custody-evidence-v1.tar.gz` (5,506,315bytes),
SHA256 `eb08d4ed62f4271596a7b8a1e982f6cd7c2b7f3ab8ec7488319c6bb9aeab5510`.
The archive and observed summary were copied to local sibling `c4_assets` and
the archive hash was independently checked after transfer. Historical v0 and
older executions remain unchanged; none were relabelled as v1 executions.

## Adversarial replay and next matrix

The new opt-in CPU replay suite consumes retained actual directories through
`ROBOT_SIM_RUNTIME_CUSTODY_EVIDENCE` (JSON list), holds their independently
captured launch inputs/environment fixed, and mutates copies only. It does not
produce execution evidence or another uploader. Remote scratch copies are
`test_runtime_custody_actual_records.py`, `_v2.py` and `_v3.py`.

Initial v1 run failed a test precondition because Python compares1 equal toTrue:
1failed/10passed/2skipped. The type-distinction assertion was corrected; production
already rejects bool integers. v2 passed78/skipped10 in37.16s on actual Warp/CUDA
asymmetric records. v3 passed170/skipped13 in197.62s on those two records plus
actual CUDA Go2 960x640. Logs are retained locally under sibling `c4_assets` as
`c2d3-9b5702b-actual-record-mutations-v{1,2,3}.log`.

Coverage includes session/bootstrap/controller substituted for canonical source,
runtime owner swaps, actual guardian adoption swaps, storage signatures with a
recomputed storage digest, empty-array invalid scalar types, cross-guard moves,
omitted observations, file spec/loader/origin/call changes, absence-requester
changes, alias descriptor and component-token substitutions. The Go2 test checks
the actual alternate retained canonical manifest, including equal arrays and
content/topology, then rejects the fully consistent alternate case and updated
downstream manifest hash against the fixed trusted launch case. Skips denote
component/Go2/zero-size occurrences absent from that specific actual input;
they are not passes or generic coverage claims.

The separately executed four-Go2 case/component subset passed20/skipped4,
220deselected in19.40s (`c2d3-9b5702b-go2-record-mutations-v1.log`). It covers
each real Go2 resolution on each lane against its own trusted inputs and the
alternate full manifest. The four skips are component occurrences absent from
the corresponding Warp executions; they do not suppress case checks.

The accompanying test-only driver change adds24 fixed fault selectors,48 lane
cases. Wrappers interrupt original NumPy/backend loader calls, initialization,
returned stream adoption, original allocation/copy/event/wait helpers, owner
registration, existing final storage audit, D2H copy, cleanup and missing exit.
Returned native values are held in the already-owned source-session graph
before injected exceptions. No production closure, catalog or policy is changed.
Default normal/previous terminal selectors do not install injection wrappers.
At this evidence checkpoint these new cases are collected, not GPU-validated.
Local parser/manifest regression after the driver edit:53passed in1.28s.

Full ACK/EOF, dual-device/ambient negative, remaining required failure windows,
fresh candidate regression and independent whole-C2d3 code review remain open.
C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host, R2
and shipping remain closed. C4 is still the active continuous objective.
