# C2d3 first actual captured Warp input path (author evidence)

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-07
Status: draft
Related Files: runtime-adoption design v2; implementation-note v1;
tests/gpu/test_native_runtime_cold_warp.py;
tests/data/runtime_environments/README.md
Owner Summary: Real cold Warp canonical upload/readback now completes, and its
terminal sync failure/hang paths have real supervisor/reap evidence. This exposed
and repaired a canonical-array operation-ID bug. Whole C2d3/C4 is NOT approved;
the remaining actual runtime, multi-device, fault and evidence gates remain open.

Candidate:022b2ee (whole C2d3 WIP, not independently code-reviewed).
Production fix:8f1c92f; additional C0 GPU regression:1e28b00.
Primary183, hd03-gpu2-0031,8H200, driver595.71.05, robot_sim Python3.12.13.
Isolated checkout:/data2/shared/ang.gao/robot_sim_c2d3_binding_wip.
Original robot_simulator/f771e61 remained clean and untouched. No101 fallback.

## Frozen author fixture and actual dependencies

The tracked host-specific author fixture has canonical contract SHA-256
ae95257eb95b4441e5ae786a5ae65f3e7a86459e30aa455abfc27ecae0ed2299.
It explicitly names86 NumPy materialization modules and25 Warp runtime modules,
not all32 inventoried wheels. It includes the one absent distributor-local probe,
file hashes/METADATA, Warp/NumPy native libraries and explicit resolved OS/driver
files. No namespace/creator exception or Torch/Go2 permission was added.
Observed111 completed file modules plus the absent probe match this Python table.

Read-only installed AST inspection supplied eager dependency edges. Native
multiarray's dependency names and the numpy.typing body supplied the remaining
explicit entries. Initial NumPy-only candidates failed closed, first on the
unlisted numpy.typing dependency and then at the deliberately unlisted Warp
entry. No runtime failure automatically changed a permission table. Candidate
files were separately frozen before each fresh invocation. The raw wheel
inventory remains facts, not permission. Author scripts/artifacts are retained
under /data2/shared/ang.gao/c2d3-* and local c4_assets.

Before the Warp probe, installed Runtime/Device code and native cuda_init were
inspected: quiet=True avoids the greeting's per-device memory query; device
registration uses metadata and defers primary-context acquisition; the explicit
get_stream(cuda:1) obtains the target context. This is source-audit support, NOT
the mandatory full native-call/multi-device observation/failure matrix. Full
native-loading closure and target1/ambient0 proofs remain incomplete.

## Failures found and corrected

At e7a07fc the actual canonical upload submitted9 H2D copies before failing:
`guardian resource role/operation is invalid`. Sorted array10 is
instance_X_body_geometry_R. C0's inherited per-array operation IDs concatenated
raw array names, while lifecycle identifiers are lowercase/bounded. Both the
capital X and R are legal array-name characters but illegal operation-ID text.
Readback had the same construction pattern. Warm in-process C0 probes originally
used only lowercase names, so they missed this integration bug.

8f1c92f uses one helper: fixed operation prefix plus SHA-256 of the exact UTF-8
array name. It does NOT lowercase/relabel arrays, merge R/r, change storage
authority or relax the lifecycle grammar. Both upload cores and validation
readback share it. Core AST comparison normalizes ONLY the five old diagnostic
ID expressions per lane; all allocation/copy/event mechanics remain compared
against the approved pre-extraction code. Catalog v4 was repinned to the two
changed source files; no bootstrap exception/activation policy was expanded.

Unit tests cover each operation prefix, distinct case-sensitive canonical names,
long names and invalid input/prefixes against actual guardian/descriptor checks.
1e28b00 adds x_world_R/x_world_r to both actual C0 GPU cores and compares returned
values with host originals. No benchmark-only allocation/copy implementation.

Separately, 10e7b26's diagnostic-prefix addition lost final worker evidence on
partial importer construction: events had not been initialized yet. Its4-test
run returned2 failed,2 passed. e7a07fc checks the actual partial object's fields
before emitting optional observations; the same4 tests then passed in12.57s.
Worker hard exit/quarantine was preserved, but the failed diagnostic run is not
counted as successful evidence. V0 import observations remain debug facts, not
the required versioned runtime-custody record.

## Repro and observed real-device results

Command from the isolated checkout:

```text
ROBOT_SIM_TEST_C2D3_COLD_WARP=1 timeout --signal=TERM --kill-after=5s 210s \
 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -m pytest \
 tests/gpu/test_native_runtime_cold_warp.py -q -s
```

At022b2ee:3 passed in82.56s, no skips. Wrapper3331540, Python3331542,
parent3331499, start23:10:14+08; actual wait/SSH exit0. Outer210s bound did not
fire. Parent stayed alive during terminal hang; the actual operation watchdog
expired and killed/reaped that worker, rather than a pytest observation timeout.

| Case | Supervisor/worker PID | Reaped exit | Actual disposition |
| --- | --- | --- | --- |
| normal |3331582/3331584|0|one target sync, guardian RELEASED, release completed|
| sync raises |3331901/3331902|74|one attempt, full quarantine, release not authorized|
| sync hangs |3332290/3332291|-9|original terminal deadline, kill requested, actual reap|

Every case first performed42 real allocations,42 H2D copies,42 validation D2H
copies,2 event creates/records/waits and canonical host comparison. No measured
sample, renderer, final readiness or result was created. The hang has only the
last pre-terminal worker prefix; tests do not invent terminal cleanup/release
from that partial record. Supervisor evidence supplies the actual terminal
deadline/kill/reap. All cases verify no ordinary atexit cleanup.

Earlier author probes retained original-failure and repair artifacts (not extra
pytest passes), each in /data2/shared/ang.gao/:

| Directory | Worker | supervisor-output.json SHA-256 |
| --- | --- | --- |
|c2d3-warp-author-v2-run1|3318097/74|5bb8a0d0c97790fc8d314340f23eca91a718e665d96e23735b42bfaa9c768a70|
|c2d3-warp-author-v2-run2|3322014/0|233912e5c74b3f595bbe06a0c656faeb5eac01b38cdec83db06e918800f0d9e8|
|c2d3-warp-author-v2-sync-raise|3326752/74|61cc219344bda549818ce2e0d825023113b37e4419184700a8c90071db116945|
|c2d3-warp-author-v2-sync-hang|3327434/-9|443133628951687555ce47c19f9bad8431be9a4b612cbd5d1d41cb2cbf26eee7|

## Regression and environment distinction

- At8f1c92f, local source-policy/upload-core/readback:222 passed in4.37s.
- At8f1c92f, original C0 GPU file:10 passed in10.63s (Python3323708,
  wrapper3323706,parent3323704,start23:02:32+08,wait0).
- At1e28b00, enhanced C0 GPU file plus upload-core/readback:65 passed in11.68s
  (Python3325949,wrapper3325947,parent3325916,start23:04:35+08,wait0).
- At022b2ee, local expanded5-file group:25 failed,270 passed in20.22s. All
  failures entered benchmark runtime-hook resolution without installed Torch;
  local importlib.find_spec confirms torch=None and warp=None. No packages were
  installed and no missing dependency was mocked to claim a pass.
- The same5 files in robot_sim on183:295 passed in11.58s, no skips. Files:
  test_native_source_policy, test_native_upload_core, test_guarded_native_readback,
  test_guarded_runtime_binding, test_optical_benchmark_contracts. Python3333432,
  wrapper3333430,parent3333428,start23:11:58+08,wait/SSH0;120s bound not fired.
- Local additional cleanup/operation-ID selection:28 passed,40 deselected
  in0.71s. Ruff checks and git show/diff checks passed. No all-suite pass claimed.

At23:13:39+08 OS inventory had no ang.gao Python/test/worker/supervisor/timeout
residual; only session/current SSH readers remained. GPU inventory had the
pre-existing MPS2238410,60MiB. Local Python/SSH residual query was empty after
the local runner exited. No unrelated kill, install, GPU reset or original
worktree mutation; no claim to repair the historical101 hang.

## Remaining authorization boundary

This is author evidence, not independent code approval of C2d3 or a substitute
sub-checkpoint. Still required: CUDA dependency/publication/absent tables and
cold path; dynamic and both real Go2 workloads; native custody evidence schema;
setup partial failure/hang/parent loss and real dual-device matrices; native
closure verification; fresh whole-candidate independent review and fixes.
C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, final readiness,
result/baseline, host payload and shipping remain closed. Goal C4 remains active.
