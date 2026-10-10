Initiative: Q54 / P12.4 / C4c2
Stage: Bounded native evidence and status
Author: Codex
Version: v1
Date: 2026-10-10
Status: implemented
Related Files: terminal-target implementation-note v1 and code-preflight-review v1; warp-exit-fault test-code-preflight-review v1; PROGRESS.md
Owner Summary: Both repaired native worker lanes now complete the small two-sample probe. Formal speed comparison has not started. Reader hardware coverage remains incomplete and the full C4 objective stays active.

## Repaired worker probe

Exact production revision: 0be9876b9778cf4826a62f45c373a38d60ff33f1.
Primary host: tileops-dev. Clean isolated checkout:
`/data2/shared/ang.gao/robot_sim_worker_0be9876`.
Campaign: `/data2/shared/ang.gao/c4-worker-0be9876b97-probe`.
Existing driver SHA256: 89ef276136c74efd830d8c4b476d62893e8d67148317f598d808dd2d93375ca3.
Start 2026-10-10T10:41:22Z; terminal exit at 10:45:49.288763Z.

Both Warp and CUDA asymmetric_synthetic_7x3_v1 primary runs pass. Each
worker returns 0, resolves two completions, reclaims two slots, records the
normal close acknowledgement and leaves its arena CLOSED. Both terminal
records report zero emergency-sync attempts. This closes the observed CUDA
RESOLVE rejection of an equal indexed device value on its original context.
It does not establish all-channel numerical correctness or full workloads.

Recorded owned PIDs 2793997, 2794004, 2794282, 2794284, 2799842 and 2799844
were all absent in fresh OS and GPU checks. The wrapper reports a clean tree.
Raw records, binary requests/replies and scoped closure observation are kept
in the remote campaign and copied to local c4_assets. The archive contains
90 raw files, 13,826,211 bytes, plus its inventory:

- `c4-worker-0be9876b97-probe-evidence-v1.tar.gz`
  SHA256 77e0eef6d18bb6e2b7687b7cb28b36141446ed6b2b4cfea6655a33114e0157e7.
- Inventory SHA256 e1e1893d43a3cebf1e9849b9ace331df74c9eefb7d39230ee648bee4646fdfbd.

The local archive hash matches the remote bytes. Driver elapsed times include
startup, preparation, auditing and shutdown; they are not renderer timings
and provide no Warp/CUDA speed conclusion. Evidence recording is an author
observation, not fresh independent whole-hardware acceptance.

## Reader matrix partial attempt

Exact revision: 22b1db0ee619f05e30f03bd5e339f03e883e97a5.
Campaign: `/data2/shared/ang.gao/c4-reader-22b1db0-native-scope-v1`.
It ended at 2026-10-10T10:37:41Z with 16 passed, 1 failed, 83 deselected;
three planned cases were not executed due to fail-fast. Seventeen live
worker/supervisor identity chains were observed without observer errors.
The 16 passing cases cover existing scope/input/Go2 and bounded reader faults.

The first foreign-stream test failed before reaching its fault marker:
canonical materialization rejected numpy.ma as an unlisted external import.
Its actual request selected historical 183_warp_synthetic_author_v5.json,
while the prior scope requests selected the intended current Warp Go2 v4
profile. The runner assigned profiles on an imported module alias, whereas
pytest collected a different instance of the fault-test module. Missing
fault_entered.json is secondary to this early setup rejection.

Required next action: bind the already pinned current profiles to the actual
collected item.module via a pytest collection hook, preserving existing test
bodies and assertions; rerun all four foreign/unknown-stream cases into a new
directory. No production admission-policy expansion is indicated by this
runner failure. This attempt is not a 20-case pass.

All 17 workers/supervisors and the wrapper/driver/pytest were absent in the
saved round_process_closure.json; no owned GPU process remained. A stale
local SSH transport PID31348, created at local18:15:15 with the exact reader
launcher command, remained after remote completion. Its identity was checked
and only that local transport was stopped. Tool session16145 then terminated
with code1; that transport code does not replace the preserved pytest result.

- `c4-reader-22b1db0-native-scope-v1-evidence-v1.tar.gz`
  SHA256 bf3da25c8688be8d827c37f515951593b8d3354498946df34724bc32ee790171.
- Inventory SHA256 192e2433232d2db45490c29922392f9e69a100552e43a66e55f1fd5d8227227a.

The raw partial attempt is preserved remotely and copied locally with matching
archive hash. No failure artifact or old cache was deleted.

## Additional fault preflight and remaining scope

Fresh independent session 01a12565-e595-7763-b65f-3c04530d6ad0 approves only
exact fc23b4e38d816d5108b61363a5372710db6a2937's Warp exit-not-restored test
for bounded native execution. Its 21 independent CPU cases passed without
skips. Full report is archived verbatim as
`q54-optical-lab-p12-4c4c2-warp-exit-fault__test-code-preflight-review__codex__v1.md`;
source and archived working-file SHA256 both equal
0ef709ac90cbe480f955272fca48e93e2d071795959442fb5e44e2d636720e50.
CLI tool49091 exited0; reviewer4292 and launcher47420 are absent. The real
new fault has not yet run. Production is unchanged from the approved repair.

Still required: corrected four-case rerun, the additional actual Warp exit
fault, whole-reader hardware assessment, final-sample physical and numerical
proof on the complete frozen workloads, bounded/preallocated control work,
remaining worker failure/close coverage, independent whole acceptance and
the remaining C4 stages. Systematic refactoring remains part of the objective;
the known per-event history copying/replay must not be mistaken for a finished
fixed-work control path. No formal performance pass, backend selection,
product activation or later gate is authorized by these results.
