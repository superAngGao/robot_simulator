Initiative: Q54 / P12.4 / C4c2 terminal target binding
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: runtime-integration design/review v3; native-worker-source-audit
code-preflight-review v1; guarded_lifecycle.py; test_integrated_worker_lifecycle.py
Owner Summary: The duplicate source check is repaired, and actual CUDA now
submits its first sample. Resolution exposes an indexed-device value versus
retained-target identity mismatch. A bounded repair is frozen for fresh review;
neither two-lane completion nor final numerical/performance acceptance is claimed.

## Actual native run at the approved predecessor

Exact clean server revision22b1db0ee619f05e30f03bd5e339f03e883e97a5, worktree
`/data2/shared/ang.gao/robot_sim_worker_22b1db0`. The unchanged worker matrix
driver SHA89ef276136c74efd830d8c4b476d62893e8d67148317f598d808dd2d93375ca3
ran `probe`: asymmetric full-shadow R0, primary mode, two samples per lane.
The complete source-audit preflight report was received and archived first.

Campaign `/data2/shared/ang.gao/c4-worker-22b1db0ee6-probe` ran from
2026-10-10T10:08:51Z to10:13:11Z. Wrapper2670393, driver2670408.

- Warp worker2670775/supervisor2670774 passed both samples, resolve/release,
  normal close and exit0. Its driver validates the actual arena/guard/worker
  path and production input-scope coverage.
- CUDA worker2678363/supervisor2678361 passed HELLO, OPEN, PREPARE and first
  EXECUTE. RESOLVE response0004 reports WorkerSetupProtocolError:
  `terminal wait is not the worker's exact pending sample`. Worker exit64;
  first sample COMPLETION_RECORDED, second PREPARED; historical submission
  remains latched. No normal resolve or close is claimed.
- CUDA terminal evidence records one successful emergency target sync under
  sequence22, release completed and guardian RELEASED, disposition
  aborted_synced_exit. It did not claim zero submitted work. Normal completed
  guards22; target-only known scope. Error response contains terminal digest
  2ae8a75f63f1f051fe7ab32389c66e0a311592e2ade9e786ae282ab3fa5da2eb.
- All six listed owned PIDs were absent at the scoped OS check. Driver was
  waited/reaped, exit1; checkout remained clean. GPU after-inventory has no
  listed owned PID; unrelated GPU processes remain untouched.

Driver elapsed83.81s/170.88s include initialization, checks and cleanup. They
are not native latency measurements and cannot rank the backends.

Raw80files18,971,513bytes preserved unchanged. Archive
`../c4_assets/c4-worker-22b1db0ee6-probe-evidence-v1.tar.gz`,900,013bytes,
SHA53629420b09408b5c48933f7850667f67b5af9244475e8814f435285fb3fd703.
Inventory SHAee6c39e870e633f805c3a20e44833e70eefec30bcb09a5dde697c03296fed4fd.
Extracted to `../c4_assets/native_worker_22b1db0_probe`; all80 sizes/hashes checked.

## Reproduction and repair

Source inspection finds CudaOpticalExecutionContext normalizes its indexed
device through torch.device(device). The factory requires value equality with
the captured target and exact stream identity. The wait controller instead
required the context's device object to be the original captured target object.
The CPU graph fixture had bypassed that constructor and shared the target.

A new fixture invokes the actual context constructor with modeled copy-by-value
device semantics before any provisional/final template is sealed. It preserves
the same modeled physical device/stream scopes. The production dispatcher then
reproduces the exact native RESOLVE rejection. Initial fixture setup did not
reach this condition because the shared fixture bypassed __init__; after
correcting setup, the exact old error reproduced. Native object identities
were not retrospectively measured: the value/object mismatch diagnosis follows
from the source and matching CPU reproduction, pending actual repaired execution.

Candidate0be9876b9778cf4826a62f45c373a38d60ff33f1 changes only the controller,
catalog and this regression file. Existing sample/transaction/event/stream
identity requirements remain. For CUDA, it additionally requires the
transaction's exact context to be the final factory context, the passed device
to be that context's device object, the exact stream and indexed-device value
to match the retained graph. It then checks adopted ownership using the
original graph target. Warp keeps target object identity. No backend context
constructor, loader, native query/ABI, event wait, handshake or release rule
is changed. No new registry or authority is introduced.

Catalog LF SHA4a400516facc458763158738153e78fda98d6ce2fc2ffe1b4f20a110103bd532;
all91 prospective LF source entries passed the production AST validator.
Ruff and whitespace checks pass. The first integrated regression run had55
passes and one test-construction failure: copy.copy of the sealed owner raised
before controller admission. That test now supplies an uninitialized exact-type
replacement. All six replacement negatives subsequently passed in11.72s.
The distinct-device two-sample positive passed in the first run. Related
guarded lifecycle/evidence, CUDA resolution/timeline and finalgraph checks are
running; their result is not yet claimed.

Fresh review is frozen at `../c4_terminal_target_review_v1`. Built-in spawn
returned agent thread limit reached; the owner-authorized equivalent fresh
CLI session01a12555-9ed3-79c3-9b48-a811e5677ae7 is reviewing it, started
2026-10-10T10:22:10Z, parent37896/reviewer42004. Request SHA
c566fee8ca14d4376da3eca348291294e64422067c9d4a18140f9708d86ecc43.
No verdict or native execution permission for the candidate is presumed.

## Concurrent reader validation, separate unchanged candidate

The approved22b1db0 reader is undergoing its existing16 input-scope cases plus
four foreign/unknown-stream cases on the real deployment. It uses the original
test functions/assertions with the current pinned Warp Go2v4/CUDA precision
environment inventories and120000ms operation timeout for both lanes. The
historical synthetic Warp test default was30000ms; the campaign explicitly
selects the current profile's120000ms deployment timeout. No native call,
assertion or successful observation producer is replaced. The external recorder
only samples /proc ownership and start identities. Test faults remain test-only.

Campaign `/data2/shared/ang.gao/c4-reader-22b1db0-native-scope-v1`, started
2026-10-10T10:15:16Z, wrapper2701942/driver2701950/pytest2701956, still running.
Driver `../c4_assets/run_native_reader_scope_22b1db0.py` SHA
b38a73ff9f57c1d9d84fb934c10198136e2d3a6f21ec5dfe7fb8b166a682bdfc.
No later GPU campaign will start over its live owned processes.

## Parent work remains

Full final-sample physical/numerical verification, all R0/R1 rows/profiles and
camera/generation coverage, bounded preallocated control traces/fixed work,
post-submit/transport/ACK/cleanup/hang and normal-close supervision remain.
The live lifecycle currently copies/replays its full event history per event;
this does not satisfy the final fixed-work/preallocation requirement. It needs
a coherent refactor under the accepted invariant, not a timing interpretation.
Fresh whole C2f/C4 review and C4c3/c4/C4d/B3 are still required. F3 predecessor
approval is separate. Activation, formal performance, results/baselines,
backend selection, R2/S/host/a3b/shipping remain closed.
