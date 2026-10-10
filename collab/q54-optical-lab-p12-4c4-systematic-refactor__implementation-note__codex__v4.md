Initiative: Q54 / P12.4 / C4
Stage: implementation-note
Author: codex
Version: v4
Date: 2026-10-10
Status: draft
Related Files: runtime-integration design-request v3 sections5-8; systematic-refactor gap-review v1 M1/M4/M5; code-review v2; implementation-note v3
Owner Summary: Continue the coherent R0 final-graph packet. This working draft now composes real arena/C3 plans over warmed storage in both lanes under CPU native doubles. Final source-session admission/barrier are now implemented; real-device acceptance remains unproved. No checkpoint or gate is complete.

Base: clean evidence commit3d77448291643ebc1a512fe8c4a9d1be2e78341d;
last approved code99eb1f3aff0436134346d08b2be4f235a910b19d.
Current changes are uncommitted implementation work, not a frozen review candidate.
No independent review of this delta has yet been requested; native evidence and
remaining failure checks for this coherent packet must be supplied first.

Implemented draft

- Warp setup and final transactions share one recorded phase/reset/event loop.
  Exact setup authority and exact live arena lease authority remain separate.
- New optics/warp_arena.py uses NativeOpticalArena for all slot/FIFO/generation/
  resource-hold transitions. It builds R0 C3 from the actual recorded bundle,
  registered prepared input/readiness and output allocations; transaction and
  scope objects are prebuilt before acquisition. No new slot state machine,
  kernels, upload path or renderer algorithm was added.
- NativeGuardedSetupLifecycle.observe_terminal_completion is the common guarded
  wait used by CUDA and Warp. Both retain their original binding rechecks before
  waiting and after DISARMED, before committing existing lease resolution.
- native_setup._prepare_fixed_final_graph consumes exact observed warmups,
  certifies new final readiness, retains its upstream origin, builds new factories
  and every frame/C3/dormant plan over the existing output/event slot. It leaves
  provisional readiness unchanged and final sample indices start at zero.
- Captured CUDA C3 preparation reuses the exact origin's already observed input
  validation, with live storage/identity checks. The existing standalone adapter
  keeps its full direct validation. Calling the old adapter for every final frame
  attempted repeated unguarded GPU-to-host copies; the new composition test caught
  this before any native campaign. No caller boolean or parsed record bypasses it.
- A composition test also caught per-frame capabilities conflicting on the shared
  terminal event. The builder now uses the existing dynamic-arena rule: one
  controlled capability per slot, with distinct frame template/plan identities.

Validation and limits

New Warp tests cover actual control-plane acquire/cancel/burn/reuse, recorded reset
order, launch/event/wait failures, copied sample/transaction rejection, stream/event/
parameter drift before or during wait, lost DISARMED, and ACK-boundary drift.
Actual acquire Python-line counts in both arena modules are equal for1/120/600
prebuilt frame catalogs. This is bounded operation-count evidence, not GPU timing.
Two composition tests use production readiness/C3/arena/plan builders and actual
warmup/guard control, but CPU doubles for native arrays/modules/commands. They
assert no additional allocation, optical submission or readback during final-plan
assembly and retained physical output/event identity across provisional/final graphs.
These do not prove actual Warp packed-command or device execution behavior.

Intermediate runs: warmup58 passed; CUDA resolution/guarded lifecycle130 passed;
new Warp adapter17 passed; final CUDA/Warp composition6 passed. A selected source,
policy, extraction, resolution, guarded-lifecycle and new-adapter run passed390
with4 local native/POSIX skips, before the final Warp composition case and formatting.
Counts overlap. Final post-format run:114 passed in97.04s, command:
python -m pytest -q tests/unit/optics/test_native_render_warmup.py
 tests/unit/optics/test_native_final_graph.py tests/unit/optics/test_warp_arena.py
 tests/unit/optics/test_cuda_arena_contract.py tests/unit/optics/test_cuda_arena_resolution.py
(the above paths were one shell command). Its owned pytest PID42008 and tool
session98757 completed with exit zero. Earlier sessions42656/84717/78619/51872/
29002 also returned exit zero; intermediate failing composition runs were terminal.
The failures above include CPU-double missing event handle/dtype metadata as well
as the substantive shared-capability and hidden-readback integration findings;
the native metadata adapters remain test-only and do not bypass C3/lease checks.
Ruff passes for the new Warp source/tests, shared timeline and CUDA C3 adapter.
All89 catalog source bodies pass the unchanged AST policy with canonical LF bytes;
current catalog SHA256c70919239b06ccbf47003dbc24596609954f4f82a61f0513d84c8818eff813f7.
Bootstrap source admission and product PREPARE are unchanged. No SSH, GPU work,
installation, native driver, process signal or performance measurement occurred.

Prior continuation checklist (superseded by the current status below)

1. Connect composition to the source session through a narrow production entry,
   with frozen-row admission and response/trace capacity checks at the required
   pre-GPU boundary. The private helper alone is not a runnable captured-worker
   path; bootstrap does not yet admit its new final phases or Warp arena import.
2. Validate/consume exact origin and complete final graph, retain immutable SEALED
   coverage once, then perform the exact-target barrier under the existing normal
   guard. Only its observed DISARMED may commit quiescence and execution admission.
   Use an explicitly versioned downstream final-setup record; preserve historical
   input-custody records and their meaning. Parsed graphs/receipts cannot admit.
3. Tie captured arena acquisition mechanically to that successful admission;
   prove before-barrier rejection and failure retention with no retry/reseal.
   Complete shared setup sequencing/failure routing as required by M1.
4. Run the actual fixed supervised native driver for the complete both-lane R0
   composition, its failure/rebinding/scope checks, then freeze one coherent
   candidate for fresh independent review. Do not review a readiness-only helper
   or claim the current CPU tests complete this packet.
5. Continue the approved R1/C2f integration, native precision/failure matrices,
   full-C4 refactor and whole independent acceptance. Performance/result/backend
   selection and later product gates remain closed. Full C4 remains the objective.

Current integration status, 2026-10-10

The production source-session entry now consumes a frozen R0 row admitted before
materialization/runtime initialization. The arena spec builder is extracted from
the benchmark package into optical_worker/arena.py; the benchmark keeps its
existing full gate validation and delegates the single construction body. No
product PREPARE dispatcher is opened. Fixed-driver samples are three; the
protocol numeric count bound alone is not full C2f response/trace budgeting.

Final composition is followed by one immutable guardian seal, exact-target
synchronization under the existing setup guard, actual DISARMED confirmation,
and only then quiescence/admission. Captured NativeOpticalArena acquisition
checks that admission in fixed work, without scanning plans or recomputing hashes.
The final lifecycle uses explicit v4; historical v3 remains unchanged. Runtime
custody replays the explicit nested lifecycle version. Its input-stage readiness
flag retains its original meaning and does not stand for final admission.

Both-lane CPU integration checks exercise success, sync failure, lost DISARMED
and stream rebinding, with no acquisition before the barrier, no emergency retry
after a failed barrier, and retained quarantine custody. Physical Warp event
handles are retained and rechecked alongside stream identity in final validation,
execution and resolution; the added event-rebinding checks are being run.

The existing fixed supervised native driver now selects the frozen row before
materialization, runs its ten warmups, composes final plans, observes target
scope/ambient restoration and acquires/cancels one plan after successful barrier.
It does not launch final samples or measure performance. No second supervisor or
benchmark execution engine is introduced. Actual native runs are still pending.

Current author checks: final graph plus native arena51 passed; expanded two-lane
final graph14 passed; source/graph/Warp arena/guarded lifecycle177 passed with4
platform skips; runtime custody/schema/source policy192 passed with70 native
fixture skips. Counts overlap and native fixture skips require Linux evidence.
Source catalog now contains90 bodies; bootstrap and source pins are regenerated
from exact LF release bytes after edits. The earlier89-file digest and unchanged
bootstrap statement above describe only the preceding draft.

Still required: actual supervised both-lane final preparation/scope/failure
evidence; frozen candidate and fresh independent review; M1 setup/failure-routing
consolidation where needed; R1/C2f integration and the full native precision and
failure matrices, C4c3/c4 and C4d/B3, then whole-C4 review. Nothing here proves
performance, opens later gates, or completes the active C4 objective.

Before native candidate freeze: the event-handle drift test exposed a real CUDA
gap (object identity was stable while the underlying handle changed). Existing
arena/sample signatures now include native event handles, matching the Warp
check. Both-lane final barrier cases pass, including this counterexample.
One intermediate resolution regression returned AttributeError for an event
replaced with an arbitrary object; signature collection now uses absent metadata
as a mismatch and preserves the existing protocol-error path. Its full21-test
resolution file passes after repair. Earlier123-test preparation/graph/Warp run
had122 passes and the original CUDA event-drift failure; the following68-test
run had67 passes and only that error-type regression. Native evidence and fresh
independent review are still pending; this freeze is an author test candidate.

First real Go2 final preparation at c59edc74d7d766eb71bd4b7f81d94045440cb889

The remote isolated tree /data2/shared/ang.gao/robot_sim_final_graph_wip was
clean at the exact candidate. Warp GPU1/ambient0, Go2960, frozen render-only
shadow row completed ten warmups, final graph construction, final target barrier
and acquire/cancel. It restored native ambient state. Original raw directory:
/data2/shared/ang.gao/c4-final-graph-c59edc74d7-warp-runtime_final_graph_success.

This run is NOT accepted end to end: the supervisor's combined-artifact validator
replayed the explicitly v4 worker trace using its default v3 and raised
GuardProtocolError: invalid final graph seal. The worker's recorded terminal
state is aborted_synced_exit with completed release, but the top-level native
acceptance failed. Author parent75699, supervisor75759 and worker75765 were all
queried absent after the command exited1. No signal or GPU reset was issued.
The repair forwards the already validated nested version through that remaining
replay call and adds a two-lane CPU combined-artifact regression. A fresh native
run is required. Original evidence is preserved unchanged.

Frozen-candidate local source/policy/final-graph/CUDA-resolution/Warp-arena check
completed262 passed,4 platform skips in87.26s; no performance assertion follows.
