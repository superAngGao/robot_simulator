# Fixed R0 setup warmup execution

Initiative: Q54 / P12.4 / C4 runtime integration
Stage: implementation, before independent code and native acceptance
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: runtime-integration design-request codex v3 sections 4, 5, 5.1, 6, 8, 9; remaining-setup gap-scan codex v1 G2; provisional-render-graph code-review codex v2; provisional-graph-native-repair code-review codex v1
Owner Summary: Connect the existing provisional R0 graph to ten actual setup transactions and physical terminal waits, using the original supervisor and renderer implementations. CPU checks cover execution order and failure semantics. Native execution, image correctness and whole C4 remain unproved.

Base: e43a86c8e63a37da7bd84a5a8f5616559c7e12a1.
Candidate: the commit containing this note; the review packet records its exact HEAD.
Implementation worktree: C:/Users/AG/新建文件夹/c4_render_execution_next.

## Behavior

The source-session entry is ordered and single-use after the provisional graph.
Its first setup guard imports the two new CPU-safe definitions and prepares all
ten transactions. The original runtime binding, compiled owners, graph, exact
prepared input and truthful warmup0 readiness remain held. The private setup
authority is registered by actual session/controller/guardian/runtime/target/
stream/graph identity, with one active transaction index, ticket and original
resource-scope operation. A copied authority or transaction does not register.
Each launch/event checks the current original setup scope; no final C4 lease or
sample token is fabricated.

Warp's private setup transaction in optics/warp_timeline.py executes the actual
recorded commands. Phase boundaries come from the validated render ABI: primary
counter reset is inside first-hit, shadow reset inside shade. Diagnostics off
does not record diagnostic events. All parameters are revalidated untimed before
enqueue. The transaction owns a precreated terminal record and records the slot
event once in finish. It has no wait or allocation in its execution loop.

CUDA subclasses its existing transaction only for private setup authority and
leaf declarations. Its original render stage, entry calls, diagnostic state,
enqueue and finish behavior are reused. The stage's existing diagnostic-start
method is also called with diagnostics off, immediately before each native
entry; the setup subclass declares warmup_launch at that boundary. Kernel or
event exceptions still propagate through the existing sticky guarded scope.
Ordinary CUDA transaction behavior is unchanged.

The setup driver runs ten nonnested guarded frames in order. It retains each
actual finish object before waiting on its actual terminal event. Only after
the wait AND the exact guard's COMPLETE/DISARMED succeed does it append that
frame's observation. Consequently reuse of the shared slot/event begins after
both completion conditions. The final frame also audits source/external custody.
Any failure leaves the original whole graph in the existing terminal lifecycle;
it grants no warmup completion record and starts no recovery guard or retry.
After the driver returns or fails, its live authority is retired.

WarmupCompletionEvidence retains the actual prepared input, provisional
readiness, compiled-module owner, graph, ten transactions/terminal objects and
successful guard tickets. Its descriptive payload names the source catalog,
configuration and ordered local object/guard identities. Only the exact local
registration passes require_observed_warmups; construction or serialization is
not issuance. The provisional readiness still truthfully says warmup0.

The source inventory gains optical_worker/warmup.py for both lanes and
optics/warp_timeline.py for Warp only. Catalog v16 uses schema v8; both cold and
production catalog selectors agree. Imports become available only in the two
new warmup phases. Existing source checks and pins are regenerated for the
literal 86-source LF release; no source-policy exception is added.

## Author verification

Local interpreter: C:/Python314/python.exe -B, pytest cache provider disabled.

- Focused graph/preparation, CUDA timeline and warmup checks initially passed
  142 tests. This preceded the final Warp state representation and source fixes.
- Final warmup suite: 50 passed in 64.50s (session 6717). It exercises the real
  source-session method, lifecycle, guard socket peer, authority, transaction
  loops, CUDA render stage and completion checks. Native returns/calls are
  explicit doubles. Warp bundle/packed-ABI construction is doubled, so this
  suite does not claim actual Warp recording or numerical rendering.
- Positive cases cover both lanes, three profiles, both shadow settings and
  diagnostics on/off. Negative cases cover launch/record/wait failure, failed
  DISARMED after a returned wait, replaced terminal, copied authority and copied
  transaction. Checks include no next-frame work, no issued completion, full
  quarantine and zero emergency synchronization after a failed normal guard.
- The first test assertion incorrectly rechecked an old completed ticket after
  later guards. The production client intentionally accepts only the most
  recently completed ticket; the driver already checks immediately after each
  frame. The test now inspects each retained normal_disarmed trace instead.
- An initial source run found the cold selector still named v15 and the lane
  inventory test omitted the new Warp file. Both are repaired; the latter now
  requires exactly the three Warp-only files. No rejection was relaxed.
- Final source policy/source/bootstrap suite: 281 passed, 13 host/asset skips
  in 161.66s (session 19348, exit0). Catalog SHA256 is
  d25e55793d1fddf2f3bf28d7cc286cde2986eb3e5129efa4a064e43bea72f5f2.
  Skips are two local Go2 asset cases, four POSIX entry/release-byte cases and
  seven pinned CPython3.12 generated-module cases; none is a native pass.
- All author CPU test handles are terminal. Final scoped Win32_Process query
  found no process matching either invocation. Observed earlier test PIDs30496
  and31172 and the final source PID45736 have exited. The unrelated native graph
  batch remains owned by its original live handle14293; it was not restarted.

## Explicit remaining work

This is private fixed R0 setup execution, not the complete measured Warp
timeline/arena. R1 scene/refit stages, final R_final/FinalReadinessOrigin, new final
factories/commands, C3 registration, one coverage seal, final target barrier,
sample/lease activation, device delivery and complete lifecycle integration are
still required. No public product PREPARE, IPC, result, baseline or performance
gate is opened. No image, physical warmup success or fair speed result is claimed
from these CPU tests.

The native graph campaign at exact6300531 is separate and still running; it
executes no optical kernels. Native warmup support and actual success/failure/
hang/output checks must exercise this candidate before native acceptance.
The owner's systematic whole-C4 refactor and fresh whole review remain required.
