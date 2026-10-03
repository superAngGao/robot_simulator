# Independent C2d1 guarded-custody design review v1

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: review
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: guarded-custody design-request codex v1; runtime-integration design-request codex v3; optical_worker/lifecycle.py; optical_worker/native_setup.py; optical_worker/wait_protocol.py; optical_worker/wait_supervision.py; native source/bootstrap policy and tests
Owner Summary: Request changes. The proposed subclass can intercept the actual upload abort without copying C0, but the no-GPU clean-cancel path lacks a specified bounded terminal-observation mechanism, and the new v3 evidence grammar is not frozen sufficiently for implementation. No candidate edits or GPU execution occurred.

## Identity and verdict

Reviewer session: `/root/c4_c2d1_guarded_custody_design_review_v1`, fresh independent session receiving a task packet rather than the author's implementation conversation. No delegated agents were used. The reviewer did not author or modify the candidate.

Verdict: **Request changes**. Findings F1 and F2 below are blocking design clarifications, not claims that an unimplemented controller failed a runtime test.

- Base: `11603cd2772aea79c07579704285767f6c5ac622`.
- Frozen HEAD: `e4aeb403703f98ec6946b7c75c328c018ae01b0a`.
- Candidate: `collab/q54-optical-lab-p12-4c4c2d-guarded-custody__design-request__codex__v1.md`.
- Independently checked SHA-256: `131de33407451bc0bd668feb83daae255da8f4bca14733e77a11d7c04c0bb133`.
- Local: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Remote: `/data3/ang.gao/robot_sim_c2d1_design_v1`, authorized fallback `ang.gao@192.168.80.101`, detached at the same HEAD and digest.
- Initial/final local and remote status were clean. The entire base-to-HEAD delta is the single added 256-line design document. Diff whitespace check passed. Explicit protected-source checks found no changes to lifecycle, native_setup, worker, wait_protocol or wait_supervision.

The packet reported primary183 transport timeout; fallback101 was used without retrying primary. The original dirty `/data3/ang.gao/robot_sim_c4c1b2` was not accessed or modified.

## Findings

### F1 — P2: clean cancellation from IDLE has no specified bounded cleanup/exit observation

Location: candidate lines 124, 140–150, 209–212; actual `wait_supervision.py::_State.deadlines`, `_transition`, and `lifecycle.py::NativeWorkerResourceGuardian.release_whole_graph`.

Trigger: construct the controller and adopt CPU owners, but enter no GPU setup leaf. A failure now satisfies section 4 step 4: healthy IDLE, clean cancel, whole-graph release, exit. Keep the parent alive. Cleanup of a CPU owner, or the fixed driver's interval before `os._exit`, hangs while the controller still holds its private socket.

Consequence: the actual C2b supervisor in IDLE has no operation, ACK or terminal deadline. It cannot expire merely because an operation budget has elapsed. A retained open channel supplies no EOF, and the live parent supplies no liveness loss. Thus this specified branch can stall indefinitely unless the implementer invents a terminal-observation policy. The emergency branch is different: its existing ARMED_TERMINAL deadline does cover cleanup and exit.

Evidence: the independent CPU probe below used the actual C0 guardian release and actual C2b core. During the adopted CPU owner's destructor the core was `IDLE`, deadline `None`, kill-required `False`. Advancing its controlled clock by 50 operation budgets and calling `tick()` left the same state. No hang was induced; this confirms the dependency premise, not an execution of the proposed subtype.

Why existing requirements miss it: section 6 explicitly tests emergency sync/cleanup/exit hangs, but not clean cancellation with no setup work. `os._exit` is a final action, not supervision of code preceding it. The C2b harness's outer failing safeguard is not a production watchdog. Parent runtime-integration section 4 also requires bounded terminal process-exit observation; that general requirement does not select a concrete mechanism here.

Requested direction: freeze the exact terminal notification/observation and ordering for this branch before release begins. For example, specify a supported private-channel terminal/EOF transition that makes the existing supervisor retain a bounded terminal obligation before potentially blocking cleanup, or explicitly identify an existing reviewed supervisor-owned terminal observation path. Preserve retained graph/client custody, no fabricated exit, no new emergency GPU sync, and no new wire/timer policy by accident. Define what happens if the notification itself fails. Do not rely on process return, destructor completion, the parent dying, or an outer test timeout.

Required proof after repair: real parent-alive subprocess cases for (a) ordinary no-GPU clean cancellation, (b) CPU-owner cleanup hang, and (c) post-cleanup/pre-hard-exit hang. Show the actual terminal trigger, unchanged/non-restarted bound, observed kill/reap where required, zero GPU sync, and truthful release/custody versus actual exit evidence. Also exercise notification failure. Preserve existing emergency-deadline tests.

### F2 — P2: the promised exact v3 lifecycle/replay contract is not yet specified

Location: candidate lines 159–186, particularly 164–180; cleanup-failure semantics at 140–150.

Trigger: implement the v3 evidence class/replayer and combined worker/supervisor validation artifact from this supplement. The text calls for an operation enum and exact operation-specific fields, but does not enumerate that enum, event fields, terminal root/summary fields, allowed incomplete tails, or the corresponding transition grammar. The parent v3 design likewise states replay obligations rather than supplying this missing schema.

Consequence: materially different safety interpretations remain possible under the same prose and `v3` label. A concrete example is failure after release authorization or during `release_whole_graph`: the frozen guardian can be `RELEASE_AUTHORIZED` or `RELEASED`, and `transfer_to_quarantine()` only accepts `PARTIAL_FROZEN`/`SEALED`. The supplement correctly proposes separate terminal custody of remaining owners, but does not freeze how the v3 replay represents and checks that condition without falsely claiming the guardian itself transferred or that cleanup completed. Similar choices remain for a pending ACK with no returned ticket and for matching a partial worker tail to the supervisor's independent terminal trace.

Why existing checks miss it: C0's exact `TransportTerminationEvidence` validator has no guard/custody grammar and must retain its old domain; C2b's exact schema has guard/process facts but no owner/cleanup facts. Neither is the missing v3 contract. A test implementation can otherwise derive its expected schema from the same new implementation and fail to constrain these choices. This conflicts with the project's design requirements to freeze high-risk state/evidence schemas and remove substantive implementation policy decisions.

Requested direction: add a compact normative v3 schema/transition table, reusing unchanged C0 resource/descriptor definitions by explicit reference. Specify the exact root, transition and derived-summary fields; field domains and version identity; each new operation's exact detail fields, preconditions/effects and guard reference rules; distinct release authorization/start/completion and terminal-custody states; legal cleanup-failure guardian states; partial versus terminal worker records; and the compatibility rules for the combined supervisor artifact. Define which events may repeat for different segment/owner identities rather than literally banning all duplicate operation names. This need not freeze incidental Python class/method names or create another wire protocol.

Required proof after repair: independently specified legal fixtures for normal segments, inner `_abort`, pre-return/unknown scope, ARM/COMPLETE ACK failure, emergency success, failure before release, and failure during/after reference release. Mutation tests must reject missing, duplicated, reordered or wrong-identity transitions and inconsistent summaries/tails. Keep the explicit limit that replay proves consistency, not authenticity of an entirely rewritten legal trace or live GPU authority.

## Confirmed design strengths and actual-code analysis

Read `AGENTS.md`, `collab/README.md`, `collab/REVIEW_PRINCIPLES.md`, the complete candidate and the complete accepted runtime-integration v3 design. Inspected prior C2b design/core review and C2d0 source design/code review records as context, not as proof that this candidate is correct. Truncated combined output was supplemented by targeted reads of the relevant source and parent sections.

Inspected actual upload functions and `_abort`, guardian ownership/release methods, lifecycle descriptor/permit/scope/termination and trace machinery, complete GuardWaitClient, supervisor state/deadline/replay transitions, fixed real-process guard harness/tests, C0 lifecycle tests, native source policy/catalog verifier, and captured-source bootstrap loader/identity audits.

- `_abort` really dispatches to `lifecycle.terminate()` inside both upload exception handlers. A subclass override therefore reaches the right boundary; an outer wrapper alone would be too late. The C0 terminating implementation really may invoke its A/K notifier and sync. The proposed override must never delegate to that terminating implementation.
- The existing methods permit reuse of one guardian and descriptor/permit implementation. Overriding the protected trace hook can associate inherited events with current guard identity without copying upload loops or mutating per-instance methods. Base construction calls that hook, so the implementation must initialize its CPU/pre-guard trace context before that call.
- GuardWaitClient retains exact ticket objects in append-only state history and rejects emergency COMPLETE. A bounded identity-observation extension is feasible without new wire messages. Current supervisor ARMED_TERMINAL preserves its deadline to observed exit. Neither ticket identity nor parsed evidence is GPU completion authority.
- The proposed precedence for compromised scope, failed/pending normal guard and emergency failure matches parent v3. In particular the inner upload abort must quarantine before outer context handling and must not COMPLETE or initiate another sync.
- The existing final-barrier helpers synchronize and commit C0 quiescence before an enclosing guard could receive DISARMED. Forbidding inherited barrier/clean-close/measured admission in this slice is necessary and correctly scoped. Full final graph/readiness remains a later dependency.
- Native source policy currently pins the C0 lifecycle bytes and the CPU bootstrap/loader definitions. Leaving C0 unchanged avoids weakening that pin. The new controller/wait-client delta needs literal source catalog repins and tests. The current captured-source finder only admits the six CPU core modules, so later guarded-role adoption remains actual implementation work, not something this CPU slice supplies automatically.
- Terminal controller/client/exception retention outside the guardian avoids reopening a sealed coverage manifest. The trusted fixed-driver and closed-process assumptions are appropriate; arbitrary in-process reflection/native-code attacks are not falsely advertised as blocked.
- The cleanup ledger/removal plan and continued same-controller source-frozen real Warp/Torch matrix are explicit. No CPU surrogate is offered for dual-GPU/foreign-pre-return/unknown-scope acceptance, and no C2d/e/f or C4 completion is inferred.

## Independent verification, commands and results

This is a documentation-only design review. There is no new guarded controller or acceptance suite to execute. No unrelated full GPU regression was warranted. Prior author test outputs were not counted as new execution evidence.

Read-only local inspection used `Get-Content -Raw/-Encoding UTF8`, targeted `Select-Object` reads, `rg`, `git rev-parse HEAD`, `git branch --show-current`, `git status --porcelain=v1`, `git diff --stat/--name-status`, `git diff --check 11603cd2772aea79c07579704285767f6c5ac622 HEAD`, and `Get-FileHash`.

Independent diagnostic, created with apply_patch outside the candidate:

`C:/Users/AG/新建文件夹/c4_assets/c2d1-design-review-probe-v1.py`

SHA-256: `f93b663e9bf719e7f8e512830eb68b1691c3ef948e1fb0520e567d0b88100051`.

Executed:

```text
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/c2d1-design-review-probe-v1.py
```

Observed exit 0 on CPython 3.14.0. It imported neither Torch nor Warp, started no child, allocated no GPU storage, and asserted actual core/guardian behavior rather than a copied state machine. Key output:

```text
pid=35484 ppid=12100
at_cleanup_state=IDLE at_cleanup_deadline=null at_cleanup_kill_required=false
after_50_operation_budgets=IDLE deadline_ns=null kill_required=false
c0_disposition=cancelled_before_submission_exit
```

The controlled clock is a supported core test input. This probe does not implement the candidate, inject an actual hang, or prove any new release safety. The potential live-parent hang follows from the observed absence of any deadline or terminal trigger in the specified path.

Remote read-only checks used:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 <read-only command>
```

Commands checked exact HEAD, detached branch, empty status, diff whitespace and `sha256sum` of the candidate. All matched the review identity. `ps -eo pid,ppid,lstart,args` with a scoped filter and `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader` checked residuals and current compute applications. Final remote compute output had no rows.

Inspection/tool limitations: the first instructions command looked for root-level `REVIEW_PRINCIPLES.md`; the correct complete `collab/REVIEW_PRINCIPLES.md` was then read before substantive review. One remote process-check command failed because remote `rg` is unavailable and quoting did not preserve its alternation; its earlier Git/digest checks succeeded, but its process/GPU suffix did not execute. That suffix was repeated successfully with `grep -E` and proper quoting. These were inspection/tool errors, not candidate failures.

## Process hygiene and scope limits

- Owned local probe PID 35484, parent PowerShell PID 12100, returned actual exit 0. A later exact CIM lookup found neither PID present. No child worker/supervisor or GPU process was launched by the probe. Its short-lived OS creation timestamp was not captured; the launch/return/PID evidence and later absence are the recorded scope.
- Successful remote process scan showed only its own shell PID 2241973, parent 2241891, and grep PID 2241995, parent 2241973, both started Sat Oct 3 08:44:33 2026 remote time. No scoped worker/test match remained. A final exact `ps` query found both absent; its exit 1 was the expected no-matching-PIDs result, not failed Git checks.
- Local GPU query returned zero entries for owned probe PID 35484. Unrelated desktop applications were not touched. Remote compute query returned no entries. These are scoped current observations, not GPU quiescence proof.
- No test timeout, kill, install, GPU reset, recursive deletion, candidate write, source normalization or cleanup of another session's work occurred. Only the external diagnostic and this report were created. Read-only SSH/command process identities were not exhaustively ledgered; no claim of exhaustive short-lived shell enumeration is made.

This review does not approve implementation of the unresolved C2d1 contract as written. Revise the two bounded design gaps and obtain fresh independent re-review. No change to C0 historical approval is implied.

Even after design approval, implementation/code review and all specified CPU real-process tests remain mandatory. Captured-source guarded-role adoption, complete module/session custody before CUDA-capable execution, source-frozen real upload tests in both lanes on target1/ambient0, foreign pre-return/unknown/fault matrices, final graph/readiness/warmup composition, full C2d/e/f and independent activation reviews remain outstanding. Product GPU preparation/submission, arena/result/baseline authority, a3b, R2/S/host/shipping/B3 and C4 completion remain closed.
