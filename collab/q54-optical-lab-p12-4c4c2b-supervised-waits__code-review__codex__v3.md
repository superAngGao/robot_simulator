# C4c2b guarded waits — independent code re-review v3

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: code-review
Author: codex (fresh independent reviewer)
Version: v3
Date: 2026-09-29
Status: in_review
Related Files: design-request v3; design review v3; code-review v1; implementation-note v1/v2; wait_protocol.py; wait_supervision.py; guarded-wait tests and cleanup ledger
Owner Summary: Request changes. The repaired supervisor publication closes the supplied lost-terminal-state counterexamples. Independent Linux regressions pass, but client failure can still be overwritten during ARM/COMPLETE, and reentrant transport can send a complete ARMED response after terminal entry. No runtime or GPU gate opens.

## Identity and verdict

- Reviewer session: `/root/c4_c2b_wait_code_review_v3`, fresh independent session with no candidate authorship and no delegated agents.
- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Overall base: `fcb9608a916d20d1c43f64a3401b3c78d823404f`.
- Exact reviewed HEAD: `f74fd9bf548bbeda8d1accdbd105e2c6d4524ced`.
- Exact tree: `e1f083e952c161c6869b8bbec9ee7948bf7374e9`.
- Independent Linux execution: permitted fallback `ang.gao@192.168.80.101`, clean detached `/data3/ang.gao/robot_sim_c4_c2b_code_f74fd9b`, same HEAD and tree, `robot_sim` Python 3.12.13.
- Local execution: Windows PowerShell, Python 3.14.0.
- Local and remote porcelain status were empty before/after verification. No candidate edits were made. Report and two repro scripts are outside the repository; `-B` and disabled pytest cache provider were used. Ignored pre-existing/normal dependency caches are not asserted absent.
- Delta from base: five new executable/test paths, cleanup ledger addition, prior review and two implementation notes; nine paths, 2,381 insertions. Existing C0/C1 worker/runtime/gate/source-policy surfaces are outside the diff.
- Read repository instructions, complete v3 design and its independent review, prior full code report, both implementation notes, actual two modules, harness and tests. Garbled initial Chinese output was reread as UTF-8 and truncated combined reads were supplemented. Initial root-level REVIEW_PRINCIPLES lookup failed; correct `collab/REVIEW_PRINCIPLES.md` was subsequently read. An initial unquoted PowerShell `HEAD^{tree}` lookup failed; quoted local and remote lookups returned the tree above.
- Approved design SHA-256 independently checked: `a3619ead4e22ad47f3133faca7788cdb85d0816f2bad8cffd8a46998fc80c872`, approval revision `ff3b4db0d4c3812c7149c97cb9cf9bb095598461`.
- Previous candidates `5dd04f8` and `9b8bc3b` were not approved. The interrupted v2 reviewer produced no verdict; none is invented here.

**Verdict: Request changes.** Two substantive findings below. Passing regressions are independent observations, not evidence that these uncovered interleavings satisfy the contract.

## R1 — P1: nested client failure is overwritten by outer ARM/COMPLETE success

Locations: `optical_worker/wait_protocol.py:224–234` and `:251–254`; exact reproduced injection lines 231 (`self._active = ticket`) and 253 (`self._active = None`). Existing incomplete coverage: `test_swallowed_nested_failure_cannot_reopen_client`.

The client checks its pending state after `_exchange`, but then separately publishes ticket/sequence/state. A same-thread signal after that check calls `client.arm("resolve")`; the nested call sees ARM_PENDING or COMPLETE_PENDING, rejects and sets FAILED. When its exception is caught by the handler, the outer operation resumes and assigns ARMED or IDLE. FAILED is therefore not irreversible.

This was independently reproduced using the actual client, an actual socketpair, and actual core/transport in its owning server thread. No candidate method was monkeypatched; no private state or ticket was modified. The tracing hook only schedules a real Linux SIGUSR1 handler at the specified line; Windows uses the same-thread handler directly.

Observed on both Python 3.14.0 (Windows) and Python 3.12.13 (Linux):

```text
phase arm: nested [('FAILED', 'guard is not idle or sequence is exhausted')]
outer_state ARMED; returned a ticket; subsequent_wait succeeded
phase complete: nested [('FAILED', 'guard is not idle or sequence is exhausted')]
outer_state IDLE; subsequent_wait succeeded
```

The real supervisor observed two normal completed waits in each case. This is not merely a stale property read: normal use is actually reopened. For ARM, fixed driver code can enter its leaf using a ticket returned after a detected nested failure. For COMPLETE, subsequent work can start instead of following the required failed-client hard-exit path. No claim is made that this candidate is already adopted by a product worker.

Why tests miss it: the current nested-client test injects from `_exchange`, before the pending-state check, and consequently passes. It does not schedule nested failure after that check or across the subsequent state/ticket publication. Cross-thread rejection cannot guard same-thread signals.

Required direction: make failed-client custody monotonic across every public entry and success publication, with a well-defined linearization boundary for ticket/state/sequence. A late success assignment must not overwrite a committed failure, and stale outer work must not return a usable normal ticket. Another plain check immediately before assignment leaves the same class of window. This is the already-approved supported-process reentrancy requirement, not a hostile-Python sandbox requirement.

Required proof: real socket/core/client tests for both ARM and COMPLETE, swallowed and propagated nested errors, `fail`/close and nested wait entry around precheck, exchange, ticket publication and final return; FAILED must remain FAILED, no usable ticket/subsequent normal request, no ordinary cleanup after a failed exchange. Rerun the original ticket/thread/clock/process regressions on the resulting exact revision.

## R2 — P2: terminal transport reentry does not fence the pending socket send

Location: `optical_worker/wait_supervision.py:614–620`, specifically `sent = self.channel.send(...)` at line 620. Existing incomplete coverage: `test_swallowed_transport_reentry_prevents_pending_ack_output`.

Prepare a normal ARM_ACK_PENDING response, enter `transport.service(writable=True)`, and deliver SIGUSR1 at line 620, after the last terminal-state check but before the send. The handler calls the public `transport.service` again and catches its reentry exception. That nested call correctly faults the core to TERMINATING and drops its pending response, but the outer transport still owns `_output` and sends a complete, valid ARMED frame. The subsequent acknowledgment commit rejects only after those bytes were sent.

Independently observed with actual socketpair and Linux SIGUSR1, also local callback:

```text
nested [('TERMINATING', 5000100, 'reentrant supervisor transport service')]
outer_error ACK write does not bind pending output
wire_after_terminal operation=ARMED sequence=0 role=setup session=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
final state TERMINATING, effective_deadline_ns=5000100, pending_ack=null
trace contains no ack_written
```

This violates design section 6's cease-output rule and explicit section 8.3 acceptance requirement: no response bytes after terminal entry. A peer can receive authorization to enter its leaf even though termination was committed before any response byte was sent. Unlike the earlier lost-deadline defect, the current core retains its deadline and valid incomplete evidence; R2 does not claim that kill protection or replay is lost. The contract failure is the actual post-terminal external write.

Why tests miss it: the existing transport test injects at the first `_service` tick, before the terminal check. The new broad line-scheduling test covers core scope/ACK/state/clock/append methods, not the transport's send boundary.

Required direction: explicitly linearize terminal cancellation with transport output initiation, so a terminal transition that wins before send cannot leave stale normal output eligible for sending. Merely checking again on a preceding Python line or rejecting acknowledgment afterward does not establish this ordering. Keep the distinction between already-written bytes that cannot be recalled and this reproduction, where terminal entry precedes the first byte. If achieving this needs a changed execution/signal-delivery contract, seek the scoped design decision; do not silently weaken the accepted rule.

Required proof: inject actual same-thread reentry/parent loss around output preparation, last-state check, send, partial-write continuation, and full-write commitment, for both ARMED and DISARMED. Assert peer-visible bytes as well as state/evidence. A terminal-before-send ordering must emit zero new bytes; a write-before-terminal ordering must retain honest deadline/custody and never commit a stale normal acknowledgment. Preserve the existing OS-backpressure and process tests.

## Prior finding closure and current positive observations

The old supervisor stale-state publication is materially repaired. `_append` computes one `(state, event)` record and uses exact builtin `dict.setdefault` under an integer index as its sole publication point. If nested termination occupies that index first, the outer record cannot replace it. If normal commitment already won, a later fault is represented as a subsequent terminal transition. Evidence snapshots derive both state and events from the same committed record sequence. This assessment is bounded to the supported CPython/process model and tested interleavings, not arbitrary object mutation or an assertion about every Python implementation.

Independent local probes of the supplied original precommit and author finalizer counterexamples produced:

```text
_append line419: TERMINATING deadline5000100 normal_complete_count0
_invoke line367: TERMINATING deadline20000100 normal_complete_count1
swallowed-clock: TERMINATING kill_requiredTrue arm_count0
```

Both traces replayed. The finalizer ordering has already legitimately completed DISARMED before parent loss and therefore has a new terminal grace deadline, not an erased fault. The precommit ordering preserves the earlier pending deadline. The installed tests additionally ran all nine precommit boundaries (propagated/swallowed), the swallowed-clock case and normally reached DISARMED/finalizer lines, using real SIGUSR1 on Linux. The specifically reported old supervisor lost-state and swallowed-clock counterexamples are closed on this candidate; this does not excuse R1/R2 elsewhere in the promised contract.

Other inspected behavior:

- Exact configuration is frozen/reconstructed; canonical bounded frames use the shared production codec. Integer domains reject bools, wrong identities and noncanonical framing. Sequence/counter/index/timestamp checks preserve the accepted capacity and no-wrap model.
- Tickets require outstanding object identity. Ordinary copied, reconstructed, retired and emergency ticket misuse fails before wire output. Same-thread final publication remains the R1 exception; foreign-thread entry rejects before mutation.
- Each client exchange has one absolute send-plus-receive deadline and readable ACK loses at equality. Client executes no blocking callable.
- ARM deadlines precede output; DISARM_ACK_PENDING keeps the original operation deadline. Full recorded DISARMED output is required to retire normal deadlines/count a completion. Fault folding retains the earliest pending/operation deadline. Ordinary partial input/output progress cannot renew budgets.
- Transport buffers are bounded and nonblocking. Full requests arriving before pending ACK commitment fault rather than queue. R2 is a terminal cancellation/output ordering gap, not an unbounded buffer or blocking-receive defect.
- Replay uses the same transition grammar and exact derived summary. Expiry, attempted kill, kill failure and actual observed exit remain distinct; a kill request cannot finalize evidence. Snapshots are not live authority or authenticated execution evidence.
- The fixed harness uses actual poll/wait return codes and retains child custody through failure; its outer fail-safe is not presented as protocol watchdog success. Ten subprocess cases include normal/emergency hangs, no/wrong/EOF ACK, failed send, stalled DISARMED, parent loss and partial input. Finalizer sentinels test hard-exit/process behavior, not CUDA ownership.
- Capacity proof remains G=(2^63−1−8)//4, S=G−1; initialization +4G+6 fits canonical signed-64-bit fields. Actual checked primitives and first-invalid boundaries are exercised; infeasible full-size traces are not claimed executed.
- Cold-import tests block Torch/Warp/NumPy/benchmark/test/tool/example import attempts. Scoped import search finds only new core/test consumers, not product adoption. The base delta leaves C0 A/K, C1 S/V/E, existing worker/native setup/lifecycle, source policy and gates unchanged.
- Cleanup ledger accurately classifies shared safety infrastructure pending separately reviewed selected-runtime adoption and removal if no selected consumer remains. The harness is not a second product supervisor.

## Independent executions and provenance

1. Local:

```powershell
python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_guarded_wait_core.py
```

Observed **93 passed in 1.68s**, exit 0, Python 3.14.0.

2. Independently initiated Linux run (not the author's 16.61s run/log):

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 "cd /data3/ang.gao/robot_sim_c4_c2b_code_f74fd9b && git rev-parse HEAD && git status --porcelain && /data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/benchmarks/test_optical_benchmark_contracts.py::test_inventory_and_cleanup_ledger_match_current_repository"
```

Observed exact HEAD, clean status and **154 passed in 16.27s**, exit 0, no skips. Includes actual POSIX process cases, core signal tests, unchanged C0 protocol/worker, C1 watchdog and cleanup inventory. Later independent remote HEAD/tree/status/Python checks matched the identity above.

3. Two independent repro scripts saved outside candidate with `apply_patch`:

- `C:/Users/AG/新建文件夹/c4-c2b-client-reentry-repro-v3.py`, SHA-256 `61ec4cc53224567eac38b01a47ba4a43a64a2e5b1c3e4b9bdb40b6385856354d`.
- `C:/Users/AG/新建文件夹/c4-c2b-transport-reentry-repro-v3.py`, SHA-256 `9335c131a0d6b65d39f11d70846ced0553e25b51ec7ac434c7433cb45662ec28`.

From the candidate cwd, each ran with:

```powershell
Get-Content -Raw -Encoding UTF8 ../c4-c2b-client-reentry-repro-v3.py | python -B -
Get-Content -Raw -Encoding UTF8 ../c4-c2b-client-reentry-repro-v3.py | ssh -o BatchMode=yes -o ConnectTimeout=10 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 "cd /data3/ang.gao/robot_sim_c4_c2b_code_f74fd9b && /data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -B -"
```

The same commands substituting `c4-c2b-transport-reentry-repro-v3.py` ran R2. All four executions exited 0 because their assertions prove the described bugs, not acceptance. They did not install, change source, fork a replacement protocol implementation or create remote tracked evidence. The core in R1's background thread is constructed/used in that thread; the client and signal handler share their owning main thread.

Complete captured R1 stdout (local then remote):

```text
python 3.14.0
{'phase': 'arm', 'signal': False, 'line': 231, 'nested': [('FAILED', 'guard is not idle or sequence is exhausted')], 'outer_state': 'ARMED', 'subsequent_wait': 'succeeded'}
{'phase': 'complete', 'signal': False, 'line': 253, 'nested': [('FAILED', 'guard is not idle or sequence is exhausted')], 'outer_state': 'IDLE', 'subsequent_wait': 'succeeded'}
python 3.12.13
{'phase': 'arm', 'signal': True, 'line': 231, 'nested': [('FAILED', 'guard is not idle or sequence is exhausted')], 'outer_state': 'ARMED', 'subsequent_wait': 'succeeded'}
{'phase': 'complete', 'signal': True, 'line': 253, 'nested': [('FAILED', 'guard is not idle or sequence is exhausted')], 'outer_state': 'IDLE', 'subsequent_wait': 'succeeded'}
```

Complete captured R2 stdout (local then remote):

```text
outer_error ACK write does not bind pending output
python 3.14.0 signal False line 620
nested [('TERMINATING', 5000100, 'reentrant supervisor transport service')]
wire_after_terminal {'device_ordinal': 1, 'operation': 'ARMED', 'protocol_version': 'q54.closed-worker.guarded-wait.v1', 'role': 'setup', 'sequence': 0, 'session': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'}
final {'active_wait': {'role': 'setup', 'sequence': 0}, 'arm_count': 1, 'complete': False, 'deadline_expired': False, 'effective_deadline_ns': 5000100, 'kill_error_kind': None, 'kill_requested': False, 'next_sequence': 1, 'normal_complete_count': 0, 'observed_returncode': None, 'operation_deadline_ns': 20000100, 'pending_ack': None, 'state': 'TERMINATING', 'terminal_deadline_ns': 5000100, 'terminal_reason': 'request_out_of_order'}
outer_error ACK write does not bind pending output
python 3.12.13 signal True line 620
nested [('TERMINATING', 5000100, 'reentrant supervisor transport service')]
wire_after_terminal {'device_ordinal': 1, 'operation': 'ARMED', 'protocol_version': 'q54.closed-worker.guarded-wait.v1', 'role': 'setup', 'sequence': 0, 'session': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'}
final {'active_wait': {'role': 'setup', 'sequence': 0}, 'arm_count': 1, 'complete': False, 'deadline_expired': False, 'effective_deadline_ns': 5000100, 'kill_error_kind': None, 'kill_requested': False, 'next_sequence': 1, 'normal_complete_count': 0, 'observed_returncode': None, 'operation_deadline_ns': 20000100, 'pending_ack': None, 'state': 'TERMINATING', 'terminal_deadline_ns': 5000100, 'terminal_reason': 'request_out_of_order'}
```

For durable reproduction if companion scripts are not archived, the essential scheduling procedure is: construct actual configured socketpair/core/transport; use `inspect.getsourcelines` to locate the exact markers above; install a SIGUSR1 handler that invokes the specified public nested call and catches GuardProtocolError; use `sys.settrace` solely to raise that signal once at the target line, immediately disabling tracing; resume outer public call; assert the observed state/wire response. R1 serves the other socket in a normal select loop in its owner thread, completes both initial and subsequent real waits, and asserts the supervisor recorded two completions. R2 supplies a constant valid clock100 and asserts the received full frame equals the pending ARMED response and evidence contains no ack_written. The complete executable companion sources with verified hashes are available at the paths above.

4. Independent original supervisor counterexample probe ran through a single-quoted PowerShell here-string piped to `python -B -`. It injected public `core.fault("parent_lost")`, catching its nested error, at `_append`'s setdefault and `_invoke`'s release line, then required TERMINATING/deadline/replay. A separate constructor clock called and caught nested `core.accept`; outer ARM was rejected, arm count stayed zero and kill became required. Results are recorded in the closure section.

5. Read-only identity/integrity checks included `git rev-parse HEAD`, quoted `git rev-parse 'HEAD^{tree}'`, `git branch --show-current`, `git status --porcelain=v1`, base diff stat/name-status, `git diff --check fcb9608... HEAD`, current repair diff from `9b8bc3b`, cleanup diff, UTF-8 source reads, scoped `rg`, and design/repro SHA-256. Diff check emitted no whitespace errors. No formatter/linter was independently run; author lint claims are not adopted as reviewer evidence.

The supplied primary-host SSH timeout was a handoff fact, not an independently reexecuted command in this session. This reviewer used the owner-authorized fallback and verified its actual clean exact revision. No environment/toolchain failure prevented the required CPU tests or repros.

## Limitations and scope boundary

Finite tests and these targeted interleavings are not exhaustive asynchronous proofs. Kill-error custody is exercised at core level; a real permission-denied OS kill is not induced by the process matrix. Sequence capacity is checked through actual primitives and arithmetic, not allocation of billions of billions of events. The functioning/scheduled supervisor assumption remains explicit. No arbitrary hostile-Python isolation or unsupported interpreter guarantee is asserted.

No GPU/native timeline, guardian completeness or ownership safety has been tested or approved. No candidate edit, install, push, merge, result/baseline modification, runtime adoption, worker gate change or host payload/shipping action occurred.

Approved implementation scope: **none yet**. Repair R1/R2 and obtain fresh independent review of the next exact revision. Existing design approval still permits only the inert stdlib CPU protocol/client/supervisor core, fixed standalone process regression harness/tests and cleanup classification. Closing the original supervisor defect does not independently approve the broader candidate.

C0 A/K and C1 S/V/E remain unchanged. C4c2 integration and its separate complete design, workload/numeric camera materialization, native setup/lifecycle/source-policy/runtime-manifest adoption, actual Warp timeline and R0/R1 lease graph, GPU preparation/submission, C4c3/R2, C4c4, C4d/B3, a3b, results/baselines, host payload/shipping, backend selection and product defaults remain closed. No parent C4c/C4 completion or broader approval follows from the passing CPU runs.
