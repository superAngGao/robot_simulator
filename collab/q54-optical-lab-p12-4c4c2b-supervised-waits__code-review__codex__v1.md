# C4c2b guarded waits — independent code review

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: review
Author: codex (fresh independent reviewer)
Version: v1
Date: 2026-09-29
Status: in_review
Related Files: C4c2b design-request v3; independent design review v3; six-path candidate delta
Owner Summary: Request changes. Independent CPU regressions pass, but same-thread supervisor reentry can overwrite a committed terminal state, clear its deadline, and corrupt replay. No runtime integration or GPU gate is approved.

## Identity, scope and verdict

- Reviewer session: `/root/c4_c2b_wait_code_review_v1`; fresh review session, no candidate authorship and no delegated agents.
- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Local branch: `implementation/c4-supervised-waits`.
- Base: `fcb9608a916d20d1c43f64a3401b3c78d823404f`.
- Exact reviewed HEAD: `5dd04f85731ebe4089bdfc01f507bd75a67761ec`.
- Remote independent execution tree: `/data3/ang.gao/robot_sim_c4_c2b_code_5dd04f8`, detached HEAD at that exact commit, on permitted fallback `ang.gao@192.168.80.101`.
- Local and remote `git status --porcelain=v1` were empty at inspection and after verification. No candidate/source edits were made. This report is outside the repository. Normal ignored interpreter/test caches are not asserted absent.
- Delta: new `optical_worker/wait_protocol.py`, `optical_worker/wait_supervision.py`, `tests/support/guarded_wait_process.py`, `tests/unit/optics/test_guarded_wait_core.py`, `tests/unit/optics/test_guarded_wait_process.py`, and a 23-line addition to `benchmarks/results/q54_p12_4/cleanup_ledger.json`; six files, 1,696 insertions.
- Requirements read: complete `AGENTS.md`, `collab/README.md`, `collab/REVIEW_PRINCIPLES.md`, v3 design and archived independent v3 design review. Initially garbled Chinese output was reread with UTF-8; truncated combined design output was supplemented with targeted reads.
- Design SHA-256 independently checked: `a3619ead4e22ad47f3133faca7788cdb85d0816f2bad8cffd8a46998fc80c872`, corresponding to approved design revision `ff3b4db0d4c3812c7149c97cb9cf9bb095598461`.

**Verdict: Request changes.** One substantive finding, R1 below. Passing test totals do not close its independently reproduced counterexample.

## R1 — P1: reentrant supervisor entry can erase terminal custody/deadlines

Locations: `optical_worker/wait_supervision.py:327–350`, especially the transition/commit gap at lines 329–333 and the clock-only rejection at line 343. Affected entry example: `acknowledge_written`, lines 418–433. Existing coverage: `tests/unit/optics/test_guarded_wait_core.py:423`.

The core checks thread identity, but has no guard spanning an entire public mutation. `_reading_clock` protects only the call to the clock. `_append` computes an updated state, then separately appends its event and assigns the previously computed state. A same-thread asynchronous callback can reenter between those steps.

Concrete trigger, independently reproduced locally and on Linux:

1. ARM a normal setup wait and record its ARMED write.
2. Accept COMPLETE, entering DISARM_ACK_PENDING.
3. During the DISARMED `acknowledge_written`, after `_transition` computed IDLE but before `_trace.append(event)`, a same-thread signal handler calls the public `core.fault("parent_lost")`.
4. The nested call successfully commits TERMINATING with effective deadline 5,000,100 ns.
5. The outer call resumes and overwrites that state with its stale IDLE result, clearing the effective deadline. The trace now has inconsistent indexes and cannot replay.

Observed Linux Python 3.12.13 output:

```text
python 3.12.13
nested terminal state [('TERMINATING', 5000100)]
outer final state IDLE deadline None
evidence error GuardProtocolError trace index drift
```

Consequence: terminal state is not monotonic under the explicitly promised same-thread reentrancy contract. Parent loss can lose its termination deadline, the core can accept subsequent normal work, and its evidence becomes invalid. The example does not write private state, replace candidate methods, forge tickets or rely on hostile arbitrary-Python memory access. A tracing hook deterministically schedules an actual signal; the signal handler invokes the supported public fault API. This is fault injection for reentrancy, not a claim that production currently installs that handler. The new core remains inert today, but this defect blocks its promised bounded core contract.

Complete Linux reproduction, run from the exact remote candidate with the `robot_sim` interpreter:

```python
import signal, sys
from optical_worker.wait_protocol import GuardConfiguration
from optical_worker.wait_supervision import GuardSupervisorCore

config = GuardConfiguration('a' * 32, 1, 5, 20)
core = GuardSupervisorCore(config, clock=lambda: 100)
core.acknowledge_written(core.accept(config.message('ARM', 0, 'setup')))
reply = core.accept(config.message('COMPLETE', 0, 'setup'))
nested = []

def on_signal(signum, frame):
    core.fault('parent_lost')
    nested.append((core.state, core.effective_deadline_ns))

old = signal.signal(signal.SIGUSR1, on_signal)

def schedule(frame, event, arg):
    if (event == 'line' and frame.f_code.co_name == '_append'
            and frame.f_locals.get('operation') == 'ack_written'
            and frame.f_lineno == 332):
        sys.settrace(None)
        signal.raise_signal(signal.SIGUSR1)
    return schedule

sys.settrace(schedule)
try:
    core.acknowledge_written(reply)
finally:
    sys.settrace(None)
    signal.signal(signal.SIGUSR1, old)
print('python', sys.version.split()[0])
print('nested terminal state', nested)
print('outer final state', core.state, 'deadline', core.effective_deadline_ns)
try:
    core.evidence()
except Exception as exc:
    print('evidence error', type(exc).__name__, str(exc))
assert nested == [('TERMINATING', 5000100)]
assert core.state == 'IDLE' and core.effective_deadline_ns is None
```

A second simpler reproduction uses only the documented constructor clock seam. A clock that catches the nested-entry exception permits the outer call to continue and commit a normal ACK; thus the attempted clock reentry is not sticky either:

```python
from optical_worker.wait_protocol import GuardConfiguration, GuardProtocolError
from optical_worker.wait_supervision import GuardSupervisorCore
config = GuardConfiguration('a' * 32, 1, 5, 20)
core = None
nested_errors = []
def clock():
    if core is not None:
        try:
            core.accept(config.message('ARM', 0, 'resolve'))
        except GuardProtocolError as exc:
            nested_errors.append(str(exc))
    return 100
core = GuardSupervisorCore(config, clock=clock)
reply = core.accept(config.message('ARM', 0, 'setup'))
print(nested_errors, reply, core.state)
core.acknowledge_written(reply)
print(core.state, core.evidence().to_payload()['trace'])
```

Observed: two `reentrant supervisor clock entry` exceptions were caught; outer `accept` returned ARMED in ARM_ACK_PENDING; acknowledgment committed ARMED_NORMAL with only initialized/arm_committed/ack_written events and no failure.

Why existing checks miss it: the reentrancy test reenters only while reading the clock and lets the inner exception escape. That exception is then caught by the outer `_now` and converted to a clock fault. It neither catches the inner error nor reenters during transition commitment. Thread identity does not reject same-thread entry. There is no operation-level sticky invalidation check before the stale state assignment.

Direction: guard complete externally callable mutations/transport service against reentry, distinguish legitimate internal method composition from external nested entry, and make a detected nested failure sticky even if a callback catches the exception. An outer in-flight operation must not overwrite a terminal state or emit/commit a success ACK after invalidation. Preserve the first terminal reason and earliest deadline and keep evidence replayable. This is within the accepted fail-closed contract, not permission to add a hostile-Python security layer or broaden the wire protocol.

Required proof after repair: independently rerun both original counterexamples (adjust the deterministic precommit injection location to the changed source), test analogous ARM/DISARM/terminal/kill/exit mutation boundaries, show terminal monotonicity and no subsequent normal acceptance/output, preserve bounded kill obligation and replayable incomplete evidence, and rerun the existing CPU/process/C0/C1 regressions. A fix must not merely remove the test clock hook or rely on callers propagating every inner exception.

## Verified high-risk behavior and independent tests

Inspected all six changed paths, the shared canonical codec, actual client call/IO flow, supervisor transition/replay authority, and fixed process loop. Positive observations are bounded to this revision and do not override R1:

- Configuration is reconstructed into an exact frozen dataclass. Identity, sequence and integer domains are checked; tickets are frozen and COMPLETE requires the outstanding object's identity. Invalid/copy/stale/emergency client use is sticky. Cross-thread calls reject before mutation/write.
- Client ARM and COMPLETE share one absolute send-plus-receive budget; readable ACKs lose at deadline equality. No blocking callable crosses the wire or is executed by the client.
- In ordinary non-reentrant flow, ARM commits deadlines before a response, DISARM_ACK_PENDING retains the original operation deadline, and only full recorded DISARMED output returns to IDLE/increments completion count. Parent loss folds the earlier pending/operation deadline before cancellation.
- Transport uses nonblocking sockets and bounded input/output. Complete requests received while an ACK is pending are faults. Terminal service drops pending output; partial progress does not reset deadlines. Real subprocess tests exercise actual OS backpressure, not just a fabricated state flag.
- Expiry, kill request, kill failure and observed exit are distinct; kill request alone cannot finalize evidence. Harness uses actual `poll`/`wait`, and its outer test safeguard is not reported as successful watchdog evidence.
- Evidence replays the same transition grammar, checks exact fields/index/time/identity/summary, and rejects malformed summaries and illegal event sequences. It is consistency evidence, not authentication or authority to mint a live ticket.
- Capacity calculation is consistent: G=2,305,843,009,213,693,949; S=G−1; maximum `1+4*G+6` events is 9,223,372,036,854,775,803. Final index and counts fit signed-64-bit canonical encoding. Actual checked-add/next-sequence primitives reject their first invalid values. The finite proof does not assert construction of an enormous trace.
- Cold-import testing blocks Torch/Warp/NumPy/test/tool/benchmark/example import attempts. Scope search found no adoption imports in existing worker/tool runtime paths. The six-path Git delta leaves C0/C1 runtime, source-policy and gate surfaces unchanged. Cleanup ledger identifies shared safety infrastructure pending separately reviewed adoption, with removal if no selected consumer remains.

Independent executions:

1. Windows PowerShell, Python 3.14.0, frozen local tree:

   `python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_guarded_wait_core.py`

   **72 passed in 1.35s.**

2. Permitted fallback SSH, fresh clean detached candidate, Linux `robot_sim` Python 3.12.13:

   ```text
   ssh -o BatchMode=yes -o ConnectTimeout=10 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 "cd /data3/ang.gao/robot_sim_c4_c2b_code_5dd04f8 && git rev-parse HEAD && git branch --show-current && git status --porcelain=v1 && /data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/benchmarks/test_optical_benchmark_contracts.py::test_inventory_and_cleanup_ledger_match_current_repository"
   ```

   **133 passed in 16.20s**, exit 0. This is the reviewer's execution, not the author's log. It includes ten real subprocess fault cases, new unit tests, unchanged C0 protocol/worker regressions, C1 watchdog regressions and the cleanup inventory check.

3. The two R1 reproductions ran through PowerShell here-strings piped to `python -B -`; the first local scheduling probe initially used nonexecuted closing-parenthesis line 331, so did not inject. Corrected line 332 reproduced the defect. The complete SIGUSR1 reproduction above was separately piped over SSH to `conda run --no-capture-output -n robot_sim python -B -`; it reproduced the same lost terminal state under Python 3.12.13, exit 0 (assertions confirm the bug, not acceptance).

4. Additional independent local transport/budget probes, exit 0:

   - Real socketpair plus a send proxy restricting each send to one byte; arm/ACK at 100 ns, COMPLETE at 19,000,100 ns. Tested both direct expiry and intervening parent loss. One DISARMED prefix byte was observed; the original 20,000,100 ns deadline remained. At equality, state was TERMINATING with kill required, completion count zero, empty transport output buffer, and no further peer-readable bytes. This deliberately injects partial sends around the actual transport implementation; it is not represented as a real OS naturally short-writing a tiny frame. Real OS backpressure is separately covered by the Linux harness.
   - Actual client socketpair with clock/readiness injection at 0, 0, 4,000,000 and 5,000,000 ns for a 5 ms ACK budget. ARM bytes were sent, but receive timed out without a second readiness wait; client was FAILED. This directly checked the shared absolute budget rather than granting a fresh receive budget.

5. Identity/integrity commands included `git rev-parse HEAD`, `git branch --show-current`, `git status --short`, `git status --porcelain=v1`, `git diff --stat fcb9608... HEAD`, `git diff --check fcb9608 HEAD`, ledger diff, scoped `rg`, source reads, design `Get-FileHash -Algorithm SHA256`, and local/remote `python --version`. Expected revision/tree status was observed; `git diff --check` emitted no errors. Final remote/local status checks were clean.

No environment or toolchain failure prevented required CPU execution. The primary-host timeout was supplied in the handoff; this reviewer used the already-authorized fallback and verified its independent tree instead of treating that handoff as execution evidence. The old dirty remote draft and author regression log were not used as proof.

## Coverage limitations and remaining gates

The installed tests do not exhaustively mutate every possible field/value of every event or independently fixture every terminal transition; passing them is not a universal proof. Additional independent probes checked specific missing budget/partial-send cases. Sequence capacity is checked arithmetically and through actual primitives, not by allocating an infeasible full-size trace. Kill-error custody is exercised at the core level; inducing a real OS permission-denied kill was not part of the real subprocess matrix. CPU sentinel results prove destructor/atexit process behavior only, not CUDA safety or guardian completeness.

No candidate edit, install, push, merge, result/baseline change, GPU import/preparation/sample, host payload or gate mutation occurred. No GPU/native runtime test is claimed or required to validate this inert CPU slice. Functioning/scheduled supervisor remains an explicit assumption.

Approved implementation scope: **none yet**, pending R1 repair and fresh independent review of the resulting exact revision. The existing v3 design approval remains a bounded authorization to implement the two inert stdlib protocol/client/supervisor modules, unit tests, fixed standalone subprocess harness and specified cleanup classification; this review does not rescind or extend that design decision.

C0 A/K and C1 S/V/E adoption/semantics remain unchanged. C4c2 runtime integration and its separate complete design, materialization/numeric camera resolution, worker/native-setup/lifecycle/source-policy/runtime-manifest adoption, actual Warp timeline and R0/R1 lease graph, GPU preparation/submission, C4c3/R2, C4c4, C4d/B3, a3b, results/baselines, host payload/shipping, backend selection and product defaults remain closed. No C4c/C4 completion or inherited broader approval follows from the CPU test results.
