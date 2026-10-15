# C4c2b supervised waits — independent design re-review

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: review
Author: codex (fresh independent reviewer)
Version: v3
Date: 2026-09-29
Status: accepted
Related Files: C4c2b design-request v3 and reviews v1/v2; C4c arena design v7; C4c1 ABI design v3; C4c2a design-request and code-review v1
Owner Summary: Approve the exact v3 CPU-only protocol-core design. The revised capacity closes the count/index counterexample, and the pending-DISARMED deadline closure remains intact. This permits bounded implementation, not runtime adoption, GPU preparation, parent-stage completion or any result gate.

## Identity and verdict

- Reviewer session: `/root/c4_c2b_wait_design_review_v3`, fresh independent session without candidate authorship or delegated agents.
- Candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `design/c4-supervised-waits`.
- Base: `3e78b481ab27db36330a9b7379c61b4e2bcecfb5`.
- Frozen HEAD: `ff3b4db0d4c3812c7149c97cb9cf9bb095598461`.
- Design: `collab/q54-optical-lab-p12-4c4c2b-supervised-waits__design-request__codex__v3.md`.
- Independently verified source-byte SHA-256: `a3619ead4e22ad47f3133faca7788cdb85d0816f2bad8cffd8a46998fc80c872`.
- Base delta: only the new 464-line v3 design and 106-line archived v2 review. No executable source or test changes.
- Archived v2 review and the outside-tree original both hash to `0c93823ea1037b05047f80a3575a3da7c93c55400f2a443922f7a3b562f10947`; the archive is verbatim.
- Working tree was clean at entry and final inspection. No candidate edits, caches or generated files were made. This report is outside the candidate.

**Verdict: Approve.** No substantive blocking or non-blocking finding remains for this exact design and bounded delivery. This is design approval only; no unimplemented behavior is reported as a passed runtime test.

## Independent closure of prior findings

### V2 R1 — Closed at design level

Section 7, especially lines 298–349, now binds every guard sequence, active/pending identity and summary next-sequence to the same narrower `Seq` domain. Let `M = 2^63 - 1`, `G = (M - 8) // 4`, and `S = G - 1`. The independent arithmetic probe gives:

```text
G = 2305843009213693949
S = 2305843009213693948
maximum events = 1 + 4*G + 6 = 9223372036854775803
maximum final index = 4*G + 6 = 9223372036854775802
M - maximum final index = 5
```

Consuming sequence S now requires exactly G ARM events, not M+1. Both derived counts are at most G. Every completed normal wait contributes exactly four events. An unfinished normal wait contributes at most three before a terminal path; an emergency contributes at most two and cannot return to normal operation. A terminal interruption therefore cannot add another cycle of ordinary events. The six listed tail events are individually single-use; repeated faults, ticks and bad clocks cannot enlarge the tail. Reserving a full four-event contribution per accepted ARM is conservative for both partial and emergency paths.

All advertised maxima fit the existing canonical signed-64-bit codec, including initialization, last-ticket completion and the full terminal/kill/error/exit tail. The null sentinel after S avoids an out-of-domain next sequence. Later ARM is terminal even if it sends zero; the client rejects locally before sending another ARM. The design prohibits wrap, saturation and trace truncation, requires checks before transition/ACK commitment, and rejects a high-sequence short trace as invalid history. A cap need not be the mathematically largest possible one to be consistent; the explicit remaining headroom is harmless.

I rechecked both old counterexamples: M+1 ARM events overflow the old count, and `4 * 2^61 = M+1` overflows the old completed-wait event index. Both sequences are beyond the new session capacity. The new protocol restriction does not modify the existing product protocol: an actual existing `validate_request` call still accepts sequence M for its unchanged CLOSE_SESSION envelope.

The proof is a numeric/grammar proof, not an attempt to allocate a billions-of-billions-element trace. Implementation must test its actual checked primitives and normal replay fixtures as specified; it may supplement those with a reduced-capacity mathematical model, not replace the production implementation with that model.

### V1 F1 — Closure independently retained

Reapplying the original trigger—successful normal leaf, COMPLETE accepted, then indefinitely partial/stalled DISARMED output with parent alive—leaves the supervisor in DISARM_ACK_PENDING. Sections 5–6 retain the original operation deadline and add one absolute ACK-send deadline. Only a fully written ACK whose outcome is committed retires both and returns to IDLE. Effective expiry is their minimum and wins over late buffered COMPLETE or output success. Partial progress never extends either budget.

Parent loss during either pending ACK folds the earlier deadline into sticky termination before cancellation; it cannot clear or extend protection. A complete next request before ACK-write commit is a fault, not queued normal work. Pending transport buffers are bounded. Terminal entry discards unsent output; a fully written ACK with failed outcome recording still retains the guard and terminates. Client COMPLETE failure has explicit no-retry, no-second-emergency and hard-exit/custody behavior. The v1 unbounded disarm-output interval is therefore no longer allowed by the design.

### V1 F2 — Closed at design level

The exact configuration, wire identity, initialization, event payloads, reason enums, deadline tie order, derived summary, snapshot/final distinction and replay rules are now frozen. The remaining numeric inconsistency identified by v2 is closed as above. Evidence distinguishes expiry, a kill attempt, kill failure and actual process exit. Neither an absent return code nor a caller-provided claim can finish evidence. Parsed evidence cannot mint a live ticket or reopen a session; a completely rewritten legal trace is correctly not represented as cryptographically authenticated.

## State, evidence and dependency checks

Read the complete repository instructions (`AGENTS.md`, `collab/README.md`, `collab/REVIEW_PRINCIPLES.md`), candidate and prior reports; read C4c2a design/code-review v1 and C4c1 ABI v3; inspected C4c arena v7 lifecycle, implementation-slice, cleanup and closed-gate sections. Truncated combined source/predecessor output was supplemented by targeted reads. The initial root-level REVIEW_PRINCIPLES lookup failed because the file is under collab; the correct file was then read in full. No rule was inferred from that failed read.

High-risk paths checked against the normative text and actual dependencies:

- IDLE → ARM_ACK_PENDING → ARMED_NORMAL → DISARM_ACK_PENDING → IDLE preserves one active identity and one ACK obligation. Counts advance only on arm commitment and recorded successful DISARMED write. Failed normal leaves never send COMPLETE; they retain the original guard.
- Emergency arming is IDLE-only and terminal-role completion is forbidden. A failed normal wait cannot create a second emergency guard. Removing the unused client TERMINAL label leaves all defined client outcomes expressible; ARMED emergency can only fail/close/exit, not disarm to IDLE.
- Exact outstanding object identity, immutable tickets, session/ordinal/role/sequence echo, thread ownership and non-reentrancy prevent supported-path copying, cross-client reuse, nesting and stale completion. The text correctly does not claim hostile same-process Python isolation.
- Clock values are non-bool integer nanoseconds. Budget conversion and every timestamp addition are checked. A rejected clock is recorded once at the last valid timestamp; its immediate obligation is preserved on repeated bad readings and does not wait indefinitely for a new good clock. The existing first terminal reason is retained. Expiry/kill observations remain distinct and single-use.
- All seven supervisor states admit honest incomplete snapshots. Only actual poll/wait exit observation permits EXITED/final evidence. Kill failure or unavailable process status retains custody with incomplete TERMINATING evidence; no timeout fabricates exit or allows a restart.
- The production codec in `optical_worker/protocol.py` is stdlib-only, has the required canonical JSON/duplicate-key/integer checks and configurable four-byte framing. The new guard-specific field/domain/state checks are additional work; codec acceptance alone is not a guard validator.
- The current product supervisor actually uses one-shot A/K with its deadline retained until exit. `tests/support/sacrificial_watchdog.py` is a distinct S/V/E test-only protocol with exact-object tickets and S-only disarm. Neither supplies the new repeated normal-wait core and neither is changed here.
- Actual native setup performs direct readiness/final-barrier synchronization. Its barrier exception passes a synchronization callable into `_abort`, and `NativeWorkerSetupLifecycle.terminate()` can call that again after the emergency notifier. The future integration gap is real; this inert core does not claim to repair or adopt that path.
- C4c2a section 8 still requires a separate full runtime integration design. C4c2b section 9 is an unapproved dependency inventory, not an approved guardian/readiness ordering, camera resolver, source policy, Warp timeline or lease graph. Those decisions remain outside this verdict.

## Commands and observations

Executed locally with PowerShell and Python 3.14.0, from the frozen candidate. Principal identity/delta commands were:

```powershell
git status --short
git status --porcelain=v1
git rev-parse HEAD
git branch --show-current
git diff --stat 3e78b481ab27db36330a9b7379c61b4e2bcecfb5 HEAD
git diff --name-status 3e78b481ab27db36330a9b7379c61b4e2bcecfb5 HEAD
git diff --check 3e78b481ab27db36330a9b7379c61b4e2bcecfb5 HEAD
git diff --no-index collab/q54-optical-lab-p12-4c4c2b-supervised-waits__design-request__codex__v2.md collab/q54-optical-lab-p12-4c4c2b-supervised-waits__design-request__codex__v3.md
Get-FileHash -Algorithm SHA256 collab/q54-optical-lab-p12-4c4c2b-supervised-waits__design-request__codex__v3.md
Get-FileHash -Algorithm SHA256 'C:/Users/AG/新建文件夹/c4-c2b-wait-design-review-v2.md'
Get-FileHash -Algorithm SHA256 collab/q54-optical-lab-p12-4c4c2b-supervised-waits__review__codex__v2.md
python --version
```

Observed expected HEAD/branch, clean tree, matching hashes, two documentation additions and no whitespace errors. The v2/v3 comparison emitted Git's LF-to-CRLF working-copy advisory, not a content/test failure. Read-only `rg` and `Get-Content -Encoding UTF8` calls inspected the files and portions identified above.

The following exact Python input ran through a PowerShell single-quoted here-string piped to `python -B -` (no generated candidate bytecode):

```python
from optical_worker.protocol import canonical_json_bytes, decode_canonical_message, encode_framed_message, decode_frame_header, validate_request, ClosedWorkerProtocolError
M = 2**63 - 1
G = (M - 8)//4
S = G - 1
E = 1 + 4*G + 6
assert (G, S, E, E-1) == (2305843009213693949, 2305843009213693948, 9223372036854775803, 9223372036854775802)
assert G < M and E < M and E-1 < M
print('capacity', {'max_guards': G, 'max_sequence': S, 'events': E, 'last_index': E-1, 'spare_index': M-(E-1)})
for field, value in [('sequence', S), ('arm_count', G), ('normal_complete_count', G), ('index', E-1), ('at_ns', M), ('old_product_sequence', M)]:
    payload = {field: value}
    encoded = canonical_json_bytes(payload)
    assert decode_canonical_message(encoded, max_bytes=1024) == payload
print('six maximum-field canonical roundtrips passed')
for field in ('arm_count', 'normal_complete_count', 'index', 'at_ns'):
    try:
        canonical_json_bytes({field: M+1})
    except ClosedWorkerProtocolError:
        pass
    else:
        raise AssertionError(field)
print('four I64+1 codec rejections passed')
record = dict(protocol_version='q54.closed-worker.guarded-wait.v1', session='f'*32, sequence=S, device_ordinal=1023, role='emergency', operation='DISARMED')
frame = encode_framed_message(record, max_bytes=1024)
assert decode_frame_header(frame[:4], max_bytes=1024) == len(frame)-4
assert decode_canonical_message(frame[4:], max_bytes=1024) == record
print('largest-field wire shape bytes', len(frame)-4, '(shape/codec probe only, not a legal emergency DISARMED transition)')
validate_request(dict(protocol_version='q54.p12.4.closed-worker-protocol.v1', sequence=M, operation='CLOSE_SESSION', arguments={}))
print('existing product request maximum remains accepted')
for bits in range(4, 17):
    limit = 2**bits-1
    guards = (limit-8)//4
    for n in (0, 1, guards):
        assert 0 <= n <= guards
        assert 1+4*n+6 <= limit
        assert 4*n+6 <= limit
print('reduced-domain capacity formula passed for bit widths 4..16')
B = (2**31-1)*1_000_000
assert B == 2147483647000000 and B < M
assert (M-B)+B == M and (M-B+1)+B == M+1
assert 4*(2**61) == M+1
assert M+1 > M
print('timestamp addition boundary and old v2 count/index counterexamples confirmed')
```

Probe exit 0. All assertions passed; maximum-field frame body was 196 bytes, below 1024. The intentionally combined longest role/operation strings probe codec shape only, not a legal emergency DISARMED transition. New Seq first-invalid values remain codec-encodable and must be rejected by the future guard schema; no existing codec behavior is misrepresented as that unimplemented validation.

## Limitations, implementation obligations and approval boundary

No new runtime code exists in this candidate. No new supervisor/client execution, subprocess hang/reap test, GPU test or implementation acceptance is claimed. Existing unchanged C0/C1c runtime tests were not rerun for this documentation-only correction. No remote host, old remote v1 tree, author pass total or prior approval was used as proof of v3. No installation, benchmark, result/baseline access, push, merge or product activation occurred.

Implementation review must establish all section 8 obligations using the actual new core/client: nonblocking framed transport and deadline checks, actual checked arithmetic and replay, exact-ticket/thread behavior, real child ACK-before-leaf and bounded kill/reap, stalled DISARMED/partial-write/parent-loss cases, repeated bad clocks without starvation, kill-failure custody, finalizer sentinels, cold-import attempts and byte-identical old product/C1c surfaces. The local arithmetic probe is not a substitute for those tests. The supervision guarantee also retains the explicit assumption that the supervisor is functioning/scheduled; it is not hardware-fault tolerance.

Approved only: implementation of the two inert stdlib-only production protocol/client/supervisor-core modules, unit tests and a fixed standalone subprocess regression harness described in v3 at the exact recorded digest/HEAD. The cleanup classification must be recorded as specified; the harness is not a second product supervisor.

C4c2 runtime integration and its separate complete design, owner-pending numeric camera resolution, worker/native-setup/lifecycle/source-policy/runtime-manifest adoption, actual Warp timeline and R0/R1 lease graph, GPU preparation/submission, C4c3/R2, C4c4, C4d/B3, a3b, results/baselines, host payload/shipping, backend selection and product defaults all remain closed. No C4c/C4 completion or inherited broader approval follows from this verdict.
