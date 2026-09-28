# C4c2b supervised waits — independent design review

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: review
Author: codex (fresh independent reviewer)
Version: v1
Date: 2026-09-29
Status: in_review
Related Files: C4c2b design-request v1; C4c arena design v7; C4c1 ABI design v3; C4c2a design-request and code-review v1
Owner Summary: Request changes on two bounded design gaps: supervisor ACK-output lifecycle/deadlines, and the exact evidence/replay contract. The CPU-only prerequisite boundary is acceptable in principle; this review does not approve runtime integration or any GPU/result gate.

## Identity and verdict

- Reviewer session: `/root/c4_c2b_wait_design_review_v1`, fresh independent session without candidate authorship. No delegated agents.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `design/c4-supervised-waits`.
- Base: `0da32dbf733e7da483db867a7fcd59595b510e61`.
- Frozen HEAD: `ddf85fcf8e757dbd7d98bcb49f58dd8bca333af1`.
- Candidate: `collab/q54-optical-lab-p12-4c4c2b-supervised-waits__design-request__codex__v1.md`.
- Independently verified source-byte SHA-256: `037bd4c5aafb2ae3979fa141c3523cea7444c75ce53295144864792835dc9558`.
- Base-to-HEAD diff contains only that new 291-line document. Working tree was clean at entry and final verification; no candidate edits or generated candidate files.

**Verdict: Request changes.** These are design-contract gaps, not claims that an unimplemented candidate already fails a runtime test. Neither requires expanding this slice into GPU integration.

## Findings

### F1 — High: disarm ACK transmission loses its explicit watchdog bound

Location: candidate lines 149–173, especially 155–158 and 164–166; acceptance lines 221–227.

The supervisor commits IDLE and retires the normal-operation deadline before producing DISARMED. It enters TERMINATING if delivery fails, but the design gives no supervisor-side absolute ACK-write deadline or pending-response state. The only explicit shared send/receive ACK deadline belongs to the client. The instruction that supervisor writes cannot block beyond the active deadline is insufficient after COMPLETE has removed that deadline.

Counterexample to the currently frozen transition contract:

1. A normal wait is armed and the fixed child reports COMPLETE before expiry.
2. The core commits IDLE and clears the normal deadline.
3. The DISARMED output remains pending (`would block`, partial-write stall, or an injected transport stall), while the parent remains alive and the worker has not exited.
4. There is no active operation/terminal deadline. No transport error or EOF need arrive. A nonblocking event loop can keep polling indefinitely without violating the specified state/deadline transitions.

An ordinary tiny ACK will often fit in an empty socket buffer; this finding does not claim that every stopped reader immediately blocks that write. The problem is the missing contract for the partial/stalled-write fault the acceptance matrix itself requires. It must also say whether another buffered ARM/COMPLETE may be accepted while an earlier response is pending; the enumerated states presently allow interpretations with different queueing and failure behavior.

Consequence: independent implementations can satisfy the prose's normal COMPLETE behavior yet disagree on bounded shutdown after disarm-ACK loss. The worker's intended ACK-timeout hard exit is not a substitute for the supervisor's independent deadline, particularly in a fixed fault child that stops progressing during completion. Tests of a stalled ARMED response retain an operation deadline and would not expose this distinct gap.

Required direction:

- Freeze a pending-response identity/phase in the core or a precisely specified transport adapter, including when that obligation is created and retired.
- Give every supervisor ACK output one non-resettable absolute send deadline, with an explicit source budget and start point; partial progress must not extend it.
- Define how that deadline composes with the operation and sticky terminal deadlines, including COMPLETE's normal-deadline retirement. Expired or failed DISARMED output must enter an irreversible terminal state with a specified bounded kill schedule.
- Specify whether incoming frames are rejected or withheld while an ACK is pending, bound any buffers, and prohibit queued responses from escaping after terminal entry.
- Freeze the client COMPLETE exchange failure disposition as explicitly as ARM's, including no retry and the fixed driver's hard-exit behavior.

Proof required after repair: deterministic clock/transport tests for DISARMED output stuck after COMPLETE has retired the operation deadline, partial writes that do not reset budgets, parent loss during that interval, no new ARM acceptance/response leakage contrary to the chosen rule, and a bounded subprocess fault demonstrating eventual kill/reap with parent alive. Assert actual states and exact earliest deadlines, not only an exception or eventual test timeout.

### F2 — Medium: replay evidence is not yet a frozen, independently implementable schema

Location: candidate lines 75–78, 100–103, 119, 175–180, 189–203, and 232–233.

Section 7 lists categories of facts and desired rejection properties, but does not freeze the schema/version literal, exact field sets and types, event enum values/payloads, timestamp unit/domain, terminal/deadline disposition enums, or legal ACK-transport event ordering. These are the production deliverable, not unrelated implementation details. For example, it is not yet possible to tell from the document alone whether `arm commit -> normal completion commit` with no ACK-transport outcome is a valid trace, how a pending ACK appears in a snapshot, or which event proves that a kill was attempted rather than merely that the deadline expired.

There are concrete boundary conflicts left for the implementer to decide:

- ARM consumes a sequence in `0..2^63-1`; evidence stores `next sequence`. After the maximum legal ARM, the natural increment is `2^63`, which the required existing canonical codec rejects. The document does not specify an exhaustion sentinel/state or the following API behavior.
- Evidence timestamps are only called monotonic. The existing canonical codec forbids floats, so directly recording `time.monotonic()` is invalid; units, representation, bounds and deadline arithmetic must be fixed.
- Configuration allows any positive Python integer timeout, whereas canonical integers are signed 64-bit and a chosen millisecond-to-nanosecond conversion may need additional checked bounds.
- The design says to record “failed to be reaped,” but also says to keep the loop alive until status is collected and gives only EXITED as the observed-exit state. No observation/cutoff or evidence meaning for an unreaped process is defined. A missing exit status must never be presented as confirmed process exit.

Consequence: a validator and producer can agree with each other while selecting materially different safety/evidence meanings. The broad instruction to mutate each evidence field does not establish an external oracle for those meanings. Existing codec reuse settles JSON syntax, not the missing state contract.

Required direction: add a compact normative schema and transition/event table. Freeze initial sequence, exact identity/config fields, optional/null rules, timestamp unit and numeric bounds, event-specific exact fields, ACK obligations, deadline creation/removal/expiry order, kill-request and actual-exit observations, terminal reason precedence, sequence exhaustion, and incomplete/unreaped evidence. Define whether snapshots are permitted and how they differ from complete final evidence. Reuse one transition authority where practical, but make the specification precise enough to check it independently. Keep the existing honest limitation that replay checks consistency, not cryptographic authenticity or GPU completion.

Proof required after repair: explicit legal and illegal trace fixtures with independently stated expected states/deadlines/counts; missing/reordered ACK-outcome events; fabricated kill/exit summaries; maximum sequence followed by exhaustion; valid/invalid time/config limits and equal timestamps; pending-response snapshots; and unreaped/unknown-exit evidence. Recomputed hashes must not rescue structurally inconsistent traces. Tests need not reject a wholly rewritten but internally valid trace as authenticated forgery—the document explicitly does not claim authentication.

## Scope, predecessor consistency and inspected high-risk areas

Read `AGENTS.md`, `collab/README.md`, and the full `collab/REVIEW_PRINCIPLES.md` as UTF-8. Inspected the C4c2b candidate in full, C4c2a design and code-review v1, C4c1 ABI v3, and relevant C4c arena v7 lifecycle, evidence, implementation-slice, cleanup and gate requirements. Truncated combined command output was supplemented with targeted reads for the relevant lifecycle and slice sections.

The C4c2a section 8 requirement remains in force: a complete integration design is required before subsequent runtime implementation. This proposal explicitly limits delivery to two inert, stdlib-only production owners plus a standalone regression harness, forbids changing/adopting them into the product worker and C0 manifest, and enumerates the missing integration dependencies in section 9. A separately reviewed CPU prerequisite can therefore proceed without pretending the integration design is complete. C4c arena v7 also explicitly permits a CPU IPC skeleton before GPU enablement. No finding demands early materialization, source-policy or GPU work here. Any actual worker/native-setup adoption would exceed this boundary and require the outstanding integration design and independent implementation review.

Code facts independently checked:

- `optical_worker/protocol.py` provides stdlib-only canonical JSON, duplicate-key rejection, signed-64-bit integer restrictions, four-byte big-endian framing and configurable body limits. It can be reused without changing the product protocol.
- `tools/optical_closed_worker_supervisor.py` implements one-shot A/K abort supervision; its guard deadline remains armed until process exit. It does not provide repeated normal waits.
- `tests/support/sacrificial_watchdog.py` is a separate test-only S/V/E protocol. It issues exact-object tickets and disarms only S. It must not become a product import or be relabeled equivalent to the proposed new protocol.
- `native_setup.py` directly synchronizes readiness events and final target barriers. Final barrier exception handling passes another synchronization callback into `_abort`; `NativeWorkerSetupLifecycle.terminate()` can invoke it after the notifier. Thus the claimed failed-wait/no-retry integration gap is real, and this slice correctly does not claim to repair it.
- The proposed exact-ticket identity, ACK-before-leaf ordering, no callback/executable control payload, private inherited link, thread ownership, original-deadline retention after failed leaf, no second emergency retry, sticky terminal deadline and no GPU meaning for CPU evidence are sensible constraints. The findings concern missing closure around them, not replacing them.
- Parent loss cannot be cleared by a later normal COMPLETE under the specified sticky-deadline rule. Late buffered COMPLETE must lose to expiry. These rules should be retained explicitly in the repaired ACK-state table.
- Cleanup classifies the common core as shared safety infrastructure pending adoption and forbids counting the new and historical watchdog protocols as one version. The harness remains a protocol regression harness, not an alternate product supervisor.

## Independent verification and limitations

Commands executed locally in the frozen candidate:

```text
git status --short
git rev-parse HEAD
git branch --show-current
git diff --stat 0da32dbf733e7da483db867a7fcd59595b510e61 HEAD
git diff --check 0da32dbf733e7da483db867a7fcd59595b510e61 HEAD
Get-FileHash -Algorithm SHA256 <candidate design>
rg --files collab
rg -n <targeted lifecycle/protocol patterns> <inspected files>
Get-Content -Encoding UTF8 <instructions, predecessors, candidate and source>
python -B - <read-only canonical codec boundary probe>
```

Observed: expected clean branch/HEAD, one-document delta, matching digest, and no whitespace errors. The codec probe actually called `canonical_json_bytes`: `sequence=2^63-1` was accepted; `next_sequence=2^63` raised `ClosedWorkerProtocolError` for the signed-64-bit limit; floating timestamp `1.25` raised the same error for forbidden float type. This demonstrates existing dependency constraints, not a failing implementation of the unbuilt core.

No new runtime code exists in this candidate. No GPU tests, remote execution, benchmark, baseline/result access or product activation were needed or performed. No candidate imports of Torch/Warp occurred. No files were edited inside the candidate; this report was created outside it using `apply_patch`.

The author later reported a 50-pass existing C0/C1c regression run on the matching fallback-host design tree. That report is supplemental author information and was not independently executed or used to resolve either design finding. Prior approvals likewise establish historical scope only, not proof for this new design.

## Approval boundary

No implementation approval is granted for v1 pending F1/F2 repair and fresh independent review. The acceptable intended scope remains only the CPU production protocol/client/supervisor transition core and bounded subprocess regression harness.

C4c2 runtime integration, camera numeric resolution, source-policy/native-setup/worker adoption, actual Warp timeline and R0/R1 lease graph, GPU preparation/submission, C4c3/R2, C4c4, C4d/B3, a3b, results/baselines, host payload/shipping, backend selection and product defaults all remain closed. The unresolved camera resolver must not be invented, substituted or changed to complete this CPU slice.
