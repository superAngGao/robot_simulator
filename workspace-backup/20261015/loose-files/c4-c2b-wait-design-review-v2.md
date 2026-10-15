# C4c2b supervised waits — independent design re-review

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: review
Author: codex (fresh independent reviewer)
Version: v2
Date: 2026-09-29
Status: in_review
Related Files: C4c2b design-request v2 and review v1; C4c arena design v7; C4c1 ABI design v3; C4c2a design-request and code-review v1
Owner Summary: The previous stalled-DISARMED safety counterexample is closed. The new schema resolves the main replay ambiguities, but its maximum-sequence promise conflicts with the newly frozen count/index domains. Request one bounded numeric-contract correction; no runtime integration or GPU work is requested or authorized.

## Identity and verdict

- Reviewer session: `/root/c4_c2b_wait_design_review_v2`, independent of candidate authorship, with no delegated agents.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `design/c4-supervised-waits`.
- Base: `ddf85fcf8e757dbd7d98bcb49f58dd8bca333af1`.
- Frozen HEAD: `3e78b481ab27db36330a9b7379c61b4e2bcecfb5`.
- Candidate: `collab/q54-optical-lab-p12-4c4c2b-supervised-waits__design-request__codex__v2.md`.
- Independently checked source-byte SHA-256: `92e4819a8e1a57aa8d5c43342b990f16c9635c674c2f76d0952ca5347c12ca94`.
- Base-to-HEAD delta: only the new 418-line v2 design and the 116-line archived v1 review. No executable source or test changes.
- Working tree was clean on entry and final inspection. The candidate was not edited; this report is outside it.

**Verdict: Request changes.** One medium-severity design-contract inconsistency remains. This is not an observed failure of an implementation: the proposed core does not yet exist. The finding concerns the explicit numeric boundary the design promises, not a claim that an ordinary process could practically run billions of billions of waits.

## Previous findings

### V1 F1 — Closed at design level

Sections 5–6 now distinguish `ARM_ACK_PENDING` and `DISARM_ACK_PENDING`; COMPLETE no longer clears the operation deadline. A DISARMED exchange retains the original operation deadline and adds a non-resettable send deadline. Only a complete write, recorded by the core, retires the guard and returns to IDLE.

Reapplying the original counterexample—normal COMPLETE followed by a indefinitely pending/partially written DISARMED response while the parent stays alive—now leaves an effective deadline equal to the minimum of the original operation and ACK deadlines. Expiry precedes input/output success, enters TERMINATING and requests a kill. A subsequent parent loss cannot extend that minimum. The client separately has one total send/receive deadline and a no-retry hard-exit disposition.

The one-response output bound, partial-input bound, rejection of complete input before ACK-write commit, terminal output cancellation, and full-write-but-unrecorded-outcome failure rule also settle the previous queueing ambiguity. Acceptance item 3 explicitly requires stalled DISARMED/partial-write/parent-loss/buffered-next-ARM controls; items 4–5 require real child-process kill/reap and finalizer evidence. These are future implementation-review obligations, not tests already passed by this document.

### V1 F2 — Substantially addressed, but not fully closed

Section 7 supplies exact root/config/identity/event/summary fields, version literals, operation/reason enums, integer-nanosecond times, checked budget arithmetic, ACK transport transitions, tie priority, sequence exhaustion, kill-attempt versus observed-exit meaning, and incomplete-versus-final evidence. A missing/reordered ACK is no longer an unspecified replay case. `kill_requested` without actual process status is explicitly incomplete; kill errors cannot fabricate exit. A legal rewritten trace is correctly described as internally consistent rather than authenticated.

The prior float timestamp, unbounded timeout and `next_sequence=2^63` examples are directly addressed. However, the added I64 counters and event indices introduce the related boundary inconsistency below. The null next-sequence sentinel alone does not make the promised maximum ARM representable as a complete session history.

## Finding R1 — Medium: maximum legal sequence has no representable evidence history

Location: v2 section 7, lines 240–242 (I64 event indices), 287–288 (I64 counts), 296–305 (maximum sequence consumption and exact derived counts); section 4's inclusive sequence domain. The existing codec constraint is in `optical_worker/protocol.py`, `_validate_canonical_value`.

Let `M = 2^63 - 1`. The design simultaneously requires:

1. Initialization at sequence 0, strict consecutive ARM consumption, and acceptance of sequence M, after which its ticket may still complete.
2. `arm_count` equal to the number of `arm_committed` events, in I64 (`0..M`).
3. Event indices starting at 0, increasing by one, also in I64; full replay from initialization rather than a truncated maximum-sequence fixture.

Consuming the promised last sequence requires exactly `M + 1 = 2^63` ARM events. The required `arm_count` therefore exceeds its schema domain. If all those normal waits complete, `normal_complete_count` also exceeds its domain. Nulling `next_sequence` does not repair either count.

The event index exhausts even earlier: one completed normal wait produces four events (`arm_committed`, ARMED `ack_written`, `normal_complete_received`, DISARMED `ack_written`). Including initialization at index 0, after N completed waits the final event index is `4*N`. At `N=2^61`, the required DISARMED event index is `2^63`, while the consumed sequence is only `2^61-1`, well within the advertised sequence range. There is no event-index exhaustion rule, capacity reservation for the required terminal/expiry/kill/exit observations, or legal wider representation.

Trigger and consequence: at these explicitly permitted boundaries, an implementation must either violate the I64 schema/canonical codec, wrap or falsify counts, silently reject earlier valid operations, or fail evidence recording without the specified replayable final history. The design gives the implementer no conforming choice that preserves all its promises. This is a formal boundary-contract defect; it is not evidence of an imminent practical watchdog failure.

Why existing checks miss it: the sequence checked-increment test allowed at lines 300–302 can prove `M -> null` in isolation while never checking the corresponding derived count or event-index obligations. Ordinary short legal/illegal fixtures exercise none of these boundaries. Timestamp-addition overflow rules do not cover counter or trace-index arithmetic.

Required direction: freeze a mutually consistent numeric/exhaustion policy for sequence, counts and trace indices. Either retain the full sequence domain and choose exact codec-compatible representations capable of expressing its whole history, or explicitly bound session operations earlier with a reviewed exhaustion transition and sufficient reserved evidence capacity for terminal/kill/exit observations. Do not silently saturate counts, truncate a trace, or redefine an exact count as a last sequence number. The chosen behavior must be stated in the design, including what happens before an ARM/ACK would exceed capacity.

Proof after repair: independent arithmetic/boundary tests for every incremented evidence field, including its largest valid value and first invalid value, with expected reject/terminal behavior and canonical encoding. Prove algebraically that the maximum promised operation sequence and its required terminal tail fit the selected domains. Use primitive boundary tests or a deliberately reduced-capacity test model; do not pretend an injected high-sequence short trace is a valid full-session trace. Normal fixtures must continue to replay exact counts and indices without overflow policy drift between producer and validator.

## Inspected safety and scope boundaries

Read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` fully; the Chinese review principles were reread explicitly as UTF-8 after the first PowerShell read rendered them incorrectly. Read the full candidate, prior v1 report, C4c2a design and code-review, and C4c1 ABI v3. Inspected C4c arena v7's implementation-slice, cleanup and closed-gate requirements. Truncated combined output was supplemented by targeted reads of the relevant source and predecessor sections.

Independently checked these actual dependency facts:

- The existing production codec is stdlib-only, rejects duplicate keys/noncanonical JSON/floats/out-of-signed-64-bit integers, and provides configurable-size four-byte big-endian frames. Guard-specific record validation must still be supplied by the new core; the codec alone is not its authority.
- The current product supervisor is one-shot A/K supervision with its deadline retained until exit, not a repeated normal-wait implementation.
- The separate S/V/E test watchdog issues exact-object tickets and disarms only S; it is not a production dependency or an equivalent version of this protocol.
- `native_setup.py` synchronizes directly, and failed final target barriers pass another synchronization callable into `_abort`; `NativeWorkerSetupLifecycle.terminate()` can invoke it after its emergency notifier. Thus future guarded-wait integration must still fix that retry path, but doing so here would exceed scope.
- Exact ticket identity, session/thread ownership, shared client send/receive budgets, guard commit before ACK, original deadline retention, no second emergency synchronization after a failed normal wait, no GPU meaning for CPU evidence, and process custody until actual return-code observation are explicit constraints.
- The C4c2a section 8 full integration design remains required. V2 section 9 is an inventory of unapproved future obligations, not approval of materialization, source admission, readiness/warmup ordering, timeline, lease authority, failure custody or idle close.
- The unresolved numeric camera resolver remains unresolved; no substitute fixture, invented value or hash rewrite is authorized.
- Cleanup keeps the new common core only as consumed shared infrastructure and the fixed harness only as a regression harness; it does not install a second product watchdog or merge the new and historical protocol versions.

## Independent verification

Executed locally in the frozen candidate with PowerShell and Python 3.14.0:

```text
git status --short
git branch --show-current
git rev-parse HEAD
git diff --stat ddf85fcf8e757dbd7d98bcb49f58dd8bca333af1 HEAD
git diff --check ddf85fcf8e757dbd7d98bcb49f58dd8bca333af1 HEAD
Get-FileHash -Algorithm SHA256 <v2 candidate>
rg --files / rg -n <targeted contract and lifecycle searches>
Get-Content -Encoding UTF8 <instructions, candidate, predecessors and actual source>
python --version
python -B -c <read-only canonical-codec and arithmetic probe>
```

Observed: expected clean branch/HEAD, exact digest, only the two documented additions, and no diff whitespace errors. The actual `canonical_json_bytes` accepted `{"sequence":9223372036854775807}`. It rejected the mathematically required maximum-history count `{"arm_count":9223372036854775808}` with `ClosedWorkerProtocolError: canonical JSON integer is outside signed 64-bit range`. The probe also checked the four-events-per-normal-wait index formula. It did not construct or claim to execute a huge trace or the unimplemented supervisor.

No new runtime code exists to execute. No GPU, remote, benchmark, result/baseline, product activation or installation action was performed. Existing C0/C1c runtime tests were not rerun: unchanged implementation regressions would not resolve this document-only arithmetic inconsistency. No remote candidate or old author test output was used as proof of v2. No candidate files or caches were generated by the Python `-B` probe.

## Approval boundary and remaining obligations

V2 is not yet approved for implementation because R1 remains. The otherwise acceptable intended boundary is exactly the two CPU-only production protocol/client/supervisor-core modules plus unit tests and a fixed standalone subprocess harness. The requested repair needs no expansion into runtime, GPU or product policy work.

After design closure, implementation review must establish actual nonblocking transport, clock/transition arithmetic, exact-ticket behavior, evidence replay, real subprocess ACK-before-leaf and kill/reap behavior, resource-finalizer sentinels, CPU import isolation and unchanged C0/C1c surfaces. Design reasoning is not that implementation proof.

C4c2 runtime integration, worker/native-setup/source-policy adoption, actual Warp timeline and R0/R1 lease graph, GPU preparation/submission, C4c3/R2, C4c4, C4d/B3, a3b, results/baselines, host payload/shipping, backend selection and product defaults all remain closed. No parent-stage completion or inherited broader approval is claimed.
