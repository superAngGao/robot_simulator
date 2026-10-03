# Independent C2d1 guarded-custody design review v2

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: review
Author: codex
Version: v2
Date: 2026-10-03
Status: accepted
Related Files: guarded-custody design-request codex v2; complete design-review codex v1; runtime-integration design-request codex v3; optical_worker/lifecycle.py; optical_worker/native_setup.py; optical_worker/wait_protocol.py; optical_worker/wait_supervision.py; native source/bootstrap policy
Owner Summary: Approve with comments for this exact bounded DESIGN. Prior F1 and F2 are closed at design level. The existing emergency-role protocol can supervise CPU cancellation without GPU synchronization, and v2 now freezes the required lifecycle/replay/custody grammar. One non-blocking whitespace comment remains. No guarded-controller implementation, GPU release safety, runtime activation or C4 completion is approved.

## Identity and verdict

Reviewer session: `/root/c4_c2d1_guarded_custody_design_review_v2`, a fresh independent session supplied the review packet and repository, not the author's implementation conversation. No delegated agents were used. The reviewer did not author or edit the candidate.

Verdict: **Approve with comments**. No unresolved blocking design finding or required design-level evidence gap was found. This is not a code review or acceptance of an implemented controller.

- Code/evidence base: `11603cd2772aea79c07579704285767f6c5ac622`.
- Previous proposal HEAD: `e4aeb403703f98ec6946b7c75c328c018ae01b0a`.
- Frozen reviewed HEAD: `f1f4394a3f84aea1107daa3a09bb8d06cee99f59`.
- Candidate: `collab/q54-optical-lab-p12-4c4c2d-guarded-custody__design-request__codex__v2.md`.
- Independently verified candidate SHA-256: `00aa16cfef0dfd83b7f44081f845c9a3397d79435b4b3cf319e27c467d840983`.
- Independently verified prior complete report SHA-256: `06df7a3a9ee2c7e0df579d6ddbe203776f13a4af4c08b3ac2929d18901740822`.
- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Remote: authorized fallback `ang.gao@192.168.80.101`, exact worktree `/data3/ang.gao/robot_sim_c2d1_design_v2`, detached at the same HEAD and digest.
- Initial/final inspected local and remote status were clean. Base-to-HEAD contains only the v1/v2 proposals and archived v1 report. Explicit protected-source diff for lifecycle, native_setup, worker, wait_protocol and wait_supervision was empty. The v1-HEAD-to-v2-HEAD delta adds only the v2 proposal and v1 report.

Primary-host SSH transport timeout was supplied in the packet and independently retried by the author during this review; this reviewer used the authorized fallback without retrying primary. The original dirty `/data3/ang.gao/robot_sim_c4c1b2` was not accessed or modified.

## Findings and prior-finding closure

### C1 — P3, non-blocking: trailing whitespace

Location: candidate line 260, the blank line after the inherited-events table.

`git diff --check e4aeb403703f98ec6946b7c75c328c018ae01b0a HEAD` reports one trailing-space line. This has no safety or semantic consequence. Disposition: retain as a non-blocking documentation-cleanup follow-up; remove the space in the author's next appropriate document change and rerun the whitespace check. This approval names the actual frozen digest above, not an uninspected replacement digest. No reviewer edit was made.

### Prior F1 — closed at design level

The old no-GPU branch really had no deadline while the supervisor remained IDLE. I reran the original external CPU probe against this unchanged code base: the owner's destructor observed IDLE, no deadline and no kill obligation; advancing its controlled clock still did not cause expiry.

V2 section 4 now selects a concrete supported mechanism: a terminal `emergency` ARM/ARMED envelope with local purpose `cpu_cancel`, before release or potentially blocking cleanup, with zero runtime import/sync. The same operation deadline remains active through cleanup and pre-hard-exit code; no COMPLETE, DISARMED or deadline restart is permitted. Failed terminal notification instead retains the graph, closes the channel and hard-exits without cleanup/sync. Section 6 expressly requires ordinary cancellation, CPU-destructor hang, pre-hard-exit hang and failed/stalled-ACK cases with a live parent.

Actual dependencies support that choice. `GuardWaitClient.arm` returns the exact ticket only after matching ACK; `complete` rejects emergency tickets. Supervisor `arm_committed` installs the operation deadline before ACK, and `ack_written` enters ARMED_TERMINAL without clearing it. Its terminal/EOF handling retains the earlier deadline when one exists. An independent bounded real-socket probe confirmed CPU guardian release can occur under ARMED_TERMINAL with the unchanged deadline and zero GPU sync; controlled expiry sets kill-required. This demonstrates feasibility of the design correction, not implementation of it or actual kill/reap acceptance.

### Prior F2 — closed at design level

Sections 5.1–5.4 now specify exact lifecycle/termination/artifact envelopes, field domains, summary keys, operation details, predecessor rules, guard/attempt references, scope replay, release phases, terminal endings and prefix/supervisor compatibility. These settle the previously missing high-risk choices without requiring a second wire protocol or changing the old validators' domain.

In particular:

- ARM_PENDING references name an attempt, not a fabricated ticket. `guard_armed` records actual ACK; normal DISARMED names the completed ticket while the resulting summary has no active reference. Failed attempts preserve their origin rather than reconstructing authority from a failed-client snapshot.
- Full descriptors and adopted-owner scope conservatively influence replay even if failure interrupts the following dirty/unknown notification. Missing notification cannot manufacture clean cancellation or exact-target release.
- Release authorization, release start and completed release are separate. `cleanup_failed` permits actual RELEASE_AUTHORIZED/RELEASED states and remaining-custody quarantine, not a false guardian transfer or claim that already-dropped owners remain live. This fits `release_whole_graph`, which clears owners before setting RELEASED, and `transfer_to_quarantine`, which cannot operate on those release states.
- Repeated operation names are allowed for distinct resources/segments; duplicate identities and illegal predecessors reject. Normal and terminal guard roles have distinct completion grammar. Terminal snapshots end in exactly one terminal intent; prefixes cannot claim exit or synthesize missing worker tails.
- Combined validation uses the unchanged supervisor replay as process/ACK authority. Worker ACKs/completions must have corresponding supervisor history; pending/killed worker prefixes can lag. Public hashes establish consistency, not authenticity of an entirely rewritten legal history or live execution authority.

The specified hand-authored fixtures and adversarial mutation tests remain mandatory code-acceptance work. They were not executed against an unimplemented v3 replayer.

## Independent high-risk analysis

Read the complete AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md, the candidate, prior complete report and accepted parent runtime-integration v3. Targeted reads supplemented truncated combined outputs. Inspected actual upload loops/internal abort, guardian ownership and release, lifecycle descriptor/permit/scope/event machinery, client state history and ACK exchange, supervisor replay/deadlines/transport, source policy/catalog inventory, captured-source loader/finder restrictions and the existing process-harness kill/observe-exit paths.

1. **Actual exception boundary:** both upload functions catch internally and invoke `_abort`, which dynamically calls `lifecycle.terminate`. A subtype override therefore intercepts the right call before outer segment exit. C0 termination may call its notifier and synchronize; v2 explicitly forbids delegating there. The fixed rejecting notifier is a tripwire, not an alternate policy. Retaining a sealed-guardian exception in terminal context avoids illegal re-adoption.
2. **Guard order and irreversible scope:** the proposed active-setup check precedes inherited permit/leaf logic. Actual C0 begins possible scope before low-level entry, adopts returned owners before ordinal checks, and latches foreign/unknown scope. V2 preserves that single implementation and requires failure before COMPLETE/new ARM/sync. Scope compromise wins over normal guard failure; active/pending/failed normal guards cannot masquerade as healthy IDLE. Reentrancy must invalidate outer success, not produce a second release or sync.
3. **Whole-graph custody:** controller/client/socket/ticket/failure context enter process-wide retained custody before potentially blocking terminal work. The existing guardian remains the sole storage owner/release implementation. Full pre-release quarantine and remaining post-release custody are intentionally different. Exception traceback owners and module/session ownership are explicit requirements, not inferred from resource labels.
4. **No readiness shortcut:** the old final-barrier helpers synchronize and commit C0 quiescence before an outer normal DISARMED ACK. They are inadmissible here; the subclass must reject the inherited commit/admission/clean-close paths. Upload success under a fixed test driver does not produce a prepared sample or final graph/barrier authority. Final seal, warmup and quiescence composition remain later work.
5. **Source restrictions:** the current source policy pins lifecycle regex-bearing bytes and captured-loader/bootstrap definitions; native_sources uses literal inventories. The current captured finder admits only six CPU-core modules. Thus CPU controller work and required source/catalog repins are feasible without pretending the guarded runtime is already admitted. Later captured-source guarded-role adoption must retain module owners before body execution and preserve actual-import audits. No broadened reflection/callback/native-call ingress is authorized.
6. **Trust and migration:** the closed trusted-process boundary is explicit; serialized evidence or a copied exact-looking ticket cannot mint runtime authority. The fixed test launcher is not product request dispatch. Reuse of actual upload loops and the existing guardian avoids a second implementation. The P12.4e cleanup obligation is explicit and must appear in the implementation ledger.

No substantive blocking finding was found in these areas for this bounded interface design.

## Verification and limitations

This is a pure documentation delta with no new controller/replayer implementation. No unrelated GPU regression, installation, push, reset, source normalization or candidate edit was warranted or performed.

Read-only commands included `Get-Content -Encoding UTF8`, targeted `Select-Object`, `rg`, `git rev-parse HEAD`, `git branch --show-current`, `git status --porcelain=v1`, `git diff --name-status`, protected-file `git diff`, `git diff --check`, and `Get-FileHash`. The sole whitespace failure is C1 above.

Independent external probe files:

- Original repro: `C:/Users/AG/新建文件夹/c4_assets/c2d1-design-review-probe-v1.py`, unchanged; rerun, not reused old output.
- New dependency probe, created using apply_patch: `C:/Users/AG/新建文件夹/c4_assets/c2d1-design-review-probe-v2.py`, SHA-256 `192f2ab262981fe9b937af1f0ef0ab259b7b4ec2f408a6a660538f27be614aa1`.

Both ran with `C:/Python314/python.exe -B <probe-path>`, CPython 3.14.0. Separate final executions each returned actual exit 0:

| Probe | PID / parent PID | Observed result |
| --- | --- | --- |
| v1 | 35284 / 20672 | CPU cleanup IDLE; deadline null; no kill obligation after 50 controlled operation budgets. |
| v2 | 16692 / 31372 | Real socket/client/transport ACK; CPU cleanup ARMED_TERMINAL; deadline 1000000100 unchanged; guardian RELEASED; controlled expiry TERMINATING/kill-required; zero GPU sync; worker thread joined. |

The v2 probe uses the actual core's supported controlled clock and actual guardian release, not a copied supervisor state machine. It starts one bounded client thread, no child worker/supervisor and no actual hang. It does not call observe_exit or claim a supervisor-observed child exit. Neither probe imported Torch or Warp. Earlier combined exploratory executions also completed (PIDs 35552 and 13240, parent 29644; aggregate command exit 0); the separate executions above give individual command exit evidence.

Remote read-only inspection used:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 <command>
```

Remote revision, detached branch, clean status and SHA matched. The remote interpreter was not used for tests. No real-device properties, guarded-controller behaviors, v3 parser mutations or source-frozen runtime adoption are claimed as executed. Required real subprocess cancellation/hang/ACK and integrated GPU matrices remain future code acceptance.

Inspection/tool limitations: one SSH command failed before executing because PowerShell stripped quotes around a shell regex; it was rerun successfully with separate `grep -e` patterns. An `rg` call using wildcard path arguments failed on Windows and was replaced by `rg --files` plus exact paths. Initial non-UTF8 display garbled Chinese instructions; REVIEW_PRINCIPLES was reread completely with explicit UTF8 before substantive review. These are tool/read errors, not candidate behavior.

## End-of-round process hygiene

- Exact CIM queries found none of owned probe PIDs 35552, 13240, 35284, 16692 or their recorded short-lived parent shells 29644, 20672, 31372 remaining. The final local GPU compute list contained no owned probe PID; unrelated desktop processes were left untouched.
- Remote scoped process scan showed only its own inspection shell PID 2257904, parent 2257824, and grep PID 2257931, parent 2257904, both starting Sat Oct 3 08:57:58 2026 remote time. A subsequent exact PID query showed both absent. No review worker/supervisor was launched remotely. Final remote status remained clean and the GPU compute query returned no rows.
- No timeout was counted as process exit; no kill, GPU reset or cleanup of other work occurred. Short-lived local process creation timestamps and every read-only shell/SSH process were not exhaustively captured. The ledger states that limitation rather than claiming exhaustive process enumeration. These checks are scoped current observations, not device quiescence proof.

## Approved boundary and remaining gates

Approval permits implementation of this exact C2d1 interface: one production-owned guarded lifecycle subtype, the narrow exact-ticket history observation extension, specified v3 evidence/replay and fixed CPU integration tests, reusing C0 guardian/descriptor/permit/upload logic and preserving protected sources/protocol semantics. Its implementation still requires fresh independent code review and all required CPU real-process evidence, including the no-GPU cleanup/exit hangs added for F1.

Captured-source guarded-role adoption, complete module/session custody before CUDA-capable execution, source-frozen real Warp/Torch uploads, both-lane target1/ambient0 and foreign-pre-return/unknown/fault matrices, final graph/seal/barrier/warmup/readiness composition and the remaining C2d/e/f checkpoints are not approved as completed. CPU probes do not satisfy those gates.

No product GPU preparation/submission, dispatcher activation, arena/result/baseline authority, a3b, R2/S/host, B3/shipping or C4 completion is opened. No historical C0 approval is changed. The unchanged complete report may be archived as review-only evidence under the collaboration protocol; semantic follow-up changes remain subject to independent delta review.
