Initiative: Q54 / P12.4 / C4c2 final observation D1
Stage: Fresh independent DESIGN follow-up review
Author: Codex, independent reviewer /root, fresh CLI session
Version: v2
Date: 2026-10-10, Asia/Shanghai
Status: accepted
Related Files: final-observation design-request v2 and original v1; complete archived final-observation design-review v1; runtime-integration design-request v3; native-state-reader design-request v1; complete archived worker-acceptance gap-design-boundary-review v1; original reader, worker, guard, supervisor, arena, timeline, readback and source-policy code
Owner Summary: Approve with comments for the exact v2 document at the verified HEAD and actual digest below. F1 coverage inflation and F2 request retirement/expiry ambiguity are closed at DESIGN level. C1/C2 are incorporated, including actual event initialization and at least three successful sentinel reuses. Required integrated implementation and native/OS evidence remain outstanding. The supplied digest does not match the document at the supplied HEAD; this report binds only the observed bytes, not that other digest.

Verdict: **Approve with comments**, confined to the D1 design supplement at HEAD 208f32c86c25b26bda6a31f66c21f9aeb0fe1ffb, committed LF SHA-256 **81111b1e548c438689e44a04129407273379184b62c69624480835964a71b7a6**. No unresolved substantive blocking design finding was found in these bytes. The packet identity discrepancy is recorded explicitly below. There is no approval of an unobserved document with the supplied different digest.

## Identity, independence and review boundary

| Item | Observed value |
|---|---|
| Frozen candidate | C:/Users/AG/新建文件夹/c4_final_observation_design_review_v2 |
| Actual HEAD | 208f32c86c25b26bda6a31f66c21f9aeb0fe1ffb |
| HEAD tree | b09609faf04af5cccf80d2ecbdd679d5d5acbcad |
| Production base and actual merge-base | cf7f5a6d9ff9b3971144410a7c3ad156a806646c |
| Documentation base and actual merge-base | 819a26f10ba32d8ce479c71acd45c8df41e6c027 |
| Candidate file | collab/q54-optical-lab-p12-4c4c2-final-observation__design-request__codex__v2.md |
| Actual committed LF SHA-256 | 81111b1e548c438689e44a04129407273379184b62c69624480835964a71b7a6 |
| Packet-supplied SHA-256, not observed | 06862ff86d56f607868930936078ef3817724cefbbdafa72a375463d27cb12cf |
| Initial tree | Clean detached HEAD, including tracked/untracked/ignored status |
| Reviewer CLI | codex.exe PID 45132, parent 20728; creation 2026-10-10T11:43:02.815Z; enclosing ancestry in reviewer_ancestry.json |
| Local report/session identifier | independent-cli-final-observation-v2-ce58f66a-7f88-446a-8be8-5f612c7a89eb |
| Interpreter | C:/Python314/python.exe -B, Python 3.14.0 |
| Sole write location | C:/Users/AG/新建文件夹/c4_assets/final_observation_design_review_scratch_v2 |

The local identifier was generated for this report; it is not a claimed platform conversation UUID. This fresh CLI reviewer did not author the candidate, received no implementation conversation, and spawned no agents. The owner explicitly authorized this independent mechanism when tool spawning was unavailable. No claim of a different vendor or stronger security isolation is made.

I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully; an explicit UTF-8 reread corrected the first Chinese console decoding. I read the complete original v1, archived v1 review, v2, runtime-integration v3, native-state-reader v1, and complete archived worker-acceptance gap review. Large outputs were followed by focused reads of omitted sections. Predecessor findings were checked against the accepted requirements and actual source, rather than adopted as new requirements.

The production-base delta consists of PROGRESS.md and seven collab documents. No production source differs from cf7f5a6. The documentation-base delta consists only of the added v2 request and archived v1 report. The original v1 is preserved. `git diff --check 819a26f... HEAD` passes. The candidate's committed bytes contain no CRLF; the digest above was computed directly from `git show HEAD:<path>`, not from a normalized working copy.

**Packet identity comment I1 (nonblocking for the explicitly named HEAD):** The HEAD, path and production base agree, but the supplied digest does not. I notified the owner and requested clarification while continuing this read-only review. The report identifies exactly which bytes were reviewed. Correct the handoff/archive metadata to this digest if the pinned HEAD is intended; if the other digest identifies a different intended candidate, this verdict does not cover it and its delta needs independent review. No candidate edit is requested merely to alter its hash. Any subsequent owner clarification is to be retained with this report, not silently substituted for these observations.

## Follow-up findings and independent closure

### F1 — CLOSED at design level: exhaustive CPU camera coverage is separated from bounded native numerical coverage

Original finding: v1 lines 190–193 naturally read as mandatory all120 full-resolution numerical comparisons, reinforced by gap-review v1 lines 89/180. Those review statements could not expand the accepted parent.

Controlling requirements checked independently:

- runtime-integration v3 lines 174–231 bind the resolved sequence, modulo camera/replay selection, frame/time metadata and actual binary32 ABI arguments;
- its section 9 item 1, lines 728–731, places all120 camera values and first/last/wrap checks under CPU moves;
- item 2, lines 732–737, separately requires asymmetric/dynamic independent CPU rays, geometry, IDs/material/RGB/intensity, actual Go2 fixture validation and unchanged parity rules;
- item 4, lines 743–745, requires six R0 and two R1 rows, both lanes, profiles/shadow/diagnostic variants, terminal-event and generation-reset behavior;
- C4c parent v7 lines 1489–1509 retain channel parity, miss/counter reset, actual scope/stage/event behavior and at least three successful slot reuses. They do not impose all120 full-resolution numerical comparisons.

V2 lines 251–268 now expressly deny the expanded numerical mandate and provide a requirement-to-coverage table. CPU camera/selection coverage cannot substitute for actual final output; actual final outputs remain required at the frozen resolution and with real Go2, applicable channels and unchanged tolerances. First/last/wrap and sentinel-selected final generations must be joined and checked before release. Original intervening FIFO generations must run when needed; they cannot be fabricated or replaced with warmup output. The exact implementation case index must name selected samples/channels and artifact paths.

This closes the ambiguity without reducing the accepted native correctness matrix or inventing a numerical image count. I do not read the erroneous archived gap-summary all120 output mandate as authority. The archived report remains unchanged; v2 lines 36–41 correctly explain its limits. No new GPU execution is required to establish this textual scope correction.

### F2 — CLOSED at design level: sequence-bound publication, success observation, retirement and expiry arbitration

Original counterexample: publish q/D; supervisor is descheduled; worker returns after D; client clears/replaces q; supervisor resumes and misses expiry. The opposite incomplete policy never retires a timely success and falsely expires it.

V2 lines 136–190 and 278–284 freeze the missing lifetime:

| Event | Required disposition |
|---|---|
| Before any q bytes are sent | Publish exact worker/session/q/start/D under the short CPU-state lock; only EMPTY or successfully RETIRED permits a new request |
| Complete response | Original codec validates version, exact sequence and successful status; capture monotonic observation time immediately before persistence |
| Matching observation before D | Client publishes RESPONSE_OBSERVED only while q remains PUBLISHED; only supervisor can retire it |
| Observation at D or after D | Terminal failure candidate, even if supervisor has not run yet; equality loses |
| Supervisor sees still-PUBLISHED q at/after D | Commit TERMINAL; later publication cannot resurrect q |
| Timely RESPONSE_OBSERVED, supervisor resumes later | May retire only if terminal/core failure has not already won |
| Partial/error/EOF/send failure | Existing terminal path; no clearing/replacement or successful retirement |
| Stale q observation | Cannot clear or replace current q+1; cannot resurrect TERMINAL |
| Subsequent request | Wait outside the lock until supervisor retirement, checking stop/terminal; no q+1 before successful retirement |
| Terminal disposition | Stop is latched; command endpoint shutdown both directions, close fallback, preserve first error; core fault is reached in finally; actual worker disposition precedes joined-client acceptance |

This rejects the exact original late-response counterexample. It also distinguishes two races that should not be conflated: a timely result already published may survive delayed supervisor scheduling, while a pre-deadline timestamp published only after TERMINAL cannot reverse expiry. The short lock must never span socket I/O, persistence, native work or a blocking wait. The core-owning supervisor is the sole core fault/kill caller.

I checked the real integration sites. `tests/support/guarded_custody_process.py:315–397` currently has only the outer harness deadline; `exchange` writes response bytes at :353 before validating at :354–360. V2 therefore requires moving successful observation before that persistence, plus adding the shared record, rather than merely renaming the existing deadline. The actual client is a thread created at :480–485. The original main loop at :494–565 owns tick, kill request, poll/observe_exit and wait. The worker is a separate child, so its native getter cannot hold the supervisor process's Python interpreter lock. These are usable existing entrypoints; no new watchdog is needed.

`GuardSupervisorCore` really admits `process_observation_failed` (wait_supervision.py:27–36), enforces original-thread ownership (:337–338), and sets terminal deadline to the minimum of prior active/ACK/emergency deadline and terminal grace (:186–193). `_fault` ticks first and does not renew terminal state (:470–493). Three fresh probes of this unmodified CPU core confirm IDLE grace and preserved armed resolve/emergency deadlines, repeated-fault nonrenewal and kill eligibility. Exit in these probes is simulated, not OS reaping.

The design model also exercised timely/equality/late response with delayed supervision, expiry before late publication, no next request before retirement, stale q after q+1 publication, partial frame, EOF, send failure, unsuccessful status and preexisting core failure. All tested schedules meet v2's rules. This model demonstrates contract consistency; it is not proof of the absent request bridge implementation or lock/socket scheduling. Actual shutdown/close-failure tests and OS blocked-getter tests remain mandatory under v2 lines 247–249 and 278–284. A request timeout in IDLE can be followed by the existing core terminal grace; do not describe total disposal as only one operation-timeout budget.

### C1 — text incorporated; implementation proof remains a nonblocking comment

V2 lines 215–222 now require the event's actual native storage to be initialized through the required first record/wait during guarded preparation, before the final barrier. This addresses the lazy-event-wrapper pitfall and follows parent v3 lines 360–386. `native_setup.py:1270–1279` currently separates wrapper creation, record and wait, confirming that mere creation is not the complete preparation operation. No native-library behavior was imported or tested here.

Disposition/proof at implementation: demonstrate initialization and retained owners before the single seal/barrier and no post-seal event/host-view/copy-scope allocation or getter cache discovery. Exact-code/native-call preflight must inspect changed placement against pinned runtime sources. This is retained evidence work, not a new design block or capability.

### C2 — text incorporated; sentinel and post-resolution failure proof remain a nonblocking comment

V2 lines 224–239 settle the order: original terminal wait; same-guard diagnostic copy/record/wait; restored-state observation; normal guard completion/DISARMED; original arena resolution; CPU persistence/comparison before RELEASE or staging reuse. Copy/wait failure is unresolved; later CPU failure preserves and poisons the real resolved-held state rather than rewriting history.

The fixed sentinel fill must follow possible-submission custody and precede the original reset/render on the exact same ordered stream. Its completion is protected by the sample's original resolve; no fill can race D2H or another generation. V2 lines 237–238 and 260 expressly require **at least three successful slot reuses**, agreeing with parent C4c v7:1495. The archived gap report's weaker two-generation wording cannot replace it. Three initial generations are not automatically three reuses: the acceptance index must demonstrate three reuse transitions of the same slot with the required fills and resulting outputs.

Disposition/proof at implementation: retain generation and pointer joins, original reset coverage, distinguishable sentinels, no stale channel/counter survival, and failure-state assertions on both sides of arena commit. Do not replace original reset bodies with a test renderer or weaken profile/miss/counter rules.

## Remaining mechanism assessment against actual source

### Exact admission and original guard

`native_scope.py:41–70` currently permits setup or the narrowly eligible emergency operation, and calls `_require_setup()` for every nonterminal query. `guarded_lifecycle.py:185–189` requires a live setup ticket. Thus final observation is genuinely a policy supplement, not a currently usable setup API. `require_reader` at native_scope.py:523–553 checks the source-owned seven-object record, reader/controller/module/runtime/target/stream/binding identity and guardian custody; those checks should be shared, not copied or bypassed.

Original `observe_terminal_completion` at guarded_lifecycle.py:346–396 checks the pending active record, owner/transaction, original event/stream and target. CUDA binds the exact transaction context/device object and compares its target value to the retained target at :362–374; v2 correctly preserves this approved value rule instead of demanding identity between distinct equivalent torch.device values. It then retains owners, enters the actual resolve guard at :387, performs the original wait, checks completed ticket and records `_worker_resolution` only afterward.

`_require_worker` at :476–494 deliberately requires IDLE and no active context. Calling it unchanged inside RESOLVE cannot work. V2 lines 72–79 explicitly require factoring identity/phase admission while keeping separate live-resolve and driver-local EXECUTE predicates. There is no need to relax all IDLE reader access, forge a setup ticket or monkeypatch `_ticket`.

`worker.py:449–480` binds the actual active sample and execute operation before original enqueue/finish. A dormant pre-seal diagnostic registration can bind those actual objects, live lease/generation, owner thread and occurrence extent. It expires at extent end and permanently on failure. Copied registrations, matching labels, source substitutions, expired tickets, stale generations and artifact tokens grant no authority. The original worker trust model is a closed trusted process, not protection against arbitrary hostile reflection. Ordinary IPC cannot install an observer/callback, and production must not import tests.

### Exact original scopes and the enqueue-exit masking case

WarpPreparedArenaSample precreates its real device/stream scopes at warp_arena.py:492–496 and uses them separately in enqueue (:521–534) and finish (:536–549). Its transaction enqueues multiple groups and diagnostic event records at warp_timeline.py:187–225; actual final launch/event leaves are :287–295. A probe around setup helpers alone misses these operations.

CUDA `CudaActiveArenaSample.enqueue` at cuda_arena.py:686–710 delegates each stage. `cuda_timeline.py:723–744` activates a context per stage; `activate` at :434–440 enters indexed device then stream. Finish independently activates at :801–806. R1 scene/acceleration/render scopes must have distinct occurrence indices. Current CUDA activation constructs contexts; the already-required G3 preparation refactor must provide the original prepared scopes, not unrelated metadata-equivalent scopes. That is a delivery dependency, not another framework.

A fresh scratch probe compiles only the unchanged original Warp adapter enqueue/finish AST methods, with explicitly artificial CPU scopes that omit enqueue restoration and repair on the next exit. After enqueue the state is wrong; after finish it is restored. Thus a post-EXECUTE-only observation loses the information. This reproduces the ordering counterexample on the actual method bodies, **not** a real pinned Warp runtime restoration defect. V2 lines 93–102 and 270–275 require immediate original after-exit observations before later entries and independent enqueue/finish omission negatives; these close the information gap at design level. Later native preflight must prove the actual placement and native negative controls. Nested intermediate scope state must not be confused with fully restored outer state.

### Failure transfer and retained custody

V2 lines 199–211 require first-error/raw-prefix/returned-owner retention, unknown-scope poison before subsequent normal native work, retained known foreign ordinals, no new query after entry/body failure, and original unwind exactly once. Necessary original context exits still execute according to their exception semantics. No fabricated successful after-row, ordinary release, new normal ARM or emergency target sync follows sticky observation failure.

This needs explicit integration: `_observe_public_operation` at guarded_lifecycle.py:208–218 auto-poisons only active/terminating operations; ordinary IDLE rejection is not sufficient for diagnostic EXECUTE. `_fail_guard` at :147–160 supplies sticky failure/retention, and `terminate` at :787–807 quarantines compromised/failed state. `NativeStateReader._fail` at :72–80 is a starting point, not proof that append/capacity or wrong-thread failures can never interrupt failure transfer. G3's reserved terminal capacity and G5's error-preserving persistence must support it. Setup leaf/adoption APIs cannot simply be invoked after seal to record new custody.

The source rule to stop querying on compromise is appropriately conservative. V2 :273–275 separately preserves the parent's healthy-IDLE, complete known-target post-submit fault path with one emergency ARM/sync. A diagnostic failure matrix that forces every unrelated fault into UNKNOWN would still fail the parent's integrated acceptance.

### Sealed staging and resolution ownership

`native_setup.py:1144–1244` holds actual warmed slot outputs, final frames, factories, arena and transactions, so one reusable validation staging set can be prepared in the existing graph. CUDA's public output accessor requires PAYLOAD_READY (cuda_arena.py:817–831); retain exact private output owners during preparation without calling that accessor prematurely or weakening its gate.

`native_readback.py:114–215` currently allocates host arrays/views, adopts them, creates an event and calls setup leaf admission. It cannot run wholesale after seal. V2's prepare/execute split is necessary and feasible with the existing copy/record/wait bodies at native_setup.py:1260–1299. CUDA copy scope preparation joins G3; Warp must preserve restoring outer scope behavior already documented at native_readback.py:86–95. No new oracle or copy algorithm is required.

Both original lane resolvers wait before committing the lease: warp_arena.py:572–603 and cuda_arena.py:734–770. The fresh original-method CPU probe confirms the existing controller wait is within its resolve segment and the completed-ticket check follows it. The diagnostic copy/wait must be inserted before that segment exits, not in a wrapper around the returned method. Raw observations and the copy event never become a second production terminal event or resolution authority.

Use applicable profile/write/reset channels: output_contract.py:120–139 distinguishes FULL, RGB_PREVIEW and RENDER_ONLY. Do not require uncomputed FULL channels in smaller profiles or omit internal channels the frozen validation requires. Existing exact discrete, floating, miss/nonfinite and lane-specific counter rules remain controlling.

### Query-free timing, source policy, evidence and finite removal

V2 lines 81–89 exclude facade/registration/staging from future timing workers independently of primary versus diagnostic_profiled pass. A correctness primary pass may require diagnostics; a timing diagnostic pass must remain free of correctness query/copy/sentinel work. Inspect and instrument acquire/enqueue/finish and the full timing path at implementation; testing only a pass Boolean is insufficient. The current fixed worker-success branch at native_runtime_process.py:1256–1279 does not install the existing later-helper facade; wire the actual worker branch, not a separate smoke orchestrator.

Source policy pins the reader/bootstrap and restricts ctypes to the reviewed reader initialization/state methods (native_source_policy.py:19–35, :83–102); production test imports are forbidden outside literal launcher policy. V2 retains the sole getter implementation, pinned ABI/library identities, no post-seal discovery/cache initialization and no new setters. Normal implementation pin/catalog updates must be coherent and exact; a recomputed hash alone cannot admit new arbitrary call targets or transfer a native-query exception to another module.

The final observation schema is distinct from input runtime-custody v3 (v2 :120–132). Its in-process joins require actual owners; serialized occurrence/request/generation tokens are descriptive. Concrete field types/edge values and incomplete-coverage rejection must be frozen in the coherent implementation. Partial getter prefixes cannot become full snapshots, and CLOSE cannot query a released reader. Later CPU lifecycle/ACK/exit evidence is distinct from the last successful physical snapshot.

V2 lines 286–296 retains one G1–G5 packet: shared atomic/prepared bounded control, staging/final validation, transport/close-exit/fault coverage and complete journal recovery. Those are inherited mandatory work, not four new policy supplements. Current immediate RELEASE (:381–384 in the fixed client), whole-history replay (`guarded_lifecycle._record`:112–130) and absent close-exit use of its ACK timestamp are not silently approved by this design verdict. One case-to-requirement index suffices. Remove superseded wrappers after equivalent original-path coverage; eventual losing-lane/selector/comparison-only cleanup follows parent v3 :785–798. No permanent dual-backend bridge, product host result or new framework is justified.

## Verification and limits

Executed read-only inspection commands used PowerShell Get-Content with explicit UTF-8, rg/rg --files, git --no-optional-locks rev-parse/status/merge-base/diff and git show via Python. A v1/v2 `git diff --no-index` returned the expected status 1 for differences. An initial Windows wildcard rg operand failed; the literal C4c v7 path was then discovered and read. Some combined outputs were truncated and followed by focused reads; these are inspection limitations, not candidate defects.

Scratch execution evidence:

- `C:/Python314/python.exe -B <scratch>/identity_probe.py`: exit 0; Python PID 30172, parent 43352. Saved identity.json and inventory_initial.json for 1,363 non-Git-metadata files, with hashes/sizes. The tool briefly returned session 91952 after producing output; explicitly polled to completed exit 0.
- `C:/Python314/python.exe -B <scratch>/cpu_probes.py`: exit 0 on first execution; Python PID 13320, parent 23884. Sixteen CPU checks passed. Three run the actual unchanged supervisor core with controlled clock/simulated exit; eleven exercise an explicitly labeled design transition model; two compile unchanged original method ASTs with declared CPU doubles. Full inputs/traces, source hashes and limitations are in cpu_probe_results.json. These are not sixteen native or OS acceptance cases.
- Scoped Win32_Process ancestry inspection: recorded only reviewer/launch ancestry and later owned-process closure. No remote process, GPU process, unrelated campaign or background job was inspected or signaled.

All Python runs use the specified interpreter with -B, PYTHONDONTWRITEBYTECODE=1, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and TEMP/TMP inside scratch. No pytest suite/cache was used. The behavioral probe rejects torch/warp/numpy/ctypes/cupy imports and verifies none is present afterward. No native-containing arena/controller module was imported to run extracted methods. Other Python work uses standard-library file/hash/Git operations.

Unexecuted by explicit scope: native getter ABI/cache/nonallocation/restoration behavior; actual OS blocked-getter disposal and joined-client tests; socket shutdown/close-failure races in implemented bridge code; real final-output/CPU numerical comparison; all120 materializer execution; Go2 assets; source-policy/native preflight; full G2–G5 implementation evidence; journal/transport fault campaign; work-count/allocation acceptance; performance. No GPU/native import/query, SSH/remote call, deployment, package installation or implementation occurred. The terminal 17-case campaign and possibly active separate cf7 framed probe were neither inspected nor interfered with. Their outcomes are not evidence in this review.

## Process closure, preservation and gate disposition

Final independently recorded preservation/process results follow below. No native worker, guard supervisor, compiler, SSH process, persistent helper or GPU workload was launched by this reviewer. Every probe was a short-lived local CPU process; no signals or resets were issued. Enclosing reviewer/launcher processes remain active infrastructure until this report is returned; their eventual exits cannot be attested from inside the session.

Only scratch files were written. No candidate edit, evidence commit, private-context edit or implementation was made. The full report and scratch probe evidence are ready for verbatim archive; evidence-only archiving does not alter this verdict or its scope.

**Scope closure:** Approval permits implementation of this exact D1 design within the already required coherent G1–G5 packet. It is not approval of any code or native call, execution/deployment permission, product activation, performance/result/baseline, backend selection, R2/S/B3 host delivery, external holds/depth-two, shipping or whole-C4 completion. Existing predecessor approvals retain their exact scopes/revisions. Exact-code/native-call preflight, required implementation evidence and fresh independent integrated review remain necessary. All code/native/product/performance/whole-C4 gates remain closed by this design review.

Final preservation check: `final_preservation.py` exited 0, PID 21084 / parent 38264, at 2026-10-10T11:49:37.767572Z. All 1,363 inventoried non-Git-metadata files have identical bytes and sizes: zero added, removed or changed files. This includes ignored/untracked working-tree files, not only tracked diffs. HEAD is unchanged; final tracked/untracked/ignored status remains clean detached HEAD. Working-copy design text matches the committed LF bytes after CRLF normalization. Git administrative metadata was excluded from recursive inventory; the .git worktree pointer was read separately during inspection and final closure. Full results are in preservation_final.json.

Final scoped process check: process_closure.json records recorded probe PIDs 30172, 13320, 21084 and their parent shells 43352, 23884, 38264, plus direct reviewer/probe children. All recorded probe processes and those shells were absent at this check. Direct reviewer children observed were conhost.exe PID 28676, codex-code-mode-host.exe PID 40592, and the synchronous closure PowerShell command PID 25744. The first two are enclosing CLI/tool infrastructure, not probe-created workers. No unresolved owned worker or test hang was observed. The reviewer CLI and enclosing launcher ancestry are active infrastructure, not residual test workers. Not every short-lived text-reading shell was separately sampled; completed tool results provide their command closure. No assertion is made about unrelated processes or either uninspected remote campaign. No process was signaled, and no live exec session remains.

This independent design-review round is complete. The report remains bound to the exact observed HEAD/digest above; all later gates remain closed.

