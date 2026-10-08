Initiative: q54-optical-lab-p12-4c4c2d-native-global-publication
Stage: implementation-gap-review
Author: codex, fresh independent reviewer
Version: v2
Date: 2026-10-08 (Asia/Shanghai)
Status: in_review
Related Files: native-global-publication design-request/design-review/implementation-gap-review/implementation-note v1; runtime-adoption design-request v2; optical_worker/native_bootstrap.py; optical_worker/runtime_environment.py
Owner Summary: Request changes for two remaining evidence-contract gaps. The four original counterexamples now reject, and the original exact-spec finding is closed. After poisoning, issuance still emits mutated facts; descriptive replay still admits contradictory same-frame return histories. This is a bounded repair re-review, not C2d3/C4 acceptance. Actual captured Go2 preparation still times out and mandatory aggregate/runtime proofs remain outstanding.

1. Independent identity, frozen scope and verdict

Reviewer session: /root/native_global_implementation_gap_v2. Fresh independent context received the review packet, requirements, prior findings and preserved repros. I did not author or edit the candidate, and did not delegate this review. Same-model context isolation does not claim a separate security boundary or external Claude review.

Verdict: Request changes. Findings V2-F1 and V2-F2 below are P2 evidence-integrity defects within the native-global repair scope. They do not demonstrate a reopened GPU execution, normal COMPLETE, sync or release path. No broader code approval is implied by the passing CPU checks.

Repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
Branch: implementation/c4-supervised-waits
Original implementation base: 8b7c9b88bcb42ddd704237395d27fa508899064c
Repair base / prior reviewed candidate: 4933b7ca46eb594f51c924ce88f8bd124bb14ba8
Intervening evidence archive: 1b597b5
Frozen reviewed HEAD: 9d1d7d025a4798cef8e7aff6318c530320ef7736
Primary remote: tileops-dev, 183.166.183.33:40031
Remote checkout: /data2/shared/ang.gao/robot_sim_c2d3_binding_wip, detached at the same HEAD
Local interpreter: C:/Python314/python.exe, CPython 3.14.0
Remote interpreter: /home/ang.gao/miniforge3/envs/robot_sim/bin/python, CPython 3.12.13

Initial and concluding inspected local/remote trees were clean at the frozen HEAD. The repair-base-to-HEAD delta is eight files, including the unchanged archived v1 report and the implementation note. No AGENTS.md, shared review protocol, deployment environment table or historical profile changed. git diff --check passed. The reviewer wrote only scratch repros and this report outside the candidate; no package changes, process signals, GPU reset, candidate repairs, push or merge were performed.

Approved design snapshot: 80387e8d8199e9dc6741924b95ff1aeeae047b45. Independently rechecked raw design-request v1 SHA-256: 5c29fad420b9d31a848f8b53df07450b46f0fee0b4e2abe9f38aad7997be8b2c. Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md in full, with explicit UTF-8 reread where PowerShell default decoding was unsuitable. Read the native-global design, its independent design-review v1, authoritative runtime-adoption design-request v2, prior implementation-gap-review v1 and current implementation note. Large combined outputs were followed by narrower reads; no truncated output is treated as a complete document read.

2. Closure of prior findings

Original F1 (coherent copied ModuleSpec): closed within this repair scope. The original exact spec is now held in the private native-call binding. _check_native_call compares both frame and index projections against that issued object; _native_call_state preserves the new field. Replacements encountered in frame/index are retained before rejection. The original coherent-copy repro now fails with SourceBootstrapError("native call binding drift"). Pending, audit and issuance variants in the candidate suite pass, with sticky session failure and retention checks. Audit now checks all native calls, including calls not referenced by a global event.

Original F2 (coherent paired token mutation): the original healthy-boundary counterexample is repaired. Immutable event facts and exact event/record/module/frame roots are retained before the mutable event projection; completion, audit and healthy issuance compare them. The original repro now fails at audit with SourceBootstrapError("native global issued event drift"). Mutation/removal/reorder/copy/checkpoint tests pass. However, the failed-state issuance path bypasses those checks, leaving V2-F1 below unresolved.

Original F3 (call phase and ancestor return): both exact original counterexamples are repaired. Each call's phase must now equal its configured module row. Ancestor exec_return events are remembered and prohibit later descendant events, including failed-prefix parsing. The wrong-create-phase repro now raises ValueError("native call relation"); parent-return-before-child raises ValueError("parent return preceded nested event"). The new exact-type parent check also rejects bool-as-int substitution. However, return constraints remain incomplete for the same frame and its terminal state; see V2-F2.

I ran the original acceptance-demonstrating repro file unchanged. Its four tests deliberately assert invalid acceptance, so the observed four failures at the intended production checks demonstrate closure of those exact old counterexamples, not four new regressions.

3. Findings

V2-F1 — P2: After the first evidence-drift rejection poisons the session, retrying issuance emits the same forged object-token history.

Location: optical_worker/native_bootstrap.py:892-916 (_native_global_observations), especially both `if not self.failed` branches at lines 894 and 914; public sticky-failure wrapper at 885-890.

Trigger: Complete a valid native-global admission using the production finder/loader and fixed native delegate. Change both mutable event object_token fields from 1 to 101 without changing any private event facts, held module, record, frame or private token. The first native_global_observations() raises "native global issued event drift" and correctly sets admission.failed and session.failed. Call the same method again. Because the session is now failed, it skips live binding/event checks and replay, copies the corrupted projection, and returns an ordinary native-global-observations.v1 payload with token 101. validate_native_global_observations(..., require_complete=True) accepts it; the actual private object token remains 1.

Consequence: Issuance in the terminal reporting path can detach reported identity from the actual held object after that very drift was detected. Sticky lifecycle failure is preserved, but sticky failure must not convert a rejected evidence projection into issuable facts. This violates the supplement section 6 requirement that issuance checks the live private registry and preserves issuer-owned token identity. It is a failure-evidence integrity problem, not a claim that a parsed component can reopen the poisoned controller or replace the mandatory aggregate guard binding.

Why current tests miss it: New issuance tests stop after the first expected exception and assert poison/custody. They never request evidence again in that terminal state. The mutable event list remains unchanged by failure intake, and the failed branch does not use the newly retained immutable event facts.

Repair direction: Keep terminal diagnostic serialization available without reopening admission, but do not export unchecked mutable call/event projections as valid issued facts. Either continue rejecting compromised observations on every attempt, or serialize only verifiable immutable issued facts with an explicit failure-compatible contract. Do not require current sys.modules presence as a blanket precondition for all legitimate post-unwind failure diagnostics; distinguish immutable issued-history integrity from live healthy-state validation. Never manufacture a missing event or admission during reporting.

Required proof: Preserve this retry counterexample. Cover failure entered by issuance, by audit and by an unrelated earlier import failure, then mutate/erase/reorder the projected history and request terminal evidence repeatedly. Verify no wrong token/root relation is emitted, legitimate failure prefixes retain actual issued facts, terminal stdlib reporting remains available, and no new admission/ARM/COMPLETE/sync/release occurs.

V2-F2 — P2: Replay rejects returned ancestors but still accepts impossible return histories within the same frame and a return observation from a still-started frame.

Location: optical_worker/runtime_environment.py:434-435 and 460-477 in validate_native_global_observations; return_native_call at native_bootstrap.py:725-731 establishes the production ordering being replayed.

Counterexample A: Configure two legal native-global rows with the same observer (the schema permits this). Publish both during that observer's exec and obtain genuine complete evidence: pending A at exec_return, pending B at exec_return, followed by both admissions. Change only the second pending event to checkpoint=request_entry/request_name=fractions. Replay accepts require_complete=True, despite the first event proving the same delegate already returned before that second first-observation request. A failed-prefix version retaining both pending events, omitting admissions and setting the call state to failed also passes. No ancestor is involved, so the new ancestor-only returned set check misses the contradiction.

Counterexample B: Obtain a normal single-global exec_return pending observation, remove the later admission, and change that call's terminal snapshot state from completed to started. Prefix replay accepts. Production return_native_call marks the frame returned before invoking its exec-return checkpoint; after that it may become completed or failed, but cannot remain started. This contradiction is available entirely within the component and needs no aggregate call clock.

Consequence: The descriptive component accepts events that cannot arise under its own fixed loader state machine. This remains an evidence replay gap under supplement sections 4/6/7, even though parsing grants no runtime authority and the aggregate runtime-custody schema is still mandatory. The active Go2 table currently has one row per observer; counterexample A uses a legal synthetic schema case, not a claim that current deployment rows were silently expanded.

Why current tests miss it: The new return-order cases exercise an ancestor versus descendant, while sibling positives have different observers. There is no multi-global same-observer return/request mix, nor a pending exec_return paired with a started terminal snapshot. Tracking only ancestors in returned does not constrain the referenced frame itself or its state.

Repair direction: Once a frame's return is evidenced, forbid later request-entry first observations for that frame while allowing multiple globals legitimately observed at its single return checkpoint. Require return observations to agree with a return-capable terminal state; retain legal failed prefixes where return occurred and later wrapper work failed. Do not forbid valid repeated globals at one return or valid unrelated sibling returns.

Required proof: Reject both preserved forms with prefix and complete replay where applicable; retain positive two-globals-at-return, request-before-return in one frame, nested request-entry/child-return/parent-return, and unrelated siblings. Bind any chronology that this component cannot represent only in the required aggregate schema, but reject contradictions already provable from these fields.

4. Preserved repros and executed verification

Original unchanged scratch source:
C:/Users/AG/新建文件夹/c4_assets/native_global_gap_review_probes_v1.py
SHA-256: 01791003d7c63b6ebfd0364acd86ec0dff4e9bac17c1c8ebaab932bf1d932fae

New independent scratch source:
C:/Users/AG/新建文件夹/c4_assets/native_global_gap_review_probes_v2.py
Remote byte copy: /data2/shared/ang.gao/native_global_gap_review_probes_v2.py
Both independently checked SHA-256: 176e1f186748fdbb09c70b3eb5ea16942d2e54e9121d77bc5c7731e9d971aa9a

All repros use the production importer/admission through the existing CPU fixture; only native extension delegate methods are fixed doubles. They are not alternate preparation implementations and prove no native binary or GPU property. The new tests deliberately assert acceptance of invalid histories, so their PASS result demonstrates the findings.

Local working directory for all local pytest commands was the frozen candidate. Environment for each: PYTHONDONTWRITEBYTECODE=1 and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1. Exact local commands/results:

python -m pytest -p no:cacheprovider -q C:/Users/AG/新建文件夹/c4_assets/native_global_gap_review_probes_v1.py
Result: 4 failed in 1.32s, exit 1, all at the intended production rejection sites described in section 2.

python -m pytest -p no:cacheprovider -q -s C:/Users/AG/新建文件夹/c4_assets/native_global_gap_review_probes_v2.py
Result: 4 passed in 0.81s, exit 0. Output:
COUNTEREXAMPLE: retry after sticky failure issues token 101 for held token 1; complete replay accepts
COUNTEREXAMPLE: same call first observed at exec_return then request_entry; complete=False
COUNTEREXAMPLE: same call first observed at exec_return then request_entry; complete=True
COUNTEREXAMPLE: exec_return observation accepted for still-started frame

python -m pytest -p no:cacheprovider -q tests/unit/optics/test_native_global_publication.py tests/unit/optics/test_runtime_environment.py tests/unit/optics/test_runtime_import_admission.py
Result: 238 passed, 3 skipped in 18.73s, exit 0. Per-skip reasons were not requested for this first suite; skips are not proof.

python -m pytest -p no:cacheprovider -q -rs tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_runtime_failure_observations.py tests/unit/optics/test_runtime_import_cold.py
Result: 296 passed, 13 skipped in 253.50s (0:04:13), exit 0. Skips: two unconfigured real-Go2 asset cases; two POSIX release-byte cases on Windows; two actual POSIX source-entry cases; six generated-module boundary cases plus one preload case requiring pinned CPython3.12 POSIX. The later remote five-test invocation covers the POSIX release-byte/source-entry cases at this HEAD. The local skipped generated-module and Go2 cases are not claimed as independently passed.

python -m ruff check optical_worker/native_bootstrap.py optical_worker/runtime_environment.py optical_worker/native_source_policy.py tests/unit/optics/test_native_global_publication.py
Result: All checks passed, exit 0.

Remote execution was deferred until the author explicitly confirmed the actual Go2 attempt was complete, to avoid host contention. All remote pytest commands used the same clean detached frozen checkout and `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -m pytest -p no:cacheprovider`:

- `-q -s /data2/shared/ang.gao/native_global_gap_review_probes_v2.py`: 4 passed in 0.40s, exit 0, same four counterexample messages as local.
- `-q tests/unit/optics/test_native_global_publication.py`: 70 passed in 4.30s, exit 0.
- `-q tests/unit/optics/test_native_source_policy.py::test_cold_bootstrap_and_production_contract_select_the_same_catalog tests/unit/optics/test_native_sources.py::test_actual_posix_checkout_catalog_matches_reviewed_git_bytes tests/unit/optics/test_native_bootstrap.py::test_actual_posix_source_entry_without_packaging`: 5 passed in 4.63s, exit 0. These independently cover exact POSIX release bytes and cold selector/source entry at this HEAD.

Static inspection covered the entire repair diff and relevant production call/data/state flow: private frame tuple creation/state update/checks; original spec retention; event fact issuance and immutable roots; pending/complete transitions; audit and issuance failure handling; normal versus failed evidence paths; phase and ancestor replay; same-frame/multi-row observations; raw metadata checks; finder wrappers and return marking; failure retention; source policy and catalog pins; notes and closed gates. Existing ordinary nested imports, thread/checkpoint failures, spec/frame/record mutation, terminal stdlib and event failure tests passed within the executed native-global suites.

5. Actual Go2 evidence and limitations

The author ran one isolated actual guarded attempt on this HEAD while the reviewer stayed local. I did not launch that worker or any GPU workload. After completion I independently read the request/identity/supervisor artifact and hashed the raw files. This is an author execution independently inspected, not a reviewer rerun.

Directory: /data2/shared/ang.gao/c2d3-9d1d7d0-go2-960-guarded-author-v10
Invoker: /data2/shared/ang.gao/run_c2d3_go2_guarded_author_v10.py
Invoker SHA-256: d2f2d978491cc0160634560932c6ed4328050ad7765fc376a9e7ae73a2ccd875
request.json: 3a2324fff5acae37f354e9ed8ecc495beed8cd2574550c359a1fa6ec67350077
identity.json: be9237220fe9ab42e96d52e0c31014636ffec7b28a0b59073d3106b9c7408315
supervisor_stdout.json: 6435888c74a24048066a7cd82263e5d302c020ab819188fa49e92eee85075a82
custody_artifact.json: 0896be3cfed5db6cb01f3e770c85d94d61c709b347baf74d5ad2002b5fe2e3da
prefix.json: 92d700303a0d5e981485129c972c850f72693d60b255019327229f8a9faea3d6
external_environment.json: cbf2569620b3546fa2a1ea00fda18373b60da47cd3f9038b53f7be3f9c66d4e7
setup_definitions.json: 04b2c24873ffd613d5ebdcc783c9ef7aa5708537d7896840aa90b5959344c18c
supervisor_stderr.txt was empty (0 bytes).

Request binds exact HEAD9d1d7d0, Warp target1, go2_static_960x640_v1, frozen environment e1c1a6b02128f329f5639af0c41b8bc6317002fd6a8c33eab469bf289f948ece, ACK1000ms and original operation120000ms. Parent/supervisor/worker PIDs: 4147753/4147759/4147760. Supervisor output records parent_alive=true, injected=false and worker_returncode=-9. Trace has setup sequence2 ARMED, original operation deadline expiry, kill_requested, process_exited(-9); summary EXITED, complete=true, deadline_expired=true, normal_complete_count=2. There is no sequence2 completion and no successful materialization/upload/readback artifact. The directory has no worker.json; aggregate worker side is an earlier prefix with next_sequence2, guard IDLE and only the two earlier completions, not post-timeout cleanup proof. No native-global captured interval success is inferred.

For historical comparison only, prior HEAD4933 attempt v9 is preserved at /data2/shared/ang.gao/c2d3-4933b7c-go2-960-guarded-author-v9. Its exact archived v1 report/note hashes are: invoker c95986ef4e21546079bb9dab2344248140ac876a09ebc8b1921b28caa3c8f038; request16aaa05dc900f33b035c4397d88eb643971d48593bda89cbcad7d7ff16d5b30b; identity5a2d5a92a0a4ec23fa4a7be647a5e183c9c3f5c440aa286cf41aed03c09b6740; supervisor84944b1fddc74594450cf32112990c3cff856024cf7535504d6a96129a16e8c4; custodyc60b2e4e2505c693f148c70242badb4ce72742aca2f517ef6b80eaeadedd95d5. These v9 values were read from the prior independently archived report, not rehashed in this round. That attempt also timed out at120s and reaped worker4118680(-9), with no success. It is not evidence for the repaired HEAD.

Timeout cause remains unisolated. No performance conclusion, deadline extension or optimization approval is made. The author mentioned a separate static hotspot hypothesis; I did not adopt it as actual worker evidence or expand this review into performance design.

Scoped read-only ps for4147753/4147759/4147760 returned only the header after completion. Read-only nvidia-smi showed only the pre-existing nvidia-cuda-mps-server2238410 at60MiB. No process was signaled by this reviewer. Short-lived SSH, Git and pytest utility PIDs were not exhaustively logged; this is a scoped residual check, not a global no-hang assertion. CPU child ownership/reaping follows the executed test harnesses. No fallback host or reviewer GPU execution was used.

6. Mandatory gaps and final scope

The two findings require repair, fresh freezing and independent re-review with both original and new repros preserved. Fixing them does not complete C2d3. Still mandatory before whole-stage code acceptance:

- Actual captured canonical materialization of both frozen Go2 assets in both lanes, using the production importer/guardian/supervisor and demonstrating the exact literal observer intervals and retained identities. CPU doubles and a supervised failure exit cannot replace this.
- Both lanes' canonical synthetic/asymmetric/dynamic cases, real allocation/upload/storage validation and supervised readback, without substitute materializers or success-by-skip.
- Real cold NumPy/backend partial create/exec/import and returned-owner/adoption/reconciliation/recording faults; retained sentinels through actual terminal disposition.
- Backend allocation/copy/submit-then-raise/event/ready-wait/register/audit faults, original-deadline hangs, ACK/DISARMED/EOF cases, and no new normal COMPLETE/ARM/sync/release after failed normal segments.
- Actual dual-GPU cold/target1/ambient0 restoration, foreign-device and UNKNOWN-scope cases with UUID/ordinal and dirty-history evidence.
- Successful sole supervised target sync and whole-graph cleanup with actual exit/reap, plus terminal-sync/cleanup/missing-exit failures and hangs.
- The final versioned exact-key runtime-custody schema binding actual imports/calls/returns, native globals, guard history, session/lane/target, runtime binding and prepared storage, with aggregate forged-success replay negatives. The present descriptive native-global component is not that schema.
- Required broader C0 A/K, C2d0/1/2, readiness/storage and full both-lane upload parity regressions for whole-stage acceptance.

The author's reported prior test counts are claims, not substitutes for the specific independent executions above. No native exception-safety, invisible between-checkpoint mutation, optical renderer, benchmark, resource budget or product activation proof is claimed.

Still CLOSED: whole C2d3/C2d/C4 acceptance, C2e/f, C4c3/c4, C4d/B3, product GPU preparation/activation, a3b, result/baseline, host payload, R2 and shipping. No final optical graph/readiness, active sample lease, renderer result or selected-backend completion is authorized. C4 is not complete.

The author may archive this complete report verbatim under the evidence-only exception. Archival cannot change this verdict, frozen scope or remaining obligations.
