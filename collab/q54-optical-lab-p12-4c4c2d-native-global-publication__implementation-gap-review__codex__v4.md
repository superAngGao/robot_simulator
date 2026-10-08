Initiative: q54-optical-lab-p12-4c4c2d-native-global-publication
Stage: implementation-gap-review
Author: codex, fresh independent reviewer
Version: v4
Date: 2026-10-08 (Asia/Shanghai)
Status: accepted
Related Files: native-global-publication design-request/design-review v1, implementation-gap-review v1/v2/v3, implementation-note v3; runtime-adoption design-request v2; optical_worker/runtime_environment.py; optical_worker/native_bootstrap.py
Owner Summary: Approve the bounded implementation-gap repair at the exact revision below. The same-call observation-after-admission counterexample now rejects in complete and prefix replay, all earlier preserved counterexamples remain rejected, and focused CPU regressions pass. No substantive finding remains established in this reviewed delta. This is not standalone CPU-stage or whole C2d3 code approval. Mandatory aggregate custody and full both-lane real-runtime evidence remain outstanding; no later gate opens.

1. Independent identity, exact candidate and verdict

Reviewer session: /root/native_global_implementation_gap_v4. Fresh independent context received requirements, the bounded task packet, candidate locations and prior reports/repros, not the implementation conversation. I did not author or edit the candidate, and did not delegate this review. Same-model independent context does not claim a separate security boundary or external Claude review.

Verdict: Approve, exclusively for this bounded IMPLEMENTATION GAP re-review of the V3-F1 repair and its affected invariants. No substantive blocking finding was established. This verdict closes this repair loop; it is not acceptance of a standalone CPU subset, whole C2d3 code, CUDA closure, C2d or C4.

Local repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
Local branch: implementation/c4-supervised-waits
Original implementation base: 8b7c9b88bcb42ddd704237395d27fa508899064c
Repair base / prior reviewed candidate: 3646630575509a075c8fc2f0d867a1dbaf025805
Intervening archive-only commit: 0c80a6458cca4901383cfd3931d876c85bbfea3c
Frozen reviewed HEAD: 8bc53bfe25acff9d384a3536d249c0e3268538d0
HEAD tree: c71fc32464d89ae70c2f47dd0e14d0f38e37b598
Primary remote: tileops-dev, 183.166.183.33:40031
Remote checkout: /data2/shared/ang.gao/robot_sim_c2d3_binding_wip, detached at the same HEAD/tree
Remote Python: /home/ang.gao/miniforge3/envs/robot_sim/bin/python, CPython 3.12.13

Initial and final local/remote working trees were clean at the frozen HEAD. No fallback host, candidate edit, package change, GPU execution, process signal, reset, push or merge was performed. Scratch report output is outside the candidate. Short-lived utility/test process identities were not exhaustively recorded.

Approved design snapshot: 80387e8d8199e9dc6741924b95ff1aeeae047b45. Independently verified raw design-request v1 SHA-256: 5c29fad420b9d31a848f8b53df07450b46f0fee0b4e2abe9f38aad7997be8b2c. Archived complete v3 report SHA-256: 9f84c943d8b99a10e1091c4c463a8972999584e00b0aed8f4fe2e2a520d74c11. Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully; explicit UTF8 reread corrected initial default-decoding output. Read the approved supplement, its review, authoritative runtime-adoption v2, full v1/v2/v3 gap reports and implementation-note v3. Narrowed subsequent reads covered truncated combined outputs; no truncated output was treated as a complete read.

2. Actual delta and prior-finding closure

Repair-base-to-HEAD contains six changed files: two new parser lines, 52 test lines (five parameterized cases), one catalog digest repin, eight PROGRESS lines, the implementation note and unchanged v3 report archive. The archive commit contains only the report. There is no loader, cache, deployment permission table, source policy, guard, watchdog-budget, shared-review-policy or private-context change.

The production change at runtime_environment.py:475-476 requires that an observed_pending event's call_sequence differ from every previously admitted event's call_sequence. The condition applies in complete and prefix modes. It does not depend on whether the new pending event uses request_entry or exec_return. The admitted map cannot lose earlier admissions; names cannot be observed twice and event sequence values are exact integers checked against their call records. A later same-call pending event therefore cannot evade the check by selecting a different configured name/token or checkpoint.

This matches the actual production ordering inspected at native_bootstrap.py:725-731, 797-866: return_native_call marks the delegate returned, removes the active delegated frame and runs the full finite-key checkpoint; only after that checkpoint and wrapper validation does complete_native_call iterate required globals to emit admissions. A new first observation cannot arise in that call after admission starts. The repair correctly tests call identity rather than prohibiting all later observations across the session. A completed child may precede its still-running parent's later pending observations; unrelated observers remain legal.

V3-F1 is closed. The original unchanged two-global repro reorders genuine production-fixture evidence from pending A, pending B, admitted A, admitted B to pending A, admitted A, pending B, admitted B. Both require_complete=False and True now raise ValueError("native observation after same-call admission") at the added check. Candidate regression tests also reject this reorder in a genuine unrelated-import-failure prefix, while leaving the admitted observer's actual completed state intact. They do not manufacture an admitted event on a failed call. Positive cases execute a completed child followed by three parent globals, both all-at-return and mixed request/return observation, then audit, issue and replay successfully.

Earlier exact counterexamples remain closed in independent current-HEAD executions:
- v1 coherent copied spec: native call binding drift.
- v1 coherently changed event token pair: native global issued event drift.
- v1 wrong create-call phase: native call relation.
- v1 ancestor return before nested child observation: parent return preceded nested event.
- v2 poisoned issuer retry after token drift: native global issued event drift again.
- v2 same-frame request after return, complete and prefix modes: native request observation after return.
- v2 exec_return observation with started call: native return observation in started call.

Inspected surrounding immutable call/event binding checks, poisoned issuance versus healthy live checks, required-row completion, nested ancestry/return replay and final complete/prefix rules. No new integrity defect was established in those affected paths. The unchanged cache's preserved projection-detachment, rebinding/eviction, trusted-digest and fresh-file checks passed. This is bounded regression evidence, not a new exhaustive cache or loader audit.

The structured catalog comparison showed exactly one source_digests row changed, for optical_worker/runtime_environment.py, from cd201e983933f970a25fc5cd3ffd3bacdc6d6f218fafbe5e8fe3ceb79e272358 to 70d3bc86e1bb5d8d22b3fb440b05c79a73df0a4003e4f89a8ed8d70fcaca2b7f. The latter independently equals SHA-256 of git show HEAD:optical_worker/runtime_environment.py. No catalog lane membership, other source digest or AST policy changed.

3. Executed commands and results

All remote pytest executions used the frozen checkout as cwd and this prefix:

PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -m pytest -p no:cacheprovider

Focused regression suffix:

-q -rs tests/unit/optics/test_native_global_publication.py tests/unit/optics/test_runtime_environment.py tests/unit/optics/test_runtime_import_admission.py tests/unit/optics/test_native_source_policy.py::test_cold_bootstrap_and_production_contract_select_the_same_catalog tests/unit/optics/test_native_sources.py::test_actual_posix_checkout_catalog_matches_reviewed_git_bytes tests/unit/optics/test_native_bootstrap.py::test_actual_posix_source_entry_without_packaging

Observed: 274 passed in 19.75s, exit 0, no skips. This includes all five added cases and actual pinned-POSIX catalog/release-byte/cold-source-entry checks. The importer fixture replaces fixed native delegate methods and exercises production admission mechanics; those fixture passes do not prove native binary or GPU behavior.

Preserved repro suffix:

-q --tb=short /data2/shared/ang.gao/native_global_gap_review_probes_v1.py /data2/shared/ang.gao/native_global_gap_review_probes_v2.py /data2/shared/ang.gao/native_global_gap_review_probes_v3.py

Observed: 10 failed, 2 passed in 1.19s, exit 1. The ten preserved tests intentionally assert invalid acceptance, so their failures at the exact rejection checks listed above establish closure of those counterexamples. They are not ten product regressions. The two passing tests are the v3 cache positives. Original files were not edited. Local and remote SHA-256 independently match:

v1: 01791003d7c63b6ebfd0364acd86ec0dff4e9bac17c1c8ebaab932bf1d932fae
v2: 176e1f186748fdbb09c70b3eb5ea16942d2e54e9121d77bc5c7731e9d971aa9a
v3: 59fd327ac3f07025c539b86ba6e0d1f0aaa73dce1e4fec2fa5615053c8c0e92a

Local static command:
python -m ruff check optical_worker/runtime_environment.py tests/unit/optics/test_native_global_publication.py
Observed: All checks passed, exit 0.

Git verification included status --porcelain=v1, rev-parse HEAD, branch --show-current, rev-parse 'HEAD^{tree}', log -3, diff --stat/--numstat, show --stat for the archive commit, and diff --check 3646630575509a075c8fc2f0d867a1dbaf025805 HEAD. Protected-path diff --exit-code against that base covered AGENTS.md, collab/README.md, collab/REVIEW_PRINCIPLES.md, native_bootstrap.py, native_source_policy.py and tests/data/runtime_environments; it passed. Structured JSON comparison, rather than visual inspection of a giant single-line diff, verified the sole catalog change.

4. Historical real-Go2 records replayed with current parser

I launched no GPU work. The two uninstrumented actual Warp Go2 executions remain historical narrow evidence from 3646630575509a075c8fc2f0d867a1dbaf025805. Their v3 report/note describe successful canonical upload/readback of 42 arrays, original 120-second operation bound, sole terminal sync/release and actual worker exit/reap. This round does not assert a new execution of those facts or promote the older instrumented diagnostic into acceptance.

Fresh CPU replay was executed over SSH by piping a Python program to:

cd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip && /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -

For each directory below, the program loaded request.json, asserted its exact historical revision, loaded RuntimeEnvironmentContract.from_payload from request.environment_path, asserted the environment digest, extracted canonical_materialization.json['native_global_observations'], and called validate_native_global_observations(observations, environment, require_complete=True) from current HEAD. Both records passed, each with six events. Fresh canonical record hashes match the archived v3 hashes:

/data2/shared/ang.gao/c2d3-3646630-go2-960-guarded-author-v11
canonical_materialization.json SHA-256 e111d7a358db1a922660a1a55c7a653f13ac723fb719982651935fce0f38ee7b

/data2/shared/ang.gao/c2d3-3646630-go2-1080-guarded-author-v12
canonical_materialization.json SHA-256 3990e3e93d685e28967c6c2fbb8d53d185964c7dcba01a3b3a076e88a539f751

This verifies compatibility of actual historical observation histories with the new parser. It is not current-HEAD GPU validation, a new materialization, full dependency-file verification or final aggregate runtime-custody acceptance. No lifecycle artifacts or native runtime were rerun in this round.

Replay inspector PID16035, parent16033, completed exit 0; final read-only ps -p 16035,16033 -o pid,ppid,stat,args returned the header only. Remote test/SSH utility PIDs were not exhaustively tracked. No host-global no-hang or GPU-empty claim is made, and no unrelated process was touched.

Two reviewer shell issues were corrected without affecting product execution: one inline remote Python -c command lost quoting and failed in bash before Python ran; piping source on stdin succeeded. One unquoted PowerShell HEAD^{tree} argument was misinterpreted; quoting it yielded the correct tree above. Initial default file encoding and oversized output issues were resolved with explicit UTF8/narrow reads. These were tooling issues, not candidate failures.

5. Limitations, remaining obligations and scope

Checks were proportional to this two-line descriptive-parser repair. I did not rerun the complete broader source/failure/cold suites, all prior local suites, actual GPU execution, fault injection, hang or dual-device cases. Existing historical outputs and author counts do not substitute for this round's explicitly listed fresh executions. No native exception-safety, invisible between-checkpoint mutation, performance, renderer, resource budget or native sandbox proof is claimed.

Before whole C2d3 code acceptance, the mandatory work remains:
- Complete CUDA deployment closure and both lanes' canonical synthetic/asymmetric/dynamic and both real Go2 allocation/upload/storage/readback evidence.
- Real cold/partial native import, returned-owner/adoption/reconciliation/recording failures and sentinels retained through process-terminal disposition.
- Both lanes' allocation/copy/submit-then-raise/event/wait/register/audit faults; original-deadline hangs, ACK/DISARMED/EOF and no new normal COMPLETE/ARM/sync/release after failure.
- Actual dual-GPU cold/target1/ambient0 restoration, foreign-device and UNKNOWN histories, with physical identity/scope evidence.
- Sole supervised terminal sync/whole-graph release and actual exit/reap, plus terminal sync/cleanup/missing-exit failures and hangs.
- Final versioned exact-key runtime-custody schema binding actual file calls/returns, native globals, session/lane/target, runtime binding, prepared storage and guard history, with aggregate forged-success negatives.
- Required full C0/C2d/readiness/storage/upload-parity regressions and whole independent acceptance review.

The descriptive component grants no live object or lifecycle authority. Approval of this bounded repair scan neither completes that aggregate contract nor accepts a standalone CPU stage. Still CLOSED: whole C2d3/C2d/C4 acceptance, C2e/f, C4c3/c4, C4d/B3, product GPU preparation/activation, a3b, result/baseline, host payload, R2 and shipping. No final optical graph/readiness, active sample lease, renderer result or selected-backend completion is authorized. C4 is not complete.

The author may archive this complete report verbatim under the evidence-only exception. Archival cannot change the verdict, exact reviewed revision, bounded scope or remaining obligations.
