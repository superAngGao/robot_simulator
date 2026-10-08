Initiative: q54-optical-lab-p12-4c4c2d-native-global-publication
Stage: implementation-gap-review
Author: codex, fresh independent reviewer
Version: v3
Date: 2026-10-08 (Asia/Shanghai)
Status: in_review
Related Files: native-global-publication design-request/design-review v1, implementation-gap-review v1/v2, implementation-note v2; runtime-adoption design-request v2; optical_worker/native_bootstrap.py; optical_worker/runtime_environment.py
Owner Summary: Request changes for one remaining same-call replay-order contradiction. All eight original counterexamples reject. Poisoned issuance and configuration-cache checks pass the bounded checks below. Two actual uninstrumented Go2 Warp runs now have successful narrow-path evidence, independently inspected here; they do not complete C2d3 or its aggregate schema and mandatory matrix.

1. Independent identity, candidate and verdict

Reviewer session: /root/native_global_implementation_gap_v3. This fresh session received the review packet, approved requirements, prior reports and original repros, not the implementation conversation. I did not author/edit the candidate and did not delegate the review. Same-model context isolation is not a separate security boundary or an external Claude review.

Verdict: Request changes for V3-F1 (P2), confined to this native-global implementation gap re-review. No other substantive defect was established in the repair delta or immutable-configuration optimization. This is not whole C2d3 code approval and does not approve a CPU subset as final acceptance.

Repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
Branch: implementation/c4-supervised-waits
Original implementation base: 8b7c9b88bcb42ddd704237395d27fa508899064c
Repair base / previous reviewed candidate: 9d1d7d025a4798cef8e7aff6318c530320ef7736
Intervening archive-only commit: ed6c9bf
Frozen reviewed HEAD: 3646630575509a075c8fc2f0d867a1dbaf025805
Primary remote: tileops-dev, 183.166.183.33:40031
Remote checkout: /data2/shared/ang.gao/robot_sim_c2d3_binding_wip, same detached HEAD
Local Python: C:/Python314/python.exe, CPython 3.14.0
Remote Python: /home/ang.gao/miniforge3/envs/robot_sim/bin/python, CPython 3.12.13

Initial and inspected concluding local/remote trees were clean at the frozen revision. The repair-base delta is ten files: code/tests/pins plus PROGRESS and the report/note. No environment deployment permission table, historical profile, watchdog budget, AGENTS, shared review protocol or private context changed. git diff --check passed. The reviewer wrote only scratch probes/report outside the candidate and copied the new scratch probe outside the remote checkout. No package install, GPU execution, process signal, reset, push, merge or implementation edit was performed by this reviewer.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully. Explicit UTF-8 rereads corrected initial PowerShell default-decoding output. Read the native-global design/design review, authoritative runtime-adoption v2, complete original reports and current implementation note. Narrower reads followed large truncated outputs. Approved design snapshot80387e8d8199e9dc6741924b95ff1aeeae047b45 retains independently checked raw design SHA-256 5c29fad420b9d31a848f8b53df07450b46f0fee0b4e2abe9f38aad7997be8b2c.

2. Prior findings and affected invariants

Original v1 F1/F2/F3 exact repros remain closed: coherent spec replacement rejects with native call binding drift; paired stored-token mutation rejects with native global issued event drift; wrong create phase rejects with native call relation; parent return before child observation rejects with parent return preceded nested event.

V2-F1 is closed for its poisoned-issuance counterexample. _native_global_observations now checks thread/call history length, exact native-call bindings, held global bindings and immutable event facts even after failure. The second issuance attempt rejects the same corrupted token instead of emitting it. The live flag skips only healthy sys.modules/raw-metadata presence checks after poison; it does not skip immutable issued-history checks. Legitimate pre-return and post-return failed prefixes survive importlib removal, and repeated terminal stdlib lookup remains available. Candidate tests cover issuance/audit/unrelated-import failure entry followed by token, erase, reorder or frame drift and repeated reporting. Failure reporting does not fabricate events or reopen admission.

V2-F2's exact counterexamples are closed. Replay rejects same-frame request_entry after an earlier exec_return, with both prefix and complete parsing, and rejects a return observation paired with a still-started call. Positive multiple-globals-at-return, request-before-return/mixed observation, legitimate failed prefixes, nested observer order and unrelated sibling cases pass. The broader same-frame chronology still has V3-F1 below.

Configuration optimization: the cache is bounded to one entry, keyed by full canonical bytes, and stores recursively immutable tagged tuples/scalars produced only after decode and _validate. Every to_payload recursively allocates fresh dict/list containers. No file, loader/spec, module identity, call/event, runtime scope, guardian or execution observation is cached. check_live still compares the immutable-contract projection against mutable payload and all indexes and checks the trusted bound digest; audit still calls verify_files, which revalidates structure and rereads filesystem/interpreter facts. Independent probes verified every nested mutable container is detached between outputs, valid byte rebinding selects a new configuration and fails the original trusted digest, cache eviction does not mix configurations, and a changed file after warming the cache still fails fresh verify_files. The candidate coherent payload/index mutation test also passes. No cache-integrity finding was established.

3. Finding

V3-F1 — P2: Replay accepts a new first observation in the same native call after that call has already begun admitting globals.

Location: optical_worker/runtime_environment.py:472-486 (observed_pending branch), with admission handling at489-503. Production ordering is established by native_bootstrap.py:725-731 (return_native_call), 797-845 (full finite-key checkpoint), and848-866 (complete_native_call).

Trigger: Use a legal environment containing two native-global rows bound to the same observer/exec/phase. The actual production finder/loader with a fixed native delegate publishes both globals during that one exec. Genuine evidence is pending A, pending B, admitted A, admitted B, with one call_sequence and exec_return checkpoints. Reorder only the detached event list to pending A, admitted A, pending B, admitted B and renumber contiguous indexes. Leave all row, call, parent, phase, token, absence and terminal completed facts unchanged. validate_native_global_observations accepts both require_complete=False and True. The preserved independent v3 repro passes on Windows CPython3.14 and pinned Linux CPython3.12.13, deliberately demonstrating this bad acceptance.

Why this is a component-level contradiction: the normal-return checkpoint visits the entire finite configured key set before complete_native_call can emit any admitted event. If B existed, its pending event necessarily precedes A's admission. If B was not yet available at that checkpoint, later wrapper work cannot establish a first observation under an already returned native interval, and complete_native_call does not create pending observations. Thus after any admission for a call, no later observed_pending event can belong to that same call. This is inferable from the component's call_sequence, operation and event order; it does not require the deferred aggregate file/guard clock. Supplement sections4/6/7 require replay to represent that production state machine, and design-review v1 explicitly requires rejection of contradictions available inside this component.

Consequence: the descriptive component accepts a history that cannot be emitted by the fixed production state machine. It remains an evidence-integrity/replay defect, not a demonstrated live issuer bypass, GPU execution, successful normal COMPLETE, extra sync or release. The current deployment table has one global per observer, so this reproduction uses a legal synthetic schema case; it does not imply permission to change the deployed table. Immutable live event checks prevent this detached reorder from being freshly issued by an unchanged healthy registry.

Why checks miss it: the new returned set blocks later request_entry events, but intentionally permits additional exec_return events. observed_pending checks duplication/token reuse and already-admitted ancestors, not admission already seen for the referenced frame itself. Both globals in this repro use exec_return, so the new request-after-return condition never rejects it. Existing two-global tests alter the second checkpoint to request_entry; they do not reorder an admission between same-return pending observations.

Repair direction: once replay has seen any admission for a call_sequence, reject a later first observation for that same sequence. Keep legitimate multiple pending observations at one return before admissions, request observations before return, child admissions before a still-running parent's later first observation, and unrelated sibling returns. Do not apply a global ban after the first admission anywhere in the session.

Required proof: preserve v3 repro; reject its complete and non-complete parser modes. Add an incomplete/failed-history variant where applicable without claiming that admitted events may be attached to failed calls. Retain genuine two-/three-global same-return and request/mixed positives, nested child-before-parent cases and unrelated siblings. Repin changed exact source bytes and obtain a fresh independent re-review. Do not defer this self-contained contradiction to the aggregate schema.

The main author independently raised this possible edge case during review. I assessed it from the approved contract and production flow, constructed the above repro independently, and did not accept the suggestion as evidence or edit the candidate. There was no requested verdict.

4. Commands and observed results

All local pytest cwd was the frozen repository. Environment: PYTHONDONTWRITEBYTECODE=1 and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1. Cache provider disabled. CPU fixture tests replace only fixed native delegate methods while exercising the production importer/admission; they certify mechanics, not native binaries or GPU behavior.

python -m pytest -p no:cacheprovider -q --tb=short C:/Users/AG/新建文件夹/c4_assets/native_global_gap_review_probes_v1.py C:/Users/AG/新建文件夹/c4_assets/native_global_gap_review_probes_v2.py
Result: 8 failed in2.22s, exit1, all at the expected production rejection sites listed in section2. These preserved tests intentionally assert bad acceptance; failure here is proof of exact repro closure, not eight new product regressions.

python -m pytest -p no:cacheprovider -q -rs tests/unit/optics/test_native_global_publication.py tests/unit/optics/test_runtime_environment.py tests/unit/optics/test_runtime_import_admission.py
Result:261 passed,3 skipped in18.36s, exit0. The three skipped interpreter-data cases require CPython3.12 POSIX; skips are not proof.

python -m pytest -p no:cacheprovider -q -s C:/Users/AG/新建文件夹/c4_assets/native_global_gap_review_probes_v3.py
Result:4 passed in0.75s, exit0. Two passes demonstrate V3-F1 in the two parser modes; two passes verify cache/freshness invariants.

python -m ruff check optical_worker/native_bootstrap.py optical_worker/runtime_environment.py optical_worker/native_source_policy.py tests/unit/optics/test_native_global_publication.py tests/unit/optics/test_runtime_environment.py tests/unit/optics/test_runtime_import_admission.py
Result: All checks passed, exit0.

python -m pytest -p no:cacheprovider -q -rs tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_runtime_failure_observations.py tests/unit/optics/test_runtime_import_cold.py
Result:296 passed,13 skipped in263.96s, exit0. Skips: two real-Go2 asset paths not configured locally; two POSIX release-byte checks; two actual POSIX source-entry checks; six generated-module cases and one preload case requiring pinned CPython3.12 POSIX. The remote five parameterized release-byte/catalog/source-entry tests above cover those exact platform checks. The skipped generated-module cases are not claimed independently passed in this round.

Remote CPU checks were postponed until the main session confirmed both actual Go2 runs had finished. Remote cwd was the same clean frozen checkout. Commands used PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -m pytest -p no:cacheprovider:

- -q -s /data2/shared/ang.gao/native_global_gap_review_probes_v3.py:4 passed in0.59s, exit0, same two counterexample messages and cache positives.
- -q -rs tests/unit/optics/test_native_global_publication.py tests/unit/optics/test_native_source_policy.py::test_cold_bootstrap_and_production_contract_select_the_same_catalog tests/unit/optics/test_native_sources.py::test_actual_posix_checkout_catalog_matches_reviewed_git_bytes tests/unit/optics/test_native_bootstrap.py::test_actual_posix_source_entry_without_packaging:96 passed in9.92s, exit0. These include actual POSIX release-byte/catalog/cold-selector/source-entry checks, not GPU initialization.

Preserved local scratch sources under C:/Users/AG/新建文件夹/c4_assets and remote same basenames under /data2/shared/ang.gao were independently hashed:
v1:01791003d7c63b6ebfd0364acd86ec0dff4e9bac17c1c8ebaab932bf1d932fae
v2:176e1f186748fdbb09c70b3eb5ea16942d2e54e9121d77bc5c7731e9d971aa9a
v3:59fd327ac3f07025c539b86ba6e0d1f0aaa73dce1e4fec2fa5615053c8c0e92a
The v1/v2 originals were not edited. v3 contains the complete new reproductions and cache checks.

Static inspection covered the full repair diff, surrounding frame/binding state transitions, return-before-wrapper ordering, finite-key checkpoint, retention before validation/adoption, event issuance and private facts, failure intake and terminal stdlib, completion/audit, parser chronology, configuration cache/data copying, full live payload/index/file checks and active source pins. Revision/status/diff checks used git rev-parse HEAD, git branch --show-current, git status --porcelain, git diff --name-status and git diff --check against the repair base. No author test count substitutes for these executions.

5. Actual narrow Go2 evidence independently inspected

The author ran two uninstrumented actual guarded Warp target1 cases sequentially while the reviewer stayed local. I did not launch either worker or overlap a remote CPU suite with them. After completion I inspected the outside-checkout invoker, request/identity/canonical/global/upload-validation/worker/supervisor artifacts, local byte-copy hashes and selected matching original remote hashes. These are author executions independently inspected, not reviewer reruns.

Both requests bind exact HEAD3646630575509a075c8fc2f0d867a1dbaf025805, environment e1c1a6b02128f329f5639af0c41b8bc6317002fd6a8c33eab469bf289f948ece, original ACK1000ms/operation120000ms, and the actual frozen Go2 model directory. Both supervisor records show injected=false, parent_alive=true, worker_returncode0, EXITED/complete,6 normal completions,7 ARMs, no deadline expiry and no kill. Worker summary reports42 allocations,42 H2D,42 D2H, two event create/record/wait each,5 runtime import/init leaves,1 successful terminal target sync, guardian RELEASED and release completed, with dirty ordinals[1] and UNKNOWN false. Validation lists42 arrays, final_execution_readiness_issued=false; sample_submission_count=0. These are setup operations, not measured samples.

960 artifact: /data2/shared/ang.gao/c2d3-3646630-go2-960-guarded-author-v11
Parent/supervisor/worker:4176494/4176527/4176529
Invoker SHA:5163f7f76be3d1c71c63dcac1eaba7ddf41a6ed534bf5d3345069da3dbc5bc19
request.json:ae860fd249eba606fc3d30cecf082ff3dabdfada86556699fa2ac30aeb46ad64
identity.json:ac82cd37c78131df358574fefdf70817b8acdeee047b57c094545571ed337545
canonical_materialization.json:e111d7a358db1a922660a1a55c7a653f13ac723fb719982651935fce0f38ee7b
input_validation_report.json:f76ba232ba326d720de31334fdab6608a265449b379a15adca1e5303e9eb9e20
worker.json:eb4eae479a56cb5391efdb62b364daa506ba41ab4653694aca1c9bbdeb5f06d1
supervisor_stdout.json:a0ab164f4d39a45adc52bf14a776b51bd075fd4c5b9fb5b3b7f7786561405158
custody_artifact.json:bf2f141750ee3140838e245a9d58592c7349112f2ddb47ba8a5f1335fab909b5

1080 artifact: /data2/shared/ang.gao/c2d3-3646630-go2-1080-guarded-author-v12
Parent/supervisor/worker:4181358/4181382/4181383
request.json:628b66067f868fedab738ca2ac1bcc25237ab0545209bc4c7752d13b95f5a65d
identity.json:9bef9dca355ee0f79bb8c25aa1a3d2651844a2bbe46b5a6fbf7dff9d47e26ff8
canonical_materialization.json:3990e3e93d685e28967c6c2fbb8d53d185964c7dcba01a3b3a076e88a539f751
input_validation_report.json:7f34a3f42ccf973f8516b75182696e380608d23d6b4910c08bf07126afaed8f5
worker.json:eb4eae479a56cb5391efdb62b364daa506ba41ab4653694aca1c9bbdeb5f06d1
supervisor_stdout.json:682484f17960c79b4ad863ad02afd97a7e72f916333dc99a4578d88c3445fe7e
custody_artifact.json:e0ea74bef803f2cdd886d9bbe1e81cfffe8ca0e5894a22a73951a23557d08cd5

Both canonical records contain the six expected native-global pending/admitted events. For960 the actual first requests are numpy.random.bit_generator for cython_runtime, numpy.random._common for _cython_3_2_8, and scipy._cyutility for _cython_3_2_5. Observer intervals and pending/admitted tokens agree with the literal table. On the pinned remote interpreter, independently ran validate_native_global_observations(..., require_complete=True) on both captured JSON records against the frozen environment; both passed. Locally ran validate_custody_artifact on both aggregate lifecycle artifacts; both passed. This existing lifecycle aggregate is not the still-required final runtime-custody schema binding every external/runtime fact.

960 manifest:ac3d950309917168d0bb5758e9916adcffcbb17bd68f6611e4fe0f3bdfed3a12
1080 manifest:4785c42146d8707453654eefbc4c94e38e683ec4b5659536240e6b1048b2279e
Both content:8e79f7061cf08e8778fd4fca57aac7ad6936ddf3e9ef67f0f6c074faf69c3c18
Both topology:3ef81f6fc01bfda47983cb77ddb7523d45b8b0291010956f400652ccb7561e69

The previous9d stack diagnostic remains diagnostic-only; this review does not promote its faulthandler/runpy prelude into normal acceptance or claim exact causal performance gains from cache changes. Previous9d v10 and4933 v9 uninstrumented timeouts remain historical failures, not successful runs or current-head proof. No timing/resource/renderer-performance acceptance is asserted.

Tooling limitations were kept separate: one local Python stdin inspection failed because PowerShell replaced non-ASCII literal path characters; relative paths fixed it. An attempted local parse of the remote environment correctly rejected POSIX paths on Windows; the actual environment/global replay was subsequently executed successfully on the pinned Linux interpreter. These are reviewer tooling/environment limitations, not new product defects. Read-only ps for all six actual-run PIDs showed headers only after completion. No reviewer GPU process existed, and no reviewer signal was issued. Utility/test subprocess PIDs were not exhaustively tracked; this is a scoped residual check, not a host-global no-hang claim.

The author subsequently reported unrelated tilesight-tl012 Python4184984 using740MiB alongside the pre-existing MPS2238410. I did not independently query NVML in this round, did not treat those other processes as review residuals and touched neither. No globally empty GPU-process claim is made.

6. Remaining mandatory scope

V3-F1 requires repair, a fresh frozen revision and independent re-review with all original repros preserved. The successful Warp Go2 paths narrow an earlier evidence gap but do not complete the parent matrix. Still required before whole-stage acceptance: both lanes' complete canonical synthetic/asymmetric/dynamic and real Go2 allocation/upload/storage/readback proofs; cold/partial native import and returned-owner/adoption/reconciliation/recording failures; actual retained sentinels; allocation/copy/submit-then-raise/event/wait/register/audit faults; original-deadline hangs and ACK/DISARMED/EOF cases; real dual-GPU cold/ambient/foreign/UNKNOWN scope; terminal sync/cleanup/missing-exit failures/hangs; full C0/C2d/readiness/storage/upload parity regressions; final versioned exact-key runtime-custody schema and aggregate forged-success negatives.

No invisible between-checkpoint mutation, trusted native internals, renderer, optical readiness or native sandbox proof is claimed. Still CLOSED: whole C2d3/C2d/C4 acceptance, C2e/f, C4c3/c4, C4d/B3, product GPU preparation/activation, a3b, result/baseline, host payload, R2 and shipping. No final optical graph, active sample lease, renderer result or selected-backend completion is authorized. C4 is not complete.

The author may archive this complete report verbatim under the evidence-only exception. Archival cannot change its verdict, exact reviewed scope or remaining obligations.
