Initiative: q54-optical-lab-p12-4c4c2d-native-global-publication
Stage: implementation-gap-review
Author: codex, fresh independent reviewer
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: in_review
Related Files: native-global-publication design-request/design-review v1; runtime-adoption design-request v2; native-self-alias design-request v1; optical_worker/native_bootstrap.py; optical_worker/runtime_environment.py
Owner Summary: Request changes for three independently reproduced implementation gaps in exact spec binding, live evidence issuance, and descriptive replay. Passing CPU mechanics do not establish the deployed native intervals or complete C2d3. The actual frozen-revision Go2 attempt expired during materialization and supplies no successful upload/readback proof.

1. Identity, candidate and bounded verdict

Reviewer session: /root/native_global_implementation_gap_v1. Fresh independent session, receiving the bounded task packet and repository/evidence locations rather than the author's implementation conversation. I did not author or edit the candidate, and did not delegate the review. Same-model independent context is not a separate security boundary or external Claude review.

Verdict: Request changes, for this bounded IMPLEMENTATION GAP REVIEW only. This is not whole C2d3 code acceptance, and it neither opens a gate nor converts a CPU subset into final acceptance.

Local repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
Local branch: implementation/c4-supervised-waits
Implementation base: 8b7c9b88bcb42ddd704237395d27fa508899064c
Frozen reviewed HEAD: 4933b7ca46eb594f51c924ce88f8bd124bb14ba8
Primary remote: tileops-dev, 183.166.183.33:40031
Remote checkout: /data2/shared/ang.gao/robot_sim_c2d3_binding_wip, detached at the same HEAD
Remote interpreter: /home/ang.gao/miniforge3/envs/robot_sim/bin/python, CPython 3.12.13
Initial and concluding local/remote candidate checks were clean at the same revision; git diff --check passed. No fallback host, candidate repair, package installation, GPU workload, process signal, push or merge was performed by this reviewer.

Approved design snapshot: 80387e8d8199e9dc6741924b95ff1aeeae047b45, collab/q54-optical-lab-p12-4c4c2d-native-global-publication__design-request__codex__v1.md. Independently checked raw SHA-256: 5c29fad420b9d31a848f8b53df07450b46f0fee0b4e2abe9f38aad7997be8b2c. Archived independent design-review v1 at the implementation base has independently checked raw SHA-256 0a8d5d31b47be7a6aed3e30c09c446b2459ff6c8c3ac1ea3f8f70eee463a3577. Its design approval is not code evidence.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully, including an explicit UTF-8 reread after PowerShell's initial default decoding produced mojibake. Read the exact supplement, its design review, the relevant predecessor contracts and gap evidence. Inspected the 17-file base-to-HEAD implementation delta and the surrounding production call/state/custody flow. Large combined outputs were followed by targeted reads for the relevant code and authoritative requirements.

2. Findings

F1 — P2: The supposedly immutable delegated-frame binding does not bind its exact spec.

Location: optical_worker/native_bootstrap.py:647-663 (_check_native_call), 700-719 (begin_native_call), with audit_native_globals at 843 and evidence issuance at 858.

Trigger/reproduction: Run a valid native-global admission with the actual production finder/loader and fixed native delegate. Shallow-copy the parent's ModuleSpec; replace admission.specs[parent], the live parent's __spec__, and the spec fields of that parent's existing create/exec frames with the copy. Leave native_call_bindings, native_global_bindings, loaders, modules, tokens and all private frame identities unchanged. The copy is a distinct, unissued and unadopted object. admission.audit() and admission.native_global_observations() both succeed, with admission.failed still false.

Cause: The private frame tuple binds frame/loader/module/operation/phase/parent/absence/wrapper/state, but omits the original spec. _check_native_call checks only self.specs[loader.name] is frame['spec']; both mutable references can move together. The module metadata check likewise agrees with the replacement index. Keeping the original spec in retained owners does not prove that the spec now asserted by the frame is that original object.

Consequence: A copied spec can stand in for the exact spec actually used by delegated execution and still pass a successful segment audit and issued observations. This contradicts supplement section 4's immutable exact frame binding and section 7's spec-drift rejection. It is an identity/custody invariant failure within the explicitly requested mutation tests, not a claim of defense against arbitrary hostile Python or of an observed native use-after-free.

Why existing tests miss it: The new suite mutates/copies the frame object or global record itself; it does not coherently replace the spec references while preserving the issued frame identity and immutable registry tuple.

Repair direction: Include the original exact spec (and any other promised exact frame roots) in the private immutable binding and compare each mutable projection against that issued identity. Keep the original held during rejection. Do not repair by accepting equivalent metadata or merely checking that a replacement is adopted.

Required proof: Preserve the scratch reproduction as a rejection test at both pending and completed boundaries, prove sticky failure before successful segment completion/evidence issuance, and retain original plus available replacement roots. Regress ordinary nested imports and unchanged repeated audits.

F2 — P2: Evidence issuance validates live records and serialized events separately, allowing event tokens to detach from actual object identity.

Location: optical_worker/native_bootstrap.py:740-751 (_global_event), 858-882 (native_global_observations); audit_native_globals at 843 does not cross-check issued event facts.

Trigger/reproduction: Complete one valid admission whose exact private live global record has token 1. Change both stored native_global_events entries' object_token values to 101, leaving the object, record, frame, private binding and token 1 untouched. admission.audit() succeeds. native_global_observations() emits token 101, and validate_native_global_observations(..., require_complete=True) accepts it.

Cause: Issuance checks each live call/global binding, then copies a mutable event list and applies a parser that knows only configuration and the internally consistent event pair. Nothing binds the copied events' token/relation/checkpoint facts back to the issuer-owned live observation record or an immutable issued event ledger.

Consequence: The production issuer can report a different object identity token from the one it actually assigned to the exact held object. The same structural gap leaves other self-consistent changes to descriptive history unauthenticated at issuance. This violates supplement section 6's issuer-owned token and live-registry issuance contract, and section 7's forged-token rejection obligation. A downstream parser by itself cannot know the correct opaque token; the failure here is specifically live issuance, not an assertion that arbitrary paired token renumbering in a detached record can be detected without an identity anchor.

Why existing tests miss it: The new negative test changes only the final admission token in a copied payload. That breaks pending/admission equality and is rejected. It does not change the stored pair consistently and request a fresh issue from the live admission object.

Repair direction: Preserve immutable issuer facts for each observation/admission and verify the projected event history against the actual private object/frame/token bindings before emission and successful completion. Do not rebuild a missing history at audit. Poison on detected drift and retain the actual owners.

Required proof: Mutate both tokens consistently in the live event history, remove/reorder/replace a live event while keeping parser-level consistency where possible, and verify issuance rejects without manufacturing new admission events. Keep a positive unchanged issuance/read-only-audit test. Do not give parsed records authority over runtime registries.

F3 — P2: Descriptive replay accepts known-impossible phase and nested-return histories.

Location: optical_worker/runtime_environment.py:419-438 (call replay), 458-474 (first-observation ordering).

Two independent accepted counterexamples:

(a) For a module configured exclusively for materialization, change its create call's phase to runtime while keeping its subsequent exec/materialization and events intact. require_complete=True accepts. The production resolver requires rule['phase'] == self.phase, so this cannot describe the same permitted call. The parser checks only membership in the global phase enum for calls without a global event; it does not check each call against its module row.

(b) Obtain a valid nested trace: parent pending at request_entry, child pending/admitted, parent admitted. Change the first parent observation to exec_return with request_name=null. Replay still accepts. This now claims the parent delegate had returned before its nested child's first observation. The actual wrapper suspends native execution after return, so a later nested child cannot occupy that parent's delegated interval. Replay blocks observations after an ancestor's admitted event, but not after the ancestor's earlier exec_return observation.

Consequence: The component accepts internally contradictory descriptive histories even before the still-required aggregate runtime-custody binding is considered. This violates supplement section 6's exact row/phase and ordering checks. It does not directly authorize execution or release, since the parser correctly remains descriptive, but it weakens the promised evidence contract that the aggregate must consume.

Why existing tests miss it: Wrong-phase tests change the global event itself, which is cross-checked against its row. Nested order tests move parent admission ahead of the child, not the parent's return observation while leaving admission last.

Repair direction: Validate every call's phase against its exact configured file row and enforce the known return-order constraints implied by exec_return observations, call ancestry, and chronological events. Keep any facts that cannot be proved from this component reserved for the explicit aggregate cross-check; do not silently accept contradictions that are already demonstrable here.

Required proof: Reject both preserved parser counterexamples with require_complete=True; cover the same invalid relationships in failed-prefix parsing where applicable. Regress real parent-request-entry/child-return/parent-return positive order and unrelated sibling observations.

3. Independent probes and verification

All counterexamples use the production _CapturedSourceFinder/_ExternalSourceLoader/_ExternalAdmission through the repository's CPU fixture, with standard ExtensionFileLoader methods replaced by fixed native doubles. They exercise actual admission mechanics but do not prove native binary or GPU behavior. The scratch tests intentionally assert the invalid states are currently accepted: their PASS results demonstrate bugs, not candidate acceptance.

Preserved source:
C:/Users/AG/新建文件夹/c4_assets/native_global_gap_review_probes_v1.py
Remote copy:
/data2/shared/ang.gao/native_global_gap_review_probes_v1.py
Both exact SHA-256: 01791003d7c63b6ebfd0364acd86ec0dff4e9bac17c1c8ebaab932bf1d932fae

Exact executed command, remote cwd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip:

PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -m pytest -p no:cacheprovider -q -s /data2/shared/ang.gao/native_global_gap_review_probes_v1.py

Observed output, exit 0:
COUNTEREXAMPLE: copied spec substituted in completed frames; audit and issuance pass
COUNTEREXAMPLE: issuer serializes token 101 for live private token 1; audit/replay pass
COUNTEREXAMPLE: replay accepts runtime create for materialization-only module
COUNTEREXAMPLE: replay accepts parent exec_return then nested child observation
4 passed in 0.43s

Independent regression command in the same cwd/interpreter:

PYTHONDONTWRITEBYTECODE=1 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -m pytest -p no:cacheprovider -q tests/unit/optics/test_native_global_publication.py tests/unit/optics/test_runtime_environment.py tests/unit/optics/test_runtime_import_admission.py

Result: 215 passed in 12.63s, exit 0. Plugin autoload was not explicitly disabled in this first invocation; subsequent probes and source checks explicitly disabled it. No backend-native/GPU proof is claimed from these CPU fixtures.

Independent source/cold regression command:

PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -m pytest -p no:cacheprovider -q tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_runtime_failure_observations.py tests/unit/optics/test_runtime_import_cold.py

Result: 306 passed, 3 skipped in 164.69s (0:02:44), exit 0. The skipped cases are not counted as proof; this run did not request pytest's per-skip reason summary. These suites include actual isolated CPU source entry and fixed CPU dependency composition, not backend/GPU initialization. Their exact source/catalog checks passed at the frozen revision.

High-risk production areas inspected: common finder entry before stdlib/external/repository dispatch; poisoned stdlib fallback; per-thread checks and checkpoint reentry; wrapper markers and frame nesting/return timing; retention before global metadata/adoption; pending and completed identity checks; live private registries; immutable table comparison; returned file roots; native exception intake and late parent deltas; required-global completion; final audit/no late admission; ordinary creator-attribute/self-alias handling; environment schema collisions/site-root/parent rules; descriptive calls/events; source policy/catalog selectors; deployment fixture migration; cleanup ledger.

The implemented finite-table checkpoint and ordinary failure retention are separate, as designed. Pending objects stay strongly retained before explicit guardian adoption. The current code does not invent a creator or native ownership path. Synthetic and Go2 profiles differ from their historical predecessors only in schema_version and native_globals. The synthetic table is empty; Go2 has exactly the three approved literal observer/exec/materialization rows. Canonical hashes independently recomputed:
synthetic v4: c01ea90e6c52728a05addd29d081f1accd8e9ff30caae15fd3996a50ce5e6da9
Go2 v2: e1c1a6b02128f329f5639af0c41b8bc6317002fd6a8c33eab469bf289f948ece
Historical profile files are untouched by the reviewed diff. Active bootstrap/catalog selectors are v10, AST policy is v11, and new tests/profiles/catalogs have cleanup-ledger entries. No product interface or later gate was opened in this delta.

4. Actual Go2 attempt and evidence limitation

The author ran the guarded attempt while this CPU/read-only review was in progress. I did not launch it or overlap any reviewer GPU work. I independently read and hashed its completed raw supervisor artifact after receiving its location. This is an author execution independently inspected here, not a reviewer rerun.

Artifact directory: /data2/shared/ang.gao/c2d3-4933b7c-go2-960-guarded-author-v9
Fixed driver: /data2/shared/ang.gao/run_c2d3_go2_guarded_author_v9.py
Independently verified driver SHA-256: c95986ef4e21546079bb9dab2344248140ac876a09ebc8b1921b28caa3c8f038
request.json SHA-256: 16aaa05dc900f33b035c4397d88eb643971d48593bda89cbcad7d7ff16d5b30b
identity.json SHA-256: 5a2d5a92a0a4ec23fa4a7be647a5e183c9c3f5c440aa286cf41aed03c09b6740
supervisor_stdout.json SHA-256: 84944b1fddc74594450cf32112990c3cff856024cf7535504d6a96129a16e8c4
custody_artifact.json SHA-256: c60b2e4e2505c693f148c70242badb4ce72742aca2f517ef6b80eaeadedd95d5

The request binds frozen HEAD4933b7c, Warp target1, go2_static_960x640_v1, environment digest e1c1a6b..., 120000ms operation and 1000ms ACK bounds. Actual parent/supervisor/worker PIDs: 4118656/4118679/4118680. Supervisor output has parent_alive=true and injected=false, worker_returncode=-9. Its authoritative trace records ARMED setup sequence2, operation deadline expiry, kill_requested, then process_exited(-9); summary state EXITED, complete=true, deadline_expired=true, normal_complete_count=2. The author's operation label is canonical_materialization. There is no successful completion for sequence2 and no canonical/upload/readback success artifact.

There is no worker.json in that directory. custody_artifact.json's worker side is the earlier available prefix (IDLE, next_sequence2, two prior normal completions), not a post-timeout worker cleanup record. It cannot override the supervisor's deadline/exit history or prove successful materialization. No recorded native-global relation success is inferred from this attempt. Cause of the timeout was not isolated: CPU correctness tests overlapped on the host, and no performance measurement or revised deadline claim is made. The author deferred further GPU attempts pending repairs.

An initial read command expected worker.json and stopped with exit1 when it was absent. A subsequent schema-aware JSON inspection read the actual supervisor/aggregate artifacts successfully. This is reviewer tooling, not an additional code failure. The earlier c8b404c/v8 failed evidence was treated as historical context, not new validation of this revision.

Read-only scoped ps found no remaining 4118656/4118679/4118680 or reviewer inspector4127489. Read-only nvidia-smi listed only pre-existing MPS server2238410 using60MiB. No process was signaled or reset. Short-lived SSH/Git/pytest utility PIDs were not exhaustively logged; this is a scoped residual check, not a global no-hang claim. CPU test subprocess ownership/reaping follows their existing test harness; no reviewer native/GPU workload was launched.

5. Unexecuted mandatory matrix and scope

This bounded gap review did not execute or complete the parent acceptance matrix. Still required:

- Successful actual captured production materialization of both frozen Go2 assets in both lanes, with guardian/supervisor evidence proving the literal native-global intervals and exact retained identities. Ordinary probes or this CPU fixture do not establish those relations in the captured worker.
- Both lanes' canonical synthetic/asymmetric and dynamic cases, actual allocations/upload/storage checks and supervised validation readback; no substitute materializer or success-by-skip.
- Real cold NumPy/backend/native partial create/exec/import faults, returned-owner/adoption/reconciliation/recording faults, and sentinels held to actual process-terminal disposition. The CPU suite covers local portions only.
- Both backends' allocation/copy/submit-then-raise/event/ready-wait/owner-register/final-audit faults, original-deadline hangs, ACK/DISARMED/EOF cases, and zero new normal COMPLETE/ARM/sync/release after failure.
- Real dual-GPU cold, target1/ambient0 restoration, intentional foreign-device and UNKNOWN scope cases with actual UUID/ordinal and dirty-history proof.
- Successful sole supervised target sync and whole-graph cleanup with real exit/reap, plus terminal-sync/cleanup/missing-exit failures and hangs.
- The final versioned exact-key runtime-custody schema binding actual file calls/returns, global observations, session/lane/target, guard history, runtime binding and prepared storage, with aggregate forged-success replay negatives. The descriptive native-global component is not this final schema.
- Remaining C0 A/K, C2d0/1/2, storage/readiness, complete both-lane upload parity and scope-specific regressions needed for whole-stage code acceptance. Passing the selected CPU suites does not replace them.

The author's separately reported 538passed/3skipped ten-file suite was not independently rerun as that exact command by this reviewer and is not counted as my observed test result. No performance, resource-budget or optical-renderer acceptance was assessed. Native pre-return exception safety and invisible between-checkpoint mutation remain explicit trusted-dependency limitations; no native sandbox guarantee is added.

After F1/F2/F3 repairs, freeze a new revision, retain these original repros, repin exact source bytes, and obtain fresh independent review of the changed implementation and affected invariants. The current candidate remains unchanged by this review. The author may archive this complete report verbatim under the evidence-only exception, without altering its verdict, scope or remaining obligations.

Still CLOSED: whole C2d3/C2d/C4 acceptance, C2e/f, C4c3/c4, C4d/B3, product GPU preparation/activation, a3b, result/baseline, host payload, R2 and shipping. No final optical graph/readiness, active sample lease, renderer result or selected-backend completion is authorized. C4 is not complete.
