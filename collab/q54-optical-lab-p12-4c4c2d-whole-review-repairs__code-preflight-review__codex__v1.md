# Independent whole-review repair code and native preflight

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: bounded independent code/native preflight
Author: codex, independent reviewer /root/whole_repairs_preflight_v1
Version: v1
Date: 2026-10-09 (Asia/Shanghai)
Status: accepted for the bounded preflight scope below
Related Files: runtime-adoption whole-code-review v1; whole-review-repairs implementation-note v1; runtime-adoption design/review v2; runtime-custody implementation-plan/gap-review v3; native-scope-aggregate design/review v2; native-scope-cache design/review v1; runtime-integration design/review v3
Owner Summary: **Approve with comments**, for exact fc86182 repair code and the unchanged 16/85/12 native reruns only. F1/F2/F3 and C1/C2 are addressed. Keep cold importer CPU fixtures separate from backend-importing collection; preserve the failed combined attempt and passing isolated rerun. No repaired native result or whole-C2d3 acceptance is claimed.

## Identity, frozen candidate and scope

Fresh independent session `/root/whole_repairs_preflight_v1`, supplied the review packet and repository, without the implementation conversation. I did not author candidate changes, delegate implementation, spawn additional reviewers, or alter the frozen checkout. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md in full; the initially garbled Chinese output was reread explicitly as UTF-8. I inspected the referenced requirements, complete prior findings and original repros, actual repair diff, and affected call/data/ownership/state flows. Author descriptions and prior approvals were treated as claims and bounded historical evidence.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Repair base: `bca0d5345244014d054f94b0ae5ec4954a54b61e`.
- Whole C2d3 base: `ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3`.
- Frozen reviewed HEAD: `fc8618268dd8f3fb10e29e32626219500751c41a`.
- Initial and repeated local tracked/untracked status: clean.
- Independently observed primary remote `/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip`: clean detached `fc8618268dd8f3fb10e29e32626219500751c41a` after coordinator transfer.

The repair delta is 12 files, 422 additions/58 deletions. This is a bounded review of F1/F2/F3 and C1/C2, affected regressions, and readiness for the unchanged 16 native-scope, 85 fault and 12 terminal cases. It is not a repetition of the complete 186-file whole-stage review and does not accept C2d3 before repaired native results and fresh whole review.

## Prior findings and inspected behavior

### F1: closed for code/preflight; repaired native execution remains required

`native_setup.py:130` is now the shared event metadata reader; readback imports that same definition. Each original upload core creates/adopts a host custody list before event creation (`:306` and `:424`), appends the exact returned event before reading its device, adopts the event with its observed creation ordinal, and checks its actual post-record ordinal before the wait (`:336` and `:454`). CUDA `device is None` is permitted only in the pre-record call. Missing/throwing/wrong-type/boolean/out-of-range metadata marks UNKNOWN and propagates failure. Warp cannot take the CUDA-unbound exemption.

For foreign creation, inherited adopt-returned-owner first retains and registers the foreign scope, then rejects. For foreign post-record metadata, the new helper registers the observed foreign descriptor before the rejecting lifecycle operation; the native wait is never entered. Its event_wait *intent* increments the existing setup-operation count, while entered-leaf count proves no corresponding low-level wait occurred. This distinction is preserved rather than misreported as an executed wait.

I traced `NativeGuardedSetupLifecycle.begin_device_leaf`, `adopt_returned_device_owner`, `mark_unknown_device_scope`, segment exception intake and `terminate`: compromised or failed armed setup goes directly to full quarantine under the original ticket. It cannot send COMPLETE, start another terminal ARM, synchronize, or authorize ordinary release. The C0 wrappers still call the sole cores and their unchanged `_abort`; C0 foreign/UNKNOWN disposition still refuses target sync/release. No lifecycle/protocol or wrapper policy changed.

The original `whole_review_v1_native_upload_event_repro.py`, run unchanged, now stops at its first CUDA case with `DeviceScopeCompromisedError` instead of returning a prepared object. Its finally path closes the peer and reports quarantine. To exercise both original lane bodies, the external review wrapper `whole_repairs_preflight_v1_event_replay.py` changes only the original script's exception reporting, adding explicit failed-closed assertions and continuing its existing loop. Both lanes reject with dirty scope `[0,1]`, zero event waits, zero normal completions, zero terminal ARM attempts, zero emergency sync attempts, and `release_status=not_authorized`; both peers close. Original source/output remain unchanged.

The candidate's production-core tests also exercise both lanes with real guarded and C0 lifecycle objects, retained-owner assertions inside the metadata property, foreign-create/record, unknown-create/record, unbound-record and boolean-record, plus one-wait positives. I ran those tests as part of the full affected selection. They are CPU doubles, not actual foreign CUDA event creation or native destructor proof.

Read-only installed source inspection confirms Warp's Event stores its resolved Device as `self.device` before native event creation; the installed Torch `ATen/cuda/CUDAEvent.h:86` getter returns an optional descriptor from existing `is_created_`/`device_index_`, without a CUDA query. The existing readback policy already used these metadata categories. The repair adds no native getter ABI, observer stream-cache action, second uploader, extra positive-path GPU leaf, or synchronization.

### F2: closed for code/preflight

`runtime_custody.py:939–941` now rejects phase completion unless every observed external file state is either `completed` or a genuine resolution-only `file_resolved`. The pre-existing transition checks still require ordered resolution/create/exec progression, actual retained returned owners, completed delegate calls and consistent module/spec/loader identities. The existing external `completed` handler requires the publisher's declared publications and any triggered disabled fallback to be confirmed. The new phase condition therefore restores the missing join; delegate completion alone cannot substitute for adapter admission.

The condition is applied at normal phase completion, not indiscriminately at record end. Truthful interrupted prefixes and immutable earlier cuts remain representable. Resolution-only probes remain legal, including deferred Triton; no invented completion rows or relaxed component parser were added.

The coordinator executed the unchanged original `whole_review_v1_replay.py` against this exact repaired parser on Linux; I inspected its code, complete parsed output, raw digest and record categories. All 113 original records roundtrip byte-identically, validate selection/attachment and join original same-run identity evidence: 16 native-scope, 85 fault, and all 12 finalized old terminal records. This includes 75 current terminal records, 32 current prefixes and six explicitly earlier validated terminal cuts. All eight original NumPy-completion-deletion counterexamples now reject specifically with `phase completed with incomplete external file admission`. The eight original positive native records retain their 1,988 native rows. The replay imported no NumPy/Torch/Warp. Output `whole_repairs_v1_replay.json` SHA256 is `3730a827dddaebe80aab9927bcf693124563f806733009aba0de3126d00673c5` (UTF-8 with BOM).

The new actual-record tests remove NumPy/backend completed, exec_started or created events, reindex descriptive references/counts and leave trusted launch/lifecycle inputs fixed. The complete actual-record mutation suite on all eight original positives finished **500 passed, 36 skipped in 832.34s**, exit0. All 48 new admission-progression mutations ran. The 36 skips are explicitly inapplicable combinations: 12 non-Go2 cross-case comparisons, 12 absent component occurrences, and 12 sources without zero-sized arrays. They are not passed cases. I inspected the completed raw stdout and summary; the runner's before/after hashes report every source artifact unchanged. Raw stdout SHA256: `6b84cea508ed8c529f5f7507a1c5e1b0a9301d9d4688cda45a920baaf6d3f501`; summary SHA256: `ae0bd1f8a031abf41a7f5732bb026e8e680a01fa491086bfde321e7fdb4a7717`.

### F3: the original twelve failures are closed

The ten replacement/alias tests now observe the actual controller `terminate` method and call it, rather than assign its getter-only termination evidence or suppress `_fail_guard`. They assert retention at the first termination, exactly one termination, actual full quarantine and the existing original-failure/callback/sticky-poison behavior. The production property was not weakened.

The environment test reads the single literal shared PROFILES table and verifies the actual cold consumer's import/selection. It checks current schema/digests and rejects filename, digest and consumer-selection drift without importing a GPU test. The historical contract test permits exactly the two previously approved CPU owners while comparing all remaining contract fields and unchanged closed product files. No duplicated digest or blanket history exemption was introduced.

My complete local affected eleven-suite run passed **684 tests, three skips in 45.60s**. The three skips are the explicit CPython3.12/POSIX generated-data-module cases, not passes. On the pinned Linux interpreter the complete fresh-process importer/global/replacement trio passed **262 tests in 20.75s**, including all 45 replacement cases and those platform-specific cases; this coordinator execution's runner and raw result were inspected directly.

### C1/C2: nonblocking prior comments addressed

C1: the cleanup ledger now explicitly includes v12 and the six omitted test paths, under existing dispositions and exit conditions. No lane/history was deleted or activation granted. I independently ran both repository inventory/cleanup tests: **2 passed in 0.65s**.

C2: implementation-note v1 explicitly records the older intrinsic-name requirement versus the later exact five-field provenance grammar, and correctly disclaims a separate intrinsic-name observation in the aggregate. I checked the older text, later plan, actual five-field emitter/parser and live `check_module_name` validation. This documents the inconsistency without inventing a new schema or claiming the absent field exists.

## Nonblocking execution-isolation comment

N1 — Run cold importer fixtures in a separate interpreter from suites whose collection imports a real backend. The coordinator's new combined 22-suite Linux invocation produced **1,058 passed / 45 failed in 160.24s**. I inspected the raw failure stack: all 45 replacement cases fail during fixture setup at `native_bootstrap.py:679`, `external dependency was preloaded`, before their scenarios.

The actual source chain is `test_cuda_arena_contract` importing `optics.cuda_arena`, which imports `cuda_timeline` (and other CUDA modules); these import installed Torch during collection. The cold importer fixture deliberately retains its original sys.modules baseline and correctly rejects preloaded names. The coordinator's separate collection probe observed the real installed Torch with `cuda._initialized=False`. This is a combined-run isolation conflict, not evidence of native execution or a repaired production admission bypass. The complete cold trio then passes in a fresh interpreter without changing the candidate or clearing/weakening production checks.

Disposition: keep the failed combined attempt, collection diagnostic and successful isolated trio separately; do not relabel the failed invocation as a pass, sum overlapping totals as unique coverage, or combine these incompatible cold fixtures in future acceptance commands. This does not block the bounded native preflight after the complete cold-trio pass. No candidate test or production change is requested for this orchestration disposition.

## Source integrity, native readiness and commands

Independent Git-blob hashing confirms only these v12 source digests changed:

- native_setup: `2c96c638ec83a0a686fd962affd9d6664c127a7977e55c82b2b354f3adda2e9c`.
- native_readback: `9aa64c71385c9db0f803081d666407e050480af208c9772eb3d4adc2e09893de`.
- runtime_custody: `7fe624b0189ab5da6abdf3ff7a24ff10dfe1f0046b85a9d6b986e5cf0ae1346f`.

All other catalog content, source permissions, profiles, observer, fixed GPU tests/support, guard/lifecycle protocol and product gate files have no repair-base delta. `git diff --check bca0d5345244014d054f94b0ae5ec4954a54b61e HEAD` passes. Local Windows source hashing used Git LF bytes rather than confusing checkout newline conversion with catalog drift.

I inspected and independently hashed the three concrete external runner scripts. They pin clean exact fc86182 and fresh output directories, preserve the approved test selections/profiles and original supervisor/reaper deadlines, capture live PID/start/ancestry evidence, reject incomplete counts/skips, and do not add author timeout/terminate/signals. Their SHA256 values are:

| Runner | Expected unchanged selection | SHA256 |
| --- | --- | --- |
| run_c2d3_native_scope_v2.py | 16 | f05b8f4d03b276637692d541f63bc6abe557d9487639b100b9d271c2777c35e8 |
| run_c2d3_runtime_fault_matrix_v5.py | 85 | 7551827e828a4119d3d232ec2e0f9327a6e56a661015d972b94c393b996bea11 |
| run_c2d3_terminal_matrix_v2.py | 12 | 3538f7403dd945798ddc3872f2930d22d659c15a49227cdeb1309e80d8c8eb7f |

The 16-case entrypoint contains both lanes' four canonical cases plus eight negatives; the 85-case fault matrix excludes the three dedicated terminal selectors; the 12-case terminal suite covers both lanes, asymmetric/replay and success/sync-raise/sync-hang. I traced observer wrappers back to the original setup helpers: they retain returned objects before observation and delegate once. The added host custody rows do not alter native helper occurrence coverage or positive operation counts. Acceptance still derives native coverage from actual entered leaves, and terminal release still requires the original sole emergency sync and observed exit.

Local executed tests used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider`, with basetemp outside the repository. The eleven files were test_native_upload_core, test_guarded_runtime_binding, test_guarded_native_readback, test_native_manifest, test_runtime_custody, test_native_contract_extraction, test_runtime_environment, test_native_source_policy, test_runtime_import_admission, test_native_global_publication and test_torch_module_replacement. Log: `whole_repairs_preflight_v1_tests.log`. The two additional node IDs were `test_optical_benchmark_contracts.py::test_inventory_and_cleanup_ledger_match_current_repository` and `test_optical_render_migration.py::test_render_cleanup_ledger_covers_durable_and_comparison_only_paths`.

Original F1 command: `C:/Python314/python.exe -B ../c4_assets/whole_review_v1_native_upload_event_repro.py .`; expected nonzero rejection log `whole_repairs_preflight_v1_original_event.log`. Both-lane assertion wrapper command uses the same interpreter and `whole_repairs_preflight_v1_event_replay.py .`; result exit0, log `whole_repairs_preflight_v1_event_replay.log`. Integrity/replay summary script: `whole_repairs_preflight_v1_integrity.py`.

Remote coordinated CPU commands use `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`, plugin autoload disabled and cacheprovider disabled. The reviewed external scripts are run via UTF-8 stdin over primary `tileops-dev`, `python -I -B -`; their pytest children use `-B -m pytest`. I did not duplicate their costly complete runs. The original replay is the unchanged saved script; the full mutation runner hashes source artifact files before/after. No new GPU/native test was launched by this reviewer.

Inspection-tool corrections included guessed absent support/test paths, unavailable remote rg (then grep), one PowerShell inline-Python quoting error, and an initial scratch catalog reader using the wrong field name. Corrected commands succeeded; none changed candidate or source evidence. These are tool errors, not candidate test failures.

## Evidence limits, process hygiene and closed gates

No substantive repaired-source finding or unresolved blocking preflight evidence gap remains. This verdict retains N1 as a nonblocking execution-isolation comment with the disposition above.

The remote CPU artifacts are under `/data2/shared/ang.gao/c2d3-fc86182-actual-record-mutations-v6`, `c2d3-fc86182-foundation-cpu-v3` and `c2d3-fc86182-cold-importer-cpu-v1`. The combined failure stdout independently hashes to `35032da9afc7254a3d1c9e11655af85ce5ee02253c3244b50f56ab5f75b05315`; successful isolated trio stdout hashes to `cccb2bc4aa5b239e2faec8890165371ee7cddbe28fe025822c9852869cbf429e`. These results overlap local selections and each other; totals are not additive unique coverage.

Independent final exact-PID `ps` returned only its header for mutation runner/pytest 3242890/3242918, foundation runner/pytest 3242909/3242915, replay/parent 3242917/3242897, preload probe 3259957, cold-trio runner/pytest 3260186/3260191 and saved diagnostic probe 3262423. Their source process-identity records and coordinator round residual evidence remain separate from this bounded exact-PID check. Reviewer local pytest/repro commands completed, both original CPU peers were closed/joined, and a final scoped Win32_Process search found no Python command matching `whole_repairs_preflight_v1`. I did not exhaustively sample every short-lived local child or utility PID; this is bounded hygiene evidence, not OS attestation.

No GPU process was launched by this reviewer, no process was signalled, no dependency installed, no private context edited and no alternate host used. All reviewer scratch/report artifacts are outside the candidate. Final local and remote integrity checks retain clean fc8618268dd8f3fb10e29e32626219500751c41a; completed read-only commands did not leave a reviewer session running.

Old real-device archives remain historical. In particular the finalized 12-case terminal archive independently hashes to `fdd875571e59a4b33092c1c220a3a2213a4ed98c0bb0c50f2f26221c50ffcde0`; its 7cde887 execution is not a repaired fc86182 result. Replaying the 16/85/12 old records proves repaired parser compatibility with those bytes, not new GPU execution, actual foreign-event behavior, native exception safety or fresh scope restoration/disposal.

The approved trusted native dependency/driver exception-safety boundary remains. CPU doubles cannot prove internal native cleanup safety. No optical rendering, numerical image correctness, performance/backend choice, warmup/final graph, readiness/sample authority or product activation was attempted or inferred.

**Verdict: Approve with comments**, restricted to the inspected repair delta at fc8618268dd8f3fb10e29e32626219500751c41a and the three exact hashed native runner scripts above. The coordinator may proceed with the unchanged 16 native-scope, 85 fault and 12 terminal cases on the pinned primary deployment, retaining actual per-run source/environment/process provenance, original supervisor deadlines and scoped OS/GPU residual checks between rounds. Do not treat a partial run, skip, old artifact or parser replay as repaired native success. Preserve failures and obtain fresh whole-C2d3 review after the required repaired results.

Still CLOSED: whole C2d3 acceptance until affected repaired native results and fresh independent whole review; C2d and C4 completion; C2e/f; C4c3/c4; C4d/B3; renderer/final graph/seal/quiescence, R_setup/R_final, warmup and prepared-sample authority; product GPU preparation/native sample dispatch; a3b; result/baseline; R2, host payload and shipping/activation. No backend selection or losing-lane deletion is authorized. An unchanged report/evidence-only commit may archive this verdict; it cannot change its scope or open a later gate.
