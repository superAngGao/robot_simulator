Initiative: q54-optical-lab-p12-4c4c2d-torch-publication
Stage: implementation-gap-review
Author: codex, independent reviewer session /root/torch_publication_implementation_gap_v2
Version: v2
Date: 2026-10-08
Status: in_review; repair required
Related Files: approved torch-publication design-request/design-review v1; runtime-adoption design-request v2; native-global-publication design-request v1; implementation-gap-review v1; implementation-note v2
Owner Summary: The three original counterexamples now reject correctly. Request changes for a new failure-routing defect: alias custody intake can execute an invalid publisher's __class__ callback, and an exception there prevents sticky failure and original-error retention. The new counterexample reproduces on both local Python 3.14 and primary 183 Python 3.12.13. No captured-runtime or whole-stage approval is given.

VERDICT: Request changes

1. Exact candidate, independence and scope

Fresh independent reviewer; no inherited implementation conversation and no candidate authoring. Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully (Chinese reread with UTF-8), the approved Torch supplement and adjacent design review, runtime-adoption v2, native-global-publication v1, mandatory prior gap report and implementation-note v2. Author test summaries were treated as claims, not proof.

Local repository: C:\Users\AG\新建文件夹\p12_4_c4_warp_timeline_design
Branch: implementation/c4-supervised-waits
Reviewed HEAD: 68e10d2b13249dfb38d6a92f4335e344602f4854
Reviewed tree: 430dd2737f32bf73ab32d95f589bc7604258a5c3
Code predecessor: 64db378dfc92996635ebb5aa644272e080e1bf80
Evidence-only base: 7ee9ef0fc9e4e5ea0038af0d815240630493d58c
Approved design file SHA-256: 2dace48ad6002db9cb6523a6852ef2ec64314130c59563762c17dd95371f7186
Prior report file SHA-256: ac6fbf54c2206721ceffbb9b718b48c2aab3b3a82e224ca421fe910f1ff037aa
Unchanged original repro SHA-256: 57bbfead7bae23deab212907e020e3ab2b9be935421c77052b783b669b79e63a
All three supplied file digests independently matched.

Scope is the bounded C4c2/C2d3 finite Torch original/exposed replacement and exact interpreter alias repairs, affected failure routing, evidence projections and source policy/catalog. This is not full code acceptance of the Torch supplement, whole C2d3, or captured-runtime approval. Initial/final local status was clean and HEAD/tree unchanged. Primary183 initially held clean predecessor 64db378; after author FF synchronization I independently checked clean 68e10d2 before remote tests and clean unchanged68e10d2 after them. No fallback host was needed.

No tracked implementation, test, document, private context or installed dependency was edited. Scratch repros/reports/logs are outside the candidate. No reviewer GPU work, package installation, reset, broad kill or backend-native execution was started. Remote tests use the same candidate but their own independent log, not the author's parallel regression results.

2. Disposition of mandatory prior findings

F1: Closed for its exact demonstrated custody-order defect. Both original bad-behavior probes now fail because the published wrapper is retained before alias failure / first controller termination. The stronger real SourceSession.fail probe observes ('interpreter alias main/identity drift', True, False): True in admission.retained, False in the controller's direct-owner list. The already adopted admission graph holds the wrapper transitively; direct adoption before validation is neither necessary nor what this repair claims. Added production-importer regressions cover request/return, malformed wrapper/frame, source audit and native-global intake. They all passed. This is available Python custody evidence, not actual native quarantine/reap proof. The new F4 below concerns a different failing intake path and prevents overall repair acceptance.

F2: Closed for the reproduced equal-but-different-type rewrite. The original main_token=True acceptance probe now raises interpreter alias facts drift. _same_facts requires an exact dict, exact str keys and exact per-field value types/values. Alias healthy/poisoned issuance, replacement events/token projections and existing native-global checks use it. The added integer-to-bool/float tests passed. Facts in these event/token projections are scalar; no new deep mutable-container equivalence claim is needed for this comparator.

F3: Closed for the reproduced final missing-publisher state. The original resolved-then-removed publisher/alias probe now raises interpreter alias publisher disappeared. Missing/None and resolution-only final states reject; untouched optional absence and the non-final no-publication interval remain legal. Standard cold import still produces exactly one stable observation and passes repeated audits.

The unchanged original external suite ran against candidate 68e10d2 and produced four expected failures in 8.12s. These tests assert OLD BAD acceptance, so the four failures are repair evidence, not four candidate failures. Their file was not rewritten to make the repairs appear passing.

3. New blocking finding

F4 - P1: Callback-bearing alias intake can prevent failure from becoming sticky and lose the original failure.

Locations: optical_worker/native_bootstrap.py:2082-2096 (new retain_import_boundary), :2098-2107 (SourceSession.fail intake before failed/error commit), :2595-2610, particularly :2603 (isinstance(publisher, ModuleType)); reached directly by SourceSession.audit at :2855 and observe_interpreter_alias.

Trigger: Put an invalid non-module publisher value at sys.modules['multiprocessing'] whose __class__ property raises. This is an adversarial wrong-type boundary test, not an assertion that the trusted worker is a hostile-Python sandbox. The approved supplement requires exact type checks before traversal, callback-free metadata auditing and sticky failure for malformed values.

Python isinstance on this nonmodule consults its __class__ property. Thus _retain_interpreter_alias invokes arbitrary metadata behavior during the operation described as intake-only. In session.audit(), the first callback exception enters the audit handler; session.fail(error) then repeats the same intake and callback before setting self.failed or appending the original error. The second exception escapes, leaving the session healthy. A direct session.fail(original) with the same publisher fails even sooner, losing the supplied original exception. Removing the invalid key then permits a successful ordinary audit.

Independent cold production source-session outputs on BOTH Windows Python 3.14 and primary 183 Python 3.12.13:
  audit: callback_count=2, failed=false, failure_count=0, original_held=false, retained=true, retry_audit_passed=true
  fail:  callback_count=1, failed=false, failure_count=0, original_held=false, retained=true, retry_audit_passed=true

The invalid publisher itself is retained, so the finding is not its collection. The consequence is a failed operation that does not poison the source session, no retained original failure in the direct-fail case, and demonstrated normal audit reuse. In a controller-bound session the callback interrupts fail before its controller failure/termination branch is reached; that ordering is static evidence, not a claimed GPU reproduction. The reviewer did not demonstrate native use-after-free or real disposal behavior.

Repro: ../c4_assets/test_torch_publication_gap_v2.py::test_bad_publisher_callback_cannot_prevent_poison[audit] and [fail]. The test uses the actual cold captured bootstrap/session helper and real session.audit/session.fail, with no replacement of those methods. It checks the desired sticky failure, and currently fails. The script prints the observed state before its expected-contract assertion. It also proves recovery after removing the bad key. SHA-256: e30aedcdbaa19a56a6c4a47c6a1127d84ebd151b9354e337fdde4e6d2c78721a.

Why existing tests miss it: Current invalid publisher cases use ordinary ModuleType instances, object(), missing entries or metadata substitutions. Those do not raise from __class__ during isinstance. The new earliest-terminal tests inject faults after intake itself succeeds. They therefore do not exercise an intake callback failure before the poison/error state commit. The isinstance was present before this repair, but adding the same intake at the start of SourceSession.fail introduces the specific non-sticky failure regression.

Repair direction: Make intake genuinely callback-free: retain every available value, then use built-in exact type introspection before any raw module traversal (or an equivalently callback-free class test when retaining subclass metadata is needed). Do not invoke an invalid value's __class__ or arbitrary descriptor. Ensure failure routing commits monotonic failed state and retains the original error even if an ancillary intake operation fails, while preserving the required full available graph before actual termination. Do not use a later catch or a second failing scan as the only protection.

Required proof: Preserve this exact repro and add invalid publisher __class__/__getattribute__ traps, healthy and poisoned entrypoints, direct source failure and externally bound failure. Assert zero metadata callback invocation, original exception retention, both relevant failure flags, finite failure routing, full available publication custody before first termination, one terminal operation, no successful new observation/admission, and no normal retry after removing the invalid key. Re-run original F1-F3 counterexamples to ensure the intake fix does not restore their old behavior.

4. Verification and high-risk inspection

Inspected actual 7ee9ef0-to-68e10d2 delta and surrounding loader/finder/frame/context, original/exposed record binding, finite intake, alias source descriptor/held resolution, typed projection checks, failure call graph and guarded controller failure handling. Confirmed source-policy/catalog repins are limited to the changed exact bootstrap bytes/classes. Reviewed pending/admitted and poisoned evidence distinctions, conditional alias exemption, package/parent identity validation and retention versus adoption. No second importer/upload path or new product gate was found in this delta.

Normal failure-routing supplemental test passed: actual SourceSession.fail -> ExternalAdmission.fail -> SourceSession.fail terminates once with both flags already failed, keeps original exception identity, retains late publication after poison, emits no replacement/global success facts, and permits terminal stdlib finder resolution without another termination. This supports the ordinary path but does not negate F4.

Local environment: C:\Python314\python.exe, Windows, CPU-only.
Executed:
  C:\Python314\python.exe -m pytest tests/unit/optics/test_torch_module_replacement.py tests/unit/optics/test_interpreter_alias.py tests/unit/optics/test_native_global_publication.py tests/unit/optics/test_runtime_import_admission.py -q -p no:cacheprovider
Observed 281 passed, 3 skipped in 75.25s. Skips are the three CPython 3.12/POSIX generated-stdlib variants on this 3.14 Windows host.
  C:\Python314\python.exe -m pytest ../c4_assets/test_torch_publication_gap_v1.py -q -s -p no:cacheprovider
Observed 4 failed in 8.12s, all expected old-bad assertions as described above.
  C:\Python314\python.exe -m pytest ../c4_assets/test_torch_publication_gap_v2.py -q -s -p no:cacheprovider
Final observed2failed,1passed in7.40s; exit 1. The two failures are F4; the passing case is normal routing/late retention/terminal stdlib behavior.
  git diff --check 7ee9ef0 HEAD
Exit 0, no output.

Remote environment: ssh tileops-dev, /data2/shared/ang.gao/robot_sim_c2d3_binding_wip, /home/ang.gao/miniforge3/envs/robot_sim/bin/python, CPython 3.12.13; verified candidate 68e10d2.
Executed independently:
  python -m pytest tests/unit/optics/test_torch_module_replacement.py tests/unit/optics/test_interpreter_alias.py tests/unit/optics/test_native_global_publication.py tests/unit/optics/test_runtime_import_admission.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_source_policy.py -q -p no:cacheprovider
Observed 491 passed, 2 skipped in 77.25s; exit 0. Both real Go2 materialization variants skip because ROBOT_SIM_TEST_GO2_MODEL_DIR was not supplied. These skips do not waive the real Go2 gate. Actual POSIX checkout/catalog and source-policy checks passed.
  python -m pytest /data2/shared/ang.gao/test_torch_publication_gap_v2.py -q -s -p no:cacheprovider
Observed2failed,1passed in3.66s; exit 1. Same F4 output on the pinned interpreter family. Scratch script is outside the repository and identical to the local final repro.

Local logs retained under c4_assets:
  torch_gap_v2_local_suite.log SHA-256 e37e4803540911c2b9d0bf28dafcd289fab7444035aaee143020a9e2af6810c6
  torch_gap_v2_original_repros.log SHA-256 3b0cded6d5879d477e0d39b541928787681e061b6673105cfdf3eab30ba6ab0d
  torch_gap_v2_additional_repros.log (final local F4 run)
  torch_gap_v2_remote_suite.log SHA-256 5ce9730b89f06c3ba617e03f03e1517bf1d2524bd4f4c644c6501e0511aa5cbd
  torch_gap_v2_remote_repros.log SHA-256 9a283a2d40a1234efb689d8fa03684c58a2f680be5392aca09f8a804e9c8f369
Remote original logs: /data2/shared/ang.gao/c2d3-68e10d2-independent-gap-v2.log and c2d3-68e10d2-independent-repros-v2.log.

All reviewer-started test/SSH commands completed; only normal temporary test releases/interpreter caches and external scratch files were produced. No tracked side effects remained. Two harmless inspection command errors were corrected: PowerShell mangled an unquoted HEAD^{tree}, then the quoted command worked; an accidentally spelled-out Select-Object count was replaced by a numeric read. These are tool errors, not candidate findings. The initial test shell printed logs after pytest, masking its shell exit code; the reported original repro failures come from the full pytest output, and final new-repro runs explicitly preserved exit 1.

5. Limits and remaining gates

No actual captured Torch import, real seven-relation custody, native _C/ops/classes publication under the supervisor, or target/ambient CUDA execution was performed or approved by this reviewer. The actual CUDA profile remains an external draft. An ordinary deferred-init diagnostic the author mentioned during review is outside this report's evidence and cannot supply captured approval.

The complete runtime-custody aggregate and replay of interrupted/failed prefixes against actual file calls, session/lane/target and guard order remain mandatory unfinished work. Component evidence is descriptive, cannot grant live import/storage/release authority, and cannot turn an admitted-event/interrupted-file prefix into completed custody. This review found no new aggregate completion claim in the repair delta and does not approve that missing aggregate by subtraction.

Full C2d3 still requires both lanes, canonical/replay and both real Go2 inputs, guarded real upload/readback, actual native scope/source audit, cold/dual-GPU/target1-ambient0 and foreign/UNKNOWN cases, fault/hang/ACK/EOF/terminal matrix, actual exit/reap and residual evidence, and fresh whole-stage review. Native internal exception safety remains a trusted-dependency obligation, not something CPU doubles establish.

Request changes for F4 at exact 68e10d2. Original F1-F3 are independently closed for their exact counterexamples, but that is not an approval of this candidate or whole Torch/C2d3 implementation. Repair F4, repin changed bootstrap bytes, and preserve both generations of repros for fresh independent re-review. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain closed. C4 is not complete.
