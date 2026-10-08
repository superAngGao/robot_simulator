Initiative: q54-optical-lab-p12-4c4c2d-torch-publication
Stage: implementation-gap-review
Author: codex, independent reviewer session /root/torch_publication_implementation_gap_v1
Version: v1
Date: 2026-10-08
Status: in_review; repairs required
Related Files: torch-publication design-request/design-review v1; runtime-adoption design-request v2; native-global-publication design-request v1; implementation-note v1
Owner Summary: Request changes for three reproduced implementation defects in earliest failure custody and interpreter-alias validation. Existing local tests pass, but these new adversarial cases do not enforce the approved contract. This bounded review is not whole C2d3 or real captured Torch approval.

VERDICT: Request changes

1. Exact candidate and independence

Repository: C:\Users\AG\新建文件夹\p12_4_c4_warp_timeline_design
Base: 60ec83af920414ebaa0ccad1a855f4870509d67a
Reviewed HEAD: 64db378dfc92996635ebb5aa644272e080e1bf80
HEAD tree: 15a57937955705a5204c3ad61f66a242d1f07662
Branch: implementation/c4-supervised-waits
Approved design SHA-256, independently rechecked: 2dace48ad6002db9cb6523a6852ef2ec64314130c59563762c17dd95371f7186

Fresh independent task; no implementation conversation inherited and no candidate authoring. Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md in full, including UTF-8 reread of initially garbled Chinese. Read the complete approved Torch design, adjacent design review, runtime-adoption v2 and native-global-publication v1, and candidate implementation note. The note and prior verdict were treated as claims and requirements, not proof of this implementation.

Initial and final git status --short were empty and HEAD was unchanged. Inspected the base-to-HEAD implementation diff and surrounding loader/finder, frame, custody, source-session, policy/catalog, environment, event/replay and diagnostic-driver paths. git diff --check passed. No tracked file, installed dependency, CLAUDE.md, remote code, GPU state or process belonging to another task was modified.

2. Blocking implementation findings

F1 - P1: Alias validation can trigger controller termination before finite published replacement values have been retained.

Location: optical_worker/native_bootstrap.py:989-1004 (enter_import_wrapper), 1042-1051 (return_native_call), 2610-2667 (observe_interpreter_alias), and 2075-2090 (SourceSession.fail).

Trigger: A Python file body publishes its wrapper, then the main/stdlib alias descriptor drifts or its validation fails at the next request/return boundary. On the healthy path, enter_import_wrapper calls observe_interpreter_alias before replacement_checkpoint or native_global_checkpoint. return_native_call likewise performs alias validation before those intake operations (and performs frame validation before intake too). The alias observer catches its error and calls SourceSession.fail immediately. That method calls the controller's _fail_guard and terminate before control returns to ExternalAdmission.fail's later module-delta retention.

Observed consequence: With actual production source execution, actual observe_interpreter_alias and actual SourceSession.fail, the first controller.terminate entry observes the published torch._VF wrapper in sys.modules but absent from both admission.retained and controller.owners. The original remains held. Later delta intake does retain the replacement, but it occurs after termination has begun, violating the approved retain-before-fallible-check/termination ordering. This is a custody-order violation, not a claimed demonstrated native use-after-free. CPU doubles do not prove the downstream native consequence.

Reproduction: external test_torch_publication_gap_v1.py::test_actual_alias_main_drift_precedes_failure_custody. It uses the existing production-importer fixture and seven tiny source modules, with a fixed native double. The VF source replaces itself normally, then changes sys.modules['__main__']. The test binds the real interpreter-alias source descriptor and observer, invokes the real SourceSession.fail, and uses a controller terminate endpoint that records custody at entry. It prints:
  actual alias validation at actual session.fail/controller.terminate: ('interpreter alias main/identity drift', False, False)
The companion injected-alias-fault test independently sees retained=False at session failure entry.

Why current tests miss it: Existing replacement fault tests inspect retained objects after import unwinding; the alias tests execute without a concurrent new external publication. Neither asserts full available custody at the first terminal transition.

Repair direction: At each supported common entry/return/audit boundary, perform unconditional finite replacement/global/alias intake into their already held graphs before any alias, thread/frame, metadata, adoption or event validation that can fail or invoke terminal handling. Preserve the original error and ensure both session and admission remain monotonically failed. Do not rely on a later catch/delta scan to repair the earliest terminal graph.

Required proof: The exact repro must show the wrapper already held at the first controller.terminate call. Add analogous request-entry, return, malformed value and native-global cases; assert no successful observation/admission, retry, normal COMPLETE, extra ARM or ordinary release follows. Retain available original/class/native/frame objects as required. Real captured fault evidence remains an additional gate.

F2 - P2: Live alias issuance accepts type-changing rewrites of immutable event facts.

Location: optical_worker/native_bootstrap.py:2643-2646 and 2673-2707, especially tuple(event.items()) equality in the live audit and issuance checks.

Trigger: After the real cold multiprocessing import and successful first audit, change interpreter_alias_events[0]['main_token'] from integer 1 to boolean True. Tuple equality considers these values equal. The event object's identity remains unchanged, so all current live checks pass.

Observed consequence: Both session.audit() and interpreter_alias_observations() succeed with session.failed still false and issue main_token=True. The pure validator in runtime_environment.py requires type(token) is int and would reject this issued payload. Thus the issuer accepts a mutated projection and produces evidence that its own exact-type schema cannot accept. The same equality shape also permits equal float replacements for integer facts; ensure the repair addresses all fields.

Reproduction: external test_torch_publication_gap_v1.py::test_alias_bool_projection_is_accepted_by_live_issuer, using the real cold production source-session helper and actual stdlib import. Output: mutated_bool_projection_issued.

Why current tests miss it: The event mutation test uses 2, which is unequal to 1, and separately tests copied dictionaries. Neither exercises Python's bool/int or int/float equality.

Repair direction: Compare exact keys and field types as well as values against frozen issued facts. The existing _check_global_events already demonstrates strict per-field type checks. Apply the same invariant to every alias audit/issuance path, including poisoned historical issuance; review replacement projection comparisons for the same pattern rather than assuming equality implies unchanged descriptive facts. This need not import a later-phase parser into the CPU core.

Required proof: Integer-to-bool/float and other equal-but-different-type mutations poison before successful audit/issuance. Unmodified cold observations remain byte-stable and parse successfully. Test live and poisoned history separately.

F3 - P2: Final alias audit accepts an unresolved held multiprocessing resolution when publisher and alias are both absent.

Location: optical_worker/native_bootstrap.py:2622-2625, reached with final=True from SourceSession.audit at line 2827 and the external segment audit.

Trigger: Resolve and execute the exact standard multiprocessing package, then remove both sys.modules['multiprocessing'] and sys.modules['__mp_main__'] before the first supported completed-alias observation. The held resolution exists, its actual spec has _initializing=False, and no event exists yet. At final audit publisher is None; the branch checks only that alias/observed are absent and returns without considering final=True. A None publisher value takes the same branch as an absent key.

Observed consequence: session.audit succeeds, session.failed remains false, and evidence contains the held resolution with no observed publisher/alias fact. A cold untouched optional publisher and a resolution that never reaches required completed observation are thereby treated alike at the completed segment boundary. This is the incomplete-publisher audit case in the approved design, not a demand to detect an unobserved transient mutation instruction. The observable bad final state is an already issued resolution with no completed publisher.

Reproduction: external test_torch_publication_gap_v1.py::test_alias_resolved_then_missing_publisher_passes_final_audit. In a real cold production source session: _verify; import multiprocessing; assert no alias events and non-null held resolution; delete publisher and alias keys; session.audit; assert healthy and zero events. Output: resolved_missing_publisher_passed_final_audit.

Why current tests miss it: Removal tests first call session.audit to establish observed state. The existing partial test sets _initializing=True on a still-present already observed publisher. Those cases correctly reject and do not exercise a held resolution with no observed object.

Repair direction: Preserve the legitimate short interval between find_spec and publication at non-final checkpoints, but reject an absent/None publisher at final audit once resolution has been issued. Keep no-resolution/no-import optional state legal. Do not repair the gap by discovering or inventing a module/event during audit. Clarify any intentionally allowed resolution-only descriptive record as an incomplete prefix, never normal segment completion.

Required proof: This repro and a resolved-then-failed/removed/None publisher reject at final audit; intermediate normal import checkpoints and untouched optional absence still work; successful standard import produces exactly one observation and later audits are idempotent.

3. Independent verification

Environment: local Windows, C:\Python314\python.exe, CPython 3.14. All executed tests were CPU-only. No Torch/NumPy/Warp backend or GPU import was used as acceptance evidence. Replacement fixtures use tiny actual Python file bodies plus a fixed native double and a custody-controller double. Cold alias cases use the production captured bootstrap/session and real stdlib multiprocessing without spawn.

Executed from the candidate repository:
  C:\Python314\python.exe -m pytest tests/unit/optics/test_torch_module_replacement.py tests/unit/optics/test_interpreter_alias.py tests/unit/optics/test_native_global_publication.py tests/unit/optics/test_runtime_import_admission.py -q -p no:cacheprovider
Observed: 257 passed, 3 skipped in 47.16s; exit 0. Skips remain skips, not runtime acceptance.

  C:\Python314\python.exe -m pytest ..\c4_assets\test_torch_publication_gap_v1.py -q -s -p no:cacheprovider
Final observed run: 4 passed in 8.96s; exit 0. These tests assert the current bad behavior to establish reproducible findings; 'passed' here does not mean the candidate contract passed. The final source contains both the simpler fault-injection probe and the stronger actual alias-validator/SourceSession.fail-to-controller.terminate probe.

  git diff --check 60ec83af920414ebaa0ccad1a855f4870509d67a HEAD
Observed: exit 0, no output.

Test side effects: ordinary pytest temporary release directories and interpreter caches only; no tracked edits. Scratch test/report files are under C:\Users\AG\新建文件夹\c4_assets. An exploratory command piping a Python script with a non-ASCII absolute path through Windows stdin failed because that path became question marks; switching to the equivalent relative path succeeded. This was a shell encoding/tool failure, not a candidate finding. All reviewer-started pytest processes completed.

The author separately reported a matching-primary183 CPU suite of 349 passed in 144.80s, zero skips, with log c4_assets/c2d3-64db378-torch-cpu-v1.log. I did not rerun or independently authenticate that remote run; it is author evidence and does not negate the new local repros. No remote/GPU task was started by this reviewer.

Inspected healthy/rejection flows: exact all-seven-or-none table and collisions; original-versus-exposed execution ownership; exact class/metaclass/bases/MRO/function/globals/code checks; raw module metadata; native VF root/value binding; own-frame request-entry and return observation, wrapper suspension and pending-parent reuse; frozen package path checks; post-import parent bindings and backend original from-import list; class/record/token drift; retained failure graph and immutable event projections; source capture/AST alias statement; held standard resolution/main/publisher identities; conditional audit exemption; source byte ledger accounting; policy/catalog pins; component parsing and diagnostic-driver outputs. The existing tests independently cover many identity substitutions and adoption/event failures. No claim of exhaustive adversarial coverage is made.

4. Known evidence/interface gaps, distinct from F1-F3

The literal actual CUDA deployment profile, real captured Torch execution of all seven relations, _C/ops/classes publication, exact alias and original/exposed custody under the supervisor remain mandatory and were not run here. CPU doubles and ordinary import/source inventories do not substitute. This review does not grant bounded full code approval in their absence.

The final runtime-custody aggregate is not implemented in this candidate. Component records lack the final session/lane/target/guard/file-identity cross-binding. They are descriptive data, not live import/storage/release authority. The note explicitly acknowledges that failure between emitted admitted event and file completion leaves an immutable raw failed prefix that completed-component validators do not accept. I found no basis to treat that admission-event prefix as completed custody. The future aggregate must represent/replay such interrupted prefixes and reject forged success using actual file and guard ordering. This is known unfinished aggregate work, not closure, and is not counted as an additional newly reproduced implementation defect in this bounded review.

Whole C2d3 still requires both lanes, canonical/replay and both real Go2 inputs, actual upload/readback, failure/hang/terminal matrix, dual-GPU scope/restoration, foreign/UNKNOWN quarantine, native scope/source audit, real exit/reap/residual evidence and fresh whole-stage independent review. Native internal exception safety remains a trusted-dependency obligation, not something the loader or these CPU tests establish.

5. Exact disposition and scope

Request changes for F1-F3 at the reviewed SHA. Repair these minimal custody/alias defects and preserve their original repros for a fresh independent review of the frozen successor. Repin affected source policy/catalog bytes as required. This review does not approve the rest of C2d3 by subtraction and does not authorize a product activation, captured Torch success claim or aggregate completion claim.

C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain closed. C4 is not complete. No separately approved non-blocking follow-up substitutes for the three repairs or the mandatory real evidence.
