# Whole C2d3 review repairs and native preflight request

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-09
Status: in_review
Related Files: runtime-adoption whole-code-review v1; runtime-adoption design v2;
runtime-custody implementation-plan v3; native-scope-aggregate design v2;
native-scope-cache design v1; terminal-matrix test-evidence v1
Owner Summary: Repair the three blocking whole-review findings under existing
contracts. Request independent code/native preflight, then affected real-device
validation and whole C2d3 re-review. No stage acceptance is claimed.

Repair base is bca0d5345244014d054f94b0ae5ec4954a54b61e (unchanged whole report
archive). Full C2d3 base remains ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3.
The review packet supplies the exact committed candidate. Original repros and
failed outputs remain in ../c4_assets/whole_review_v1_*.

F1: both original upload cores adopt a host custody list before event creation,
append the returned event before inspecting metadata, adopt its actual ordinal,
and check its actual post-record ordinal before waiting. CUDA None is allowed
only before record. Invalid metadata marks UNKNOWN; foreign scope reaches the
existing rejecting lifecycle. The shared metadata reader moves from readback
to native_setup; no second uploader, native getter, guard protocol or C0 abort
policy is introduced. Both lanes and actual guarded/C0 lifecycle fixtures cover
normal, foreign-create/record, unknown-create/record, unbound-record and boolean
ordinal cases. Metadata access verifies prior retention; failure assertions
include no later wait/sync/release and guarded no COMPLETE/new terminal ARM.

F2: successful phase completion now requires every external file state to be
completed or resolution-only file_resolved. Delegate completion alone cannot
stand for final adapter admission. This preserves deferred Triton probes and
truthful incomplete failure prefixes. Six mutations per actual positive record
delete NumPy/backend completed, exec_started or created events, reindex all
descriptive references/counts, and keep trusted launch/lifecycle facts fixed.
Remote execution of these new actual-record cases is still pending here.

F3: ten replacement tests observe the real terminate method rather than assign
its getter-only termination_evidence. They check retention at first termination,
one termination, actual quarantine, original failures and existing callback/
poison properties. The environment test reads the shared PROFILES literal and
its actual consumer selection, with filename/digest/consumer drift negatives.
Historical C0 equality allows exactly the two already-approved CPU owners and
continues comparing all remaining historical contract fields and closed paths.

Only three source catalog v12 entries are repinned to Git LF source bytes:
native_setup 2c96c638ec83a0a686fd962affd9d6664c127a7977e55c82b2b354f3adda2e9c;
native_readback 9aa64c71385c9db0f803081d666407e050480af208c9772eb3d4adc2e09893de;
runtime_custody 7fe624b0189ab5da6abdf3ff7a24ff10dfe1f0046b85a9d6b986e5cf0ae1346f.
No source permission/schema/profile/native observer changes.

C1 disposition: add current v12 catalog and six omitted test paths to existing
cleanup entries and their existing terminal dispositions. No deletion now.
C2 disposition: older native-module-name design v1 asked for separate observed
intrinsic __name__ in final custody, while later approved custody plan v3 freezes
five-field provenance without it. Current code checks name live and follows
the later grammar; aggregate does NOT contain that separate observation. This
documentation records the inconsistency and claims no schema expansion.

Local Windows Python3.14, plugin autoload/cache disabled: the complete affected
eleven-suite selection (native_upload_core, guarded_runtime_binding,
guarded_native_readback, native_manifest, runtime_custody,
native_contract_extraction, runtime_environment, native_source_policy,
runtime_import_admission, native_global_publication, torch_module_replacement)
passes 684 with three skips in43.09s. Earlier focused runs: upload55pass,
environment55pass, replacement+historical-contract46pass. These overlap, not
additive coverage. Two existing repository inventory/cleanup checks pass1.27s.
This is CPU proof, not repaired-candidate native validation or whole acceptance.

Next bounded native request uses the unchanged approved drivers, fixed profiles,
observer and supervision: rerun16 native-scope cases, affected full85 fault
matrix, and12 terminal cases on the repaired frozen source, after independent
preflight. Before that, run actual-record mutations and replay all113 original
records with the repaired parser, retaining original bytes and provenance.
The old12 terminal result is separately archived; it cannot close F1/F2/F3.

C4 stays active. C2d3 is not accepted. C2e/f, C4c3/c4, C4d/B3, renderer/final
graph/readiness/warmup/sample authority, product GPU preparation, a3b,
result/baseline, host payload/R2/shipping remain closed. No backend selection,
performance verdict, dependency change or losing-lane deletion is made.
