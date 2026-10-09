# Named prepared-array adoption join repair

Initiative: q54-optical-lab-p12-4c4c2d-array-adoption-join
Stage: implementation and pending independent review
Author: codex, implementation coordinator
Version: v1
Date: 2026-10-09
Status: in_review
Related Files: runtime-adoption whole-code-review v2; runtime-custody implementation-plan/gap-review v3; optical_worker/runtime_custody.py; optical_worker/lifecycle.py; optical_worker/native_setup.py
Owner Summary: Repair the existing aggregate's missing named-array allocation
join. This closes a contradictory-evidence counterexample without new authority,
native observations or schema fields. Current runtime matrices and fresh
independent review remain required; C2d3 and C4 are not complete.

The complete independent Request changes report is archived unchanged at
`runtime-adoption__whole-code-review__codex__v2`, evidence commit83cfef6.
Reviewed production was fc8618268dd8f3fb10e29e32626219500751c41a under evidence
HEAD9c1899898bf753644f691c5340336c35ca964a9b. On all8 actual new positive records,
exchanging two array tokens and recomputing storage_sha256 passed while their
unchanged guardian sequences still named opposite allocation operations. The
original repro and output hashes remain in that full report. This was an offline
consistency defect, not evidence of swapped live uploads or granted authority.

`prepared_storage()` now compares each actual adopted array's existing
operation_id to the exact lane/name allocation ID, in addition to prior role,
scope and occurrence checks. The single existing `_array_operation_id` function
moves unchanged from native_setup to lifecycle, which both upload and aggregate
code already depend on. The setup import preserves its existing helper name
for readback and callers. AST comparison against fc86182 confirms the function
body/signature/docstring are identical. No allocation, upload, event, native
query, barrier, supervisor, deadline or terminal code changes.

This placement preserves a CPU-only aggregate import: a fresh stdlib interpreter
imported runtime_custody without loading native_setup, native_contract, NumPy,
Torch or Warp. There is no new policy definition, allowlisted module or source
capability. Current catalogv12 repins only lifecycle, native_setup and
runtime_custody bytes; its schema, lanes, paths and AST policy remain unchanged.

The actual-record regression covers exchanging array tokens and exchanging
their guardian_sequence references while fixing the original lifecycle,
manifest, storage signatures, source and trusted launch inputs. Both still have
unique valid-role/scope owners and valid first-use indices, so rejection must
come from the missing allocation join. The original record must still roundtrip
and attach. The remote actual-record execution is pending at this snapshot.

Other available storage joins were inspected: prepared and ready-event tokens
have distinct lane-specific direct adoption roles; source joins the original
canonical owner; array names/order/extent/layout follow the exact manifest.
Array-specific operation IDs also prevent one directly adopted array token
from satisfying two distinct names. CUDA storage-owner objects are held through
the actual registered storage projection; the aggregate does not record a
separate owner-to-array native identity observation. This fix does not invent
one or claim that arbitrary coordinated descriptive edits can be independently
attested. Live storage registry checks remain the authority, as the approved
schema requires. Warp storage_owner_token remains null.

Local Python3.14, plugin autoload/cache disabled, complete affected11suites:
684passed/3platformskips in42.58s. The command uses native_upload_core,
guarded_runtime_binding, guarded_native_readback, native_manifest,
runtime_custody, native_contract_extraction, runtime_environment,
native_source_policy, runtime_import_admission, native_global_publication and
torch_module_replacement under tests/unit/optics. The skipped platform cases
are not runtime evidence. git diff --check passed. No GPU work was executed
for this repair. Original guarded event tests still pass.

The remote fc86182 full85 run continues on its unchanged clean checkout. It
must finish and retain its actual result; the approved fc86182 terminal12 run
is still pending. New parser replays must identify original runtime revisions
truthfully, and must not relabel those runs as execution of this repair.
Fresh review will assess the exact delta, original counterexample, positive
and failure-cut compatibility, source capture and required remaining evidence.

No compilation/warmup/recording/rendering/final readiness or product GPU
preparation is enabled. C2e/f, C4c3/c4, C4d/B3, a3b, result/baseline, host,
R2 and shipping remain closed. Performance and backend choice remain later
gates; selected-backend cleanup requirements are unchanged.
