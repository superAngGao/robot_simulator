# Torch replacement implementation candidate: bounded gap review

Initiative: q54-optical-lab-p12-4c4c2d-torch-publication
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: torch-publication design-request/design-review v1; runtime-adoption design v2; native-global-publication design v1
Owner Summary: Implement the approved finite replacement/stdlib-alias relations as a candidate for independent gap review. This is not code approval, captured Torch success, a2 acceptance or C2d3 completion.

Base:60ec83af920414ebaa0ccad1a855f4870509d67a. The candidate is the commit containing
this note. Design candidate292811101a48f5c43e519db782604111f1e4c548 and its unchanged
independent approval are preserved. Reviewers must inspect code and reproduce
counterexamples; this author note is not approval evidence.

## Implemented candidate

- Schema v5 has all-seven-or-none literal replacement rows. New Warp synthetic
  v5 and Go2 v3 profiles add only an empty table; historical profiles are untouched.
- Original file module/spec/delegate/call remain separate from the exposed
  wrapper. Immutable bindings preserve exact original/class/functions/code/MRO,
  m or native vf identity, pending own-call interval and event facts. Final audit
  checks importlib parent publication and original backend from-import bindings.
- The exact cold-main alias has separate source-session custody, standard
  multiprocessing resolution, raw metadata and completed-import observation.
  It is never a generic stdlib-name exemption or witnessed stdlib execution.
  The source byte ledger counts the identical __mp_main__/__main__ object once;
  actual permission remains the source-session observation.
- Evidence issuance rechecks live bindings while healthy, retains immutable
  historical facts after poison and cannot repair missing observations. Pure
  replay validators confer no authority. Alias evidence works in the CPU core
  without importing the later external-contract phase. The fixed sacrificial
  driver records both new components. AST policy v12/catalog v11 are repinned.
- Cleanup ledger covers new tests/catalog/profiles with the existing P12.4e
  selected-backend retention and losing-lane removal obligations.

## Executed author checks and limits

Local Windows C:\Python314\python.exe (no real backend/GPU in these tests):

1. `python -m pytest tests/unit/optics/test_interpreter_alias.py tests/unit/optics/test_torch_module_replacement.py tests/unit/optics/test_runtime_import_admission.py tests/unit/optics/test_native_global_publication.py -q`
   252 passed,3 skipped in33.87s on the intermediate implementation. Subsequent
   evidence-issuance/driver changes have their affected checks below.
2. `python -m pytest tests/unit/optics/test_interpreter_alias.py -q`
   14 passed in42.41s after repairing a test-only missing json global.
3. `python -m pytest tests/unit/optics/test_torch_module_replacement.py tests/unit/optics/test_native_runtime_driver_process.py tests/unit/optics/test_runtime_environment.py tests/unit/benchmarks/test_optical_benchmark_contracts.py::test_inventory_and_cleanup_ledger_match_current_repository -q`
   82 passed,4 skipped in4.82s after the final replacement-issuance/driver change.
   The process cases require POSIX; their Windows skips are not acceptance.
4. Cold alias/generated-stdlib/bootstrap/source-policy combined regression:
   250 passed,22 skipped,1 failure in143.05s. The sole failure was the test-only
   `json.dumps` global, corrected to `entry.json.dumps`; the exact positive test
   was rerun after the final source repin:1 passed in10.63s. The earlier full
   result is recorded rather than relabeled as a clean rerun.
5. `git diff --check` passed. All listed local pytest processes exited.

The new production-importer tests use tiny Python packages and a fixed native
double, preserving importlib behavior but not proving native/GPU lifecycle.
They cover seven exact relations, early backend parent, coherent projection
rewrite, original/spec/function/code/token/parent drift, wrong wrapper/backlink,
missing replacement, publication/adoption/event failures and terminal no-new-event
behavior. Cold alias tests use real stdlib multiprocessing import without spawn,
and cover preload/injection, wrong main/publisher/spec, disappearance, partial
publisher, descriptor/projection drift and poison-time observation.

## Still required; not waived by this candidate

The finite actual CUDA deployment profile is not yet authored. Ordinary import
censuses and static/linker/map inventories in the predecessor evidence remain
diagnostics, not permissions. Actual captured Torch must prove all seven
relations,40 _C paths,ops/classes and the stdlib alias under the supervisor.
Additional adversarial tests and actual runtime failures may expose gaps.

Component parsers currently validate healthy completion relations. A failure
between an emitted admission event and the enclosing file-call completion is
preserved as an immutable raw failed prefix, not rewritten into success or
accepted as a completed component. Final aggregate schema must reconcile that
prefix with actual file/guard/failure ordering; neither components nor the v0
driver observations substitute for it. The independent gap review should
explicitly assess this remaining interface and implementation counterexamples.

Both-lane canonical/replay/real-Go2 upload/readback, complete fault/hang/dual-GPU
matrix, native scope audit (including compiled-in MAGMA), final versioned
runtime-custody aggregate and whole independent C2d3 review remain outstanding.
C2e/f,C4c3/c4,C4d/B3,product GPU preparation,a3b,result/baseline,host,R2/shipping
remain closed. C4 remains active; this candidate is not the terminal objective.
