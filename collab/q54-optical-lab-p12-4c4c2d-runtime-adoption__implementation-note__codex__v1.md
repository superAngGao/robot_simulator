# C2d3 implementation work in progress

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-07
Status: draft
Related Files: approved runtime-adoption design v2/review v2; native_setup.py;
native_contract.py; runtime_environment.py
Owner Summary: Initial implementation toward the whole approved C2d3 checkpoint.
Not a completed CPU substitute, code approval or runtime activation. The complete
importer/binding/environment freeze and real-device acceptance still must follow.

Base:956d1fa8d51b85c25ac17825adef17640f0e69b5. Approved design:f668b5b.
This draft records ongoing work, not a request to approve a smaller checkpoint.

## Actual changes so far

- Added CPU-only RuntimeEnvironmentContract: canonical immutable configuration,
  interpreter/file hashes and distribution METADATA checks, explicit file-module,
  absent-probe, creator/alias and namespace tables, contained paths and fixed
  plugin/quiet settings. Site roots are distinguished from native-library roots;
  containment roots may overlap but the same physical path cannot be inventoried
  twice. Parsing/verification returns no import/readiness capability. No actual
  deployment inventory or permission table has yet been frozen.
- Extracted each existing CUDA/Warp native_setup upload body into its single
  internal core. C0 wrappers retain existing signatures, fresh-epoch entry and
  catches/abort policy. A registered exact internal context is held before upload
  and burned once at consumption. This context is not RuntimeSetupBinding,
  guarded execution authority or readiness; the guarded importer/driver still
  needs its proper registered binding from the approved design.
- Extracted the original registered-storage validation from full prepared/
  readiness validation. The new storage-only entry does not mint readiness; full
  C3/C4 checks still require the exact registered readiness object separately.
- Added CPU negative tests and AST comparison against the committed original
  upload bodies, proving the allocation/copy bodies were moved without a second
  algorithm. These are not GPU ownership/correctness or cold-admission proofs.

## Author verification, not independent approval

Local C:/Python314/python.exe; no local Torch/Warp installation. No GPU tests
were attempted or skipped as acceptance. Completed runs:

- Initial environment contract:34 passed in0.93s.
- Environment/core/guarded lifecycle:152 passed in7.70s.
- Added storage-only tests:166 passed in7.73s. Windows Start-Process reported no
  readable ExitCode on that first invocation; its stdout summary was observed,
  but that invocation was not claimed to have a separately verified exit0.
- Fresh confirmation, process handle retained before waiting:166 passed in7.70s,
  HasExited=True, ExitCode=0. PID36820, parent PowerShell31508, start
  2026-10-07T20:39:02.0426861+08:00. Earlier PID24716/parent13704 also finished.
- Final environment root-kind amendment plus directly affected tests:58 passed
  in1.30s, tool command exit0. The guarded controller was unchanged by that last
  amendment. Ruff check and git diff --check pass on all changed/new code/tests.

An initial Ruff E501 in the new test was corrected by formatting, then the full
changed-file check passed. Initial exploratory reads used two nonexistent test
paths and one Windows rg wildcard argument; rg subsequently located the actual
files. Those read/lint issues are not represented as product acceptance failures.

Author logs outside the repository:
C:/Users/AG/新建文件夹/c4_assets/c2d3-core-cpu-author-confirm.log and .err.
Earlier no-exit-code invocation: c2d3-core-cpu-author.log/.err.

## Required next implementation work (still incomplete)

1. Freeze actual deployment/native inventory, missing-probe and publication tables;
   implement real captured-session external loader/adoption and evidence replay.
2. Construct exact single-use guarded RuntimeSetupBinding; share the cores without
   C0 abort fallback; adopt actual readback destinations and supervised waits.
3. Extend literal repository admission and pin the new bootstrap/policy/catalog.
   Current source catalog is intentionally NOT repinned for these unfinished
   edits: it must reject changed source bytes rather than silently admit them.
   Source-frozen entry acceptance is therefore not claimed at this WIP revision.
4. Fixed supervised actual Torch/Warp cold setup and all required real assets,
   dual-device and fault/hang tests; existing C0/C2d0-2 regressions. No CPU mock or
   preloaded process substitutes for that matrix.
5. Update exact-path cleanup inventory, freeze the complete candidate and start a
   fresh independent code review. Repair/re-review until that scope is approved.

No code review was requested for this incomplete intermediate state. Existing
approvals do not cover these code edits. Product worker, gate.json, source-policy
admission, sample/result/baseline activation and shared collaboration rules are
not expanded. Full C2d/C4 and all previously closed later paths remain incomplete.

## Round-end hygiene

Final local Win32_Process query showed no python.exe/python3.exe/ssh.exe residual.
The confirmation test process was explicitly waited and observed exited0; other
short-lived test utility/child PIDs were not all sampled, so no complete process
ledger is claimed. Each subprocess test has bounded wait and successful return.

Primary183 remained reachable. Read-only check at20:43:10 showed no ang.gao
Python/test/worker/supervisor processes, only user-session/current SSH readers;
GPU inventory showed existing MPS2238410,60MiB, untouched. Remote design checkout
remains clean f668b5b; no WIP code was installed/copied there and no GPU work was
launched in this turn. No101 fallback, unrelated process kill, install, reset,
original-worktree edit or historical-hang-resolution claim.
