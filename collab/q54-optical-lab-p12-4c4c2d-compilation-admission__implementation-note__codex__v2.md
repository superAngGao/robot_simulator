# CUDA compilation draft: scope and failure-unwind follow-up

Initiative: Q54 / P12.4 / C4c2 / remaining parent C2d
Stage: implementation note, not independent approval
Author: codex
Version: v2
Date: 2026-10-09
Status: draft
Related Files: compilation-admission design-request/design-review v3; implementation-note v1; systematic-refactor implementation-note v1
Owner Summary: The compilation draft now places dependency binding inside the indexed compile leaf and gives build unwind one owner. CPU counterexamples reproduce and close overwritten original errors and lost exit-audit diagnostics. The whole C4 systematic refactor remains required; this local repair does not fulfill it.

## Changes relative to 3542b3242ae4d59054bb54a3c48bc2cdb695c179

The first cpp_extension import can perform native availability checks. Dependency
binding and hook installation now occur inside `_torch_compile_exact`, after
entering the explicit target device and held stream. Target architecture is
checked there. The fixed environment records an absent LD_LIBRARY_PATH as null,
distinct from an explicitly present empty string. Toolchain/PCH checks also run
immediately before installing the hook.

Previously, nested build/finally handlers allowed hook-drift and exit-audit errors
to replace a build error; the exit-audit error could escape without entering the
collector's failure objects. `_finish_build` now owns hook identity checking, the
single restoration attempt, exit environment audit and error ordering. Every exit
audit runs even after restoration failure. The earliest error propagates, later
errors remain held, and failed journal issuance cannot turn failure into success
or replace the original exception. `failed` latches failure and holds its exception
before attempting token/journal issuance. No retry or successful restoration step
is recorded after failure.

This is a bounded repair under the accepted design. It does not alter a schema,
deadline, admission requirement or product gate. Exact source/catalog pins are
still pending the integrated candidate; these changes have no native approval.

## Observed verification

Windows, C:/Python314/python.exe -B, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1,
pytest -p no:cacheprovider:

- New production-orchestration counterexamples initially gave 2 failed / 5 passed:
  both hook-drift cases lost the exit-audit diagnostic. After consolidation, the
  combined unwind/custody/job/transport/recipe/runtime subset gave 61 passed in
  2.52s.
- Two further counterexamples cover exactly one failed restoration attempt with
  exit audit still reached, and failed diagnostic journal issuance preserving the
  original exception. The resulting unwind suite gave 9 passed in 0.38s.
- The scope tests invoke the production compile helper with CPU context stand-ins;
  they prove ordering and unwind only. They do not prove CUDA context restoration,
  a real dependency import, actual compiler execution or complete custody replay.
- `git diff --check` passed. All local test processes returned; no native worker,
  compiler supervisor or GPU process was launched by this follow-up.

Read-only primary-host inspection confirmed the separate CPU tree remains clean
at b42475e976c568b6a56a10a016a96e78bdf5311c and the prior pytest PID3786836 is absent.
The prior log `/data2/shared/ang.gao/cuda-compilation-b42475e-cpu-v2.log` has SHA256
dd0c3fbffe03e9d0fca0ba4726180e0b4b0c94504ff21b312b26e781232caf18: 36 passed in
11.31s, no Torch/Warp imports, all peer threads joined. This is previous-revision
CPU evidence, not execution of the present unwind changes.

## Remaining work

All unfinished integrated/native work listed in v1 remains required, including
complete deployment inventory, the fixed integrated driver, native observer,
source policy/catalog, collector/source/parser proof, fresh independent preflight,
real Linux process and CUDA cold-build/failure matrices, and whole checkpoint
review. Subsequent parent C4 work and the owner-requested whole-C4 structural pass
remain active. Shared exception ownership here is one concrete improvement, not
an assertion that accumulated C4 hotfixes have been systematically removed.
