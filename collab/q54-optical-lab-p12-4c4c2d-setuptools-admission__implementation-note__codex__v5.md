# Compilation source closure and installed deployment candidate

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: implementation progress; execution preflights remain pending
Author: codex
Version: v5
Date: 2026-10-09
Status: draft
Related Files: setuptools-admission design-request/design-review v2; compilation-admission design-request/design-review v3; setuptools-admission implementation-note v3/v4; test-evidence v3
Owner Summary: Add the compilation sources to the captured CUDA lane and freeze the target's installed dependency profile. Fix an initialization helper placed in the shared prelude: it now belongs only to CUDA compilation. Target file/schema/resolution checks pass; actual setuptools initialization and compilation have not executed.

Base: `d69c281b647f2c101eb340516d2019e2a2fde341`. This remains a work-in-progress
candidate, not an independent implementation approval or permission to execute
real dependencies. The existing design approvals permit implementation only.

## Source capture

Catalog v13/schema v7 includes 85 distinct files (Warp73/CUDA84). The shared
compilation contract is in both lanes because the custody prelude imports it.
The initialization helper, collector/parser and exact CUDA recipes/source owners
are in CUDA only. Compiler-job/supervision code stays outside the captured worker.
The helper was accidentally put in `_CUSTODY_PRELUDE` in b18d50c. Moving it into
`_CUDA_COMPILATION_SOURCES` follows the approved design; a new regression joins
that source map to the shared continuation closure and excludes the prelude.
The stale source pins had kept actual execution closed before this correction.

AST policy v14 refreshes exact bootstrap file/class pins, includes the compilation
root/direct loader, and adds the exact initialization helper and five classes.
Only this pinned helper may load its two named importlib modules, use the three
fixed classes, and read function `__closure__`. Other files gain no such access.
Rehashed source mutations still fail the independent literal release pins.
Authoring packages LF bytes; the production verifier never normalizes input.

Catalog bytes SHA256:
`17387af6b8188c40707da7e1835f639e87d1464ffa664b7cdcdb301eedd9d399`.
Policy SHA256:
`84273d8b655f3330e8fe425cbea71c53911cc0e485e5170d1a24c423c59e879d`.

## Installed profile

`tests/data/runtime_environments/183_cuda_compilation_author_v1.json` extends
the accepted CUDA Go2 v5 profile to v6, preserving every predecessor rule. New
module/namespace/absence rows are runtime-only; no original input permission,
publication, native-global rule or replacement is changed. It has 1,791 files,
1,657 modules, 25 distributions, five namespaces and 38 absent probes.
File SHA256: `1725bfd714059d60010644e9160ef9022d27ce4d32ee89e2bae229979cb8dea8`.

The 84 added modules include the actual top-level imports plus source-audited
lazy imports: setuptools `monkey.patch_all -> _patch_distribution_metadata ->
_core_metadata`, and Torch `_jit_compile`'s first-build imports of hipify and its
GeneratedFileCleaner even for CUDA. `get_config_var('LDSHARED')` uses stdlib
sysconfig on this interpreter; its generated module remains under the existing
interpreter rule. Distutils child files execute under their requested names;
only the root has the approved alias transaction.

Top-level packaging26.1 is resolved from site before vendor. Jaraco and
more_itertools use the explicit vendor files. Python3.12 takes stdlib tarfile in
jaraco.context, so backports is not admitted. Windows-only win32com/JNA branches
and jaraco.text's older-importlib fallback are excluded. Fixed absent requests
are Cython from build_ext, dl from build_ext, and `_distutils_system_mod` from
setuptools._distutils. These actual target probes were absent. Namespace paths
for jaraco and the two distutils compiler parents were checked without importing
or fabricating parent modules.

The existing wheel inventory tool is unchanged. Setuptools source/data/metadata
bytes are joined to the separately captured Conda installed-file facts; Torch
and packaging bytes are joined to the wheel inventory. The full profile passed
the production `RuntimeEnvironmentContract.verify_files` on target Python3.12.13,
plus exact new-name resolution and stat-stable hashes of 4,701,253,155 bytes.
These facts do not prove the actual dependency initialization or completeness of
every reached runtime import. That requires the next reviewed CPU execution.

## Validation and remaining work

The current local focused source/profile run passed180/skipped1 (16.32s). The
skip is the actual POSIX profile parser on Windows; target production parsing
and file verification passed separately. Earlier source-only bootstrap checks
on the same production candidate passed238/skipped9; the later six helper pin
counterexamples were covered by the focused run. These are CPU/source checks,
not a native execution preflight or backend result.

Next prepare the production-boundary dependency-only runner, including explicit
outer lifecycle stand-ins and the RESTORED observation stop, and obtain a fresh
exact execution preflight before importing actual setuptools. The complete
compilation/native/process preflight and real build/failure matrices remain
required afterward. Full C4 implementation, whole-C4 systematic refactor,
affected real regressions and independent acceptance remain outstanding.
No readiness, timing, result/baseline or higher product gate is opened.
