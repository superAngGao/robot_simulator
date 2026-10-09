# Pinned dependency owner and bootstrap integration WIP

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: implementation progress, not a completed review checkpoint
Author: codex
Version: v3
Date: 2026-10-09
Status: draft
Related Files: setuptools-admission design-request/design-review v2; implementation-note v2; compilation-admission design-request/design-review v3; systematic-refactor implementation-note v1
Owner Summary: The private initialization owner is now connected to the existing bootstrap, compiler entry, token retention and failure path. Local CPU helper regressions pass. Complete recording/replay, installed closure, source pins and real dependency/native evidence remain unfinished; no execution or acceptance gate opens.

Base: 73ed3b45ca02d92b43655d98e8813e5980b408ce on implementation/c4-supervised-waits.
This note describes its accompanying local WIP commit. No independent code review
has approved this revision. The remote compilation tree remains at52fb36a; this
turn did not transport files or execute anything remotely.

## Changes

- Added the private setuptools initialization owner and fixed helper objects.
  It uses the existing observation tokens, compilation retention root and guardian,
  with the shared field-specific operation anchors. It retains partial construction
  before initialization/adoption and uses one source-session identity registration.
- Wired first compiler dependency import to begin the owner before cpp_extension,
  and to require actual completion before installing the compiler hook. Later
  compiler audits verify held hooks, alias, helper identities and resource bytes.
  Read-only post-guard audits do not reopen a setup guard or adopt another owner.
- Ordinary source loaders hold the compiled code used for actual execution, then
  notify the owner after ordinary completion. The owner verifies original function
  code/globals/closures and interposes only the two approved shim globals. Source
  loader finally invokes the sole restoration owner on override return or partial
  hook installation failure. Repeated finalization does not assign hooks twice;
  failed or missing adoptions are never manufactured during restoration checks.
- Added the bounded vendor-path transition, site-before-vendor lookup, original-P
  stdlib resolution, exact alias-root dispatch, source/alias metadata checks and
  ordinary child/namespace resolution. The singleton finder and sys.path object
  remain checked, including terminal topology checks after poison.
- Added the one read-only in-memory resource reader and owned named metadata
  distribution. Actual stdlib resource adapters/version parsing consume them;
  traversal, filesystem-path export, changed files and wrong execution contexts
  fail permanently. No distribution scanner or separate importer was added.
- Frozen toolchain environment now requires SETUPTOOLS_USE_DISTUTILS=local.
  The compilation-mode fixed worker reads and verifies the deployment before
  bootstrap, enters its absolute, owned0700 run root with pybuilddir.txt absent,
  checks directory identity, and later rechecks the same deployment bytes.
  This worker behavior has been parsed but has not been executed this turn.

## Verification and limits

Local Windows Python3.14, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1:

```
C:\Python314\python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_runtime_environment.py tests/unit/optics/test_setuptools_contract.py tests/unit/optics/test_setuptools_admission.py tests/unit/optics/test_runtime_custody.py tests/unit/optics/test_cuda_compilation_custody.py tests/unit/optics/test_cuda_toolchain_environment.py tests/unit/optics/test_cuda_compilation_unwind.py
```

Final result:159passed in12.76s, exit0. All local test processes returned.
AST parsing of the seven changed Python files and git diff --check passed before
the final two topology counterexamples; pytest subsequently loaded that revised
test file and exercised the corresponding production topology method.

The new helper tests use actual production initialization/helper methods, the
actual guardian, observation registry and compilation retention root. Their outer
session and source records are explicitly descriptive stdlib-only stand-ins.
Tests exercise real stdlib resource adapters/UTF-8 reads, Distribution parsing,
distinct retained handles, caught denials, wrong source contexts, changed files,
closed handles, function-code/closure changes, recompiled lookalikes, partial
installation, failed adoption, once-only restoration, coherent original-slot
swaps against actual adoption anchors, copied-owner rejection and real PathFinder
resolution excluding a vendor shadow. Topology identity tests separately call
the production checker with a descriptive interpreter surface.

These do not prove the real installed setuptools transaction, complete same-object
alias flow, exact target Python3.12 behavior, complete bootstrap integration,
compiler entry, actual worker cwd, process supervision, native context or speed.
No actual setuptools, Torch, Warp, compiler, supervisor, subreaper or GPU process
was launched. No test emits a real cpp_extension COMPLETE claim. Existing C2d3
native evidence is not reused as proof of this changed bootstrap.

## Remaining integration

1. Finish compilation record/selection v2 and full action/source/lifecycle/retention
   replay joins. Current v1 grammar still cannot accept the new journal events.
   Preserve the original input prefix and trace; add complete alias/read ordering,
   failure-prefix and coherent permutation counterexamples against the real parser.
2. Freeze the full installed Python/namespace/absence/data closure in the v6 target
   profile. Refresh source policy/catalog pins and capture definitions, including
   this module and the changed bootstrap/worker. Those pins remain stale WIP now.
3. Complete production-boundary CPU integration and failure selectors. Obtain the
   dedicated exact dependency-only preflight before importing actual setuptools;
   that proof stops before full cpp_extension COMPLETE. Current helper tests do
   not substitute for this required integration.
4. Complete the original compiler/native/process proof packet and fresh preflight,
   then actual cold builds, target1/ambient0 observations and failure matrices,
   followed by independent whole-checkpoint review. Continue later C4 phases.

The owner's speed question was answered explicitly: no fair CUDA/Warp performance
result exists yet. Compilation, warmup and equivalent native-ready inputs must be
established before the authorized performance measurement stage. Prioritize that
functional prerequisite chain. The owner-requested whole-C4 systematic refactor
still follows functional implementation and real verification, before acceptance;
this local restoration cleanup does not fulfill it.
