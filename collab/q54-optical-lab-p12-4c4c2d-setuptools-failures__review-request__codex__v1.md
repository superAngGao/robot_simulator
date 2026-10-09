# Fixed CPU dependency failure matrix preflight

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: independent bounded code and execution preflight
Author: codex
Version: v1
Date: 2026-10-09
Status: in_review
Related Files: setuptools-admission design v2 and design-review v2; setuptools-cpu-preflight test-evidence v2 and review v3; compilation-admission design v3
Owner Summary: The actual dependency positive has passed. Review the fixed negative selectors in the same CPU driver before one positive control and 32 isolated negative runs. They close the accepted dependency failure-path prerequisite for real compilation; they do not open compiler, native, readiness or performance execution.

## Candidate and requirements

Base: `7794c32` (the handoff supplies its full ID and exact candidate HEAD).
Operative delta: `tests/support/setuptools_dependency_process.py` only. This
request records scope and does not amend the accepted contracts. No production
source, profile, catalog or dependency bytes change. Read AGENTS.md and the two
shared collaboration documents, then accepted setuptools design v2 section8
and its referenced lifecycle/alias/resource/metadata requirements. The existing
positive evidence remains at0177a86; it is not evidence for these new selectors.

The runner still creates the same real source session, finder, external loaders,
initialization owner and guarded host custody. Input/device/toolchain remain
explicit outer stand-ins. All package source execution uses the installed files
and real ordinary loader. Injection changes an actual callback's argument or
state at a fixed boundary, or raises after its real call; it never supplies an
alternate importer or successful synthetic package. Preload selectors use a
foreign ModuleType solely to test rejection before installed dependency entry.

Each child has one case. The existing positive selector must reach RESTORED and
retain its unchanged alias/version/resource/audit assertions. A negative must
reach its named injection, preserve the specified first production error (and
identity for injected errors), poison owner/session/admission, reject subsequent
real checks, have exactly one failed action with no later initialization success,
and complete the actual host-only cancellation without device work. Installed
hooks must each have a restoration attempt; assignment-injection cases count
actual restoration assignments exactly once. The restoration-before fault keeps
the failing binding and may not claim successful restoration. The late-publication
case requires retention of the actual source object after poison.

The negative proof writes a separate descriptive negative.json and exits0 only
after those assertions; this means the test succeeded at proving a denial, not
that initialization succeeded. An unexpected exception, missed boundary or
failed assertion writes failure.json and exits74. Parent wait150s and exact
owned-group cleanup remain unchanged. The foreign-thread selector uses a single
bounded joined test thread, not a second importer or process owner.

## Finite execution scope requested

Review-only: source/AST/stdlib, CPU guardian/helper/parser tests, filesystem and
profile verification. Do not execute the installed dependency, Torch, Warp,
NumPy, SciPy, compiler, native observer or GPU during this preflight.

If approved, authorize one invocation per listed case at the exact candidate:

```
positive
preloaded_setuptools preloaded__distutils_hack preloaded_distutils preloaded_backports
extra_path copied_path vendor_first repeated_begin early_shim repeated_shim
foreign_thread foreign_shim function_code wrapper_closure
alias_spec alias_closure alias_module alias_loader missing_alias unlisted_child
resource_traversal resource_caught resource_changed metadata_path metadata_enumeration
partial_install_before partial_install_after partial_create partial_exec
restoration_before restoration_after late_publication
```

Each is a separate invocation of:

```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/robot_sim_cuda_compilation_wip/tests/support/setuptools_dependency_process.py parent /data2/shared/ang.gao/robot_sim_cuda_compilation_wip /data2/shared/ang.gao/setuptools-failures-<full-approved-head>-v1/<case> <case>
```

The campaign parent directory must also be new and exactly0700. No old run is
reused or modified. Run serially, archive each outcome including failures, and
check all recorded parent/child PIDs absent afterward. No rerun on changed code
inherits this approval. Logs and raw descriptive records must accompany the
subsequent checkpoint review; a preflight does not accept the matrix in advance.

## Evidence and limits

Local existing runner/helper regressions:25 passed,4 Linux-only skips on Python
3.14. These exercise restrictions and production unit boundaries, not the new
installed-source failure matrix. Target source-only regressions and exact hashes
are supplied in the handoff. Full actual v2 compilation-record integration,
compiler-tree OS proof, cold builds, device matrices and fresh whole-checkpoint
review remain required. CPU failure coverage must not substitute for them.

The owner requires a systematic whole-C4 refactor after functionality and real
validation, before final acceptance. This temporary test driver remains outside
product code; no additional production admission layer is introduced here.

Freeze the candidate. Use a fresh independent session and write the unchanged
full report outside tracked files to
`C:/Users/AG/新建文件夹/c4_assets/setuptools_failures_preflight_review_v1.md`.
Report actual revisions/tree status, findings and repros, commands/results,
unexecuted checks, standard verdict and exact approved or denied execution scope.
