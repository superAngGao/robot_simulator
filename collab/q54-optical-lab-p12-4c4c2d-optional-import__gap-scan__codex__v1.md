# C2d3 optional-import compatibility — independent gap scan v1

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption  
Stage: gap-scan  
Author: codex, independent reviewer  
Version: v1  
Date: 2026-10-07  
Status: in_review  
Related Files: runtime-adoption design-request v2 and design-review v2; native_bootstrap.py; runtime_environment.py; workloads.py; optics/mjcf_visual.py  
Owner Summary: The pinned trimesh dependency exhibits started-module failures that its parents intentionally catch. The approved absent-leaf exception does not cover those enclosing failures. Current sticky handling is consistent with the accepted policy; the gap is between that policy and mandatory real Go2 materialization.

## Identity and verdict

Reviewer session: `/root/c4_c2d3_optional_import_gap_review_v1`.

Inspected candidate: `a502ef1e01f16c45ab90faf58e3b3692122cafb2`; parent: `3f8c7630e2bb8c60d5757b34b7c6202a403626ab`.

- Local: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Remote: `/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`, primary183 through `tileops-dev`, detached HEAD at the same candidate.
- Both working trees were clean on initial and final checks.
- Approved design v2 SHA-256 independently matched `1a0d3eed4dd182ddd5ea4ad0c19344cad735aabbdd317d77a14aad1932cc0224`.

Verdict: **Request changes — a bounded design decision is required.**

This verdict concerns the discovered design/dependency compatibility gap only. It is not C2d3 code review, overall implementation rejection, or approval of any runtime/GPU property.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely. No subdelegation, candidate edits, policy changes, package installations or alternative materializer were performed.

## F1 — P2, blocking required Go2 acceptance: absence of a leaf does not make its already-started importer failure non-failing

### Requirements and implementation locations

Runtime-adoption design v2:

- Lines 163–196, section 4.1: optional absence requires no spec and zero create/exec/body execution **for the absent request**.
- Lines 187–192: started-import failures remain sticky, including when swallowed.
- Lines 330–336, section 7: any failure after admitted create/exec begins poisons the session; only the section 4.1 pre-execution absence is non-failure.
- Lines 160–161 and 377–379: retain the canonical materializer and demonstrate both real Go2 materializations.

Candidate implementation:

- `optical_worker/native_bootstrap.py:313–320`: the captured finder turns the absent sentinel into the ordinary missing-module exception.
- `native_bootstrap.py:562–585`: exact absent rules require an absent spec and record `absent_probe`.
- `native_bootstrap.py:404–434`: an exception escaping an external module’s execution invokes failure handling, regardless of whether an outer dependency catches it.
- `native_bootstrap.py:527–546`: failure is sticky and preserves available owners.
- `native_bootstrap.py:737–758`: auditing requires completed external module records; retaining an incomplete module is not successful admission.

The canonical call chain is:

`materialize_native_source` (`native_bootstrap.py:1020`, call at 1059) → `resolve_render_source` (`render_parameters.py:243`) → `materialize_render_source` (`render_parameters.py:265,279`) → `_materialize_canonical_scene` (`workloads.py:46`, Go2 branch 66–90) → `import_mjcf_visual_scene` → `import trimesh` (`optics/mjcf_visual.py:41`).

Mesh loading subsequently uses the existing trimesh OBJ path at `optics/mjcf_visual.py:151–163`. Neither selected image size avoids the initial trimesh import.

### Independently verified dependency behavior

Installed root:

`/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages`

Installed sources establish two distinct failures:

| Started module | Failing statement | Parent that catches it |
|---|---|---|
| `trimesh.ray.ray_pyembree` | `ray/ray_pyembree.py:9`, `from embreex import rtcore_scene` | `ray/__init__.py:4–12`, catches `BaseException`, installs `ExceptionWrapper`, sets `has_embree=False` |
| `trimesh.path.polygons` | `path/polygons.py:2`, `from shapely import ops` | `path/path.py:60–63`, catches `BaseException`, installs `ExceptionWrapper` |

Both modules execute `import numpy` before reaching the failing statement. Neither is an absent module: both have real, found Python source files whose bodies start.

A fresh reviewer-written observer, supplied over stdin, wrapped the standard loaders only for these two exact modules. It ran under:

```text
timeout 45s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -
```

Observed results:

```text
trimesh version: 5.0.0

trimesh.ray.ray_pyembree:
  create_started
  exec_started
  exec_failed: ModuleNotFoundError, name=embreex

trimesh.path.polygons:
  create_started
  exec_started
  exec_failed: ModuleNotFoundError, name=shapely

top-level trimesh import completed: true
trimesh.ray.has_embree: false
failed module names remaining in sys.modules: none
loaded torch/warp roots: none
```

Independent fixed-root resolution found no specs for `embreex`, `shapely`, `manifold3d` or `fcl`.

This independently confirms the author’s observation. It is normal dependency-import evidence, not guarded-worker admission or full Go2 execution evidence.

### Consequence

Adding exact absent rows for `embreex` requested by `trimesh.ray.ray_pyembree`, and `shapely` requested by `trimesh.path.polygons`, can legalize the **leaf absence** only.

The resulting exceptions still escape the enclosing modules’ `exec_module` calls. Section 7 therefore requires sticky failure. A parent’s later catch cannot erase that history.

Consequently, no completion of the existing absent-leaf table alone can reconcile this observed import behavior with both:

1. the unchanged universal started-import-failure rule; and
2. successful required Go2 materialization through this pinned dependency.

If these modules are omitted from admission, their requests fail admission instead. Declaring their present files “absent” would violate section 4.1. Admitting them normally reaches the demonstrated started-body failures.

### Scope classification: mandatory change versus false positive

This is an independently verified **design compatibility gap**, not a defect caused by implementation ignoring the accepted policy.

The loader’s decision to poison these failures is consistent with the current design. A request to suppress that poisoning merely because the exception is `ModuleNotFoundError`, the module is optional, or trimesh eventually catches it would be an incorrect implementation-only fix.

The design-review v2 closure considered missing optional leaves such as manifold3d/fcl. Its claim that the absent outcome closes the required Go2 issue does not establish coverage for an intervening module whose body fails. This newly observed behavior needs a new scoped decision; historical approval is not execution evidence.

## Existing checks and their limits

Independently ran these exact candidate tests:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 45s \
  /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest \
  -q -p no:cacheprovider \
  tests/unit/optics/test_runtime_import_admission.py::test_optional_absence_executes_no_loader_and_is_not_poison \
  tests/unit/optics/test_runtime_import_admission.py::test_swallowed_started_import_error_is_sticky_and_retains_partial_module
```

Result: **2 passed in 0.20s**.

The first test catches absence in the same module that requested the missing leaf, allowing that module to complete. The second correctly verifies sticky failure when a child body fails and its parent catches the exception. They validate the distinction; they do not demonstrate compatibility with real trimesh.

Inspected `test_native_sources.py:130–146`: the real Go2 test calls the ordinary CPU materializer and checks loaded repository sources. It does not instantiate the new external admission session. Therefore the author-reported two passing tests are not proof that guarded runtime admission accepts Go2. Those Go2 tests were not rerun here.

## Bounded design options

None of the following is authorized by this report or legal merely as an implementation adjustment under unchanged v2.

1. **An explicit pre-execution disabled-optional-module outcome.**  
   A separately reviewed design could intentionally disable an exact, **present** optional adapter before its create/exec begins. It must truthfully record presence and deliberate disabling—not `absent_probe`. Freeze exact module/file hash, trusted requester, phase, prerequisite condition, resolver behavior and caller-visible exception semantics. Preserve universal stickiness for every module that actually starts. This requires proving the trusted callers’ fallback behavior and enumerating every required requester, with no wildcard or arbitrary import suppression.

2. **An explicitly authorized dependency revision.**  
   Separately review a pinned dependency change that prevents these optional adapters from escaping module execution on expected absence, while preserving the sole canonical materializer and frozen Go2 outputs. This entails dependency provenance and regression obligations. No installation, fork, patch or version substitution is authorized now.

3. **A new narrowly enumerated expected-module-abort category.**  
   A design could distinguish exact audited optional module aborts from arbitrary started-import failures. This changes the central section 7 rule and carries greater proof burden: actual execution provenance, partial owners and traceback retention, truthful incomplete-module records, side-effect limits, retry behavior, and terminal ownership. It must never be a blanket `ImportError`/missing-name exemption. This is not the recommended first approach.

Recommended next decision: keep current failure handling and all gates unchanged; submit a narrowly scoped design amendment evaluating option 1 first. Its value is preserving the existing “once execution starts, failure is sticky” boundary. If explicit disabling of present adapters is unacceptable, keep Go2 admission blocked and request authority for option 2. Do not silently implement either.

## Required evidence after a decision

Before claiming the gap closed:

- Exercise both real Go2 cases through the actual captured, guardian-first admission path.
- Freeze the full source-audited table; this scan does not establish that these are the only required optional-import cases.
- Verify actual create/exec counts and truthful outcome records.
- Reject wrong requester, phase, source identity, unexpected dependency presence/absence and changed outcomes.
- Preserve sticky handling for unrelated or injected started create/exec/body failures, including swallowed failures.
- Verify custody of partial objects and original errors through the prescribed terminal disposition.
- Preserve the canonical materializer and frozen content/layout/topology outputs.

## Verification limitations and hygiene

The independent probe exited successfully. Its PID was `3351709`, parent timeout PID `3351708`, start ticks `225118845`; subsequent exact PID checks found neither process. Final targeted process search showed no remaining `test_runtime_import_admission` test process, only the inspection shell.

No GPU tests, backend imports, full guarded Go2 positive, native exception-safety audit, installation, asset mutation, process signalling or GPU reset occurred. The author’s running session was not touched. Unit tests used their normal temporary fixtures outside the tracked candidate.

Some inspection commands initially had Windows quoting/path or display-encoding errors; relevant reads were repeated successfully. An initial pytest `-k` invocation was misquoted and ran no tests; the explicit-node invocation above is the successful result. No tool error was treated as a product failure or pass.

This report changes no approval or activation boundary. Complete C2d3/C2d, C2e/f, product GPU preparation, a3b, result/baseline, B3 and other previously closed gates remain closed. **C4 is not complete.**
