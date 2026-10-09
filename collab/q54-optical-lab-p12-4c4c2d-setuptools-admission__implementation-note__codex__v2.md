# Versioned setuptools deployment facts and alias-parent validation WIP

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: implementation WIP
Author: codex
Version: v2
Date: 2026-10-09
Status: draft
Related Files: setuptools-admission design-request/design-review v2; implementation-note v1; optical_worker/runtime_environment.py; optical_worker/compilation_contract.py; tests/unit/optics/test_runtime_environment.py
Owner Summary: Environment v6 now represents the approved pinned dependency layout while retaining v5's exact field grammar and interpretation. It validates the alias root, ordinary namespace/file descendants, resource/metadata references and fixed site-before-vendor search projection. No initialization object or actual import authority is issued by this code; live integration and all required preflights/real evidence remain pending.

## Revision and implementation

Base: clean `2e29443aa253562080d2dba1db7f26a1dfe0b849` on
`implementation/c4-supervised-waits`. Approved design remains v2 atd62da69,
SHA256e8343ed7c9217a47ad45da5f055801cf43537a67f8059240453ed14078e27db8.
No new approval or broader authority is inferred from this WIP.

- `compilation_contract.py` now returns a detached copy of the one literal
  setuptools descriptor. There is one implementation of its field/value layout.
- `runtime_environment.py` accepts exact v5 or v6 grammars. V6 requires the new
  setuptools_initialization field, either null or the exact pinned descriptor.
  V5 forbids that field and continues to reject an undeclared distutils parent.
  RUNTIME_ENVIRONMENT_VERSION remains the existing v5 authoring constant for
  unchanged callers/fixtures; RUNTIME_ENVIRONMENT_COMPILATION_VERSION names v6.
  No existing profile is silently upgraded or rewritten.
- Non-null v6 requires the pinned Python3.12.13 description, setuptools83.0.0
  PKG-INFO distribution binding, one contained vendor-root layout, declared Python
  source bindings, the required resource-data row and the actual ordinary namespace
  parents jaraco/distutils.compilers/distutils.compilers.C. Reserved distutils root
  cannot collide with other rule families. Descendants use ordinary file/namespace
  declarations; a legacy publication cannot smuggle in another alias child.
- The alias root joins parent validation as a reserved descriptive relation; it is
  not a fabricated source-file row. Declared alias file descendants retain their
  requested intrinsic names and point to their individual vendored source files.
- Search-layout projection excludes the vendor root while inactive and returns
  site-before-vendor while active, independent of lexically sorted root IDs.
  Existing native string formatting/order is preserved for v5. Projection changes
  no sys.path, module, finder or live session; the activation argument is descriptive,
  not an authority token. Bootstrap consumption still needs the issued live owner.

The descriptor and these checks establish schema/layout integrity, not a complete
source closure or proof that hashes came from the installed dependency. Trusted
launcher bindings and frozen actual inventory/preflight remain required. Tests use
explicit synthetic inert files and do not masquerade as the real setuptools wheel.

## Verification

With PYTEST_DISABLE_PLUGIN_AUTOLOAD=1:

`C:\Python314\python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_runtime_environment.py tests/unit/optics/test_setuptools_contract.py tests/unit/optics/test_runtime_custody.py tests/unit/optics/test_cuda_compilation_custody.py`

Final result:112passed in3.60s, exit0. Earlier focused86passed preceded the additional
legacy-publication counterexample and broader runtime-custody regression. The final
run covers it; the earlier count is not separate acceptance evidence.

Positives include unchanged v5 canonical bytes, v6-null file verification, detached
descriptor copies, alias child intrinsic names, required namespace parents and
fixed root precedence. Negatives include descriptor expansion, wrong interpreter/
distribution version, redirected vendor root/resource/source path, resource kind,
wrong phase, missing helper source, extra namespace location, reserved-root collision,
legacy-publication child bypass and attempting v6 semantics under v5. Synthetic
inventory source bytes raise if executed; the positive projection test verifies
sys.path and sys.modules unchanged. Existing role/guardian and runtime/root custody
regressions pass. No backend/dependency/compiler/native/process-supervision execution
occurred, no remote tree was changed, and all local test processes returned.

## Integration still required

Next implement the actual `_SetuptoolsInitialization` owner from the approved design
and connect it to the source session's existing compile retention root/identity
registry. Retain partial construction before fallible initialization. The live
identity binding must not be replaced by a caller-supplied `setuptools_active` flag.
Preserve one actual token registry and one source/controller failure path.

Integration points to complete together: first compile's dependency entry before
cpp_extension; source loader's held compiled code tree and completed-file callbacks;
the one vendor/topology transition; exact shim interposition/restoration; alias spec/
loader and metadata matrix; ordinary alias-parent lookup; finite resource reader and
named metadata resolver; and compilation-v2 observation/replay using the shared
action/anchor vocabulary. Existing `_ExternalAdmission` search-root initialization/
audit and source finder must consult that actual owner. This note does not claim
they already do.

Then freeze full installed import/namespace/optional/data closure, complete CPU
positive/failure integration and obtain the exact dependency-only preflight before
running the real dependency. Its CPU proof stops before full cpp_extension COMPLETE.
Original compilation collector/parser/process proof, source/catalog/runner/observer/
ABI preflight, real cold builds and faults, fresh whole-checkpoint review and later
C4 stages still remain. Source pins are intentionally stale WIP, and no new native
execution gate is open. Whole-C4 systematic refactor remains mandatory before final
acceptance; neither this schema nor its CPU tests completes the persistent goal.
