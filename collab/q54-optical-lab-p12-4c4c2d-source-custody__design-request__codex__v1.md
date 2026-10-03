# C2d2: captured repository modules join the guarded resource graph

Initiative: q54-optical-lab-p12-4c4c2d-source-custody
Stage: design-request
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: source-consumption design v1; guarded-custody design/repair v2;
guarded-custody code-review v5; runtime-integration design v3 sections4/5/8/9
Owner Summary: Connect the two approved CPU mechanisms before admitting a real
runtime. Use the same captured-source loader and the same guarded guardian;
freeze the narrow loader-policy change and partial-module ownership. This
checkpoint cannot import Torch/Warp, upload inputs, or approve full C2d/C4.

Base: eef7a5052bed6c7699bd98d847fa7ed27147620e. CPU custody implementation is
approved at854c2ea; eef7a50 only archives its unchanged report and progress.
This is an interface/policy supplement, not a replacement of parent v3.

## 1. Existing code and bounded gap

native_bootstrap currently executes only its literal six-module _CORE. Each
module is strongly registered before exec_module, with held source bytes and
exact loader/spec/package checks. Its _SourceSession is CPU-only and has no
guardian. The guarded controller is catalogued, but that is not execution
admission: the finder still rejects it. Merely importing the controller through
normal sys.path would lose the approved source-consumption boundary.

NativeGuardedSetupLifecycle already provides terminal custody and actual
normal/emergency supervision. It must be constructed from its exact captured
module, not a second copy loaded by a test or benchmark package. Its dependency
modules must exist before it can instantiate a guardian/client. This requires
an explicit CPU bootstrap prelude, not a false claim that the guardian preceded
execution of its own class definition.

## 2. Delivery and non-delivery

Extend native_bootstrap's existing finder/loader/session, not a second loader.
Add a fixed internal guarded-custody binding method and one fixed CPU import
operation used by isolated supervised tests. The existing five-argument C2d0
direct-entry source check remains CPU-only and behavior-compatible; there is
no new product CLI role, IPC operation, plugin or arbitrary module selector.

This slice covers repository-module custody only. It explicitly does NOT solve
third-party extension create_module/exec_module custody, site-package admission,
real Torch/Warp import/init or uploads. Those remain the immediately following
C2d runtime-adoption checkpoint, with its source-policy/environment review and
mandatory real-device matrix. No general external import wrapper is approved
implicitly here. Do not call this CPU work completed runtime integration.

No edits to protected C0 lifecycle/native_setup/worker/launchers, guarded-wait
wire protocol, guarded lifecycle grammar, arena authority, gate.json, result
schema or shared collaboration rules. No readiness, sample token, warmup or
quiescence evidence is generated. R2/host/S and all product/result gates close.

## 3. Literal execution phases

Keep the six-module verification core and its successful C2d0 report unchanged.
After its full snapshot/AST audit succeeds, a single-use internal transition
may load exactly these additional CPU-only custody definitions from the same
captured bytes:

```
optical_worker.wait_protocol
optical_worker.wait_supervision
optical_worker.guarded_evidence
optical_worker.guarded_lifecycle
```

Their transitive repository closure must equal the core union this list;
unexpected repository imports fail before body execution. Stdlib dependencies
remain trusted. No NumPy/Torch/Warp or other site packages are admitted. The
existing isolated -I -S -B environment and no-repository-sys.path rule remain.

This finite trusted CPU prelude is allowed before guardian construction. Its
failure is still CPU-only bootstrap failure, bounded by the fixed launcher's
process deadline and actual reap, as in C2d0. It cannot be mislabeled as a
normal setup guard or a CUDA quiescence proof. Test sentinel imports must prove
runtime roots are not entered even on prelude failure.

Construct one controller using the exact class obtained from the captured
guarded_lifecycle module and one exact GuardConfiguration from the captured
wait_protocol module. Configuration/channel are trusted-launcher inputs, never
worker IPC or serialized lifecycle evidence. Do not accept caller controllers,
classes, notifier callbacks, duck-typed clients or pre-imported class aliases.
The existing constructor still owns the actual GuardWaitClient.

Before admitting any further source body, adopt into that guardian the exact
source session (which owns bootstrap/module/loader records and held bytes),
captured snapshot, bootstrap module and all completed prelude/core module owners.
The source-session binding is one-shot and bound by object identity to that
controller/guardian/configuration/lane, not a caller-filled hash. Partial bind
failure retains the session and controller graph through controller termination;
it cannot roll back to CPU verification or retry with a new controller.

The only post-bind execution operation in this slice is a fixed method that
loads optical_worker.native_setup and its sole additional repository dependency
optical_worker.native_contract. Both are stdlib-only at module import. It takes
no module name, callable or source parameter. It executes inside the controller's
own setup segment: ARMED before import/compilation, COMPLETE/DISARMED only after
successful import and exact source/module audit. A failure poisons the session
and reaches existing guarded terminal custody before unwinding to the driver.
The fixed driver hard-exits for quarantine; it never resumes a request loop.

All other catalogued modules remain execution-ineligible in this slice. In
particular, importing native_setup's definitions does not authorize calling
prepare_* or importing its deferred runtime dependencies. Test driver must not
return its module or callable as a product execution API.

## 4. One loader and before-body ownership

Generalize _CapturedSourceLoader's lookup from hard-coded _CORE indexing to the
session's current literal admitted subset. The subset derives solely from the
phase tables above and captured snapshot. Do not accept arbitrary maps from a
caller. Keep one loader per name, exact source object, origin and package
layout; retain all stale-pyc/reload/replacement tests and no fallback semantics.

For post-bind module execution, order is:

1. Check live session, exact allowed name/loader/spec and active setup guard.
2. Register the newly created exact module/loader as started in the session's
   strong registry BEFORE its body. The already-adopted session keeps it alive
   even if the following explicit guardian adoption fails.
3. Adopt the exact module as a host owner in the same guardian before source
   compilation/body execution. No second module-owner list or release authority.
4. Compile the held bytes through the existing stdlib SourceLoader mechanism
   and execute exactly once. No filesystem/pyc reread and no dynamic eval API.
5. Recheck exact module/spec/loader identity and mark completed. At operation
   boundary audit all expected completed modules, dependency closure and source
   snapshot. Only then may the guarded segment complete.

Normal import cycles can reference an already started module without executing
it again. Explicit reload/re-execution, an unexpected dependency, metadata
drift, replaced loader/finder, or missing owner fail closed. Source session
failure is sticky, including errors caught by a module body; no later operation
or successful audit can reopen admission.

If import removes a failed module from sys.modules, the session registry and
guardian retain the original partial object, loader and traceback. Never infer
absence of CUDA work from removal; this slice proves only CPU behavior and is
not the later runtime's safety claim. Exception retention uses the existing
controller failure/termination path, with no new sync or alternate cleanup.

## 5. Audit and terminal semantics

Use actual session/loader/module identities and the existing guardian coverage,
controller evidence and supervisor evidence. Existing started/completed/failed
source records remain descriptive facts, not live authority. Extend session
audit to its exact phase closure; do not change the C2d0 report to claim more.
No new serialized source-custody schema or opaque boolean capability is needed.

The fixed test observation may inspect both actual registries and recorded
coverage before/after the operation. It must check each exact module is held
before its body, not merely that names/hashes appear in an output record.
Parsed records cannot construct a bound session or a live guard.

Terminal cleanup uses the existing C2d1 rules. A failed normal source operation
quarantines without COMPLETE/new ARM/sync. Healthy CPU-only completion then
explicit terminal cancellation uses the existing emergency-role cpu_cancel
ACK, zero runtime sync, and whole-graph release under the same deadline; the
driver exits and supervisor observes/reaps it. No source import/audit is allowed
after terminal intent, even if native controller state still has held metadata.
Repeated binding/import/termination rejects without duplicate body execution.

There is no guardian seal/readiness construction in this slice. Binding a
sealed/terminal/foreign controller is impossible because binding constructs
its own exact controller once. Module graph mutation after eventual future
coverage seal must be forbidden by that later integrated setup contract, not
assumed safe by this CPU checkpoint.

## 6. Explicit source-policy amendment

Introduce native AST policy v4 for the changed pinned bootstrap implementation;
v3 remains historical, not an executable fallback. Retain the exact current
importlib.abc/importlib.util uses only (SourceLoader, MetaPathFinder,
spec_from_loader). No importlib.machinery, import_module, arbitrary loader,
reflection path, external native call or callback is added. Keep ctypes/regex
exceptions unchanged. Pin full bootstrap and changed loader/finder physical LF
bytes after implementation; immutable policy identity then drives catalog v3.
The source snapshot validator follows only that current catalog, no negotiation.
Literal component inventory is unchanged unless implementation demonstrates a
necessary CPU production dependency and reviews it; the expected closure is
the twelve module files listed above including the original six.

Bootstrap source -> policy digest -> catalog -> trusted configured release is
still a DAG. Tests must show repacking an unapproved changed source hash cannot
enlarge the AST exception. New table entries alone cannot admit another body.

## 7. Validation and scope of approval

Before code approval, independently run:

- Existing C2d0 source-only direct entry, stale pyc and path/metadata/reload
  attacks; exact six-module report remains unchanged in that mode.
- Both lane CPU bound sessions, exact twelve-module closure, object identity
  with the captured classes, one guardian and actual socket/core supervision.
  Assert registered/adopted-before-body using fixed test observation at the
  real loader, not a duplicate implementation or a production callback.
- Wrong class/session/configuration, second bind/import, restored failed
  session, replaced module/loader/spec/package, same bytes wrong object and
  unauthorized catalogued body reject before its execution.
- Fail during prelude, partial bind, before explicit module adoption, source
  compilation/body, nested dependency and final audit. Keep sentinels in partial
  modules and check no premature destructor, exact original exception retention,
  no normal completion, no retry and terminal graph custody before unwind.
- Real supervised process: CPU import operation hang, parent loss, missing/wrong
  ACK, failed DISARMED, CPU cancel cleanup/exit hang. The original guard deadline
  must kill/reap; the test outer timeout is a failing safety bound, not success.
- Import-attempt sentinels for NumPy/Torch/Warp; no CUDA-capable import or GPU
  leaf is executed. The external runtime matrix remains explicitly untested.
- Policy/catalog/release byte validation, protected C0 no-diff, C2d1 regressions
  and clean-tree checks. Every round records real process identities, actual
  exits/reap, scoped residuals and GPU process inventory; no unrelated kill.

Test-only fixed injectors may inspect the actual loader/controller in isolated
children; they are not admitted source modules or callbacks in production.
Tests must distinguish a deliberately mutated test release used to reach a
failure location from the unchanged normal source release. No mutant can mint
product readiness or count as successful frozen-runtime execution.

Design approval permits only this bounded CPU source-custody implementation.
Fresh code review is required. The next C2d checkpoint must explicitly design
and prove third-party partial-import/native-create ownership and the real
Warp/Torch upload faults before any GPU-capable role is admitted. Final graph,
warmup/barrier/readiness, C2e/f, C4c3/c4, C4d/B3, product preparation, a3b,
result/baseline and shipping remain closed.

## 8. Cleanup

No duplicate loader, guardian, upload loop or lifecycle stays behind. Extend
the existing exact-path cleanup ledger for new fixed test support, including
tests/support/guarded_peer.py from the approved CPU tests. P12.4e removes the
temporary direct CPU probe and fixed sacrificial entry after equivalent tests
cover the selected worker; the selected captured-source loader, ownership and
supervision remain. Do not remove the probes or either candidate backend now.
