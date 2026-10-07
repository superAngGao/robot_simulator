# C2d3: guarded native-runtime admission and actual a2 upload

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: design-request
Author: codex
Version: v1
Date: 2026-10-07
Status: in_review
Related Files: runtime-integration design v3; source-custody design v1 and
code-review v2; optical_worker/native_bootstrap.py; native_setup.py;
guarded_lifecycle.py; native_contract.py; native_input.py
Owner Summary: Extend the approved CPU custody into real, supervised Torch/Warp
initialization and input uploads. No product dispatcher or sample execution is
enabled. Do not mistake module retention for control of third-party native
internals, or upload readiness for completion of the future optical graph.

Base: ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3. C2d2 implementation approval
binds 0dd684867a8d60a69978e878b26d98169d2b9b1c; ff2209e is its evidence archive.
This supplements, not replaces, runtime-integration v3. The shared collaboration
rules and already approved C0 A/K protocol are unchanged.

## 1. Gap scan against actual code

The captured-source session currently admits only twelve CPU modules. Its
finder rejects NumPy/Torch/Warp, and native_setup is only imported, never called.
NativeGuardedSetupLifecycle can already own that exact session and supervise
normal setup segments and terminal cleanup. A catalogue entry is not permission
to execute an additional module or native library.

The existing native_setup functions already separate destination allocation
from upload and register the destination before copy. They still perform backend
imports themselves, use a fresh NO_SETUP_GPU_WORK entry condition, and contain
the old C0 exception/abort policy. They cannot simply be wrapped after a separate
runtime initialization: that would either reject the epoch or import/initialize
outside the intended custody. Their event waits are blocking and must execute
inside the already armed setup segment, not start a nested watchdog.

Read-only inspection on primary 183, robot_sim, found Torch 2.9.1+cu128,
Warp 1.12.0 and NumPy 2.5.1. These observations are not a new frozen environment
manifest or evidence of successful runtime execution. In those installed sources:

- torch/__init__.py loads native extensions and, by default, invokes entry-point
  backend plugins through _import_device_backends().
- Warp Runtime construction calls wp_init, builds device metadata and prints
  per-device total_memory unless config.quiet is true. The native implementation
  of wp_cuda_device_get_memory_info temporarily retains a device primary context
  when it is not already acquired. Restoring the previous context does not undo
  the fact that another device was touched.
- Warp Device.context/stream and Torch CUDA lazy initialization defer additional
  work. A successful Python import is not a proof that initialization is finished.
- Existing in-process GPU tests preload backends before the lifecycle. They are
  regressions, not proof of the required cold, guardian-first import boundary.

## 2. Bounded delivery and closed gates

C2d3 delivers one production-owned runtime admission/custody path, called only
by fixed supervisor-owned sacrificial validation drivers in this checkpoint.
It extends the existing captured session and guarded controller. It performs
real canonical CPU materialization, one lane's actual a2 allocation/upload and
input-ready event wait, exact prepared storage registration, and terminal exit.

No new product IPC operation, caller callback, arbitrary module name, arbitrary
source array, pickle or runtime handle is accepted. No active C4 lease, renderer,
warmup, R_setup/R_final readiness, final graph seal/quiescence, prepared sample
token, terminal optical event or benchmark result is constructed. The existing
prepared-input object is upload authority only, not final execution readiness.

C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, all R2,
host payload and shipping paths remain closed. C2d3 approval is not full C2d
completion. Compilation/recording/warmup and final graph adoption follow the
parent design after this real input boundary is independently accepted.

## 3. Trust and external-runtime ownership

The trust boundary remains the sealed trusted worker, not hostile Python in
the same interpreter. Repository code is captured and AST-checked; third-party
Python/native libraries and the OS/driver are trusted, versioned deployment
dependencies. Do not advertise a Python import wrapper as a native sandbox.

There are three distinct resource classes:

1. Repository modules: retain the exact partial module before body execution,
   execute held approved bytes, as C2d2 already requires.
2. External modules: retain exact spec/loader and import intent before delegating
   create_module; retain its returned module before exec_module, then retain the
   completed module and its actual dependency owners for the worker lifetime.
3. Returned runtime/device resources: retain the exact runtime root, device,
   stream, host staging/view, destination and event before the next GPU leaf.
   The prepared-input registry freezes actual physical storage, not just labels.

A native extension may initialize internal state before create_module returns.
The enclosing worker process is the terminal owner of that as-yet-unreturned
native state; the guardian holds the loader/spec/session and all available roots.
An exception in create_module or partial runtime initialization is UNKNOWN scope
and incomplete coverage: hard quarantine/exit, never target sync plus ordinary
release. Retain the exception and returned partial objects before unwinding the
repository driver. An importer must not discard a returned object because its
subsequent registration/audit fails.

This does NOT prove that a failing third-party C/C++ constructor performs no
internal cleanup before it returns to Python. Correct exception safety inside
the pinned dependency is an explicit trusted dependency obligation, not a
property enforced by our guardian. If source inspection or real fault evidence
finds native code freeing storage still in use, stop admission and fix/replace
that dependency in a separately reviewed change; do not claim quarantine repairs
an already completed native use-after-free. No new native allocator/interposer
or library fork is authorized by this checkpoint.

## 4. Frozen environment admission

Before runtime entry, a CPU-only deployment descriptor supplies the interpreter,
approved site-package roots, distribution/native-library inventory and digest.
It is a trusted launch configuration, not a request or self-certified receipt.
Freeze actual versions, file hashes and resolved contained paths in the code
review candidate. Include the required transitive external Python distributions
and native binaries, not just the three top-level version strings. Record driver
identity separately; do not pretend wheel hashes include the driver.

Reject alternate roots, unlisted external modules/loaders, site customizations,
editable package paths and repository modules reached through site packages.
Keep -I -S -B, no repository sys.path, no site.main(), .pth execution, user site,
sitecustomize/usercustomize, entry-point plugin or arbitrary import hook admission.
Set TORCH_DEVICE_BACKEND_AUTOLOAD=0 before Torch import and verify it remains so.
Set and validate Warp config.quiet=True before any wp.init/get_device/get_stream.
Configuration changes are worker-local and identical on every repeated run.

Extend the existing finder with a fixed external admission adapter; do not build
a second repository loader. Only delegated standard file-based loaders under the
frozen environment are permitted. Explicitly cover native create_module as well
as exec_module. Pre-existing non-stdlib modules are rejected at cold start.
Pin the new bootstrap AST exception and full bytes in the next policy/catalog
versions; no blanket importlib/ctypes exemption. Third-party native loading is
within the frozen dependency trust boundary, not the repository AST exception.

External files are deployment-owned and immutable for the worker lifetime. Audit
their origins/digests and actual import ledger at admission and segment success.
This is not a TOCTOU defense against a privileged actor replacing a loaded .so.
Such an actor is outside this trusted-deployment model. Unexpected import,
replacement or metadata drift observed by the supported path is sticky failure.

Before code review, freeze the literal repository closure required by the existing
render_parameters/native_input/workload materializer, including package initializers.
Every repository body still uses captured bytes and before-body custody. No
benchmarks, examples or arbitrary tools dependency may enter. Generic input
constructors cannot replace the canonical materializer merely to reduce imports.

## 5. One upload implementation and a runtime binding

Refactor native_setup only as far as necessary to expose a shared internal upload
core for each backend. Existing C0 public functions retain their signatures,
fresh-epoch checks and A/K failure semantics, and call the same cores. Their
existing tests remain mandatory. No copy of either allocation/upload loop may
live in the new worker driver or tests.

The new fixed admission path uses the captured production classes and controller:

CPU core/prelude -> guardian/source custody -> guarded external admission and
canonical CPU materialization -> guarded indexed runtime initialization ->
guarded shared a2 upload/input-ready wait -> prepared storage validation ->
terminal supervised disposal -> observed process exit.

These are nonnested setup segments with normal ARMED before work and successful
COMPLETE/DISARMED before advancing. A segment may cover several leaves but may
not renew its deadline mid-call. The initial guardian precedes any NumPy/Torch/
Warp import; CPU module definitions needed to create it remain the C2d2 prelude.

Create one exact, non-serializable RuntimeSetupBinding after successful indexed
initialization. Its production registry binds session, controller, lane, target,
actual module/runtime roots and stream. Constructor is internal to this fixed
admission path; no class-name, duck-type, supplied hash or parsed record mints it.
It is single-use for this source upload. Copies, foreign bindings, repeat upload,
changed stream/device/module and cross-session use fail before their next leaf.
The binding is registered and adopted before the upload core can consume it.

The core receives the fixed binding rather than caller-selected functions. C0's
wrapper constructs its own internal lane context; it cannot accept the new live
binding as a shortcut around its fresh-epoch rule. Shared mechanics do not share
abort policy: guarded failures terminate through the guarded controller, with
zero C0 notifier, fallback sync or second ARM.

For every potentially device-touching leaf, commit a pure-CPU descriptor BEFORE
the call. Before every copy, hold exact source/view/destination/stream. Validate
actual returned device; an unexpected ordinal expands dirty scope before any
further work. Freeze physical allocation signatures at registration and recheck
untimed after preparation, including set_()/pointer, shape/stride and owner drift.

## 6. Multi-device scope is a proof obligation

Do not query a CUDA runtime to construct a device descriptor. Use configured
integer ordinal plus the audited fixed call's scope. Metadata enumeration alone
is not an allocation/queue operation, but primary-context acquisition is device
work. The installed Warp greeting above demonstrates why these are different.

The admitted import/init call chain must be audited through its native dependency
entrypoints: normal cold setup may create resources only on the HELLO target.
Use explicit indexed device/stream arguments and restore ambient state through
the runtime's scoped operations; do not use a getter that creates an ambient
stream just to observe it. Deferred initialization is included in this audit.

Before entering a helper whose possible device scope is not established, mark
UNKNOWN rather than optimistically {target}. UNKNOWN and foreign-device history
cannot be cleared by a successful return or a later target barrier. If the pinned
runtime cannot meet target-only normal setup, this implementation cannot pass;
return for a scoped design decision, not a hidden per-device sync expansion,
CUDA_VISIBLE_DEVICES remapping or reclassification of context work as metadata.

Real target1/ambient0 tests are mandatory for both backends. The test must record
how ambient state was established. Any fixture initialization touching device0
after guardian creation belongs in dirty history; it must produce quarantine,
not be erased to manufacture a positive target-only case. A normal restoration
case must establish only ambient selection, without unaccounted foreign work;
the dependency source and actual observation must substantiate that distinction.
If this cannot be demonstrated, report the mandatory case blocked, not skipped
or passed by an ordinal mock. Cold startup and intentionally foreign-touch
negative tests remain separate cases with their actual scope histories.

## 7. Failure, terminal cleanup and evidence

Any failure inside an armed setup segment is sticky. Retain the available graph,
original exception/traceback and partial module/returned owners, then quarantine
and hard-exit under that original supervisor deadline. No COMPLETE, new ARM,
emergency retry, normal destructor cleanup, resumed request or reused session.
An importer error swallowed by dependency code still poisons the import session.

On successful upload and audit, the guard is genuinely IDLE. This checkpoint has
NOT sealed the future optical graph or established final setup quiescence. Its
fixed driver terminates using existing guarded terminal rules: exact-target,
complete known coverage -> one emergency ARM/ARMED -> sole target sync -> whole
graph release -> process exit. No COMPLETE/DISARMED on emergency. The same deadline
covers cleanup/exit; failed/hung sync or cleanup cannot retry. Foreign/UNKNOWN,
incomplete ownership or unexpected external holds -> quarantine without sync.

No external payload lease is created here. A nonzero external hold is nevertheless
a release rejection, not an assumption to ignore. Only the supervisor's actual
wait/reap proves exit. If evidence cannot be written, keep supervisor partial
evidence; never invent worker cleanup or exit. No destructor is the safety owner.

Keep existing guarded lifecycle evidence as the state authority. Add an observed
runtime-custody record (versioned, CPU-safe, exact-key parsed) with environment
digest, session/lane/target, ordered external import create/exec outcomes, actual
origin/loader identities, runtime binding identity, prepared manifest/storage
digest and successful guard sequences. It is descriptive evidence only; parsing
it cannot register modules/bindings/prepared arrays or authorize release. Include
the record in termination's retained graph, not as a cyclic prerequisite of the
earlier source/manifest hashes. Final field order/enums and replay cross-checks
must be frozen in the implementation schema, with negative tests for invented
imports, out-of-order completion and forged summaries. No result schema change.

## 8. Required validation before code approval

Run the actual captured production path in -I -S -B children under the actual
guarded supervisor, on both lanes. Fixed test fault selectors stay outside the
production import closure/IPC. Tests may observe and fault real entrypoints;
they may not replace them with a second preparation implementation.

- Cold import ledger: guardian first, one source session/controller, exact
  environment, no benchmark/package/plugin/.pth execution. Wrong environment,
  preloaded module, foreign origin, copied binding and repacked policy reject.
- Real NumPy/backend partial import: before/after create_module, before/within
  exec_module, import failure before module return, returned owner before adopt,
  nested dependency/audit failure. Sentinels must survive until process-terminal
  disposition; distinguish available Python ownership from trusted native internals.
- Canonical asymmetric fixed fixture and dynamic replay, plus both real Go2
  materializations: per-array content/layout/storage/topology validation using
  actual prepared allocations; no substitute assets or success-by-skip.
- Both backends: first allocation, first copy, submit-copy-then-raise, event
  creation/record and ready wait, owner-register and final storage-audit failure.
  Check zero additional sync/ARM/COMPLETE after a failed normal segment, no
  premature host/device owner destruction, no returned usable prepared input.
- Import/init/copy/ready-wait hangs with parent alive; ACK absence/invalid ACK,
  DISARMED loss, parent EOF at each phase. Validate original watchdog expiry,
  actual child exit/reap, not merely an outer pytest timeout.
- Real dual-GPU cold and ambient/foreign cases from section6. Deliberate helper
  device0 work before return with target1 must quarantine without target sync;
  unknown pre-return scope must do the same. Record actual UUID/ordinal mapping.
- Successful input preparation followed by one supervised terminal sync/release;
  terminal sync failure/hang, cleanup hang and missing exit are bounded and never
  counted as successful disposal. Measured-sample submission count remains zero;
  setup copy/event work is nonzero and is not called 'no GPU work'.
- Regress C0 A/K, C2d0/1/2 custody/supervision, exact readiness/storage authority,
  source-policy/catalog and existing upload parity. Full audits stay untimed;
  do not put storage traversal in C4 acquire.

Every round records PID/start/parent identity, observed exit/reap and scoped OS
plus GPU residual checks. Prefer 183; use 101 only on unreachable transport.
Do not kill others' work, reset a GPU or install packages to manufacture a pass.
Missing hardware/environment is an explicit unmet gate, not code success.

## 9. Review, activation and cleanup

This document requires independent design review before implementation. Code
approval requires the real setup matrix, environment freeze, source-policy delta
and absence of product activation. No approval of a CPU subpiece substitutes for
this checkpoint's GPU proofs. Update the existing cleanup ledger for exact new
paths; retain one loader, guardian, runtime binding and upload core.

Keep selected-backend environment/source custody and supervised resource ownership
long-term. P12.4e removes the losing runtime and temporary fixed validation entry
only after equivalent selected-worker failure tests exist. Do not remove either
candidate now or edit historical approvals/shared workflow to enlarge this scope.
