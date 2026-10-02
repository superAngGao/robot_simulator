# C2d0: execute captured repository source, not unchecked cached code

Initiative: q54-optical-lab-p12-4c4c2d-source-consumption
Stage: design-request
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: runtime-integration design codex v3 sections4/5/8/9;
source-catalog review codex v2; optical_worker/native_sources.py;
optical_worker/native_source_policy.py; tools/optical_closed_worker.py
Owner Summary: Resolve the source-consumption policy before C2d integration.
Add a strictly bounded source-only bootstrap, not a generic code loader or GPU
entry. C0 stays unchanged; all product/measurement gates remain closed.

Base: `129697a969e69d7b75f11094828694b5e80b25e5`.
This is a narrow addendum to the accepted runtime-integration v3 design, not a
replacement lifecycle design. Approval authorizes the CPU C2d0 code below only.

## 1. Code facts and gap

Approved C2c6 captures exact source bytes and checks actual imported module
metadata. Its public documentation explicitly does not claim to establish the
bytes previously consumed by Python. Existing C0 bootstrap sets
`sys.dont_write_bytecode=True` before normal imports; it has no source-only loader.
That old closed CPU protocol is not being changed or newly approved here.

Local CPython3.14 `Lib/importlib/_bootstrap_external.py::SourceLoader.get_code`
reads valid cached bytecode before consulting `dont_write_bytecode` for cache
writes. A disposable probe reproduced the same behavior on robot_sim Python3.12:
valid same-size/timestamp `.pyc` produced `STALE!` while actual source was
`MARKER = 'FROZEN'\n`. `-B` is not a no-cache-read policy. C2c6 evidence v1
records the probe, not an approved implementation of its toy loader.

The current native AST policy forbids the importlib root. Therefore silently
adding an importlib loader would contradict the frozen restricted dialect.
This addendum proposes one explicit, release-pinned exception. It does not
weaken repository source checks or claim a hostile-Python sandbox.

## 2. Deliverable, trust and non-deliverables

Deliver a stdlib-only `optical_worker/native_bootstrap.py`, entered directly as
a source script by `python -I -S -B`, and its CPU tests. It owns the sole native
source-only finder/loader used by later C2d worker/supervisor integration. C2d0
only runs a fixed CPU source-verification path and exits; it cannot negotiate
the native worker protocol, prepare GPU inputs, invoke arbitrary imported
functions, submit a sample, emit benchmark results or choose a backend.

The closed trusted execution domain remains unchanged. Interpreter, stdlib,
configured reviewed Git revision, root and trusted launcher are the bootstrap
trust base. Neither a self-computed hash nor a different self-consistent Git
commit proves owner approval. No root, revision, catalog, module, callable,
source bytes, bytecode, loader or import policy is accepted from worker IPC.
Same-UID arbitrary code/filesystem/native-library compromise is not sandboxed.

The interpreter necessarily executes the bootstrap script before it can verify
its own file. That script is an explicit initial trust root, launched only from
the configured release. It verifies its current bytes against the same release
catalog as all other sources; do not call this retroactive execution authority.
Package import of `optical_worker.native_bootstrap` is not a second supported
entry. The process has only one exact bootstrap owner, retained until exit.

C0 files, its A/K protocol, allowlist and activation remain byte-for-byte
unchanged. No edits to collab/README or prior approved designs. New native
bootstrap is a direct production-package source entry, not another unrestricted
tools import. The source ledger gains one exact allowed `__main__` path for it;
the existing two named tools entry exceptions are not generalized.

## 3. Bounded cold-start order

1. Require isolated/no-site/dont-write-bytecode flags, direct exact script path,
   supported CPython environment and no already-loaded repository modules.
   Initially keep only the interpreter's isolated stdlib search paths. Reject
   an already-installed native source session; never replace an active loader.
2. Read trusted root, full lowercase40-hex revision, literal Warp/CUDA lane and
   expected catalog SHA-256 from the fixed launcher configuration, not IPC.
   C2d0 test launcher passes these as four positional arguments, plus one already
   opened result-pipe FD; no module/function selector, callback or arbitrary
   execution mode. The script accepts exactly those five arguments.
3. Before any repository import, read the catalog from the fixed path and Git
   object, compare configured revision with actual HEAD, expected catalog hash,
   canonical bytes and disk/Git equality. Every selected source path must be a
   normalized contained `.py` with no symlink component or alias. Capture each
   byte string once and verify its catalog digest. Include the bootstrap itself.
   No source is executed during this capture.
4. Install the source-only finder/loader. Initially allow repository execution
   only for the exact CPU verification core below. Load that core from the
   captured byte strings, never from normal filesystem/bytecode import. Its
   reviewed release hashes are the initial trust, before its own AST audit runs.
5. Invoke the sole existing `native_sources` full verifier: exact literal lane
   inventory, source/schema/policy validation and snapshot equality against the
   captured bootstrap map. Run the full AST check on all selected sources.
   Do not duplicate the substantive source-policy algorithm in bootstrap code.
6. Audit actual loaded modules against the captured sources, exact loader/module
   identities and execution records. Emit a bounded canonical CPU-only report
   to the inherited result pipe, then exit. No site-package path or CUDA-capable
   module is admitted in this C2d0 path, even if present in the source catalog.

The initial CPU verification core is exactly:

```text
optical_worker/__init__.py
optical_worker/lifecycle.py
optical_worker/protocol.py
optical_worker/manifest.py
optical_worker/native_source_policy.py
optical_worker/native_sources.py
```

This includes the real eager package initializer dependencies, not an assumption
that importing a package is empty. Its transitive repository imports must equal
this reviewed core or fail before execution. No NumPy/Torch/Warp import is needed.

The first-pass catalog parser only performs the bootstrap checks needed before
loading that exact core. It does not define a second full catalog schema or
source policy. The canonical codec, literal lists and full validator remain
single production sources; after loading them, byte-for-byte replay is mandatory.
Bootstrap's bounded JSON/duplicate-key/size checks are tested against the full
validator. Use the existing protocol's source-manifest size bound; do not enlarge
wire limits. Malformed paths must reject before reading outside the root.

## 4. Exact source-only loading

The finder is first in `sys.meta_path`; the repository root is never inserted
into normal `sys.path`. For repository roots already reserved by native_sources,
unknown/unlisted names fail before a later finder can run. Listed GPU modules
also fail in C2d0's fixed CPU execution subset. For installed dependencies outside
these names, trusted stdlib resolution remains unchanged. Do not implement an
alternate arbitrary-file, `.so`, namespace-package or zip fallback for repository
modules. A repository module under a foreign checkout cannot win resolution.

Each loader instance is bound to one canonical full module name, one captured
source object and one absolute origin. Fixed package names/`__path__` come from
the literal `__init__.py` layout. `get_code(fullname)` requires exact equality,
then calls stdlib SourceLoader.source_to_code with the held source bytes and
that origin. It never calls ordinary SourceLoader.get_code, reads `.pyc`, reads
source again, or writes bytecode. `get_source` returns the held UTF-8 source;
`get_filename` returns the frozen origin; alternate names/paths are rejected.
No caller-supplied code object, byte string or filename is a loader input.

Module execution uses stdlib loader semantics, including relative imports and
package cycles. Before execution, retain the exact module/loader in a local
registry and append a started record. Successful return appends completed;
exceptions append failed and permanently invalidate the session. A parent module
can import a different allowed core module during its execution. Re-entry for
the same executing module through Python's normal `sys.modules` cycle handling
does not execute it again. Explicit loader re-execution, reload, replacing
sys.modules entries, or a second loader for the same name fails closed.

After successful core loading, cross-check actual `sys.modules` with this registry:
exact module, `__loader__`, `__spec__.loader`, name, origin, package layout and
completed execution record. A descriptive source hash or forged metadata alone
does not count as a loader observation. The registry is internal process fact,
not a serializable C4 authority. Source snapshot and execution evidence are
distinct; the latter references the former, never the reverse.

If source changes on disk after capture, the loader still cannot execute the
new bytes or stale bytecode. The later unchanged-source audit rejects the session
before it reports success or, in later integration, admits work. The filesystem
is trusted, but drift is not silently tolerated. No rewrite of cached bytecode,
deletion of another run's cache, or baseline subtraction is part of this design.

## 5. Deliberate native AST-policy amendment

Introduce native AST-policy v3; preserve old C0 AST-policy v1 and native v2 history.
Only `optical_worker/native_bootstrap.py` may import `importlib.abc` and
`importlib.util`, for this fixed finder/loader. Allow only SourceLoader,
MetaPathFinder and spec_from_loader use in its frozen bootstrap implementation.
No import_module, arbitrary spec_from_file_location, loaders selected by name,
runpy, eval/exec/compile builtin, pickle or direct native launch/event calls.
Inherited stdlib compilation/execution is trusted implementation of this exact
loader; do not claim no compilation occurs. No dynamic compilation in renderer
or timed code is authorized.

Pin the bootstrap's complete LF source and loader definitions' exact physical
source bytes in the native policy, as with the existing renderer snapshot. AST
uses locations only, never interpreter-dependent ast.dump hashes. Its expected
pins are frozen in the implementation candidate after code exists; altered source
plus a caller-recomputed digest cannot expand this exception. Policy rejects
aliases, rebinding, added uses, relocation and escaping loader entrypoints.

Keep the existing ctypes and exact regex exceptions unchanged. Version the native
catalog for the new policy/inventory and extend its single snapshot validator;
do not create a concurrent executable old/new admission fallback. Old artifacts
remain historical. No self-hash cycle: bootstrap bytes -> policy digest ->
catalog -> configured release; bootstrap contains no expected digest of itself.

## 6. Failure, supervision and later C2d composition

C2d0 does no GPU work or CUDA-capable imports. On capture/import/audit failure,
emit a failure report if possible and exit nonzero, no retry or partial success.
The fixed test launcher owns a bounded process deadline, closes/reaps the child
on pipe failure or timeout, and records actual exit, not merely a report field.
The report is descriptive CPU verification, never a setup quiescence receipt.
If the bootstrap unexpectedly attempts a non-core repository import, reject it
before its body; do not try to prove GPU cleanup using these CPU-only rules.

Later C2d lifecycle adoption must use this same source loader before importing
the native driver, not revert to default imports. It may extend the literal
executed subset only in the reviewed worker role under the actual guardian and
supervisor guard. Guardian creation precedes all CUDA-capable import/init;
module owners and partial imports join its retained graph before executing them.
Any failing armed import follows runtime-integration v3 section8: quarantine,
no COMPLETE, new ARM or emergency retry. Actual source-ledger checks at startup,
HELLO and every request remain required. C2d0 does not implement that GPU role.

The CPU self-check entry is temporary validation scaffolding, not a second
shipping worker. C2d integrated bootstrap replaces that fixed terminal path
with its reviewed role dispatcher, retaining source-consumption regressions in
isolated test drivers. No arbitrary callback mode survives. P12.4e retains only
the selected native loader/source closure and deletes unused lane/probe paths.

## 7. Acceptance before CPU code approval

- Actual isolated process, source-entry bootstrap, both lane catalogs, actual
  POSIX release bytes and local interpreter portability; no GPU imports/work.
- Timestamp-matched and unchecked-hash stale `.pyc` for actual core source,
  including package initializer: frozen source executes, cached sentinel does
  not. Repeat with matching file metadata; merely changing mtime is insufficient.
- Source-only module retains normal name/spec/package identity and permits
  actual relative imports. Every completed core module has exactly one observed
  execution record; second execution/reload and registry/module replacement
  reject. No report accepts caller-filled execution facts.
- Wrong revision/catalog, changed source or core, symlink/escape, preloaded
  module, foreign checkout, unlisted name, missing source, sourceless `.pyc`,
  repository native-extension fallback and post-capture drift all fail before
  success. For denied modules assert their sentinel body never executes.
- Actual production AST policy checks the bootstrap and rejects valid-syntax
  rehashed mutants adding imports, aliases, arbitrary call/loader paths or code
  execution. The existing renderer/regex and C0 tests remain unchanged and pass.
- Parent remains alive while child/response hangs: bounded launcher termination,
  observed child reap, no orphan; ordinary CPU failure needs no emergency CUDA
  sync and cannot be reported as GPU quiescence. No broad process kills.
- No product dispatcher, GPU preparation, R2/S/host path, C4 result authority,
  a3b or baseline becomes reachable. Code review states the exact approved subset.

## 8. Remaining C4 work

CPU bootstrap approval is not C2d completion. The accepted next work remains
integrated guarded lifecycle/termination and its fixed sacrificial drivers,
Warp timeline/arena C2e, same-driver real-device C2f acceptance, separate R0/R1
activation review, C4c3/c4 and C4d/B3 joint lifecycle obligations. All original
scope boundaries and independent-review/hang-check requirements remain in force.
