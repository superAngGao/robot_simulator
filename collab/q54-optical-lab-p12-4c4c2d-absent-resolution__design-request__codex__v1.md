# C2d3: normal absent resolution under one closed import authority

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: design-request
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design-request v2; optional-adapter design-request v2; native-self-alias design-request v1; metadata-probe gap-scan v1; optical_worker/native_bootstrap.py
Owner Summary: A missing optional module must be absent for both ordinary import and metadata-only find_spec. Preserve the dependency's normal semantics without allowing a second finder to supply the missing module. No Go2 deployment table or unfinished C2d3 implementation is approved here.

## 1. Bounded change and facts

The inspected implementation is `306842c994228c80cae31dc79ce01f0748d4ee1b`.
Independent gap scan `/root/c4_c2d3_metadata_probe_gap_review_v1` has SHA-256
`63f2e31893490b2b43f3f5d0af7a69289640108d7904fe4dc319eb7576ae96c2`.
This supplement changes only runtime-adoption v2 section4.1's requirement that
the captured finder itself always raise ModuleNotFoundError for frozen absence.
All exact-name/requester/phase/source checks, started-failure custody, native
trust obligations and whole C2d3 acceptance requirements remain in force.
Optional-adapter denials remain exact ImportError token consumption, not absence.
Creator-self publication remains the approved schema-v3 relation.

Installed trimesh5.0.0 `util.has_module()` calls ordinary
`importlib.util.find_spec(name) is not None`, without an exception handler.
`trimesh.creation` calls it for mapbox_earcut, manifold3d and triangle;
`trimesh.exchange.dae` calls it for collada. These are metadata-only requests
inside their actual importing owner's body, not import attempts that catch a
missing-module exception. Current _CapturedSourceFinder turns the verified
absent sentinel into ModuleNotFoundError, which escapes these bodies. Sticky
poison is correct after that failure, but the induced failure is incompatible
with the dependency's required missing-spec semantics.

Merely returning None while leaving later meta-path finders installed is not a
fix: importlib would consult them. A found spec there could contradict the frozen
absence. Do not patch trimesh, replace has_module, install optional packages,
catch generic dependency errors or disable sticky failure.

## 2. One finder, explicit standard-library delegation

Use the existing captured finder as the **sole** installed sys.meta_path entry
from successful cold installation until one-shot process exit. No second
repository/external loader or late util.find_spec monkeypatch is introduced.

Before installing it, the -I -S -B bootstrap must verify the original chain is
exactly, by identity and order:

1. importlib.machinery.BuiltinImporter;
2. importlib.machinery.FrozenImporter;
3. importlib.machinery.PathFinder.

Reject extra/reordered/replaced entries before invoking one. This is a pinned
CPython deployment prerequisite; another interpreter configuration fails closed
instead of adding a compatibility finder. Keep the original exact delegate
classes and callable identities in the source-session-owned import topology.
The delegates are internal stdlib resolution tools, never another installed
meta-path chain for external imports.

Freeze the cold sys.path string tuple and sys.path_hooks callable-identity tuple
at installation. They must remain unchanged. The normal -I -S interpreter search
paths contain its stdlib/zip/dynload roots, not site packages or the repository.
Reject empty/relative entries and repository/site-package search entries;
when binding the frozen external environment, also check that none of its site
roots is a sys.path entry. Do not add site roots to sys.path. The startup hooks
are those of the trusted isolated interpreter before dependency/repository body
execution; holding their actual identities is not authentication by function
name, class name or inspection of a closure. No caller can supply replacement
startup hooks through the fixed launcher.
Standard-library path caches may populate normally through the pinned standard
hooks; this does not authorize arbitrary path hooks or plugins. The interpreter
and its standard-library implementation remain trusted deployment dependencies,
not an adversarial same-process sandbox. Existing interpreter path-cache contents
are trusted state, not independently authenticated executable capabilities; no
request can install a cached finder. This change does not claim to sandbox
malicious modification of standard-library implementation internals.

Dispatch in the sole finder is explicit:

- For a top-level name belonging to the frozen interpreter's
  sys.stdlib_module_names or sys.builtin_module_names, invoke the captured
  BuiltinImporter, FrozenImporter and PathFinder find_spec methods in their
  original order, stopping at the first non-None spec. These delegates are
  never invoked for a non-stdlib name. Pass through the standard-library
  package path/target semantics; this creates no external admission record.
- Repository roots use the current captured-source allowlist and loader.
- Other names use the current external admission only after its proper
  configuration/phase; before that, reject rather than consult a delegate.

The existing fixed PathFinder used **inside external admission** continues to
resolve only the declared site roots or exact held parent locations. This is
not the stdlib fallback and cannot extend runtime permission.
Do not forward find_distributions or install a metadata/plugin resolver on the
sole finder. Frozen distribution METADATA verification remains the explicit
CPU file-contract check, not entry-point discovery or execution.

Check the complete singleton meta-path identity, fixed sys.path, hook identities
and original delegate bindings at finder entry, after delegated resolution and
at existing session/segment audits. A trailing finder is drift even if the
captured finder is still first. Recheck before returning a verified absent result
so a resolver-side mutation cannot open a following finder. Do not restore a
changed chain or retry resolution after failure.

Topology checks are separate from session liveness: after poison, trusted stdlib
imports needed for bounded error encoding/termination remain available if the
same topology is intact. They cannot permit repository/external imports,
clear failure, advance a guard or authorize cleanup. If the topology itself is
untrustworthy, retain available state and let the existing supervisor enforce
terminal exit; do not temporarily restore broad import access for cleanup.

## 3. Absent result semantics

Keep the existing exact absent row keys `name, requester, phase` and environment
schema v3. No new public probe API or caller-supplied mode is necessary.

For an allowed request, the production admission must still prove:

- the live exact source session/controller and actual active requesting module;
- the frozen rule, source, phase and unchanged fixed resolver configuration;
- no matching module in sys.modules, and no spec under the approved paths;
- no target create_module, exec_module, body or native-create invocation.

Record the no-spec outcome from that actual check. The admission returns its
internal absent outcome to the same captured finder. After validating its name
and import topology, the sole finder returns None. Therefore ordinary CPython
provides the intended two behaviors without a second resolver:

- `importlib.util.find_spec(name)` returns None;
- an actual import raises CPython's normal ModuleNotFoundError, with its normal
  from-list semantics, after the singleton chain has no spec.

An importer that catches that normal missing import may complete. An exception
escaping a started enclosing module still poisons it; this supplement grants no
module-abort exception. A resolver/adoption/recording/check failure is not an
absent result and remains sticky even if dependency code catches the exception.
No exception-name/message test can legalize a started failure.

Do not infer whether the caller invoked a query or import by inspecting frames
or supplied strings. The same proven absence supports both normal APIs. Repeated
identical probes may record repeated absence; a later found module, sys.modules
insertion, wrong requester or resolver drift fails before target execution.

The future versioned runtime-custody record reports `absent_probe`/no-spec and
zero execution, not a fictitious thrown exception for metadata queries. It is
descriptive and cannot mint admission. The four concrete metadata-probe rows
remain separately frozen deployment-table obligations; this design is not an
approval of an observed module list or the unbuilt Go2 environment.

## 4. Required implementation and negative proof

Use real production finder/admission and actual importlib entrypoints. Unit
doubles may isolate custody but do not prove cold-worker or GPU properties.
Require before whole C2d3 acceptance:

- Successful ordinary import/catch and metadata query of the same frozen absent
  leaf; both zero create/exec/body, truthful absence records and unchanged state.
- The four actual installed has_module call sites reach their original False
  branches. Cold captured real Go2 must subsequently match both frozen cases,
  then complete actual Warp/CUDA a2 upload/readback; ordinary imports alone are
  not proof of captured admission.
- A sentinel finder installed before bootstrap, appended after bootstrap or
  inserted during resolution must never execute for the rejected request.
  Check exact side-effect counts, failed state and no returned usable spec.
- Changed sys.path, hooks, delegate binding or parent package path; known optional
  presence; unexpected absence; wrong requester/phase; preloaded None/module;
  record/audit failure and swallowed started-module errors all fail closed.
- Poisoned external/repository requests remain rejected while a previously
  unloaded stdlib module can still be resolved for terminal reporting. Such a
  stdlib import must not change the failure disposition or guard sequence.
- Cold bootstrap/core verification, source-custody, creator publications,
  disabled-adapter fallback, environment/storage authority and C0 A/K regress.
  C0 uses its existing path and protocol, not this source-session installer.
- Real supervised failure/exit/reap evidence remains required by the parent
  checkpoint. No new sync, watchdog ARM, unbounded wait or cleanup path is added.

Repin exact Git source bytes/classes in a new policy/catalog version; retain
historical artifacts. No general AST importlib exemption. Keep this change out
of measured acquire/enqueue. Update relevant packaging tests and fixed source
catalog digests; do not rewrite old evidence as if it ran the new importer.

## 5. Approval and cleanup boundary

Independent design review permits this bounded implementation only. It does not
approve current C2d3 code, a CPU subset in place of the required real matrix,
or the proposed Go2 table. Whole C2d3 still needs its versioned custody evidence,
both complete frozen lanes, all mandatory failure/hang/multi-device validation
and independent code review.

C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, R2 host
payload and shipping remain closed. No shared README or private agent context
changes. The selected backend retains one production import authority; P12.4e
removes losing-lane fixtures and temporary validation drivers only after
equivalent selected-worker failure coverage exists. C4 is not complete.
