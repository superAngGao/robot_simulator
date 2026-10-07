# C2d3 native creator-self aliases: bounded contract supplement

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: design-request
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design-request v2; optional-adapter design-request v2; native-self-alias gap-scan v1; optical_worker/runtime_environment.py; optical_worker/native_bootstrap.py
Owner Summary: Express the observed native self-aliases without inventing a second file load or attributing creation to a parent importer. This is a small extension of the frozen publication relation, not automatic dependency permission. Whole C2d3 implementation acceptance and all later gates remain pending.

## 1. Scope, facts and precedence

Base code/evidence HEAD is `56acbd375180f5ea89cf8fca3e3e7f854dbc4954`.
The separately archived gap scan has exact report SHA-256
`235ae6a5b085c1cfe32e92e0bd4992a8192028275ba631ea20c56ecca90b296e`.
Its F1 is a missing representable relation; F2 is an implementation defect under
the already approved sticky-failure custody rule. This supplement resolves F1.
F2 still needs a code fix and tests; no policy relaxation is needed for it.

The independent observer called unchanged canonical Go2 materialization. Four
SciPy native extensions publish an additional short `sys.modules` key for the
very module returned by their own standard extension loader. Publication occurs
inside delegated exec, not create. Parent-package attributes are bound later.
The existing nonempty `attributes` path cannot select the creator root itself.
Using a convenient parent attribute would misstate the actual creator/publisher.

This supplement changes only publication-selector semantics and the checks and
evidence needed for that relation. The runtime-adoption v2 requirements otherwise
remain in force, including its native pre-return trust obligations, all mandatory
real-device tests and whole-stage review. The optional-adapter v2 exception is
unchanged; these native aliases are neither optional denials nor absent probes.

No new request, callback, package resolver, runtime discovery permission, source
rewrite, library install, GPU operation, readiness or executable sample authority
is introduced. No shared collaboration policy, C0 A/K protocol or metric changes.

## 2. Versioned selector contract

The active environment schema becomes `q54.p12.4.runtime-environment.v3`.
Keep historical v1/v2 artifacts unchanged; the active parser rejects old versions
and never silently upgrades them. Updated author/test launch fixtures receive
new versioned paths and canonical digests. Existing nonempty publication rows
retain exactly their existing raw-attribute semantics.

Keep the exact publication row keys:

`name, kind, creator, publisher, attributes, phase`.

`attributes` remains an exact list. A nonempty list contains identifier strings
and means `creator_attribute`: follow raw `vars()` entries without invoking
dependency `__getattr__`. An empty list means `creator_self` ONLY if all of these
CPU schema conditions hold:

- `kind == module_alias`; never `created_module` or namespace.
- `creator == publisher`; both identify the same admitted file-module row.
- That row is a non-package and references an `extension` file under a site root.
- Publication phase equals that file-module row's phase.
- Alias name differs from the qualified creator and collides with no file,
  namespace, absent or disabled rule. Existing unique-name/order checks apply.

There is no alias-to-alias selector or self-certified creator, and no arbitrary
empty-path waiver for Python creators. These rules are generic schema relations,
not permission to discover or add an alias. Names, files, hashes and the phase
are still literal, separately reviewed entries of the trusted frozen environment.
The IPC caller cannot supply or extend this table or mint a live publication.

The proposed Go2 deployment candidate may use precisely these four new
creator-self relations, all in `materialization`, with `attributes=[]`:

| Alias name | Creator and publisher |
| --- | --- |
| `_csparsetools` | `scipy.sparse._csparsetools` |
| `_cyutility` | `scipy._cyutility` |
| `_moduleTNC` | `scipy.optimize._moduleTNC` |
| `_ni_label` | `scipy.ndimage._ni_label` |

Their complete file/native dependency closure and frozen deployment table remain
implementation review obligations, not approved by this four-row mapping. A
further discovered relation requires a versioned table review before acceptance;
no observed `sys.modules` enumeration may add permission. Schema mechanics may
be tested with isolated synthetic fixtures, which are not deployment evidence.

## 3. Exact execution and ownership checks

The captured admission registry, already held by the guardian before import,
owns the actual standard delegate, spec and returned creator module. Keep the
existing file bytes/path/loader checks. The following checks supplement them:

1. Before calling a creator's native `create_module`, every declared self-alias
   key for that creator must be absent from `sys.modules`, including a `None`
   entry. Preserve the checked boundary as a registry fact, not caller evidence.
2. Retain the actual returned module in that creator's file record before any
   fallible explicit adoption or further native call, as already required.
   Immediately after create return, and again before delegated exec, require
   the self-alias keys still absent. This supplement does not permit create-time
   self-publication. A failure after return still retains that exact root.
3. At exec entry, verify the exact spec, standard delegate binding, module and
   qualified `sys.modules` key. Record exec-started before entering the delegate.
4. Immediately after that delegate returns and before recording creator
   completion or allowing the next repository leaf, reconcile its table rows.
   For a self-alias require the creator file record is the same executing record,
   the publisher is that exact record, and both qualified and short keys refer
   to its held returned module by `is`. A matching name, hash, `ModuleType`, copied
   object, parent attribute or caller-written evidence does not satisfy this.
5. Preserve the creator's qualified `__name__`, package, file, spec and loader
   identity checks. The alias has no invented short-name spec or source file.
   Store the exact creator/publisher/root identity and checked relation, then
   emit one alias-publication event. This is not another allocation, create or
   exec, and does not repeat the native delegate call.

Nested imports are checked at their own actual creator return, not deferred to
an outer package. Alias publication is attributed to this observed native
delegate boundary; no claim of instruction-level tracing inside the extension.
The pinned dependency is trusted within that boundary as specified by the parent
design, not arbitrary Python/native callbacks in an open process.

Audit must be read-only with respect to admission: it may revalidate previously
registered aliases but cannot create a missing publication record after the
publisher returned. For an unrequested publisher, its declared alias must remain
absent; configured-but-unrequested rows are not successful observations. For a
requested completed publisher, all its aliases must already be recorded and
unchanged. Repeated audits emit no duplicate create/exec/publication events.
Apply this no-late-admission rule to the existing attribute publications too;
normal first admission remains at the exact publisher completion, not audit.

Missing/early/extra/unlisted/replaced aliases, changed qualified storage,
wrong phase/creator/publisher, invalid create/exec order or failed reconciliation
poison before successful segment completion. Available-object delta scans remain
failure custody only. In particular, an unexpected alias cannot be converted to
an admitted alias by recomputing a hash or observing it at a later audit.

## 4. Failure custody and evidence

Any exception after native create/exec starts follows the existing guarded
failure disposition: hold all available owners and original errors, keep the
original deadline, prohibit COMPLETE/new ARM/retry/normal cleanup, quarantine
and exit. This does not make native allocation before return inspectable or
undo native cleanup; the parent design's dependency/process obligation remains.

The gap scan F2 fix must separate one-time terminal notification from repeated
failure intake. Later owners and errors from an already-executing enclosing body
must append to the already-held resource graph even after the first failure.
Never use normal guardian adoption to reopen a poisoned lifecycle. Repeated
intake must not sync, ARM, release, restore eligibility or notify termination a
second time. Retain native-failure context conservatively; a previous Python
failure cannot make a later incomplete native call a safe-release path.

The whole C2d3 runtime-custody record remains mandatory. Its publication record
must derive `creator_self` versus `creator_attribute` from the actual checked
row, reference the existing creator file-record identity/sequence, phase and
publication boundary, and retain alias/root equality as observed by the issuer.
File create/exec facts belong only to the qualified root. Alias spec/loader/file
origin are `not_applicable`; do not synthesize a second file hash or invocation.
Evidence references flow from file record to publication to aggregate custody;
no self-hash cycle and no serialization that itself mints live authority.
Extend the versioned evidence validator and its negative replay tests alongside
the implementation; old artifacts cannot assert this newly introduced relation.

## 5. Required proof and implementation boundary

Before whole C2d3 code acceptance, require:

- Schema positives for both selectors; negatives for Python/package creator,
  mismatched publisher/phase, created-module empty path, wrong list type,
  collisions/cycles and unsupported old versions. Parsing must import no backend.
- Captured production importer tests for retained-root equality, exact boundary,
  missing/early/unlisted alias, qualified/alias replacement with equal metadata,
  wrong publisher/phase and preexisting `None`. Check calls, side effects and
  terminal state, not only exception types. Existing attribute tests still pass.
- Removal of a publication record before audit must fail, not recreate evidence;
  repeated audit leaves identity, record counts and invocation counts unchanged.
- Create/exec/reconciliation/adoption/record failure tests retain returned roots,
  available deltas and original exceptions, including late parent publication
  after swallowed inner failure. Remove or replace the mutable global key and
  prove the held object survives. No duplicate terminal operation or continued
  repository GPU work. Synthetic delegates prove local mechanics only.
- Cold real observations of all four roots through the captured production
  importer, preserving real file/spec/delegate identities and once-only native
  execution. Ordinary observation scripts are not admission evidence.
- Complete approved C2d3 matrix: both backends, canonical asymmetric/dynamic and
  both Go2 workloads, actual uploads/readbacks, multi-device scope and failure/
  hang/reap behavior, under the frozen complete environment and source closure.
  Distinguish cold evidence from caches, skip/environment failure and code failure.

Repin the source policy/catalog against exact Git file bytes when implementation
changes. Preserve historical versions and absence/disabled-adapter regressions,
and rerun C0 regressions. Do not patch site packages or add a second importer.
This relation is part of the shared frozen import-admission contract, not an
optical backend bridge. Backend selection removes losing-lane fixtures and
comparison-only deployment profiles according to the existing cleanup ledger;
retain only the selected path's required contracts and audited dependencies.

Design approval allows implementing this supplement only. There is no standalone
CPU-subset approval of unfinished C2d3. C2e/f, C4c3/c4, C4d/B3, product GPU
preparation, a3b, result/baseline, host payload, R2 and shipping remain closed.
No performance result or final optical readiness is produced by these tests.
