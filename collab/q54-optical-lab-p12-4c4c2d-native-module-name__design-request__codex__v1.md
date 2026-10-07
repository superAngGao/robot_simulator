# C2d3: one exact native module-name metadata contract

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: design-request
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design v2; native-self-alias design v1; generated-stdlib design/review v1; native module-name gap-scan v1; runtime_environment.py; native_bootstrap.py
Owner Summary: Distinguish one frozen native module's runtime __name__ from its Python import key without relaxing spec, file, loader or object authority. This is metadata validation, not a new import, alias, creator-publication or arbitrary caller exception.

## 1. Code fact and scope

Implementation base f97186cd9c3746ccf9ea0c98e9753076d2072011. Real supervised
Go2 attempt c2d3-f97186c-go2-960-guarded-author-v3 fails at external event2326:
`scipy._lib._uarray._uarray`. Its actual retained module has:

```text
import key / spec.name: scipy._lib._uarray._uarray
module.__name__:        uarray._uarray
module.__package__:     scipy._lib._uarray
exact spec:             true
exact wrapper loader:   true
spec.loader is wrapper: true
```

The retained file is the correct frozen SciPy extension. Current check_module
requires module.__name__ == import key for every file module, hence rejects this
valid pinned extension metadata. The existing sticky failure/quarantine is the
correct response under the current rule. Do not catch or forgive that error.

An ordinary CPU diagnostic independently observes the same name difference.
The previously recorded complete Go2 CPU import trace has this sole mismatch
among exec_returned file-module observations. Neither ordinary observation is
captured-worker acceptance; the actual guarded failure supplies that counterexample.
No missing package or extra sys.modules alias is implied by the runtime name.

This design changes only that exact __name__ expectation. It does not alter
which module may load, where it resolves, whose allocation/module is retained,
how native code executes, or when a started failure becomes terminal.
For this single declared file module, this supplement takes precedence over
earlier generic wording requiring runtime __name__ to equal the import key.
All other metadata and creator-self publication requirements remain unchanged;
none of the four approved creator-self alias roots is this module.

Independent gap-scan v1 from /root/c4_c2d3_native_module_name_gap_review_v1
returned Request changes for F1 (P2), not approval of this proposed supplement.
Its unchanged report SHA-256 is
3ecd6b80366b03c7e2ed784b82a4379d7e3cc9e6dc7e94b86fafcd23a2acf4ec.
The actual pinned extension is 115080 bytes, SHA-256
ca83664f94c09db3b054260ec762112b16fc6543d259b3477f26fa6f6639c7c0;
these are evidence references, not a second runtime inventory definition.

## 2. Single production metadata policy, no new table authority

Define one literal immutable metadata relation in the existing CPU-safe
runtime_environment contract module:

```text
scipy._lib._uarray._uarray -> uarray._uarray
```

There are no other rows, prefixes, suffix matching or inferred short-name rules.
The fixed relation is reviewed production policy, analogous to the existing
literal disabled-adapter policy, not a mutable launch/request preference.
Its source bytes are pinned by the next policy/catalog version. Do not duplicate
this mapping in bootstrap, tests, a benchmark schema or the deployment table.

The contract exposes a CPU-only expected-file-module-name query. It accepts only
a file module already declared in the validated environment. Normally it returns
the declared full import key. For the one relation above, it requires the row's
referenced inventory file to be kind=extension and the row to be nonpackage;
otherwise the contract fails closed. The mapping cannot authorize a Python file,
namespace, publication or undeclared module. Environment validation rejects an
incompatible declared row before dependency execution, not at the final audit.

The environment continues to bind exact interpreter, distribution version,
module import key, parent/package path, relative extension path, physical digest
and size. No second copy of file hashes or binary-version knowledge is stored in
the metadata policy. Existing origin/loader/parent checks must still mechanically
match that one frozen file row. A future different binary with a different native
name needs an explicit versioned policy review; do not accept the full import key
as an alternative just because it was the former default.

Runtime environment schema v3 and existing literal tables remain unchanged:
this is no new caller-settable field, optional default, wildcard or self-reported
module-name override. The source-pinned contract owns the expectation. The
runtime parser/hash still cannot mint an import capability; current fixed launch
configuration and production source-session authority remain necessary.

## 3. Preserve exact resource and execution binding

ExternalAdmission.check_module obtains the expected __name__ exclusively from
that exact bound environment's formal policy. Every other comparison remains:

- sys.modules[declared import key] is the held returned module;
- exact spec object, exact wrapper and original delegated loader binding;
- spec.name equals the declared full import key;
- spec.parent and module.__package__ equal the declared package parent;
- exact module.__file__, spec.origin and loader.origin match the frozen file;
- package/nonpackage shape, original create/exec sequence and stage authority;
- before/after native-call retention and all existing failure custody.

Do not rewrite module.__name__, spec.name, package, sys.modules or the extension
binary. Do not remove a comparison, compare names by last component, accept two
names, infer an exception from observed runtime metadata or normalize metadata
after an error. The same exact check applies at completion and later audits.

This is separate from creator-self aliases: those explicitly publish a second
sys.modules key pointing to an actual creator-owned object. Here no such key is
permitted. `uarray` remains the existing absent optional probe, and a new
`uarray._uarray` sys.modules entry is unlisted publication and must poison even
if its object is the retained extension. The metadata string is never resolver
input or proof of identity.

Check the native module's raw __name__ immediately AFTER retaining the exact
create-returned object and BEFORE any further native leaf or successful adoption
completion. Missing, non-string, empty or unexpected values fail. Do not require
spec/package/loader metadata before importlib has initialized it, and do not
fabricate it. At exec entry, require the same held object and the fully initialized
spec/loader/package/file relationships above, then recheck at exec return and
later audits. A validation failure uses existing sticky custody: retain actual
returned owners and the original error, no successful COMPLETE/new ARM/retry/
normal release or fallback sync. Even if an enclosing dependency swallows it,
the source session cannot recover. Qualified-key publication during native create
is not a short alias and retains its existing native-loader contract.

Descriptive runtime evidence retains the full import key as record identity and
records actual runtime __name__ separately when that schema is finalized. It
must not synthesize an import/creator-publication event for the metadata string.
Expected values come from the single formal policy, observations from the held
module, and equality never substitutes for spec/loader/object authority.
The still-required versioned C2d3 runtime-custody schema must explicitly carry
these separate fields and validate rule identity, kind and boundary ordering;
old diagnostic event dumps cannot assert them through an implicit upgrade.

## 4. Verification requirements

Before whole C2d3 approval:

- CPU contract tests: normal file-name expectation is the import key; the one
  declared native nonpackage returns the literal runtime name; undeclared,
  Python-file and package variants reject before an executable loader is called.
- Production importer tests use the exact create-returned object and actual
  completed module record. Correct runtime name passes while import key,
  arbitrary short name, empty name and another extension's name fail. Default
  modules retain exact full-name comparison; no global relaxed rule.
- Independently replace spec, loader, module object, file/origin or package;
  accepted __name__ does not mask any such mismatch. Repeat audits detect late
  __name__ mutation and identity replacement. Failed validation is sticky.
- Inject missing/non-string names and mutation at create return, exec entry,
  exec return and later audit, including a changed-back value observed at a
  supported boundary. Retain returned owners after failure, even if qualified
  sys.modules is removed or an enclosing importer swallows the exception.
- Insert `uarray._uarray` as an extra sys.modules alias to the held module:
  fail unlisted publication rather than treating the runtime name as permission.
  Existing uarray absence and disabled-adapter behavior remain unchanged.
  Include a None entry, an undeclared extension with the same suffix/name,
  wrong phase, preload/reload and altered frozen file bytes. No boundary may
  learn a new permission or emit another create/exec on a repeated audit.
- Real pinned extension through the captured driver: actual name difference is
  observed without rewriting, native create/exec each occur exactly once, full
  source/spec/loader/module retention and later audits pass. No ordinary CPU
  import or test-native double replaces that proof.
- Rerun both real Go2 canonical materializations and actual guarded Warp/CUDA
  upload/readback; any subsequent missing dependency/metadata requirement still
  fails closed and is separately reviewed, not automatically added here.
- Regress native self aliases, namespace/creator publication, generated stdlib,
  unexpected/swallowed native failures, watchdog/quarantine/reap and C0 A/K.
  Full parent failure/hang/dual-device matrix and runtime-custody schema remain
  mandatory. This metadata fix cannot approve a CPU subset as whole C2d3.

Test fixtures may fault/observe production entries, but cannot duplicate the
metadata mapping or install a benchmark-only import implementation. No module
object or expected-name string supplied by a coordinator is execution evidence.

## 5. Phase and cleanup boundaries

Design approval permits only this bounded metadata-policy implementation and
its tests. No live arena lease, recorded renderer, final readiness, product
dispatch, benchmark sample or output delivery is added. Whole C2d3 is unapproved.
C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline,
host-payload and shipping remain closed.

Retain the policy with the selected worker only while its pinned required
dependency needs the relation. Remove a no-longer-required relation through
reviewed dependency/cleanup changes; losing-lane deployment fixtures and temporary
drivers follow existing P12.4e obligations. No shared README/private instructions
are modified and no runtime observations become permissions automatically.
