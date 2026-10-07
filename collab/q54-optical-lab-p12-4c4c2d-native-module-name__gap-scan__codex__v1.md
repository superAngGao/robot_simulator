# C2d3 native module-name contract: independent bounded gap scan v1

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: gap-scan
Author: codex, independent reviewer
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: in_review
Related Files: runtime-adoption design v2; native-self-alias design/review v1; absent-resolution design v1; generated-stdlib design/review v1; Go2 deployment-table review v1; optical_worker/native_bootstrap.py; optical_worker/runtime_environment.py
Owner Summary: Request changes for one substantiated contract gap. The pinned SciPy extension legitimately has a different intrinsic module name from its import/spec name. A narrow, versioned, independently reviewed metadata supplement is appropriate; deleting name validation or treating the name as an alias is not. This is not whole-C2d3 review or approval.

## Verdict, independence and frozen identity

Verdict: **Request changes**, limited to the missing native module-name contract
described as F1 below. This verdict does not reject the already accepted designs
within their approved scopes and does not approve a repair that has not been
designed, implemented or reviewed.

Reviewer session: `/root/c4_c2d3_native_module_name_gap_review_v1`.
The reviewer did not author the candidate. The supplied packet was a fresh,
bounded gap-scan assignment, not the implementation conversation. Read
`AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` in full; the
Chinese principles were reread with explicit UTF-8 after default PowerShell
decoding was garbled.

Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.

Primary candidate: `tileops-dev:/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`.

Exact inspected baseline and HEAD on both machines:
`f97186cd9c3746ccf9ea0c98e9753076d2072011`.

Exact Git tree: `a84bc770d44e158b0a133425d6530afe9b2b8042`.

Both working trees were clean at opening and closing checks. No candidate
changes, dependency changes, commits or installs were made. This was a snapshot
gap scan, not approval of a base-to-HEAD implementation delta. The HEAD commit
adds failure diagnostics in the fixed runtime driver and associated test;
those diagnostics are observations, not admission authority.

The coordinator mentioned a new outside-candidate design draft late in this
review. I did not inspect or approve it. It requires its own fresh design review.

## F1 — P2: the frozen file-module contract cannot represent the real native name

**Locations:** `optical_worker/native_bootstrap.py:1143-1182`, especially
`module.__name__ == name` at line 1177; file-module schema at
`optical_worker/runtime_environment.py:177-191`; the existing Go2 row for
`scipy._lib._uarray._uarray` in
`tests/data/runtime_environments/183_warp_go2_author_v1.json`.

**Trigger:** the unchanged canonical Go2 materializer reaches SciPy's vendored
uarray extension through `scipy.fft` and `scipy._lib.uarray`. The standard native
loader returns a module whose intrinsic `__name__` is `uarray._uarray`, while
the requested import key, spec name, loader name, package and origin are the
correct fully qualified SciPy identities.

**Consequence:** after actual delegated exec returns, `check_module` rejects
this legitimate pinned extension as `external module metadata drift`. The
current captured Go2 path cannot complete materialization, let alone its
required upload/readback. Sticky failure and quarantine are the correct
response under the current contract; they must not be disabled to hide the gap.

**Why existing checks/tests do not close it:** the schema has only the file
row's `name`, root, path, package flag and phase; it cannot state a distinct
expected intrinsic name. The literal-table review verified spec/path/loader
resolution and inventory, expressly not captured execution. The synthetic
native-self-alias fixture creates `ModuleType(spec.name)` and explicitly checks
`module.__name__ == module.__spec__.name` (test lines 770 and 811); that is a
valid alias fixture but cannot demonstrate this different native behavior.
Passing that fixture or the inventory verifier cannot disprove F1.

**Repair direction:** distinguish the immutable import identity from one
explicitly declared expected native `__name__`, preserving exact equality to
each expected value. Default behavior for ordinary file modules must remain
`__name__ == import_name`. Only a reviewed literal exception for a pinned,
non-package native extension is justified by the present evidence. This is
metadata for an already admitted file module, not a new import, namespace,
absent outcome or creator publication.

**Design decision required:** yes. The accepted schema does not represent this
relation; native-self-alias design section 3.5 explicitly retains qualified
creator-name checking. A scoped supplement must state its precedence over the
blanket file-module-name assumption, while preserving the four existing
self-alias mappings and all other parent invariants. Adding a new metadata
field/table and acceptance semantics is a small policy/schema change, not an
unreviewed one-line implementation repair. Freeze versioning, exact keys,
defaults, applicability, check boundaries, evidence and negative tests before
implementation. Historical descriptors/evidence must not be silently upgraded.

There is no second substantive finding from this bounded scan. In particular,
the fact that the frozen code fails closed is not a separate custody defect.

## Independently verified dependency and failure facts

### Pinned identity and source

The actual installed distribution reports SciPy `1.18.0`. Its installed
`scipy/version.py` records git revision
`54ef5423f2e4376230ec3bfda6912a07a50958e3`.

The exact admitted extension is:

```text
import key: scipy._lib._uarray._uarray
root: site
relative path: scipy/_lib/_uarray/_uarray.cpython-312-x86_64-linux-gnu.so
absolute path: /home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages/scipy/_lib/_uarray/_uarray.cpython-312-x86_64-linux-gnu.so
kind: extension
is_package: false
phase: materialization
size: 115080
SHA-256: ca83664f94c09db3b054260ec762112b16fc6543d259b3477f26fa6f6639c7c0
```

Read the actual binary and independently matched its size/hash to the frozen
inventory. Its bytes contain the null-terminated `uarray._uarray` literal once.
This string alone would not prove execution or ownership; the source and actual
CPU observation below provide separate corroboration.

Installed `scipy/_lib/uarray.py:13` first probes the separate optional `uarray`
package, then lines 28-29 use the vendored `._uarray` fallback. The installed
vendored initializer imports `._backend` at line 115 and declares version
`0.8.8.dev0+aa94c5a4.scipy`; `_backend.py:5,11` imports its `._uarray` extension.
These unchanged source branches explain the qualified import key. Source hashes:

| Installed source | SHA-256 |
| --- | --- |
| scipy/_lib/uarray.py | 62d01f5d49e821a30fd01069f0da9532d12475ef5d6b0f19d38e32d1d48a6395 |
| scipy/_lib/_uarray/__init__.py | 470c3bc0b03b147e98a2783ba0c825fec1cfa63711b2545a4e387ad56ff1560e |
| scipy/_lib/_uarray/_backend.py | 2d99d22c9d942b6d3d8eb32da1c38ca1ce60ae5368a1bded6dbd476d6d179404 |
| scipy/version.py | 6ab6e3ea672f5a4ed8afb71c711f75a35e82a11cac2b5dcc03267e054a71470b |

The wheel does not install the C++ source beside this extension. I independently
read SciPy's primary upstream source at the exact revision declared by the
installed version file. Its module definition sets `m_name` to `uarray._uarray`
and its `PyInit__uarray` calls `PyModule_Create` with that definition (source
lines 1675-1691, one-based). This explains the intrinsic name rather than a
captured-loader rewrite. [SciPy source at the installed declared revision](https://raw.githubusercontent.com/scipy/scipy/54ef5423f2e4376230ec3bfda6912a07a50958e3/scipy/_lib/_uarray/_uarray_dispatch.cxx).

The upstream source is corroboration, not a reproducible-build attestation of
this binary. The actual binary hash and live observation are the local facts.
Neither a remote source string nor the metadata name is authority to add an
import permission.

### Fresh ordinary CPU reproduction, not captured-admission proof

In a disposable robot_sim interpreter with `-I -S -B`, added the exact installed
site directory only to this diagnostic's `sys.path`, then called:

```python
name = 'scipy._lib._uarray._uarray'
m = importlib.import_module(name)
s, loader = m.__spec__, m.__loader__
```

Observed:

| Property | Value/result |
| --- | --- |
| m.__name__ | uarray._uarray |
| s.name and loader.name | scipy._lib._uarray._uarray |
| m.__package__ and s.parent | scipy._lib._uarray |
| m.__file__, s.origin and loader.path | exact inventoried extension path above |
| exact standard ExtensionFileLoader type | true |
| s.loader is m.__loader__ | true |
| sys.modules[qualified_name] is m | true |
| all sys.modules keys referring to m | only scipy._lib._uarray._uarray |
| uarray._uarray key present | false |
| uarray parent key present | false |
| any torch/warp module imported | false |

No module attributes were renamed, no fake alias was inserted, and no site file
was edited. This establishes normal dependency behavior in a fresh CPU process,
not guardian ownership, captured execution, once-only captured invocation,
target-device scope or GPU success.

### Historical CPU closure cross-check

Independently hash-checked
`/data2/shared/ang.gao/c2d3-go2-960-cpu-dependencies-v4.json`:
`ee48dc09dfb8ddd06f1b171148dd923aa19ee7a22f8186243a36bea7b786ce5a`.

The artifact has 901 `exec_returned` events. Exactly one has `module_name != name`:
this extension. Its create-returned, exec-started and exec-returned events all
already report `uarray._uarray`; the short name is not merely a late Python
parent's mutation. Its recorded same-object keys contain only the qualified
SciPy key. The full-key observation at native create return is not a short-name
publication and does not require inventing a self-alias rule.

This is an independent scan of historical author evidence, not 901 new imports
by this reviewer, nor proof that every future workload/version has no other
name difference. Newly discovered differences must still fail closed and
require versioned review.

### Actual captured failure, independently inspected historical evidence

Directory:
`/data2/shared/ang.gao/c2d3-f97186c-go2-960-guarded-author-v3`.

| File | SHA-256 |
| --- | --- |
| external_failure.json | 85ef3338a51beadca0ca05307dcb24a4e28d25222cdb5c99ed0ca81247cf5b5c |
| failure.json | 55fab24bf3a637d363b37a0c8288f14c44307585d64fa8bae75841e793c9aed6 |
| custody_artifact.json | c28e0ee245812a335b7b95bf7680f7a92a45b3cd311201829e669f61127a6007 |
| request.json | 67a00e3e3ba09d76cc1a4a223485369dcfea949a6f313d5ce59f61949c8ac9fd |
| identity.json | ffb8a08f3c7349f45569da0eeb2146afcdfb04790cadb1dd3d5a133a70b51fba |
| supervisor_stdout.json | 586badbade5867619c23b38c03c11c200fdc5e1910a1cb358a4e2f66592fd68e |
| worker.json | c0d030b7c46e3b65646374ccef5bdb35e7c7d090ee20f227d42c6b861136dcfb |

The request binds HEAD f97186c, Warp lane, target 1, fixed Go2 960x640 case,
catalog digest `e71e429f20bcb5c30ee20d1c7d5ba7ba40f8813f7646a8cae6240b7f60f6fcdb`,
and environment canonical digest
`d2b1edc48c5a28a4ef71237fdf16d92d089f02f3fe7926c858d3e0c651146d3b`.
The environment's raw digest remains the previously reviewed
`747e680530d2a02b8de372f7f0c660840b22b4cd3a0e375656a84e82a0898d60`.
I ran the candidate's CPU `RuntimeEnvironmentContract.from_payload` and
`verify_files` against that independently supplied canonical digest: pass,
without NumPy/SciPy/Torch/Warp imports. This checks the existing inventory,
not the missing metadata contract.

Captured events 2322-2325 record resolved/create-started/created/exec-started for
the qualified extension. Event 2326 is its first failure. Failure metadata[0]
reports the intrinsic short name, correct actual file/package, and
`exact_spec`, `exact_loader`, `spec_loader_is_wrapper` all true. The fixed
diagnostic obtains those facts from the retained module record and actual
spec/loader references. The production code then explains the rejection:
post-exec `check_module` insists on qualified `__name__` equality.

Later failure contexts are enclosing import unwind, not eight independent name
mismatches. The outer SystemExit text says the preview requires trimesh; it must
not be misdiagnosed as a missing trimesh installation or permission to install
an optional engine. The earlier generated-stdlib blocker is no longer the first
failure in this artifact, but that alone is not full validation of that repair.

Worker identity is PID 3594989 / parent 3594988. Supervisor evidence records
EXITED, observed code 74, three normal ARM entries but only two earlier normal
completions, no timeout or kill request. Worker terminal summary records failed
sequence 2, full quarantine, unknown scope, release not authorized, zero
emergency sync attempts and zero terminal ARM attempts. These are historical
supervised failure records, not a new execution by this reviewer and not a
complete audit of all custody properties. A final scoped ps found none of the
packet's parent/supervisor/worker PIDs 3594945/3594988/3594989 alive.

## Required bounded supplement and exact negative proof

The following are obligations for a proposed repair, not an already approved
schema or implementation. A new design must make them unambiguous.

1. **Single frozen authority.** Bind the literal expected native name to the
   already frozen file-module rule, inventory identity and permitted phase.
   The presently justified mapping is only
   `scipy._lib._uarray._uarray -> uarray._uarray` for the file above. A sparse
   exception table or explicit per-row expected-name representation may work,
   but schema/version/default behavior must be selected by the design. Do not
   add a second mutable source of truth or accept runtime-observed values as
   expected values. Reject duplicate/unknown/ill-typed entries, wrong file kind,
   package/Python/namespace/alias targets and ambiguous old-schema upgrades.

2. **Keep import authority qualified.** Resolution, file lookup, registry key,
   spec name/parent, package, wrapper/delegate name and `sys.modules` lookup
   remain the fully qualified SciPy identity. Hold the exact returned object
   before any fallible validation/adoption can discard it. Preserve exact spec,
   loader, held-object, file origin, hashes, parent paths and standard delegate
   constraints; matching intrinsic names or copied metadata are never enough.

3. **Check, do not learn or normalize.** The actual native return must have the
   one declared intrinsic name; check it after retaining create's returned
   object, and recheck at exec entry/return and later audits with the applicable
   importlib metadata initialization order made explicit. Required full spec/
   loader fields need not be fabricated before importlib initializes them.
   No later boundary may redefine the expected name from observed state.
   Both an arbitrary wrong name and the otherwise qualified import name must
   reject for this exceptional row; unchanged ordinary rows still require their
   ordinary exact names. Test missing/non-string/empty/lookalike names and
   mutation between boundaries, including changed-back behavior where the
   supported boundary observes it. Do not claim instruction-level sandboxing.

4. **No alias fiction or extra resolution.** The metadata string grants zero
   permission to import `uarray`, `uarray._uarray`, descendants or another SciPy
   leaf. Do not insert either short key, create a parent, call another loader,
   patch site packages or rewrite module.__name__/spec.name. The existing frozen
   absent probe for standalone uarray remains absent; unrelated requests remain
   rejected. Test an injected short key (including None), fake same-object alias,
   extra publication and absent optional package becoming present. Existing
   publication rules, not the metadata row, govern any independently approved
   aliases. Do not impose pre-create absence on this extension's legitimate
   qualified native key by confusing it with a new short self-alias.

5. **Identity and drift negatives.** With the correct expected intrinsic name,
   substitute a same-name/same-metadata module at the qualified key, replace the
   held module, spec, wrapper or delegate, alter spec.name/package/origin/file/
   parent paths, swap same-basename binaries/roots, change inventory/table/hash,
   use wrong phase, preload or reload. Each must fail through the production
   entrypoint without permission learned during audit. Test another undeclared
   extension with the same suffix or same intrinsic name to defeat a general
   prefix/suffix/basename exemption. Repeated valid audits must be read-only,
   preserve object identity and issue no second native create/exec/publication.

6. **Sticky failure and ownership.** Name-validation failure after native return
   must retain that exact object, original exception and any available enclosing
   owners even if sys.modules is removed/replaced during unwind. Exercise
   create-return/adoption, exec, post-exec check, evidence-record and later-audit
   failures, including swallowed inner failure. Keep the original deadline,
   no COMPLETE/new ARM/retry/normal release/fallback sync, and no usable binding.
   Existing native pre-return trust/process-owner limitations are unchanged.

7. **Truthful evidence and regression scope.** Record requested/spec name and
   actual checked intrinsic name as distinct descriptive facts referencing the
   existing file-module identity, exact rule/environment and execution boundary.
   Do not emit a fictitious alias/file record or use parsing as live authority.
   Version the affected evidence schema/validator and test invented name,
   metadata, sequence and identity claims. Repin bootstrap source policy/catalog
   exactly; preserve original artifacts. Regress ordinary file modules, the four
   creator-self aliases, attribute publications, absent resolution, disabled
   adapters, generated stdlib, singleton topology and C0 behavior.

8. **Real proof remains necessary.** A small production-importer test must first
   prove this real extension is retained/admitted with the two names, unchanged
   spec/file/loader and no short alias, with once-only actual native execution.
   A synthetic fixture tests mechanics only; the ordinary CPU import above is
   provenance only. Both fixed Go2 workloads must then complete cold captured
   canonical materialization and the required guarded upload/readback, alongside
   the complete both-lane C2d3 custody/fault/hang/multi-device/exit-reap matrix and
   independent whole-stage code review. Reaching the next failure is progress,
   not a pass or permission to learn another exception.

Removing `module.__name__` validation globally, accepting all native mismatches,
using `endswith`, stripping a vendoring prefix, accepting either name loosely,
renaming the module, or converting ordinary observation into an automatic
allowlist would all be dangerous weakening, not this bounded repair.

## Commands, environment, limitations and side effects

Read-only local inspection used UTF-8 Get-Content, rg, git status/rev-parse/show.
An initial unquoted PowerShell `HEAD^{tree}` argument was parsed incorrectly;
it was rerun quoted and the exact tree above independently matched remote Git.
Some initial broad output was truncated; the relevant source and artifact
summaries were subsequently read in bounded sections. No truncated output was
treated as full runtime validation.

Remote diagnostics were supplied over stdin, creating no script or artifact in
the candidate:

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 tileops-dev \
  'timeout 20 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -'
```

Artifact/inventory diagnostics used a 30-second timeout instead of 20. Actual
interpreter: CPython 3.12.13, Anaconda build. Fresh recorded diagnostic PIDs:
3602743 (read-only inventory/source/history), 3603588 (ordinary SciPy CPU import),
3605333 (artifact summaries and candidate CPU inventory verification). All
completed normally; final scoped ps returned no rows for those PIDs. The first
artifact-reader diagnostic also completed normally but did not emit its PID.
No background reviewer process remains.

Only the ordinary-import diagnostic explicitly appended site-packages; only the
inventory verifier explicitly inserted the candidate root to import the CPU
contract. Neither is presented as an isolated captured worker. -B prevented new
bytecode writes; ordinary imports may read installed caches. No Torch/Warp
import, GPU API, GPU query, device initialization, package installation, process
signal, environment-file modification, candidate write or other-user process
interaction occurred. The coordinator's concurrent GPU/C0 regression totals
were not rerun or counted as independent evidence in this report.

I did not run canonical materialization anew, any captured-import positive,
pytest suite, GPU upload/readback, fault/hang matrix or a proposed repair. This
scope does not establish whole closure, successful real-device behavior or full
native exception safety. The sole write is this requested report via apply_patch
outside the frozen candidate; its final SHA-256 is reported separately to avoid
a self-referential digest.

## Scope and gates

Established: F1 is a legitimate, reproducible dependency metadata-contract gap;
a narrow explicit contract repair is defensible; a reviewed design supplement
is required before implementation. No proposed supplement or implementation is
approved by this report, and no CPU subset substitutes for whole C2d3 approval.

Whole C2d3, full C2d/C4, C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b,
result/baseline, R2/host payload and shipping remain closed. No renderer, sample
authority, final readiness, performance result, product activation or release
is created. Existing cleanup/losing-lane obligations remain unchanged.
