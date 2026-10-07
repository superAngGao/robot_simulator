# C2d3: truthful pre-execution exclusion of two optional mesh adapters

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: design-request
Author: codex
Version: v1
Date: 2026-10-07
Status: in_review
Related Files: runtime-adoption design-request v2 and design-review v2;
optional-import gap-scan v1; runtime-cold-warp test-evidence v2;
optical_worker/native_bootstrap.py; runtime_environment.py; optics/mjcf_visual.py
Owner Summary: Preserve sticky failure for every module whose execution starts.
Propose one separately named pre-execution outcome for two present optional
trimesh adapters whose missing prerequisites are already part of the frozen
deployment. No dependency install/fork, alternative mesh loader or product gate.

Base:a60c836e52b203c42bf29c58e439b2181cf425d9. This document is a proposed
bounded supplement, NOT an implementation approval. Once independently approved,
it supersedes only runtime-adoption v2's statement that absent_probe is the sole
permitted non-executing import rejection. All started create/exec/body failures
remain sticky. Every other requirement in v2 remains mandatory.

## 1. Facts and selected approach

Independent gap-scan v1 verified two real, present Python files enter execution
and then raise missing-prerequisite exceptions; their trusted parents catch the
exceptions and install disabled-feature placeholders. Adding absent leaf rows
alone cannot reconcile that behavior with v2 section7. The current implementation
correctly follows v2 and must not be weakened by catching generic ImportError.

Proposed outcome: `disabled_optional`. Resolve and verify the present adapter,
prove its named prerequisite absent, then intentionally reject the adapter BEFORE
create_module, exec_module or any body instruction. Record present+disabled,
never absent, completed, loaded, partial-success or dependency-created module.
The trusted parent's existing fallback runs unchanged. This is a deployment
restriction inside the sealed trusted worker, not a hostile-Python sandbox.

Only these literal rows are in scope:

| Present adapter | Exact active requester / fallback owner | Absent top-level prerequisite | Fallback attribute |
| --- | --- | --- | --- |
|trimesh.ray.ray_pyembree|trimesh.ray|embreex|ray_pyembree|
|trimesh.path.polygons|trimesh.path.path|shapely|polygons|

Both rows have phase `materialization`, non-package Python source only. They are
not permission for another optional module or requester. Both Warp and CUDA use
identical rows. A new required row is another reviewed design/table change, not
automatic admission following a failed run.

Pinned trimesh5.0.0 source SHA-256 facts:

| File relative to site root | SHA-256 |
| --- | --- |
|trimesh/ray/ray_pyembree.py|8a447ee15074f43bd04d5bf0f35657c416e5eafba46c13c99188c4e4aa2a4a78|
|trimesh/ray/__init__.py|e653733282a543280fdbcb6721081736ea01f6c8d674ef76652744e3897f108d|
|trimesh/path/polygons.py|76f0e2deb52beda42f102e2a2a35a0b2e245e43fa56b1acbde0f954e0e45973a|
|trimesh/path/path.py|94feaf1e162850c0334e939682d0d00a12056f0dcb15f63cdb522e8db5a847fb|
|trimesh/exceptions.py|1b353dc4771bb93b8c330d7f0267203da481f10ec598251f29adf1d57bf16f70|

The first caller catches BaseException and also sets has_embree=False. The second
caller catches BaseException and assigns ExceptionWrapper. Neither requires a
ModuleNotFoundError specifically. ExceptionWrapper stores `(type(error), args)`,
not the original error identity; its __getattribute__ raises. The implementation
must not use generic getattr/hasattr on this placeholder to inspect it.

## 2. Environment contract and fixed tables

Version the environment schema to `q54.p12.4.runtime-environment.v2`; add the
required top-level `disabled_optional_modules` ordered list (empty is valid).
No silent v1-to-v2 interpretation or missing-field default. Retain historical
v1 fixture evidence; active tests/drivers receive explicitly versioned v2 fixtures
and new trusted configuration digests. Parsing either a row or digest is not an
execution authority. A v1 payload sent to the v2 entry rejects before imports.

Each disabled row has exactly these fields, sorted by `(name, requester, phase)`:

```text
name: exact full adapter name
root: existing site-root ID
path: normalized contained Python file path
requester: exact admitted file-module name
phase: "materialization"
absent_prerequisite: exact top-level name, only "embreex" or "shapely" as above
fallback_attribute: exact one-component attribute from section1
```

Adapter identity/bytes/size come from the single files inventory, not a second
hash table. Requester and ExceptionWrapper owner use their admitted module/file
records. Environment verification binds the literal section1 mapping and hashes
for this deployment profile. A caller cannot add a syntactically valid third row
or swap the two requesters/prerequisites merely by recomputing a manifest hash.

Disabled names are disjoint from modules/publications/namespaces/absent_probes.
Disabled file paths cannot also be executable module paths under another name.
No disabled package, native extension, namespace, repository module, standard
library name or top-level runtime root. Their package parents must already be
admitted packages. Prerequisites are top-level only: no implicit parent import.
Requesters must be admitted materialization Python modules, not aliases.

The environment is frozen before the worker runs; no plugin, callback, request
option, caller-selected exception, generic deny pattern or root wildcard.
No site-package mutation. Pin changed repository bootstrap bytes/AST exceptions
and catalogue in the usual reviewed candidate; don't add broad import exemptions.

## 3. Exact resolver and controlled rejection

Use the existing captured finder/external admission instance and live guardian
controller. Do not install a second repository loader or a later global patch.
For a request for a disabled name, before any delegate create/exec:

1. Require the exact registered live session/controller, materialization phase
   and active requester object from the admission stack. Check that its held
   module/spec/loader/source identity matches its frozen record; no requester
   string supplied by a caller. Reject reload, request after phase transition,
   any previous denial of this row, or an earlier session failure.
2. Require no disabled name or descendant in sys.modules (including a None
   entry), and no existing executable loader/module record for that name. Cold
   admission rejects preloaded disabled names as well as ordinary admitted ones.
3. Derive paths from the held package parent's frozen source record. Resolve via
   the fixed PathFinder only. Require a found non-package SourceFileLoader spec
   at the exact contained path with matching bytes/size. Missing file, changed
   origin/hash, symlink, extension/package/namespace substitution all fail.
   Retain the spec/delegate as CPU resolution evidence; never call create/exec.
4. Check prerequisite and descendants absent from sys.modules; resolve its
   top-level name against the approved site roots without executing any loader.
   Require no spec, including no namespace spec. Presence is deployment drift,
   not permission to execute the adapter. Resolution/audit exceptions fail.
5. Retain the denial record, requester, resolved spec/delegate and one actual
   built-in ImportError in this session's guardian-owned graph. Its exact args
   are `(f"Frozen optional adapter disabled: {name}",)`, name is adapter fullname,
   path is verified adapter origin. It is deliberately NOT ModuleNotFoundError:
   CPython from-list handling must not reinterpret it as ordinary missing child.
6. Return an internal single-use rejection token to the same captured finder.
   A session registry binds its object identity to this request/record/error.
   Exact type alone is insufficient; copied/parsed/foreign/reused tokens fail.
   The finder consumes that exact token and raises the retained ImportError.
   Do not exempt arbitrary exceptions from external.find_spec's failure handler.

The deliberate raise is the only non-poisoning edge. If the parent does not
catch it, or its fallback subsequently fails, the parent's started module
failure poisons normally. No started adapter module exists to forgive. Native
create/exec, begun ordinary source modules, and all unrelated errors retain v2
sticky failure and terminal ownership. No COMPLETE or retry after poison.

At most one denial per row/session; an unexpected second request fails closed.
Repeated audits are read-only observations, not reissued denials. Fresh Go2
cases use fresh sessions, never a warmed module cache shared between workloads.

## 4. Fallback confirmation and lifetime

After the exact requester completes its original body, BEFORE recording it as
successfully completed, confirm its pending denial using only held objects:

- the declared fallback attribute from the requester's raw module dictionary
  has exact type equal to the class object in the held completed
  trimesh.exceptions module's raw dictionary;
- use `object.__getattribute__(placeholder, "__dict__")` to inspect the fixed
  `exception` tuple, require exact ImportError class and the issued args above;
  do not invoke the placeholder or its overridden attribute access;
- for trimesh.ray, its raw has_embree value is the singleton False;
- neither adapter name/descendant was published to sys.modules, and its create,
  exec, body and native-create counts are zero.

Retain and bind the exact fallback placeholder identity and requester to the
denial record. The placeholder's copied args are descriptive fallback evidence,
not proof of execution or a minted capability. The trusted, frozen caller body
and actual issuance trace establish the causal path; no coordinator fills it.

Segment audits require every issued denial confirmed, the same fallback object
and values, unchanged file/resolver/prerequisite outcomes, zero adapter execution
and absence from the module ledger. A still-pending denial, replacement, missing
fallback, changed has_embree or leaked token fails. Audits do not execute a
fallback, query CUDA, load a missing module or create a new authority.

Keep resolution objects, original ImportError/traceback, requesters, placeholders
and source/environment owners until the existing terminal graph disposition.
If validation/adoption/recording fails after partial construction, retain every
available object and fail through the already armed controller. The existing
failure-before-return, UNKNOWN/native-init and watchdog rules are not changed.
No early cleanup or second emergency sync is introduced by this CPU outcome.

## 5. State and evidence

Per-row internal state is:

```text
unrequested -> resolution_verified -> rejection_issued -> fallback_confirmed
                       any error -> session_failed (irreversible)
```

The first arrow is complete only after both present-source and absent-prerequisite
checks. Retain partial evidence before any subsequent operation that can fail.
The second arrow requires exact token consumption; no duplicate token use.
Only fallback_confirmed is acceptable at a successful segment boundary. Parsing
records cannot advance this state, publish a module, mint a binding or release
resources. Repeated audit is idempotent, request/issue/confirmation is not.

Extend the future versioned runtime-custody record with kind disabled_optional,
never a fake completed file_module. Freeze these record facts mechanically from
the live admission registry: sequence, name, phase, requester source identity,
environment/rule digest, present adapter origin/file hash/loader kind,
absent_prerequisite/no-spec observation, operation (disabled_optional or
disabled_fallback_confirmed), request correlation ID, fallback owner/attribute
identity token, and zero create/exec/body/native-create counts. Local identity
tokens are evidence only, not portable Python pointers or executable authority.
Fallback fields are null on issuance, required on confirmation. Include original
error type/args/name/path as descriptive facts. Events referring to a nonexistent
issuance, wrong row, duplicate confirmation, nonzero execution or false completion
reject during CPU-safe replay. No cyclic dependency on the final aggregate hash.

This adds a sixth outcome to v2's five-kind acceptance matrix. It does not weaken
any requirement to observe the five existing outcomes or full runtime custody.

## 6. Proof required in implementation, not inferred from this design

Before whole C2d3 code approval:

- CPU contract tests: exact keys, enum/row/path collisions, reordered/duplicate
  rows, changed fixed mapping, v1 rejected, copied/foreign token and replayed
  confirmation rejected. Schema/hash text never grants permission.
- Actual fixed Python import fixtures through the existing admission session:
  present adapter+absent prerequisite triggers zero create/exec and a verified
  parent fallback. An instrumented adapter body that would fail if entered must
  never execute. Use the same finder, not a test-only denial implementation.
- Negative matrix: wrong active requester, phase, source hash/path or package
  search path; preloaded adapter/prerequisite (including None and descendant
  keys); installed prerequisite; missing adapter; extension/namespace/package
  replacement; copied/reused token; second request; fallback replacement or
  wrong error args; altered has_embree; pending denial at audit. Require sticky
  failure and no further repository GPU leaf/usable prepared input.
- Existing absent-probe behavior stays distinct: genuinely absent requests still
  record no spec; found optional leaves reject. Started create/exec/body errors,
  injected ImportError and ModuleNotFoundError (even same name/text), swallowed
  failures and failed parent fallback remain sticky with retained owners.
- Fresh captured guardian-first workers on BOTH lanes and BOTH real Go2 cases
  must exercise these exact two denials, confirm fallback, perform sole canonical
  materialization and actual a2 upload/readback. Compare every array, layout,
  content, scene/fixture/topology and physical storage with frozen case contracts.
  No surrogate scene, ordinary-import-only proof or success-by-skip.
- Independently compare canonical CPU output with the unchanged ordinary trusted
  dependency import path. This diagnoses unintended fallback effects; it does
  not replace frozen gate digests or real GPU validation. No mesh-loader copy.
- Source-audit/observe the remaining dependency closure. This proposal does not
  assert those two rows solve every possible Go2 or CUDA import requirement.
- Actual guard failure tests cover denial recording/adoption/confirmation failure,
  with graph retention, original deadline, terminal quarantine and real reap.
  Regress C0 A/K and all existing custody/environment/upload/fault tests.

All observation and denial work is untimed setup and identical for both lanes;
no change to benchmark boundaries or the actual optical renderer. In particular,
the disabled trimesh ray adapter is not the Warp or CUDA optical renderer.

## 7. Delivery, review and cleanup

This is a design-only review checkpoint for the newly found policy gap. It does
not accept the unfinished C2d3 implementation, request approval of a CPU subset,
or remove any whole-checkpoint mandatory real-device test. After approval,
implement in runtime_environment/native_bootstrap and their existing tests;
repin source catalogue/policy, freeze v2 environment fixtures, then continue the
whole C2d3 matrix and independent code review. No test driver or private callback
may bypass the production contract.

No changes to shared README/REVIEW_PRINCIPLES, C0 IPC, production worker request
surface, canonical workload/gate digests, asset importer or dependency files.
C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline,
host-payload and shipping remain closed. C4 remains incomplete.

The selected-worker dependency admission contract is durable. P12.4e deletes
losing-lane and temporary author fixtures/drivers only with equivalent selected
worker failure coverage, as previously required. An obsolete optional rule is
removed only with a reviewed dependency/closure change and canonical regression;
it never becomes a permanent generic "ignore import errors" escape hatch.
