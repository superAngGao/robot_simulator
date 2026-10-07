Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: independent absent-resolution design review
Author: codex, independent reviewer
Version: v1
Date: 2026-10-08
Status: accepted
Related Files: absent-resolution design-request v1; runtime-adoption design-request v2; optional-adapter design-request v2; native-self-alias design-request v1; metadata-probe gap-scan v1; optical_worker/native_bootstrap.py
Owner Summary: Approve the bounded absent-resolution DESIGN only. A singleton captured finder with explicit pinned stdlib delegation can preserve normal find_spec absence without a trailing resolver. No C2d3 implementation, deployment table, CPU subset, Go2, GPU, or product approval.

## Verdict and identity

Verdict: **Approve**.

Reviewer session: `/root/c4_c2d3_absent_resolution_design_review_v1`. Fresh independent session; this reviewer did not author the candidate. Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` completely. Author evidence and prior verdicts were treated as claims, not approval proof.

Exact candidate HEAD: `91b11d524f6fc4aad8a2b779327e5c4277cee73c`.

Comparison base: `306842c994228c80cae31dc79ce01f0748d4ee1b`.

Reviewed design: `collab/q54-optical-lab-p12-4c4c2d-absent-resolution__design-request__codex__v1.md`.

Design SHA-256: `91c0b70ade9f0f3ec5c5b8f3ad222facf300436d2c8d09368f61947eec022cfb`.

Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.

Remote candidate: `/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`, primary SSH `tileops-dev` (183.166.183.33:40031).

Both candidate trees were clean at opening and closing checks and had the exact HEAD above. Both design hashes matched. The base-to-HEAD diff contains only the 192-line design and the 149-line archived metadata-probe gap scan. No implementation or shared workflow changes are included. The candidate was not edited.

## Findings

No substantive blocking or non-blocking design findings.

The current implementation still has the reported metadata-probe failure. That is the explicitly identified pre-implementation state, not evidence that this proposed repair is implemented or accepted as code.

## Independent assessment

1. **The motivating failure is real.** `_CapturedSourceFinder.find_spec` currently converts `_AbsentExternalSpec` to `ModuleNotFoundError`. `_ExternalAdmission.find_spec` first establishes exact permitted absence and records it. Installed trimesh `has_module` uses ordinary `importlib.util.find_spec(name) is not None` without catching an exception. The three creation checks and the dae check therefore require a no-spec result, not an exception from the finder. Fresh miniature requester reproductions through the unchanged production importer failed for all four names, with sticky parent failure and no target loader/record. Retaining poison after the started parent actually fails is correct.

2. **Singleton resolution is a coherent repair.** CPython naturally gives metadata queries `None` and real imports `ModuleNotFoundError` when its sole finder returns no spec. The design does not add caller-controlled query modes, frame heuristics, or generic exception exemptions. It removes the tail that would otherwise receive a returned `None`. A resolver/adoption/recording error remains a failure, not a permitted absence. Normal from-list semantics remain Python's own semantics rather than a promise to synthesize one exception class for every syntactic import form.

3. **The topology is available on the actual deployment.** Fresh isolated Python started with precisely BuiltinImporter, FrozenImporter, and PathFinder, in that order. Its paths were the interpreter's zip, standard-library, and lib-dynload roots; hooks were the standard zipimporter and FileFinder hook. Holding their actual identities, validating startup before replacement, and checking the complete singleton afterward are feasible. The current first-entry-only check at `native_bootstrap.py:1746` and insertion at line 1862 are insufficient for the new design and must change in implementation.

4. **The stdlib boundary is explicit rather than accidental fallback.** The union of frozen stdlib and builtin root names is the permitted internal delegation domain. A frozen external absence cannot reach those delegates or another installed finder. External PathFinder use remains bounded by declared site roots/exact held parent locations and existing origin/loader/source checks. Package-parent integrity, hooks, paths, delegate bindings, and singleton identity have stated negative-test obligations. Recognized stdlib lookup is not permission to forward arbitrary non-stdlib names or metadata/plugin discovery.

5. **Trust assumptions are appropriately stated.** This is a sealed trusted worker, not a hostile same-process Python sandbox. Initial interpreter hooks and existing path-cache state are trusted deployment inputs; ordinary cache population through those hooks is allowed. Cache-internal tampering and hostile mutation of trusted stdlib internals are not proven safe by identity checks on the public delegate alone, and the design does not claim they are. This approval does not broaden that boundary. Mutable external request input cannot supply hooks, cached finders, roots, or a replacement chain.

6. **Failure and terminal ownership remain closed.** Separating topology integrity from session liveness permits needed unloaded stdlib imports after poison without making repository/external access live again. No import success may clear failed state, arm another segment, authorize synchronization, or release quarantined owners. If topology fails, the prescribed result is retained state and supervisor-enforced exit, not restoration of a broad import chain. Existing optional-denial tokens and creator-self publication relations remain distinct and unchanged.

7. **The scope remains bounded.** The four exact requester/name/phase rows still need their separately frozen source/deployment authority. A helper call from `trimesh.util` does not grant helper-wide requester permission. The design preserves the active importing owner and no-execution facts. Required full production worker, guard, custody, canonical materialization, actual upload/readback, fault/hang/reap, and independent code evidence are not replaced by these unit/protocol checks.

## Verification performed

All execution was CPU-only on the primary host using `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`, Python 3.12.13 packaged by Anaconda. Installed site root: `/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages`.

Read-only repository checks:

```text
git status --short
git rev-parse HEAD
git diff --stat 306842c994228c80cae31dc79ce01f0748d4ee1b..HEAD
Get-FileHash -Algorithm SHA256 <design>       # local
sha256sum <design>                          # remote
```

Inspected the actual finder, repository/external loaders, absence dispatch, exact parent-path construction, external configuration/audit, sticky failure intake, source-session checks/audit, cold installer, and direct bootstrap terminal handler. Also inspected relevant accepted predecessor contracts and existing optional/started-failure tests. The archived gap scan was independently corroborated rather than copied as fresh execution evidence.

Fresh focused test command:

```sh
cd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip
PYTHONDONTWRITEBYTECODE=1 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_runtime_import_admission.py -k 'optional_absence or optional_probe or swallowed_started'
```

Observed: **5 passed, 77 deselected in 0.36s**. These are baseline mechanics tests, not tests of an implemented singleton change.

Fresh installed-source checks confirmed:

| Source | SHA-256 |
| --- | --- |
| trimesh/util.py | 251f2cb47c5e3547d34141aa5839b02638d4b4921f857108a5ed685af6213f76 |
| trimesh/creation.py | e43a84d731e4f64a8dfb5e49beab4e691d3a5aff707ebbdaf4b053605ccc32ea |
| trimesh/exchange/dae.py | 8d80c3a3835c8c7aca94c149624f1b4d0fcbec2800cb4ef857cc885ce87ca453 |

`util.py:107-124` defines the uncaught metadata query; `creation.py:35-40` checks mapbox_earcut, manifold3d, triangle; `exchange/dae.py:447` checks collada and selects the original missing-format wrappers. Source inspection also checked metadata-version consumers in installed trimesh, SciPy, Warp, NumPy, and Torch: the inspected version lookups catch missing metadata or are deferred diagnostic routines. No need to introduce a distribution/plugin resolver was established. This was a bounded source check, not certification of the whole transitive dependency closure.

Fresh production-importer reproduction, piped over SSH to `python -B -`:

- Loaded the existing `tests/unit/optics/test_runtime_import_admission.py` fixture through `runpy`.
- Extracted the actual installed `has_module` function using its AST source coordinates; did not rewrite the helper or production finder.
- For each target independently, confirmed `PathFinder.find_spec(target, [actual_site_root]) is None`.
- Created a temporary miniature trimesh package with the extracted helper and an active `trimesh.creation` or `trimesh.exchange.dae` requester whose body calls that helper.
- Configured the exact absent row for that requester and materialization phase, then invoked ordinary `importlib.import_module(requester)`.
- Asserted exact-name `ModuleNotFoundError`, failed admission and source session, exactly one target absent_probe event, and no target loader or execution record.

Observed all four cases:

```text
mapbox_earcut / trimesh.creation: ModuleNotFoundError, sticky, zero target loader/record
manifold3d / trimesh.creation: ModuleNotFoundError, sticky, zero target loader/record
triangle / trimesh.creation: ModuleNotFoundError, sticky, zero target loader/record
collada / trimesh.exchange.dae: ModuleNotFoundError, sticky, zero target loader/record
```

The fixture uses unit source-session/controller prerequisites and real production finder/admission/loader methods. It does not execute complete installed trimesh or a cold guardian-backed Go2 worker. Temporary fixture directories were removed by `TemporaryDirectory`; only reviewer-created disposable files were removed.

Fresh protocol feasibility experiment, piped to `python -I -S -B -`, installed one scratch finder in that disposable interpreter. It held the initial three stdlib delegates, path tuple, and hook tuple; asserted singleton/path/hook identity on lookup; delegated only stdlib/builtin roots; returned `None` for one synthetic absent non-stdlib leaf while live; and rejected external lookup after a synthetic failed flag. Observations:

```text
util.find_spec(absent_review_fixture): None
ordinary import absent_review_fixture: exact-name ModuleNotFoundError
stdlib imports succeeded: dataclasses, ast, typing, email.parser, importlib.metadata, ctypes
previously unloaded traceback imported after synthetic poison; failed flag remained True
external metadata lookup after synthetic poison: rejected
```

This confirms ordinary CPython protocol compatibility and representative delegation feasibility only. The scratch finder is not production code and is not evidence that production topology enforcement, custody, mutation rejection, or post-poison behavior is already implemented. It was run in a fresh process, not reused output or a previously populated external module cache.

## Required implementation evidence and limitations

The design's section 4 matrix remains mandatory. In particular, production code must verify full singleton identity before/after relevant resolution, reject sentinel drift before invoking a forbidden callback, retain exact path/hook/delegate invariants, use the validated absent outcome, and prove sticky failure even when callers catch errors. Cold install/core verification must be rerun after exact source-policy/catalog repinning. Classmethod identity pinning must compare stable callable bindings rather than ephemeral bound-method wrapper identity; that is an implementation detail, not a new design policy.

Actual stdlib package lookup and external parent-path checks must be distinguished in tests. Ordinary Python `sys.modules` fast paths are not intercepted by a meta-path finder; the existing cold/preload/registry/audit obligations remain necessary. This approval is not a new claim of instruction-level interception of hostile same-process mutations.

No implementation repair, package installation, site-package mutation, GPU operation, complete Go2 materialization, cold captured guardian execution, supervised fault/reap experiment, multi-device check, benchmark, or broad code-approval matrix was run. Those omissions are intentional for this design-only review. No required real-device property is claimed from a mock or from the scratch protocol experiment.

## Exact approval boundary

Approved: implementation of the bounded v1 absent-resolution design at the exact digest above, supplementing only the parent's unconditional finder-raised absence behavior while retaining exact requester/source/phase admission, zero optional target execution, sticky custody, and terminal stdlib availability under intact topology.

Not approved: the current C2d3 code, a standalone CPU subset, any new Go2 deployment table, full C2d3, or any later stage. Both frozen cases and complete parent lane/device/failure evidence remain required for their own checkpoint.

C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, results/baselines, R2 host payload, and shipping remain closed. C4 is not complete. Temporary validation artifacts retain the design's P12.4e cleanup condition; no second long-term import authority is approved.

This report is saved outside the frozen candidate at `C:/Users/AG/新建文件夹/c4_assets/c2d3-absent-resolution-design-review-v1-verbatim.md` for verbatim archival. Shared workflow and private agent instruction files were unchanged.
