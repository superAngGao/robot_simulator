# C2d3 optional-adapter supplement — independent design review v1

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: design-review
Author: codex, fresh independent reviewer
Version: v1
Date: 2026-10-07
Status: in_review
Related Files: optional-adapter design-request v1; runtime-adoption design-request v2/design-review v2; optional-import gap-scan v1; native_bootstrap.py; runtime_environment.py
Owner Summary: Request one bounded state-contract correction. Independent CPU probes support the proposed exact pre-execution ImportError and pinned trimesh fallback behavior. The design nevertheless contradicts the existing pre-materialization configuration boundary by allowing only fallback_confirmed at every successful segment boundary. No implementation or product gate is approved.

## Identity and verdict

Reviewer session: `/root/c4_c2d3_optional_adapter_design_review_v1`.

Verdict: **Request changes** for F1 below.

This is a design-only verdict. The selected pre-execution exclusion approach appears implementable; the finding concerns when its per-row states are legal, not a need to forgive started-module failures.

Reviewed:

- Base: `a60c836e52b203c42bf29c58e439b2181cf425d9`.
- Frozen and actual HEAD: `65319cb728c3b9db0f6928c7aab6e3d95b6023e2`.
- Design: `collab/q54-optical-lab-p12-4c4c2d-optional-adapter__design-request__codex__v1.md`.
- Independently verified SHA-256: `34ff608ef9c653ef94ab32002506a24b0a2923a073af25f34b02877e84495dfa`.
- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Remote repository: `/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`, detached at the same HEAD.
- Primary host only: `tileops-dev`, `183.166.183.33:40031`.
- Interpreter: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`, observed CPython 3.12.13.

Both working trees were clean initially and finally. The complete candidate delta is one added design document, 272 lines. Local and remote candidate `git diff --check` passed. Protected code, tests and collaboration-rule paths have no candidate delta.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely. Inspected predecessor requirements, prior review and gap evidence as claims/history rather than fresh proof. No subdelegation, candidate mutation, dependency installation or GPU work occurred.

## F1 — P2, blocking design ambiguity: unrequested rows have no legal pre-materialization successful boundary

### Locations

Optional-adapter design v1:

- Lines 184–194 define per-row initial state `unrequested`, then state: “Only fallback_confirmed is acceptable at a successful segment boundary.”
- Lines 169–173 instead require confirmation of every **issued** denial.
- Lines 235–239 require both exact denials during real Go2 materialization.

Existing implementation:

- `optical_worker/native_bootstrap.py:992`, `configure_external_environment`.
- Lines 1002–1015 construct external admission inside a setup segment, run `admission.audit()` at line 1014, and complete that segment.
- `materialize_native_source` begins at line 1020 and requires phase `runtime_configured`.
- The canonical importer reaches `import trimesh` at `optics/mjcf_visual.py:41`, during the later materialization call.

### Trigger and consequence

Use a valid proposed v2 environment containing both disabled rows for a fresh Go2 worker:

```text
configure_external_environment
  construct external admission
  both rows = unrequested
  admission.audit()
  successful configuration segment boundary

materialize_native_source
  import trimesh
  resolve, issue and confirm the two denials
```

At the first boundary neither requester has executed, so neither row can be `fallback_confirmed`.

A literal implementation of line 194 must reject configuration before required Go2 materialization can start. An implementation that permits `unrequested` would instead need to qualify the unconditional rule using its own interpretation of section 4.

This is an existing call-order fact and a contradiction in the proposed requirements, not a claim that the unimplemented supplement failed a runtime test.

### Why existing checks do not close it

The current code does not implement `disabled_optional_modules`, so its passing admission tests cannot settle the proposed state rule.

The new positive tests described at lines 221–224 begin with an actual adapter request; they need not exercise configuration’s earlier successful audit with configured but unrequested rows. The later real Go2 matrix would expose a literal implementation, but the design should not leave this policy decision until implementation.

### Required correction

Explicitly define legal row states at each applicable boundary:

- Configured but unrequested rows must be allowed at the pre-materialization configuration audit without issuing a denial or executing an adapter.
- Every issued denial must be confirmed before its enclosing successful segment boundary; `resolution_verified` and `rejection_issued` cannot silently pass.
- Both specified rows must be confirmed at the required Go2 materialization acceptance boundary.
- State the treatment of unused rows for workloads that do not import trimesh, consistently with the existing empty-table allowance.
- Preserve irreversible failure and the prohibition on manufactured confirmations, early imports or merged/renewed setup deadlines.

### Proof needed after repair

A revised design should make the above trace unambiguous. Before implementation approval, require:

1. A same-session production-path CPU test configuring the two rows, successfully auditing/completing configuration with both unrequested, and observing zero denial/create/exec/body events.
2. Subsequent materialization using that session, issuing each denial once and confirming both fallbacks.
3. Negative tests rejecting pending issued denials and rejecting missing required Go2 confirmations.
4. Repeated audits that neither issue denials nor advance row state.

No broader architectural change is needed to resolve F1.

## Independently verified compatibility and high-risk areas

### Pinned dependency facts

Read the installed files directly under:

```text
/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages
```

All five SHA-256 values in the proposed design matched independently.

Observed source behavior:

- `trimesh/ray/__init__.py:4–12` catches `BaseException`, constructs `ExceptionWrapper`, and sets `has_embree=False`.
- `trimesh/path/path.py:60–63` catches `BaseException` and stores the wrapper in its own `polygons` attribute.
- `trimesh/exceptions.py:20–35` stores `(type(exception), exception.args)` and overrides attribute access to raise a reconstructed exception.
- Consequently, the proposed raw module dictionary and `object.__getattribute__` inspection avoids invoking the wrapper’s raising behavior. Original exception identity must be retained separately, as the design requires.

### Original gap reproduced independently

A fresh stdin observer wrapped the standard loaders only for the two exact adapter names. Under `python -I -S -B`, it observed:

```text
trimesh.ray.ray_pyembree:
  create_started
  exec_started
  exec_failed: ModuleNotFoundError, name=embreex

trimesh.path.polygons:
  create_started
  exec_started
  exec_failed: ModuleNotFoundError, name=shapely

top-level trimesh import completed: 5.0.0
adapter names remaining in sys.modules: none
torch/warp roots loaded: none
```

Thus these are present adapters with started-body failures under ordinary import, not absent leaf modules. Existing sticky handling remains appropriate.

### Proposed exception semantics exercised independently

A separate fresh CPU process installed a narrow observational finder for those two names. It:

1. Resolved each adapter through `PathFinder`.
2. Observed a non-package `SourceFileLoader` spec.
3. Observed no top-level prerequisite spec in the installed site root.
4. Raised the exact proposed built-in `ImportError` before returning a spec or invoking a loader.

Observed:

| Adapter | Actual nearest trimesh requester | Requests | Fallback |
| --- | --- | ---: | --- |
| `trimesh.ray.ray_pyembree` | `trimesh.ray` | 1 | Exact installed `ExceptionWrapper`, exact ImportError class and issued args |
| `trimesh.path.polygons` | `trimesh.path.path` | 1 | Exact installed `ExceptionWrapper`, exact ImportError class and issued args |

Top-level trimesh 5.0.0 import completed. Both adapter names remained absent from `sys.modules`. `has_embree` was `False`. Neither Torch nor Warp was imported.

These results establish pinned-caller compatibility with the proposed exception. The observational finder is not the production admission implementation, a custody proof, or guarded Go2 acceptance evidence.

### CPython from-list behavior checked

Read the installed CPython `importlib/_bootstrap.py:1390–1424`.

`_handle_fromlist` specially catches `ModuleNotFoundError` and may ignore it for the requested child name. It does not apply that handling to a built-in `ImportError`.

The proposed exception distinction therefore has a concrete implementation basis. The runtime probe above independently demonstrated that it reaches the intended fallback once per row.

### Actual production insertion points and failure closure

Inspected:

- `_CapturedSourceFinder.find_spec`, `native_bootstrap.py:313–340`.
- `_ExternalSourceLoader.create_module/exec_module`, lines 379–434.
- `_ExternalAdmission.find_spec`, failure intake and audit, lines 527–758.
- `_SourceSession.fail`, line 897 onward.
- Environment exact-key parsing, inventory checks and filesystem containment in `runtime_environment.py`.
- Canonical MJCF import and unchanged `trimesh.load_mesh` calls.

The existing finder already separates a pre-execution sentinel return from its eventual caller-visible exception. Extending this location with an identity-registered, single-use token is feasible without exempting arbitrary exceptions inside external admission.

The proposed contract correctly requires:

- Truthful present-source evidence, not a fabricated absent probe.
- Literal two-row mapping and inventory-backed source identity.
- Exact live session/controller/requester authority rather than parsed fields.
- No adapter loader execution.
- Confirmation before requester completion.
- Retention of resolution objects, original error/traceback and actual fallback objects.
- Sticky failure for all started modules, unrelated resolution failures, recording/adoption failures and failed fallback.
- Versioned descriptive evidence with no capability minted by parsing.
- Unchanged canonical importer, benchmark boundaries and product gates.

No additional substantive finding was established in these areas.

## Verification commands and actual results

Local inspection used UTF-8 `Get-Content`, `Select-Object`, `rg`, `Get-FileHash`, and:

```text
git status --short
git status --porcelain=v1
git rev-parse HEAD
git branch --show-current
git diff --stat a60c836e52b203c42bf29c58e439b2181cf425d9 65319cb728c3b9db0f6928c7aab6e3d95b6023e2
git diff --name-status a60c836e52b203c42bf29c58e439b2181cf425d9 65319cb728c3b9db0f6928c7aab6e3d95b6023e2
git diff --check a60c836e52b203c42bf29c58e439b2181cf425d9 65319cb728c3b9db0f6928c7aab6e3d95b6023e2
git diff --exit-code a60c836e52b203c42bf29c58e439b2181cf425d9 65319cb728c3b9db0f6928c7aab6e3d95b6023e2 -- AGENTS.md collab/README.md collab/REVIEW_PRINCIPLES.md optical_worker tests optics
```

Remote identity, cleanliness, digest and diff checks were repeated through `ssh tileops-dev`.

Dependency source inspection and both independently written probes were supplied over stdin using:

```text
timeout 45s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -
```

All three Python inspections/probes completed with exit 0. No scratch source file was created.

Executed current production-import-mechanics regressions:

```text
cd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 45s \
  /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest \
  -q -p no:cacheprovider \
  tests/unit/optics/test_runtime_import_admission.py::test_optional_absence_executes_no_loader_and_is_not_poison \
  tests/unit/optics/test_runtime_import_admission.py::test_swallowed_started_import_error_is_sticky_and_retains_partial_module
```

Result: **2 passed in 0.20s**, exit 0.

These tests use real importlib mechanics with unit doubles for source-session/guardian prerequisites. They validate the existing absent-versus-started-failure distinction; they do not validate the proposed supplement or real guarded lifecycle.

Initial default-encoding document output was garbled, and combined large reads were truncated. Relevant documents/ranges were reread with UTF-8 and smaller outputs. No display/tool issue was treated as a product failure or pass.

## Hygiene, limitations and remaining obligations

Observed probe identities:

- Pre-execution denial probe: PID `3368112`, parent timeout PID `3368111`.
- Original-failure observer: PID `3369314`, parent timeout PID `3369312`.

Final exact `ps -p 3368112,3368111,3369314,3369312 -o pid,ppid,lstart,args` returned only its header. Its no-process exit status is not a test failure. This is scoped process absence following successful commands, not guarded supervisor reap proof.

Pytest and the source inspector returned normally; their individual PIDs/start ticks were not captured. No complete process ledger or global-idle claim is made.

No package installation, site-package patch, file deletion, process signal, GPU reset, backend initialization, asset mutation, alternative mesh loader or candidate write occurred. Pytest used its existing temporary-fixture mechanism outside tracked files.

Not executed:

- The unimplemented v2 schema/token/fallback tests.
- Guarded canonical Go2 materialization.
- CPU canonical-output equivalence comparison.
- Actual a2 upload/readback or storage validation.
- GPU, dual-device, fault/hang/parent-loss or terminal-reap matrices.
- Complete dependency-closure audit or final custody-schema replay.

These remain implementation/code-approval obligations, not design-review successes. The two independently verified adapters are not proof that the remaining dependency closure needs no further work.

## Scope and disposition

Revise the bounded design to resolve F1, freeze the new version, and obtain fresh independent re-review. Do not modify current sticky failure handling to work around the ambiguity.

This report grants no implementation approval and no CPU-subset approval in place of whole C2d3 validation.

Whole C2d3 code review and its full real-device matrix remain pending. Complete C2d, C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, R2/host-payload and shipping remain closed. No final optical readiness, sample authority or benchmark result is established. **C4 remains incomplete.**
