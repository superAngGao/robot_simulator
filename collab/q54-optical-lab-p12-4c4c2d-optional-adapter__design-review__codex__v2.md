# C2d3 optional-adapter supplement — independent design re-review v2

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: design-review
Author: codex, fresh independent reviewer
Version: v2
Date: 2026-10-07
Status: accepted
Related Files: optional-adapter design-request v2/design-review v1; runtime-adoption design-request v2/design-review v2; optional-import gap-scan v1; native_bootstrap.py; runtime_environment.py; guarded_lifecycle.py; optics/mjcf_visual.py
Owner Summary: Approve the exact bounded design supplement. Prior F1 is closed at the design-contract level: configuration permits unrequested rows, issued denials cannot cross a successful segment boundary unconfirmed, and both Go2 cases require both confirmations at materialization completion. No new substantive finding was established. This is not implementation approval, GPU validation or a product-gate opening.

## Identity and verdict

Reviewer session: `/root/c4_c2d3_optional_adapter_design_review_v2`, fresh independent session supplied the review packet and prior findings, not the author conversation. No subdelegation.

Verdict: **Approve**, for the exact design and implementation scope stated below.

- Base: `65319cb728c3b9db0f6928c7aab6e3d95b6023e2`.
- Frozen and actual HEAD: `c60d6277ff94d31cdba9d10336b5a73fd78672ed`.
- Candidate: `collab/q54-optical-lab-p12-4c4c2d-optional-adapter__design-request__codex__v2.md`.
- Independently verified candidate SHA-256: `877da7e5476ff85e9273b591481447a5c76ba4636bcdee10100af8098ae675eb`.
- Archived v1 review SHA-256: `aaee0d3ff0c67ee56a1e6f74771e946e6af1d8a06d0d5fc8c477119b24045b40`, matching the supplied immutable report identity.
- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Remote repository: `/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`, detached at the same HEAD.
- Primary host only: SSH alias `tileops-dev`, `183.166.183.33:40031`.
- Interpreter: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`, observed CPython 3.12.13.

Initial and final local/remote working trees were clean. The base-to-HEAD delta contains only the 324-line v2 design and 270-line archived v1 report. Intermediate `c065ce2af0f940c1b2b14ee154c7093d575f3834` adds only that report. Local and remote diff checks passed; protected implementation, tests and collaboration rules have no candidate delta.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely. Inspected the candidate, v1-to-v2 diff, prior finding/reproduction, runtime-adoption predecessor requirements/review and optional-import gap scan. Historical reports and author claims were treated as context, not proof of this round's behavior.

## Findings and prior F1 disposition

No new substantive finding or unresolved blocking design gap was established.

**Prior F1 (P2): closed in the revised design.**

Relevant v2 locations are lines 181–189, 209–239 and 263–272. The correction addresses the actual production sequence rather than assuming a different setup order:

1. `native_bootstrap.py:993–1018` configures the external environment inside its own setup segment. It runs the source and external audits before changing the session to `runtime_configured`.
2. `materialize_native_source`, beginning at line 1020, requires that phase and opens a later, separate setup segment. Its canonical render-source call is at line 1059.
3. The Go2 branch in `workloads.py:66–90` reaches `optics/mjcf_visual.py:41`, where trimesh is imported. Configuration does not perform that import.
4. `guarded_lifecycle.py:193–239` completes/disarms only after the guarded body returns successfully and the live-ticket checks pass. Thus materialization's required confirmation check can and must run before leaving its body, not after COMPLETE/DISARMED.

Under v1's unconditional boundary rule, the first successful configuration boundary had no legal state for the configured rows. V2 now expressly permits `unrequested` there, with zero denial/fallback/execution events and no early requester import. General successful audits permit only `unrequested` or `fallback_confirmed`; a requested row cannot evade confirmation, and `resolution_verified`/`rejection_issued` are forbidden at those boundaries.

The Go2 obligation is not weakened into an optional audit: both literal rows must be present before either frozen Go2 materialization, and both must be issued and confirmed exactly once before successful materialization ends, before COMPLETE/DISARMED and runtime initialization. A required row remaining unrequested fails at that boundary. The requirement is derived from the actual fixed case selected through `materialize_native_source`, not a parsed receipt or coordinator boolean.

Both synthetic cases explicitly allow an empty table or configured unused rows remaining unrequested, including terminal audit. No synthetic path is required to import trimesh merely to consume a rule. Later audits cannot reset confirmed state, forget Go2 requirements or relabel failed/incomplete materialization as success. Separate nonnested segments and original deadlines remain required.

Section 6 now requires the original counterexample as a same-session production-path test: successful configuration with both rows unrequested and zero counters, followed by actual materialization and exactly one confirmation each, observing real separate guard sequences. It also requires pending-state, missing-row and unrequested-Go2 negatives plus read-only repeated audits. These are explicit future implementation obligations. Their successful execution is not claimed here because the supplement is not implemented.

## Independently verified dependency compatibility

Read the installed sources under `/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages` and independently hashed all five files listed by the design. Every digest matched:

- `trimesh/ray/ray_pyembree.py`: `8a447ee15074f43bd04d5bf0f35657c416e5eafba46c13c99188c4e4aa2a4a78`.
- `trimesh/ray/__init__.py`: `e653733282a543280fdbcb6721081736ea01f6c8d674ef76652744e3897f108d`.
- `trimesh/path/polygons.py`: `76f0e2deb52beda42f102e2a2a35a0b2e245e43fa56b1acbde0f954e0e45973a`.
- `trimesh/path/path.py`: `94feaf1e162850c0334e939682d0d00a12056f0dcb15f63cdb522e8db5a847fb`.
- `trimesh/exceptions.py`: `1b353dc4771bb93b8c330d7f0267203da481f10ec598251f29adf1d57bf16f70`.

The ray requester catches BaseException, installs ExceptionWrapper and sets has_embree=False. The path requester catches BaseException and stores ExceptionWrapper in its own `polygons` dictionary slot. ExceptionWrapper stores `(type(exception), exception.args)` and overrides attribute access. The proposed raw module-dictionary and `object.__getattribute__` checks are compatible with that implementation and avoid invoking the placeholder. The wrapper does not preserve original error identity; the separate retained original error remains necessary.

A stdlib-only source/resolver inspection found no spec for either `embreex` or `shapely` in the fixed installed root and loaded none of NumPy/Torch/Warp/trimesh. The installed CPython `_handle_fromlist` specially handles ModuleNotFoundError for a missing requested child, not a built-in ImportError. This supports the proposal's deliberate exception distinction.

Two fresh reviewer-written CPU probes then used real importlib resolution and installed trimesh 5.0.0:

| Probe | ray_pyembree | polygons | Top-level result |
| --- | --- | --- | --- |
| Ordinary loader observation | create, exec, ModuleNotFoundError for embreex | create, exec, ModuleNotFoundError for shapely | trimesh import completed with original fallbacks |
| Exact pre-execution denial | one request by trimesh.ray; exact ImportError fallback | one request by trimesh.path.path; exact ImportError fallback | trimesh import completed, has_embree=False |

The denial probe resolved actual non-package SourceFileLoader specs and absent prerequisites, then raised the design's exact built-in ImportError, including name and verified path, before returning any adapter spec to importlib. The event list contained only the two request events, no adapter create/exec events. Both final placeholders had the exact installed ExceptionWrapper type and exact `(ImportError, issued_args)` tuple. The original errors retained tracebacks. Neither adapter nor descendants remained in sys.modules, and neither Torch nor Warp was loaded. Both modes observed exactly one request for each adapter.

These are fresh compatibility observations, not production admission, guardian custody or canonical Go2 acceptance proofs. The observational finder is not the proposed implementation and was not substituted into a production worker.

## Other affected high-risk invariants

- **Truthful outcome and exact authority.** The design distinguishes a present disabled adapter from a genuinely absent probe and from a completed module. It freezes the two literal mappings, source identities and materialization phase; requires disjoint names/paths and admitted package parents; and forbids generic patterns, a third caller-selected row, installed-prerequisite execution or hash-only authority.
- **One importer and narrow non-poisoning edge.** `_CapturedSourceFinder.find_spec` already separates the absent sentinel from external admission's failure handler. The proposed identity-registered, same-session, single-use token can be consumed at this existing boundary without exempting arbitrary ImportError exceptions. Exact type alone is explicitly insufficient. Wrong/copy/foreign/reused tokens, second requests and prior failure reject.
- **Completion placement.** `_ExternalSourceLoader.exec_module` has a post-body, pre-completed-record insertion point. This permits fallback confirmation on held exact requester/class/placeholder objects before marking the requester complete. The actual frozen callers and fresh probe support the required causal path; copied error args alone do not mint authority.
- **Sticky started failures and retention.** Existing external fail intake retains available module roots and exceptions and forwards to source-session failure, which poisons and selects terminal handling. The supplement does not forgive adapter execution, parent fallback failure, unrelated resolution errors or recording/adoption failure. Partial resolution objects, errors/tracebacks, requesters and placeholders must remain in the guardian-owned graph before a subsequent fallible operation. No retry, COMPLETE after poison, second emergency synchronization or premature cleanup is introduced.
- **Schema and evidence migration.** V2 requires explicit schema versioning and the ordered disabled table, with no missing-field default or silent v1 interpretation. Inventory-backed identity remains the single source of file facts. The additional outcome has issuance/confirmation correlation and rejects nonzero execution, false completion and replayed confirmations. Parsing descriptive evidence cannot advance live state, register resources or authorize release.
- **Canonical path and scope.** Existing MJCF mesh loading remains `trimesh.load_mesh` with captured OBJ bytes, process=False and the fixed resolver. No alternative importer, dependency installation/fork, benchmark-boundary change or product request surface is proposed. Removal of obsolete optional rules still requires reviewed closure changes and canonical regressions; no permanent generic ignore-import-errors policy is created.

No additional substantive design finding was established in these areas. Exact implementation mechanics, production negative tests and full custody evidence remain reviewable obligations, not facts established merely by this design.

## Verification commands and results

Local inspection used UTF-8 Get-Content/Select-Object, rg, Get-FileHash and:

```text
git status --short
git status --porcelain=v1
git rev-parse HEAD
git branch --show-current
git log -3 --oneline
git show --stat c065ce2
git diff --stat 65319cb728c3b9db0f6928c7aab6e3d95b6023e2 HEAD
git diff --check 65319cb728c3b9db0f6928c7aab6e3d95b6023e2 HEAD
git diff --exit-code 65319cb728c3b9db0f6928c7aab6e3d95b6023e2 HEAD -- AGENTS.md collab/README.md collab/REVIEW_PRINCIPLES.md optical_worker tests optics
git diff --no-index <optional-adapter-design-v1> <optional-adapter-design-v2>
```

Remote identity, cleanliness, design SHA-256 and diff-check were independently repeated through `ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev`. No fallback host was used.

The source inspection and both fresh probes were supplied over stdin, without scratch source files:

```text
timeout 45s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -
timeout 45s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B - original
timeout 45s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B - deny
```

All three completed with exit 0. No backend initialization occurred.

Executed unchanged current production-import-mechanics regressions on the primary host:

```text
cd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 45s \
  /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest \
  -q -p no:cacheprovider \
  tests/unit/optics/test_runtime_import_admission.py::test_optional_absence_executes_no_loader_and_is_not_poison \
  tests/unit/optics/test_runtime_import_admission.py::test_swallowed_started_import_error_is_sticky_and_retains_partial_module
```

Result: **2 passed in 0.20s**, exit 0. Inspection confirmed these use real importlib mechanics with source-session/guardian unit doubles. They check the existing absent-versus-started-failure distinction; they do not test the unimplemented optional-adapter extension or real guarded lifecycle.

Initial default-encoding display was garbled, and some combined reads were truncated. Relevant instruction/doc/source ranges were reread with explicit UTF-8 and smaller output. The no-index diff emitted normal Windows line-ending advisories. These were tool/display issues, not product failures or passes.

## Hygiene and limitations

Observed fresh process identities:

| Activity | PID | Parent timeout PID | Start ticks |
| --- | ---: | ---: | ---: |
| Source/resolver inspection | 3376189 | 3376188 | 225263355 |
| Ordinary import observer | 3376638 | 3376637 | 225266203 |
| Pre-execution denial probe | 3376836 | 3376835 | 225266309 |

Final exact `ps -p 3376189,3376188,3376638,3376637,3376836,3376835 -o pid,ppid,lstart,args` returned only the header. Its no-process exit status 1 is not a test failure. This is scoped absence following normal command completion, not proof of guarded supervisor reap behavior. Pytest returned normally; its individual PID/start ticks were not captured. No complete process ledger or global-idle claim is made.

No candidate edit, site-package mutation, dependency installation, file deletion, process signal, GPU reset, backend import, asset mutation or GPU work occurred. Pytest used its ordinary temporary fixtures outside tracked files. The sole intentional persistent reviewer output is this report outside the candidate repository. Final candidate HEAD, digest and cleanliness remained unchanged.

Not executed: unimplemented v2 schema/token/fallback and same-session production tests; guarded canonical Go2 materialization; canonical CPU-output equivalence; actual a2 upload/readback/storage validation; GPU, multi-device, failure/hang/parent-loss and terminal-reap matrices; complete remaining dependency-closure/native exception-safety audit; final custody-schema replay. These remain mandatory implementation/code-approval evidence. The two observed adapters do not prove completeness of the rest of the dependency closure.

## Approval scope and remaining gates

Approval binds only optional-adapter design v2 SHA-256 `877da7e5476ff85e9273b591481447a5c76ba4636bcdee10100af8098ae675eb` at `c60d6277ff94d31cdba9d10336b5a73fd78672ed`.

It permits implementation of this bounded supplement: the two exact present Python trimesh adapters may be deliberately denied before any create/exec/body work, with verified absent prerequisites, truthful disabled_optional evidence, exact live identity/token authority, original-parent fallback confirmation, phase-specific boundary rules and retained terminal ownership. The existing captured finder/controller remains the sole path. Every started-module failure remains sticky. All other runtime-adoption v2 requirements remain mandatory.

This is not approval of current or future C2d3 code, not acceptance of a CPU subset in place of full validation, and not proof that Go2/runtime admission now succeeds. Full C2d3 real-device matrices, native custody, environment/source-policy freeze and independent code review remain pending.

Complete C2d, C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, R2/host-payload and shipping remain closed. No final optical readiness, prepared sample authority or benchmark result is established. **C4 remains incomplete.**

The unchanged report may be archived under the evidence-only exception. Such recording cannot alter the approved design, verdict, requirements, scope or gates.
