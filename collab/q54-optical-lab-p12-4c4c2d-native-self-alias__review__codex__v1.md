# C2d3 native creator-self alias design review v1

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: design-review
Author: codex, independent reviewer
Version: v1
Date: 2026-10-08
Status: accepted
Related Files: native-self-alias design-request v1 and gap-scan v1; runtime-adoption design-request v2; optional-adapter design-request v2; optical_worker/runtime_environment.py; optical_worker/native_bootstrap.py
Owner Summary: Approve the bounded design supplement only. Its restricted empty selector truthfully denotes the already-held native creator, with explicit publication timing, immutable identity, read-only audits and failure custody. This is not approval of the current implementation, a CPU subset, a deployment closure or whole C2d3.

## Verdict and exact scope

Verdict: **Approve**, design-only.

No substantive blocking design findings were established. Approval permits implementing this supplement and its required tests; it does not certify that the unchanged importer already implements it. In particular, the late-failure custody defect remains an implementation obligation under the accepted parent contract.

Reviewer session: `/root/c4_c2d3_self_alias_design_review_v1`. This was a fresh independent session; I did not author the candidate and did not subdelegate.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Base: `56acbd375180f5ea89cf8fca3e3e7f854dbc4954`.
- Reviewed HEAD: `29077305468e2cd1147aef4c80b502548629f174`.
- Design: `collab/q54-optical-lab-p12-4c4c2d-native-self-alias__design-request__codex__v1.md`.
- Design SHA-256: `f50bd95ddcd5c3f8eff8d50a62edc96d6c4742283879743867933e7b88cec17f`.
- Gap-scan SHA-256 independently matched: `235ae6a5b085c1cfe32e92e0bd4992a8192028275ba631ea20c56ecca90b296e`.
- Initial and final working trees: clean. Base-to-HEAD contains exactly the two stated new documents, 428 inserted lines, no code or shared-policy edits. `git diff --check` passed.

Read AGENTS.md, collab/README.md and REVIEW_PRINCIPLES.md fully, the two accepted requirement documents, the candidate design and gap scan. Large combined displays were reread in smaller chunks where truncated; README was reread explicitly as UTF-8 after the first display's punctuation decoding issue.

## High-risk design checks

1. **Representability without false provenance.** The current schema at runtime_environment.py:220-244 rejects an empty path, while native_bootstrap.py:951-994 begins resolution from the exact retained creator module. The proposed empty list therefore denotes a genuinely missing relation without requiring a second importer, synthetic self attribute, second file record, or parent-attributed creation. The admission file record already holds the native create return before explicit adoption; self-alias checks can use that same object. The new relation is restricted to module_alias, equal file creator/publisher, a non-package extension under a site root, matching phase and disjoint name. Nonempty attribute semantics and disabled/absent outcomes remain separate.

2. **Actual boundary and nested import feasibility.** Inspected _ExternalSourceLoader.create_module/exec_module, finder resolution and loader/spec custody (native_bootstrap.py:349-661). There are usable boundaries immediately before/after delegated create and immediately after delegated exec. Reconciliation already precedes creator completion. The supplement requires absence including None at pre-create, post-create and pre-exec; then exact qualified/short-key equality with the held returned root, actual executing record and publisher. It explicitly forbids create-time publication and checks nested creators at their own return. It preserves qualified metadata and does not invent short-name file/spec provenance. This is delegate-boundary attribution within the pinned dependency trust model, not native instruction tracing or a native sandbox.

3. **No late authority through audit.** Current audit calls the same admission-capable reconciliation method (native_bootstrap.py:1043-1051). The supplement expressly disallows this behavior for both self and existing attribute publications: only publisher completion can first admit a row; audits require an existing record and preserve counts/identity. It also distinguishes configured-but-unrequested publishers from completed ones. Removal of a publication record must fail, not recreate evidence. This closes a concrete counterexample rather than relying on a copied hash or permissive rescan.

4. **Failure ownership and terminal behavior.** Inspected _ExternalAdmission.fail/hold, _SourceSession.fail and the guarded lifecycle's host adoption, termination and quarantine paths. The source/session and admission object can remain strongly held while dependency code unwinds after quarantine; terminate returns evidence, so a swallowed inner failure does not imply all enclosing Python code has ceased. The candidate correctly separates monotonic late owner/error intake from one-time termination and prohibits reusing normal adoption to reopen a poisoned controller. It keeps UNKNOWN/incomplete native failures conservative, the original deadline, no new ARM/sync/COMPLETE and no normal cleanup. Native pre-return internal exception safety remains a trusted dependency/process obligation, not a property purportedly repaired by Python retention.

5. **Schema/evidence migration and trust boundary.** v3 explicitly rejects old versions rather than interpreting v2 empty paths permissively. Historical artifacts remain unchanged, active fixtures/digests must be versioned, and changed captured source policy/catalogue bytes must be repinned. The four literal materialization mappings are bounded proposal scope, not dynamic discovery or proof of complete Go2 closure. Evidence distinguishes selector kinds and points to real file-record identity/sequence and observed publication equality; alias file/spec/loader facts stay not_applicable. Parsing cannot mint live authority. Final implementation evidence layout and replay validation remain mandatory under the parent contract.

6. **Verification and phase gates.** Section 5 requires positive and adversarial schema/importer tests, exact retained-root equality, early/missing/replaced/unlisted aliases, preexisting None, record deletion and audit idempotence, recording/adoption failures and late nested custody. It requires cold real observations through the captured production importer, not just ordinary observation wrappers. It retains the complete both-lane/canonical/both-Go2/upload/readback/multi-device/fault/hang/reap matrix. There is no CPU-subset acceptance escape. Cleanup remains governed by the existing selected-backend ledger, without introducing a second importer or an unbounded generic exception policy.

## Fresh executed checks

Read-only inspection used UTF-8 Get-Content, Select-Object and rg, including the actual schema, loader/finder/admission methods, source-session failure forwarding, guarded lifecycle methods, current importer/schema tests and all three prior reviewer probe sources named in the gap report. Prior observer results were treated as prior evidence, not executions performed by this session.

Repository identity/integrity commands:

```text
git status --short
git status --porcelain=v1
git rev-parse HEAD
git branch --show-current
git diff --stat 56acbd375180f5ea89cf8fca3e3e7f854dbc4954..HEAD
git diff --name-status 56acbd375180f5ea89cf8fca3e3e7f854dbc4954..HEAD
git diff --check 56acbd375180f5ea89cf8fca3e3e7f854dbc4954..HEAD
Get-FileHash -Algorithm SHA256 <design>,<gap-scan>
```

Local environment check:

```text
C:/Python314/python.exe -I -S -B -c "import sys, importlib.util; print(sys.version); print(importlib.util.find_spec('pytest'))"
```

Observed CPython 3.14.0, pytest unavailable on that isolated path. No install was attempted. I wrote and executed one small independent mechanics probe outside the candidate:

```text
C:/Python314/python.exe -I -S -B C:/Users/AG/新建文件夹/c4_assets/c2d3-self-alias-design-review-v1-probe.py
```

Exit 0. Observations through actual production parser/methods, with explicit unit custody prerequisites:

- A natural non-package extension creator-self row with attributes=[] is rejected by the current v2 schema: `publication requires a fixed nonempty attribute path`.
- Existing attribute reconciliation recreates a removed publication record and emits a second event (count 2). This directly corroborates the audit-callable method's current behavior; the probe did not pretend to execute a real guarded audit.
- After an initial fail(), adding a new module object and calling fail() with a second error retains neither that late identity nor the outer error. This directly corroborates the current early-return defect; it is not a real guarded/native failure run.
- No NumPy/SciPy/Torch/Warp imports or native delegates were executed.

These are expected counterexamples in the unchanged implementation, not new defects in the proposed requirements. They confirm concrete implementation work that the design explicitly requires; they do not prove its future implementation correct.

Probe SHA-256: `369a492db4bf9b39fadb413d631cb47ae07684bec7a60c16bf7b9c47f8564ca0`.

The author additionally reported a separate actual-cold-custody F2 repro during this review. I did not execute or rely on that report as independent evidence and do not claim its outcomes verified here.

## Process and workspace hygiene

The probe ran foreground as Python PID 34028 and completed normally. Afterwards:

```text
Get-Process -Id 34028 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,StartTime
```

returned no process. No background task, remote SSH job, native workload or GPU process was launched. This is scoped OS absence, not the product supervisor's wait/reap proof. The probe modified only its own temporary in-process module table, restored its three keys in finally, and created no fixture files or caches. -B disabled bytecode writes. Persistent reviewer artifacts are only the probe and this report under c4_assets, outside the candidate. No candidate, dependency, asset or shared instruction edits, installs, commits, process signals or material deletions occurred.

## Limitations and remaining obligations

No new real SciPy alias observation, generated native-source exception audit, captured Go2 admission, whole pytest suite, guardian/supervisor runtime test, GPU initialization/upload/readback, dual-device test, hang/reap test or GPU residual check was executed here. The local isolated interpreter differs from the pinned deployment and has no pytest on its isolated path. That limits runtime evidence, not this document-only design review. The four actual dependency behaviors still require the specified cold production-importer proof and reviewed complete pinned closure before code acceptance.

F2 repair, no-late-admission implementation, v3 schema/fixtures, custody evidence validator/replay tests, source repinning and the entire original C2d3 real matrix remain open implementation requirements, not optional comments and not waived by this approval.

Whole C2d3 and full C2d remain unapproved. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain closed. No sample authority, final optical readiness, performance result, deployment activation or whole-stage approval is issued.
