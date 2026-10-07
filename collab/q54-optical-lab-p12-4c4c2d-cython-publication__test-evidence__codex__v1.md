# C2d3 captured unrecorded-module evidence and review handoff

Initiative: q54-optical-lab-p12-4c4c2d-cython-publication
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-08
Status: draft
Related Files: runtime-adoption design-request v2; native-module-name
implementation-note v1; optical_worker/runtime_environment.py;
optical_worker/native_bootstrap.py; tests/support/native_runtime_process.py;
tests/unit/optics/test_runtime_failure_observations.py
Owner Summary: Actual captured Go2 failure now identifies three fileless
Cython modules held in the failure graph. This is diagnostic evidence, not
admission, a new ownership rule, independent review or C2d3 acceptance.

## Frozen candidate and diagnostic delta

Base: 2e41e16215434463ddd8e1bcdbe178bc65bd93cb.
Candidate: c8b404c54e3135d615727ee6e4a43c63822ba257.
Local: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design.
Primary183: /data2/shared/ang.gao/robot_sim_c2d3_binding_wip.
Both trees were clean at the candidate. Production still equals a50b227;
commits 08f1a25 and c8b404c change only failure diagnostics/tests. No environment
table, production admission, source policy, catalog or activation changed.

The helper records up to256 actual sys.modules entries absent from the
admission baseline/records, with total/truncation, raw metadata and exact
identity membership in retained failure owners. It deliberately does not
classify them as forbidden: newly imported stdlib/repository modules can also
appear. No arbitrary module __getattr__ or admission is performed. Tests cover
same-metadata distinct objects, None, already-recorded exclusions, no registry
mutation, and bounded truncation. These tests do not prove creator provenance.

Author tests: local CPython3.14 helper tests2passed, latest rerun0.08s; primary183
robot_sim helper plus test_native_runtime_driver_process6passed13.04s under90s
outer bound, no timeout. These overlapping counts are not summed and are not
independent approval. Local Ruff and git diff --check passed.

## Actual guarded Go2 attempt v8

The existing fixed sacrificial driver used the unchanged frozen Go2 960x640
asset/environment, Warp lane, target1, operation120000ms and ACK1000ms.
No timeout increase, alternative importer/uploader or automatic permission.
Invoker run_c2d3_go2_guarded_author_v8.py SHA-256:
7bc310e2c18e94e7361ec6e535989ce4b3f48580c4e3d6443557c3707e5a4b75.
Parent3669870, supervisor3669902, worker3669903. Worker exited74; supervisor
reported EXITED/complete, no deadline expiration and no kill request. Invoker
asserted failure (exit1), as intended for an unsuccessful guarded attempt.

Artifact root on183:
/data2/shared/ang.gao/c2d3-c8b404c-go2-960-guarded-author-v8.

| Artifact | SHA-256 |
| --- | --- |
| external_failure.json | 93fa4f97c92dd324d190f7839e3213dd048020b82b061f2b92f7cad2af432a1d |
| failure.json | e95ad9c6c1efbdc031f9c8a92aa2bb676f95a77c87fc9bef075f8e6309a62f9a |
| worker.json | fe3a7556df870d287b5291f672ddf60d8fdd0a3de2ed9cbeaefd9a7d305349a2 |
| custody_artifact.json | 42c76f61e9a083b8a1820c0ee5ec2c4e4329f11d6f28204409bfaf29d956d0e1 |
| supervisor_stdout.json | cde7946e60b6bacc46d7a0bf38e11bb56368f9805fd7d849e0666aa38fa8a9a9 |
| request.json | af4eee4c2a97fa7605c1032badace1307e0a9cdc886a75aa6f742eed6e633adb |
| identity.json | a7a600a53f892e86ac453018a27514575280a15372dcdac4f0d8dbd9839a225e |

Failure remains SourceBootstrapError/unlisted module publication in canonical
materialization. unrecorded_modules has113entries, truncated=false. Among them:

| Actual sys.modules key | Exact module held in failure graph | File/spec origin/loader |
| --- | --- | --- |
| cython_runtime | true | null/null/null |
| _cython_3_2_5 | true | null/null/null |
| _cython_3_2_8 | true | null/null/null |

Their raw module names equal those keys; packages/specs are null. This identifies
actual rejected-scope objects, not the instruction that created them or a
Python-visible owner path. Other113entries must not be wholesale allowlisted.

Worker lifecycle: setup sequence2 failed, full_quarantine,
quarantined_process_exit, guardian transferred, release not_authorized;
emergency_sync_attempt_count=0 and terminal_arm_attempt_count=0. Dirty ordinals
are[1], unknown_device_scope=false. Do not recast this as unknown-device evidence
or proof of successful GPU initialization. No successful canonical boundary,
runtime initialization, input upload/readback or optical sample resulted.

## Code facts and unresolved contract questions

Current runtime_environment publication rows allow created_module/module_alias
with an exact file creator/publisher and fixed raw attribute path. Empty paths
are restricted to a creator-self native alias. native_bootstrap resolves those
paths using vars(), then checks exact object identity. The three objects above
cannot be admitted merely by filling their observed keys into that table.

The prior ordinary CPU observer (see native-module-name implementation-note v1)
records native boundary sys.modules deltas. Nested deltas can repeat a child's
publication; an outer return is not exact creator proof. Its observations must
not silently become frozen ownership declarations.

Primary-source corroboration, not installed-wheel build attestation: Cython3.2.8
generates a module-state reference obtained by adding/retrieving cython_runtime
via __Pyx_PyImport_AddModuleRef. This is distinct from exporting a Python module
attribute. See [Cython ModuleNode.py](https://github.com/cython/cython/blob/3.2.8/Cython/Compiler/ModuleNode.py#L3484-L3488).
The version-like names alone do not establish the source used to build either
installed binary, nor a complete native ownership proof.

Before changing production, independent gap review must settle:

1. Whether a separately versioned native-global publication relation is needed,
   versus an existing mechanically demonstrable creator attribute relation.
2. Exact first-observation boundary under nested native create/exec, scope/phase
   identity, required absence before creation, later reuse by other extensions,
   and how creator/observer facts are named without overclaiming causation.
3. Holding exact objects before fallible reconciliation; partial native failure,
   replacement, removal, early insertion, wrong boundary and sticky failure.
4. Which native internals remain the explicitly trusted deployment obligation;
   no Python wrapper, hash, prefix or version-like name may imply a native sandbox.
5. Literal deployment rules, versioned evidence and negative tests required for
   the real both-lane/canonical workloads, without granting unobserved modules.

These are open design obligations, not an approved new schema. Keep current
failure closed. No new architecture is implemented by this evidence record.

## Independent session limitation and continuation packet

Requested a fresh fork_turns=none reviewer named
/root/c4_c2d3_cython_publication_gap_review_v1 with exact candidate, accepted
designs, artifacts, scope and instructions. Creation failed with
"agent thread limit reached". No review session or verdict exists. Do not
reuse a prior reviewer or relabel author inspection as independent review.
Resume with a fresh independent session when available; candidate packet is
this record plus the named accepted predecessor designs and artifact paths.
The diagnostic delta itself also remains subject to review.

At2026-10-08T04:10:16+08 scoped ps on183 showed no ang.gao Python/pytest/worker/
supervisor process. NVML compute-process listing contained only pre-existing
MPS server2238410. No broad kill, reset, dependency install or101 fallback.
This is this round's process-reap check, not resolution of the historical101 hang.

C2d3 remains WIP. Full mandatory both-lane/real-Go2/upload/readback/failure/hang/
dualGPU matrix and final versioned runtime-custody schema are still outstanding.
C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host
payload and shipping remain closed. No C4 completion or code approval claimed.
